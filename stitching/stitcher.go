package stitching

import (
	"errors"
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Errors returned by the stitching pipeline.
var (
	// ErrNoImages is returned when Stitch is called with no images.
	ErrNoImages = errors.New("stitching: no images to stitch")
	// ErrChannelMismatch is returned when the inputs do not share a channel
	// count.
	ErrChannelMismatch = errors.New("stitching: all images must have the same channel count")
	// ErrNotEnoughMatches is returned when two images cannot be related because
	// too few reliable correspondences survive matching.
	ErrNotEnoughMatches = errors.New("stitching: not enough matches to estimate a transform")
	// ErrHomography is returned when a homography could not be estimated from the
	// available correspondences.
	ErrHomography = errors.New("stitching: failed to estimate homography")
	// ErrCanvasTooLarge is a guard against a degenerate transform blowing the
	// output canvas up to an unreasonable size.
	ErrCanvasTooLarge = errors.New("stitching: composed canvas is unreasonably large")
)

// Stitcher builds a panorama from overlapping images. It runs the classic
// feature-based pipeline — corner detection, patch description, ratio-test
// matching, RANSAC homography estimation, warping and blending — and every stage
// is deterministic given the same inputs. The zero value is not ready to use;
// construct one with [NewStitcher] and adjust the exported fields as needed.
type Stitcher struct {
	// MaxCorners caps the number of corners detected per image (<=0 means no
	// cap).
	MaxCorners int
	// QualityLevel is the Shi–Tomasi quality threshold relative to the strongest
	// corner.
	QualityLevel float64
	// MinDistance is the minimum spacing, in pixels, between detected corners.
	MinDistance float64
	// BlockSize is the structure-tensor window size for corner detection.
	BlockSize int
	// DescriptorRadius sets the patch descriptor half-size; the window is
	// (2·DescriptorRadius+1)² samples.
	DescriptorRadius int
	// RatioThreshold is Lowe's ratio-test cutoff for accepting a match.
	RatioThreshold float64
	// RANSAC configures the robust homography estimator.
	RANSAC RANSACParams
	// MinMatches is the minimum number of ratio-test matches required before
	// attempting homography estimation.
	MinMatches int
	// MaxCanvasArea guards against runaway output sizes; a composed canvas larger
	// than this many pixels aborts with [ErrCanvasTooLarge]. Zero disables the
	// check.
	MaxCanvasArea int
	// Blender combines overlapping warped images. Defaults to [Feather].
	Blender Blender
}

// NewStitcher returns a Stitcher preconfigured with robust defaults for small to
// medium overlapping images and a feather blender.
func NewStitcher() *Stitcher {
	return &Stitcher{
		MaxCorners:       500,
		QualityLevel:     0.01,
		MinDistance:      5,
		BlockSize:        3,
		DescriptorRadius: 4,
		RatioThreshold:   0.8,
		RANSAC: RANSACParams{
			Iterations:      2000,
			ReprojThreshold: 3.0,
			Seed:            1,
		},
		MinMatches:    8,
		MaxCanvasArea: 64 << 20, // 64M pixels
		Blender:       Feather{},
	}
}

// blender returns the configured blender or the feather default.
func (s *Stitcher) blender() Blender {
	if s.Blender == nil {
		return Feather{}
	}
	return s.Blender
}

// EstimateTransform estimates the homography that maps the coordinate system of
// imgB into that of imgA, so that warping imgB by the result overlays it onto
// imgA. It detects and describes corners in both images, matches them with the
// ratio test, and fits the homography with RANSAC. It returns [ErrNotEnoughMatches]
// or [ErrHomography] when the images cannot be reliably related.
func (s *Stitcher) EstimateTransform(imgA, imgB *cv.Mat) (cv.PerspectiveMatrix, error) {
	grayA := toGray(imgA)
	grayB := toGray(imgB)
	featsA := detectAndDescribe(grayA, s.MaxCorners, s.QualityLevel, s.MinDistance, s.BlockSize, s.DescriptorRadius)
	featsB := detectAndDescribe(grayB, s.MaxCorners, s.QualityLevel, s.MinDistance, s.BlockSize, s.DescriptorRadius)
	if len(featsA) < s.MinMatches || len(featsB) < s.MinMatches {
		return cv.PerspectiveMatrix{}, ErrNotEnoughMatches
	}
	// Query B against A: each match relates a point in B to a point in A.
	matches := matchFeatures(featsB, featsA, s.RatioThreshold)
	if len(matches) < s.MinMatches {
		return cv.PerspectiveMatrix{}, ErrNotEnoughMatches
	}
	src := make([]pointF, len(matches)) // points in B
	dst := make([]pointF, len(matches)) // corresponding points in A
	for i, m := range matches {
		fb := featsB[m.queryIdx]
		fa := featsA[m.trainIdx]
		src[i] = pointF{float64(fb.x), float64(fb.y)}
		dst[i] = pointF{float64(fa.x), float64(fa.y)}
	}
	h, _, ok := estimateHomographyRANSAC(src, dst, s.RANSAC)
	if !ok {
		return cv.PerspectiveMatrix{}, ErrHomography
	}
	return h, nil
}

// Stitch assembles the images into a single panorama. The images are assumed to
// be given in adjacency order (each overlapping the next), which is the common
// left-to-right or top-to-bottom capture order. Pairwise homographies are
// estimated between neighbours and chained into the coordinate frame of the
// first image; every image is then warped into a shared canvas and blended.
// A single image is returned as a clone. It returns an error if the images have
// mismatched channels or a neighbouring pair cannot be related.
func (s *Stitcher) Stitch(images []*cv.Mat) (*cv.Mat, error) {
	if len(images) == 0 {
		return nil, ErrNoImages
	}
	if len(images) == 1 {
		return images[0].Clone(), nil
	}
	channels := images[0].Channels
	for _, im := range images {
		if im.Channels != channels {
			return nil, ErrChannelMismatch
		}
	}
	transforms, err := s.chainTransforms(images)
	if err != nil {
		return nil, err
	}
	return s.ComposePanorama(images, transforms)
}

// chainTransforms estimates the neighbour-to-neighbour homographies and composes
// them so that transforms[i] maps image i into the coordinate frame of image 0.
func (s *Stitcher) chainTransforms(images []*cv.Mat) ([]cv.PerspectiveMatrix, error) {
	transforms := make([]cv.PerspectiveMatrix, len(images))
	transforms[0] = identityH()
	for i := 1; i < len(images); i++ {
		// h maps image i into the frame of image i-1.
		h, err := s.EstimateTransform(images[i-1], images[i])
		if err != nil {
			return nil, fmt.Errorf("stitching: pair %d-%d: %w", i-1, i, err)
		}
		transforms[i] = matMul3(transforms[i-1], h)
	}
	return transforms, nil
}

// ComposePanorama warps each image by its transform into a common canvas and
// blends the results. transforms[i] must map image i into a shared reference
// frame (as produced by chaining [Stitcher.EstimateTransform]); the canvas is
// sized to the union of all warped image bounds and translated so it starts at
// the origin. The number of transforms must match the number of images.
func (s *Stitcher) ComposePanorama(images []*cv.Mat, transforms []cv.PerspectiveMatrix) (*cv.Mat, error) {
	if len(images) == 0 {
		return nil, ErrNoImages
	}
	if len(images) != len(transforms) {
		return nil, errors.New("stitching: images and transforms length mismatch")
	}
	channels := images[0].Channels

	// Union bounds of every warped image's corners.
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for i, im := range images {
		for _, corner := range [][2]float64{
			{0, 0}, {float64(im.Cols), 0}, {float64(im.Cols), float64(im.Rows)}, {0, float64(im.Rows)},
		} {
			x, y, ok := applyH(transforms[i], corner[0], corner[1])
			if !ok {
				return nil, ErrHomography
			}
			minX, minY = math.Min(minX, x), math.Min(minY, y)
			maxX, maxY = math.Max(maxX, x), math.Max(maxY, y)
		}
	}
	// Snap bounds that are within a tiny epsilon of an integer before rounding,
	// so floating-point noise near exact pixel edges does not inflate the canvas
	// by a whole row or column.
	const eps = 1e-6
	loX := math.Floor(minX + eps)
	loY := math.Floor(minY + eps)
	hiX := math.Ceil(maxX - eps)
	hiY := math.Ceil(maxY - eps)
	offX := -loX
	offY := -loY
	cols := int(hiX - loX)
	rows := int(hiY - loY)
	if cols <= 0 || rows <= 0 {
		return nil, ErrHomography
	}
	if s.MaxCanvasArea > 0 && rows*cols > s.MaxCanvasArea {
		return nil, ErrCanvasTooLarge
	}

	offset := translationH(offX, offY)
	layers := make([]Layer, len(images))
	for i, im := range images {
		final := matMul3(offset, transforms[i])
		color := cv.WarpPerspective(im, final, cols, rows, cv.InterLinear)
		// Warp the source feather ramp with nearest-neighbour so the weight (and
		// hence coverage) stays crisp and non-zero at the image interior.
		srcW := featherWeight(im.Rows, im.Cols)
		warpedW := cv.WarpPerspective(srcW, final, cols, rows, cv.InterNearest)
		weight := cv.NewFloatMat(rows, cols)
		for p := range weight.Data {
			weight.Data[p] = float64(warpedW.Data[p])
		}
		layers[i] = Layer{Color: color, Weight: weight}
	}
	return s.blender().Blend(layers, rows, cols, channels)
}

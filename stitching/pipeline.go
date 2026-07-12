package stitching

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// StitchMode selects the geometric model a [Pipeline] assumes for the scene.
type StitchMode int

const (
	// ModePanorama assumes the images were taken by a camera rotating about its
	// optical centre (a true panorama) and relates them with full projective
	// homographies. A surface [Warper] may be set to reproject onto a cylinder or
	// sphere for wide fields of view.
	ModePanorama StitchMode = iota
	// ModeScans assumes a flat scene photographed from shifted positions (for
	// example a scanned document or whiteboard) so the images are related in a
	// plane. It defaults to a [PlaneWarper].
	ModeScans
)

// Pipeline is a configurable, higher-level panorama builder that wires together
// the surface [Warper], [ExposureCompensator], [SeamFinder] and [Blender] stages
// around the feature-based [Stitcher]. It mirrors OpenCV's cv::Stitcher: construct
// one with [NewPipeline] for a mode, swap individual stages with the SetX methods,
// and call [Pipeline.Stitch]. Unset stages fall back to sensible no-op or default
// implementations, so a freshly created Pipeline already stitches.
type Pipeline struct {
	mode     StitchMode
	stitcher *Stitcher
	warper   Warper
	seam     SeamFinder
	exposure ExposureCompensator
	blender  Blender

	// Focal is the focal length, in pixels, used when a curved [Warper] is set.
	// Zero selects a default derived from the image width.
	Focal float64
	// MaxCanvasArea guards against a runaway output size; zero disables the check.
	MaxCanvasArea int
}

// NewPipeline returns a Pipeline for the given mode with default stages: no
// exposure compensation, no seam finding, feather blending, and (for
// [ModeScans]) a plane warper.
func NewPipeline(mode StitchMode) *Pipeline {
	p := &Pipeline{
		mode:          mode,
		stitcher:      NewStitcher(),
		seam:          NoSeamFinder{},
		exposure:      NoExposureCompensator{},
		blender:       Feather{},
		MaxCanvasArea: 64 << 20,
	}
	if mode == ModeScans {
		p.warper = PlaneWarper{}
	}
	return p
}

// Stitcher returns the underlying feature-based [Stitcher], whose detection,
// matching and RANSAC parameters may be tuned directly.
func (p *Pipeline) Stitcher() *Stitcher { return p.stitcher }

// Mode returns the pipeline's geometric mode.
func (p *Pipeline) Mode() StitchMode { return p.mode }

// SetWarper selects the surface warper. Passing a [CylindricalWarper] or
// [SphericalWarper] reprojects every image onto that surface before stitching,
// which is what makes very wide fields of view join without extreme stretching.
func (p *Pipeline) SetWarper(w Warper) { p.warper = w }

// SetSeamFinder selects the seam finder used to partition overlaps before
// blending.
func (p *Pipeline) SetSeamFinder(s SeamFinder) {
	if s == nil {
		s = NoSeamFinder{}
	}
	p.seam = s
}

// SetExposureCompensator selects the exposure compensator applied to the warped
// images before blending.
func (p *Pipeline) SetExposureCompensator(e ExposureCompensator) {
	if e == nil {
		e = NoExposureCompensator{}
	}
	p.exposure = e
}

// SetBlender selects the blender that merges the warped, compensated, seam-cut
// images into the final panorama.
func (p *Pipeline) SetBlender(b Blender) {
	if b == nil {
		b = Feather{}
	}
	p.blender = b
}

// Stitch assembles images into a panorama, running the full configured pipeline:
// optional surface warping, pairwise transform estimation and chaining, warping
// into a shared canvas, exposure compensation, seam finding and blending. Images
// must be given in adjacency order and share a channel count.
func (p *Pipeline) Stitch(images []*cv.Mat) (*cv.Mat, error) {
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
	prepared := p.prepareImages(images)
	transforms, err := p.stitcher.chainTransforms(prepared)
	if err != nil {
		return nil, err
	}
	return p.compose(prepared, transforms, channels)
}

// prepareImages reprojects each image onto the configured surface when a curved
// warper is set; otherwise it returns the images unchanged.
func (p *Pipeline) prepareImages(images []*cv.Mat) []*cv.Mat {
	if p.warper == nil || p.warper.Name() == "plane" {
		return images
	}
	out := make([]*cv.Mat, len(images))
	for i, im := range images {
		focal := p.Focal
		if focal <= 0 {
			focal = float64(im.Cols)
		}
		warped, _ := p.warper.Warp(im, focal)
		out[i] = warped
	}
	return out
}

// compose warps every prepared image into a common canvas and merges them,
// applying exposure compensation and seam finding along the way.
func (p *Pipeline) compose(images []*cv.Mat, transforms []cv.PerspectiveMatrix, channels int) (*cv.Mat, error) {
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
	const eps = 1e-6
	loX := math.Floor(minX + eps)
	loY := math.Floor(minY + eps)
	hiX := math.Ceil(maxX - eps)
	hiY := math.Ceil(maxY - eps)
	cols := int(hiX - loX)
	rows := int(hiY - loY)
	if cols <= 0 || rows <= 0 {
		return nil, ErrHomography
	}
	if p.MaxCanvasArea > 0 && rows*cols > p.MaxCanvasArea {
		return nil, ErrCanvasTooLarge
	}

	offset := translationH(-loX, -loY)
	n := len(images)
	colors := make([]*cv.Mat, n)
	coverage := make([]*cv.FloatMat, n)
	feathers := make([]*cv.FloatMat, n)
	corners := make([]image.Point, n)
	for i, im := range images {
		final := matMul3(offset, transforms[i])
		colors[i] = cv.WarpPerspective(im, final, cols, rows, cv.InterLinear)
		ones := fullMaskMat(im.Rows, im.Cols)
		warpCov := cv.WarpPerspective(ones, final, cols, rows, cv.InterNearest)
		warpFea := cv.WarpPerspective(featherWeight(im.Rows, im.Cols), final, cols, rows, cv.InterNearest)
		coverage[i] = cv.NewFloatMat(rows, cols)
		feathers[i] = cv.NewFloatMat(rows, cols)
		for pix := range coverage[i].Data {
			if warpCov.Data[pix] > 0 {
				coverage[i].Data[pix] = 1
			}
			feathers[i].Data[pix] = float64(warpFea.Data[pix])
		}
		corners[i] = image.Point{X: 0, Y: 0}
	}

	// Exposure compensation over the full overlaps.
	p.exposure.Feed(corners, colors, coverage)
	for i := range colors {
		p.exposure.Apply(i, corners[i], colors[i])
	}

	// Seam finding partitions the overlaps in place (starting from full coverage).
	seamMasks := make([]*cv.FloatMat, n)
	for i := range coverage {
		seamMasks[i] = cv.NewFloatMat(rows, cols)
		copy(seamMasks[i].Data, coverage[i].Data)
	}
	p.seam.Find(colors, corners, seamMasks)

	// Blend: weight is the feather ramp restricted to each image's seam region.
	layers := make([]Layer, n)
	for i := range colors {
		w := cv.NewFloatMat(rows, cols)
		for pix := range w.Data {
			if seamMasks[i].Data[pix] > 0 {
				w.Data[pix] = feathers[i].Data[pix]
			}
		}
		layers[i] = Layer{Color: colors[i], Weight: w}
	}
	return p.blender.Blend(layers, rows, cols, channels)
}

// fullMaskMat returns an all-255 single-channel Mat used as a warp source for
// coverage detection.
func fullMaskMat(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = 255
	}
	return m
}

package xfeatures2d

import cv "github.com/malcolmston/opencv"

// GFTTDetector wraps cv.GoodFeaturesToTrack (the Shi–Tomasi "good features to
// track" corner detector) and returns the corners as [KeyPoint]s. It mirrors
// OpenCV's cv::GFTTDetector.
type GFTTDetector struct {
	// MaxCorners is the maximum number of corners to return; values <= 0 mean no
	// limit.
	MaxCorners int
	// QualityLevel is the minimal accepted corner quality as a fraction of the
	// strongest corner's measure (e.g. 0.01).
	QualityLevel float64
	// MinDistance is the minimum Euclidean distance, in pixels, between returned
	// corners.
	MinDistance float64
	// BlockSize is the size of the neighbourhood over which the structure tensor
	// is averaged. It also becomes each keypoint's Size.
	BlockSize int
	// AnnotateHarris, when true, sets each keypoint's Response to the Harris
	// response (cv.CornerHarris) at its location instead of leaving it zero. The
	// corner selection itself always uses the Shi–Tomasi measure.
	AnnotateHarris bool
	// HarrisK is the Harris free parameter used when AnnotateHarris is true.
	HarrisK float64
}

// NewGFTTDetector returns a GFTTDetector with defaults matching OpenCV's
// (up to maxCorners corners, quality level 0.01, minimum distance 1, block
// size 3).
func NewGFTTDetector(maxCorners int) *GFTTDetector {
	return &GFTTDetector{
		MaxCorners:   maxCorners,
		QualityLevel: 0.01,
		MinDistance:  1,
		BlockSize:    3,
		HarrisK:      0.04,
	}
}

// Detect finds Shi–Tomasi corners in img and returns them as keypoints. Each
// keypoint's Size is BlockSize and Angle is -1. img may be single- or
// three-channel; a colour image is converted to gray.
func (g *GFTTDetector) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	block := g.BlockSize
	if block < 1 {
		block = 3
	}
	pts := cv.GoodFeaturesToTrack(gray, g.MaxCorners, g.QualityLevel, g.MinDistance, block)

	var harris *cv.FloatMat
	if g.AnnotateHarris {
		harris = cv.CornerHarris(gray, block, 3, g.HarrisK)
	}

	kps := make([]KeyPoint, len(pts))
	for i, p := range pts {
		var resp float64
		if harris != nil {
			resp = harris.At(p.Y, p.X)
		}
		kps[i] = KeyPoint{
			Pt:       p,
			Size:     float64(block),
			Angle:    -1,
			Response: resp,
		}
	}
	return kps
}

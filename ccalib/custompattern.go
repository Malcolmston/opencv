package ccalib

import (
	cv "github.com/malcolmston/opencv"
)

// CustomPattern is a feature-based calibration target built from an arbitrary
// textured template image, mirroring OpenCV's cv::ccalib::CustomPattern. Once
// created from a template and its physical size, it can locate itself inside a
// captured scene by matching the template's features, yielding the 3D↔2D
// correspondences a calibration or pose solver needs.
//
// The zero value is unusable; construct with [NewCustomPattern] and initialise
// with [CustomPattern.Create].
type CustomPattern struct {
	initialized bool
	physWidth   float64
	physHeight  float64
	refW        int
	refH        int
	minMatches  int
	tol         float64

	refBlobs []blob
	refDesc  []descriptor
}

// NewCustomPattern returns an empty, uninitialised custom pattern.
func NewCustomPattern() *CustomPattern {
	return &CustomPattern{minMatches: 8, tol: 6.0}
}

// Create initialises the pattern from a template image whose printed size is
// physWidth×physHeight (in any consistent physical unit). It extracts the
// template's features and reports whether enough were found to support later
// detection. Calling Create again re-initialises the pattern.
func (p *CustomPattern) Create(template *cv.Mat, physWidth, physHeight float64) bool {
	p.physWidth = physWidth
	p.physHeight = physHeight
	p.refW = template.Cols
	p.refH = template.Rows
	p.refBlobs = detectBlobs(template, 6, 0)
	p.refDesc = buildDescriptors(p.refBlobs)
	p.initialized = len(p.refBlobs) >= p.minMatches
	return p.initialized
}

// IsInitialized reports whether the pattern has been successfully created.
func (p *CustomPattern) IsInitialized() bool { return p.initialized }

// KeypointCount returns the number of template features extracted by
// [CustomPattern.Create].
func (p *CustomPattern) KeypointCount() int { return len(p.refBlobs) }

// SetMinMatches overrides the minimum number of consistent correspondences
// required for [CustomPattern.FindPattern] to succeed (default 8).
func (p *CustomPattern) SetMinMatches(n int) {
	if n >= 4 {
		p.minMatches = n
	}
}

// FindPattern locates the template inside a scene image via feature matching and
// returns the matched object points (3D, Z = 0, in the template's physical
// units) and their image points (pixels in the scene). ok is false when the
// pattern is not confidently found or the pattern was never initialised.
func (p *CustomPattern) FindPattern(scene *cv.Mat) (objPts [][3]float64, imgPts [][2]float64, ok bool) {
	if !p.initialized {
		return nil, nil, false
	}
	queryBlobs := detectBlobs(scene, 6, 0)
	queryDesc := buildDescriptors(queryBlobs)
	matches := matchBlobs(p.refDesc, queryDesc)
	inliers, _, okH := filterMatchesByHomography(p.refBlobs, queryBlobs, matches, p.tol)
	if !okH || len(inliers) < p.minMatches {
		return nil, nil, false
	}
	sx := p.physWidth / float64(p.refW)
	sy := p.physHeight / float64(p.refH)
	for _, m := range inliers {
		rb := p.refBlobs[m.ref]
		qb := queryBlobs[m.query]
		objPts = append(objPts, [3]float64{rb.X * sx, rb.Y * sy, 0})
		imgPts = append(imgPts, [2]float64{qb.X, qb.Y})
	}
	return objPts, imgPts, true
}

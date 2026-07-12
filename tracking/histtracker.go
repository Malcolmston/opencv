package tracking

import cv "github.com/malcolmston/opencv"

// MeanShiftTracker tracks an object by the histogram of its hue. [Tracker.Init]
// builds a 256-bin hue histogram of the object region with [cv.CalcHist]; each
// [Tracker.Update] converts the frame to HSV, back-projects that histogram with
// [cv.CalcBackProject] to a probability image, and runs [MeanShift] from the
// last window. The window keeps a fixed size, so this tracks colour blobs whose
// apparent size is roughly constant; use [CamShiftTracker] for adaptive size.
//
// It requires 3-channel RGB frames. Construct it with [NewMeanShiftTracker].
type MeanShiftTracker struct {
	// MaxIter is the mean-shift iteration cap per frame.
	MaxIter int

	hist   []int
	window cv.Rect
	inited bool
}

// NewMeanShiftTracker returns a MeanShiftTracker with a default MaxIter of 10.
func NewMeanShiftTracker() *MeanShiftTracker {
	return &MeanShiftTracker{MaxIter: 10}
}

// Init builds the hue histogram of the object region inside bbox.
func (t *MeanShiftTracker) Init(frame *cv.Mat, bbox cv.Rect) {
	hsv := toHSV(frame)
	b := clampRect(bbox, hsv.Rows, hsv.Cols)
	region := hsv.Region(b.Y, b.X, b.Height, b.Width)
	t.hist = cv.CalcHist(region, 0)
	t.window = b
	t.inited = true
}

// Update back-projects the stored hue histogram and runs mean-shift from the
// last window. The confidence flag is true when the converged window contains
// any probability mass. It panics if called before Init.
func (t *MeanShiftTracker) Update(frame *cv.Mat) (cv.Rect, bool) {
	if !t.inited {
		panic("tracking: MeanShiftTracker.Update called before Init")
	}
	hsv := toHSV(frame)
	prob := cv.CalcBackProject(hsv, 0, t.hist)
	win := MeanShift(prob, t.window, t.MaxIter)
	t.window = win
	return win, windowMass(prob, win) > 0
}

// CamShiftTracker is like [MeanShiftTracker] but runs [CamShift], so the tracked
// window adapts its size (and a rotated-rectangle fit is available). Each
// [Tracker.Update] returns the upright bounding box; the fitted rotated rectangle
// from the most recent update is available via [CamShiftTracker.RotatedRect].
//
// It requires 3-channel RGB frames. Construct it with [NewCamShiftTracker].
type CamShiftTracker struct {
	// MaxIter is the mean-shift iteration cap per frame.
	MaxIter int

	hist   []int
	window cv.Rect
	last   cv.RotatedRect
	inited bool
}

// NewCamShiftTracker returns a CamShiftTracker with a default MaxIter of 10.
func NewCamShiftTracker() *CamShiftTracker {
	return &CamShiftTracker{MaxIter: 10}
}

// Init builds the hue histogram of the object region inside bbox.
func (t *CamShiftTracker) Init(frame *cv.Mat, bbox cv.Rect) {
	hsv := toHSV(frame)
	b := clampRect(bbox, hsv.Rows, hsv.Cols)
	region := hsv.Region(b.Y, b.X, b.Height, b.Width)
	t.hist = cv.CalcHist(region, 0)
	t.window = b
	t.last = cv.RotatedRect{CenterX: float64(b.X) + float64(b.Width)/2, CenterY: float64(b.Y) + float64(b.Height)/2, Width: float64(b.Width), Height: float64(b.Height)}
	t.inited = true
}

// Update back-projects the stored hue histogram and runs CamShift from the last
// window, adapting the window size. The confidence flag is true when the window
// contains any probability mass. It panics if called before Init.
func (t *CamShiftTracker) Update(frame *cv.Mat) (cv.Rect, bool) {
	if !t.inited {
		panic("tracking: CamShiftTracker.Update called before Init")
	}
	hsv := toHSV(frame)
	prob := cv.CalcBackProject(hsv, 0, t.hist)
	rr, win := CamShift(prob, t.window, t.MaxIter)
	t.window = win
	t.last = rr
	return win, windowMass(prob, win) > 0
}

// RotatedRect returns the rotated-rectangle fit from the most recent Update (or
// from Init if Update has not run yet).
func (t *CamShiftTracker) RotatedRect() cv.RotatedRect {
	return t.last
}

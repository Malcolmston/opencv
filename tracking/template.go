package tracking

import cv "github.com/malcolmston/opencv"

// TrackerTemplate is a normalised-cross-correlation template tracker: it stores
// the object patch from the first frame and, on each subsequent frame, searches
// a window around the last known position for the location whose patch best
// correlates with the stored template. It is the most robust baseline in this
// package but assumes the object's appearance and scale stay roughly constant.
//
// Construct it with [NewTrackerTemplate]. The exported fields may be tuned before
// [TrackerTemplate.Init].
type TrackerTemplate struct {
	// SearchMargin is how many pixels beyond the last box (on each side) are
	// scanned for the new position. It bounds the trackable per-frame motion.
	SearchMargin int
	// MinScore is the [cv.TmCcoeffNormed] correlation (in [-1,1]) below which
	// Update reports low confidence.
	MinScore float64

	templ  *cv.Mat
	box    cv.Rect
	inited bool
}

// NewTrackerTemplate returns a TrackerTemplate with sensible defaults
// (SearchMargin 8, MinScore 0.3).
func NewTrackerTemplate() *TrackerTemplate {
	return &TrackerTemplate{SearchMargin: 8, MinScore: 0.3}
}

// Init stores the luma template inside bbox from frame.
func (t *TrackerTemplate) Init(frame *cv.Mat, bbox cv.Rect) {
	gray := toGray(frame)
	b := clampRect(bbox, gray.Rows, gray.Cols)
	t.templ = gray.Region(b.Y, b.X, b.Height, b.Width)
	t.box = b
	t.inited = true
}

// Update finds the best template match in a search window around the last box
// and returns the new box together with whether the peak correlation reached
// MinScore. It panics if called before Init.
func (t *TrackerTemplate) Update(frame *cv.Mat) (cv.Rect, bool) {
	if !t.inited {
		panic("tracking: TrackerTemplate.Update called before Init")
	}
	gray := toGray(frame)
	win := searchWindow(t.box, t.templ.Cols, t.templ.Rows, t.SearchMargin, gray.Rows, gray.Cols)
	region := gray.Region(win.Y, win.X, win.Height, win.Width)
	res := cv.MatchTemplate(region, t.templ, cv.TmCcoeffNormed)
	_, maxVal, _, _, mx, my := cv.MinMaxLoc(res)
	box := cv.Rect{X: win.X + mx, Y: win.Y + my, Width: t.templ.Cols, Height: t.templ.Rows}
	t.box = box
	return box, maxVal >= t.MinScore
}

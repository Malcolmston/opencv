package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TrackerKCF is a "KCF-lite" correlation tracker. The original KCF (Kernelised
// Correlation Filter) learns a ridge-regression filter over all cyclic shifts of
// the patch in the Fourier domain. This implementation instead approximates a
// correlation filter with an online-adapted normalised-cross-correlation
// template: like [TrackerTemplate] it locates the object with [cv.MatchTemplate]
// ([cv.TmCcoeffNormed]) in a search window, but after each successful frame it
// blends its appearance model toward the newly found patch (see LearnRate). That
// online adaptation is what distinguishes a correlation tracker from a fixed
// template; the DFT/kernel machinery is intentionally omitted (see the package
// "Deferred" notes).
//
// Construct it with [NewTrackerKCF].
type TrackerKCF struct {
	// SearchMargin is how many pixels beyond the last box (on each side) are
	// scanned for the new position.
	SearchMargin int
	// LearnRate in [0,1] is the fraction of the freshly located patch blended
	// into the model each frame; 0 disables adaptation (making this behave like a
	// fixed template), higher values adapt faster but drift sooner.
	LearnRate float64
	// MinScore is the correlation below which Update reports low confidence and
	// the model is not updated.
	MinScore float64

	model  *cv.Mat
	box    cv.Rect
	inited bool
}

// NewTrackerKCF returns a TrackerKCF with sensible defaults (SearchMargin 8,
// LearnRate 0.1, MinScore 0.3).
func NewTrackerKCF() *TrackerKCF {
	return &TrackerKCF{SearchMargin: 8, LearnRate: 0.1, MinScore: 0.3}
}

// Init stores the initial luma appearance model from the region inside bbox.
func (t *TrackerKCF) Init(frame *cv.Mat, bbox cv.Rect) {
	gray := toGray(frame)
	b := clampRect(bbox, gray.Rows, gray.Cols)
	t.model = gray.Region(b.Y, b.X, b.Height, b.Width)
	t.box = b
	t.inited = true
}

// Update correlates the model against a search window, moves the box to the
// peak, and (when the peak reaches MinScore) blends the model toward the new
// patch by LearnRate. It returns the new box and the confidence flag, and panics
// if called before Init.
func (t *TrackerKCF) Update(frame *cv.Mat) (cv.Rect, bool) {
	if !t.inited {
		panic("tracking: TrackerKCF.Update called before Init")
	}
	gray := toGray(frame)
	win := searchWindow(t.box, t.model.Cols, t.model.Rows, t.SearchMargin, gray.Rows, gray.Cols)
	region := gray.Region(win.Y, win.X, win.Height, win.Width)
	res := cv.MatchTemplate(region, t.model, cv.TmCcoeffNormed)
	_, maxVal, _, _, mx, my := cv.MinMaxLoc(res)
	box := cv.Rect{X: win.X + mx, Y: win.Y + my, Width: t.model.Cols, Height: t.model.Rows}
	ok := maxVal >= t.MinScore
	if ok && t.LearnRate > 0 {
		patch := gray.Region(box.Y, box.X, box.Height, box.Width)
		lr := t.LearnRate
		for i := range t.model.Data {
			m := float64(t.model.Data[i])
			p := float64(patch.Data[i])
			v := (1-lr)*m + lr*p
			t.model.Data[i] = uint8(math.Round(v))
		}
	}
	t.box = box
	return box, ok
}

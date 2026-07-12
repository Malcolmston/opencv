package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TrackerTLD is a lightweight Tracking-Learning-Detection tracker (Kalal et al.,
// 2012). It couples three components:
//
//   - a short-term tracker (the package's [TrackerMedianFlow]) that follows the
//     object frame to frame;
//   - a detector — a nearest-neighbour classifier over a growing set of
//     normalised object templates (patch NCC), scanned across the whole frame
//     with a variance pre-filter — that can find the object anywhere; and
//   - an integrator that trusts the tracker while it is confident but falls back
//     to the detector to re-detect the object after the tracker fails or when the
//     detector is markedly more confident somewhere else, re-initialising the
//     tracker there.
//
// Learning grows the template model (P-expert) with confident, novel views. This
// re-detection is what lets TLD recover from full occlusion, unlike the other
// trackers here. Construct it with [NewTrackerTLD].
type TrackerTLD struct {
	// TemplateSize is the side length templates are normalised to for NCC.
	TemplateSize int
	// AcceptSim is the NCC similarity at or above which a detection/track is
	// considered a valid object view.
	AcceptSim float64
	// NoveltySim is the similarity below which a confident patch is added to the
	// model as a new template (keeps the model diverse and bounded).
	NoveltySim float64
	// MaxTemplates caps the template set.
	MaxTemplates int
	// SearchStepDiv sets the detector scan step as box size / SearchStepDiv.
	SearchStepDiv int
	// MinVariance is the minimum patch variance a detection window must have.
	MinVariance float64

	flow    *TrackerMedianFlow
	pos     [][]float64
	box     cv.Rect
	ts      int
	inited  bool
	lastSim float64
}

// NewTrackerTLD returns a TrackerTLD with sensible defaults (TemplateSize 15,
// AcceptSim 0.6, NoveltySim 0.8, MaxTemplates 20, SearchStepDiv 6, MinVariance
// 25).
func NewTrackerTLD() *TrackerTLD {
	return &TrackerTLD{
		TemplateSize: 15, AcceptSim: 0.6, NoveltySim: 0.8,
		MaxTemplates: 20, SearchStepDiv: 6, MinVariance: 25,
	}
}

// normPatch crops the box from the single-channel image, resizes it to
// TemplateSize², and returns the zero-mean unit-norm template together with the
// raw patch variance (used by the variance pre-filter).
func (t *TrackerTLD) normPatch(gray *cv.Mat, box cv.Rect) ([]float64, float64) {
	b := clampRect(box, gray.Rows, gray.Cols)
	patch := cropClamped(gray, b.X, b.Y, b.Width, b.Height)
	small := cv.Resize(patch, t.ts, t.ts, cv.InterLinear)
	vals := toFloat(small)
	var mean float64
	for _, v := range vals {
		mean += v
	}
	mean /= float64(len(vals))
	var ss float64
	for i := range vals {
		vals[i] -= mean
		ss += vals[i] * vals[i]
	}
	variance := ss / float64(len(vals))
	norm := math.Sqrt(ss) + 1e-6
	for i := range vals {
		vals[i] /= norm
	}
	return vals, variance
}

// simTo returns the maximum NCC of the template against the stored positives.
func (t *TrackerTLD) simTo(tmpl []float64) float64 {
	best := -1.0
	for _, p := range t.pos {
		var dot float64
		for i := range tmpl {
			dot += tmpl[i] * p[i]
		}
		if dot > best {
			best = dot
		}
	}
	if len(t.pos) == 0 {
		return 0
	}
	return best
}

// addTemplate stores tmpl, evicting the oldest when the set is full.
func (t *TrackerTLD) addTemplate(tmpl []float64) {
	cp := make([]float64, len(tmpl))
	copy(cp, tmpl)
	if len(t.pos) >= t.MaxTemplates {
		t.pos = append(t.pos[1:], cp)
		return
	}
	t.pos = append(t.pos, cp)
}

// detect scans the frame at the current box size and returns the best-matching
// window and its similarity.
func (t *TrackerTLD) detect(gray *cv.Mat) (cv.Rect, float64) {
	bw, bh := t.box.Width, t.box.Height
	step := bw / t.SearchStepDiv
	if step < 1 {
		step = 1
	}
	best := -1.0
	bestBox := t.box
	for y := 0; y+bh <= gray.Rows; y += step {
		for x := 0; x+bw <= gray.Cols; x += step {
			cand := cv.Rect{X: x, Y: y, Width: bw, Height: bh}
			tmpl, variance := t.normPatch(gray, cand)
			if variance < t.MinVariance {
				continue
			}
			sim := t.simTo(tmpl)
			if sim > best {
				best = sim
				bestBox = cand
			}
		}
	}
	if best < 0 {
		return bestBox, best
	}
	// Fine per-pixel refinement around the best coarse window to align the peak
	// (a coarse step can leave the true centre a few pixels off, deflating NCC).
	for dy := -step; dy <= step; dy++ {
		for dx := -step; dx <= step; dx++ {
			x := bestBox.X + dx
			y := bestBox.Y + dy
			if x < 0 || y < 0 || x+bw > gray.Cols || y+bh > gray.Rows {
				continue
			}
			cand := cv.Rect{X: x, Y: y, Width: bw, Height: bh}
			tmpl, variance := t.normPatch(gray, cand)
			if variance < t.MinVariance {
				continue
			}
			if sim := t.simTo(tmpl); sim > best {
				best = sim
				bestBox = cand
			}
		}
	}
	return bestBox, best
}

// Init primes the median-flow tracker and seeds the template model.
func (t *TrackerTLD) Init(frame *cv.Mat, bbox cv.Rect) {
	t.ts = t.TemplateSize
	gray := toGray(frame)
	b := clampRect(bbox, gray.Rows, gray.Cols)
	t.box = b
	t.flow = NewTrackerMedianFlow()
	t.flow.Init(frame, b)
	tmpl, _ := t.normPatch(gray, b)
	t.pos = nil
	t.addTemplate(tmpl)
	t.inited = true
}

// centerDist returns the distance between the centres of two boxes.
func centerDist(a, b cv.Rect) float64 {
	ax, ay := rectCenter(a)
	bx, by := rectCenter(b)
	return math.Hypot(ax-bx, ay-by)
}

// UpdateConfidence runs the tracker and detector, integrates them (re-detecting
// after failure), learns, and returns the chosen box with its NCC confidence. It
// panics before Init.
func (t *TrackerTLD) UpdateConfidence(frame *cv.Mat) (cv.Rect, float64) {
	if !t.inited {
		panic("tracking: TrackerTLD.Update called before Init")
	}
	gray := toGray(frame)

	flowBox, flowOK := t.flow.Update(frame)
	flowSim := -1.0
	if flowOK {
		tmpl, variance := t.normPatch(gray, flowBox)
		if variance >= t.MinVariance {
			flowSim = t.simTo(tmpl)
		}
	}

	detBox, detSim := t.detect(gray)

	final := flowBox
	conf := flowSim
	valid := flowOK && flowSim >= t.AcceptSim

	reinit := false
	switch {
	case (!flowOK || flowSim < t.AcceptSim) && detSim >= t.AcceptSim:
		// Re-detection after tracker failure or drift.
		final, conf, valid, reinit = detBox, detSim, true, true
	case detSim >= t.AcceptSim && detSim > flowSim+0.1 && centerDist(detBox, flowBox) > float64(t.box.Width)/2:
		// Detector confidently disagrees with the tracker: trust the detector.
		final, conf, valid, reinit = detBox, detSim, true, true
	}

	if reinit {
		t.flow.Init(frame, final)
	}
	t.box = clampRect(final, gray.Rows, gray.Cols)

	// P-expert learning: grow the model with confident, novel views.
	if valid {
		tmpl, variance := t.normPatch(gray, t.box)
		if variance >= t.MinVariance && t.simTo(tmpl) < t.NoveltySim {
			t.addTemplate(tmpl)
		}
	}
	t.lastSim = conf
	return t.box, conf
}

// Update satisfies [Tracker]; the flag is true when the chosen box's similarity
// reaches AcceptSim.
func (t *TrackerTLD) Update(frame *cv.Mat) (cv.Rect, bool) {
	box, conf := t.UpdateConfidence(frame)
	return box, conf >= t.AcceptSim
}

// NumTemplates returns the current size of the learned template model.
func (t *TrackerTLD) NumTemplates() int { return len(t.pos) }

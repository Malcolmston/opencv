package objdetect

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// WeightedRect is a rectangle with a signed weight, the building block of a Haar
// feature: the feature value is the weighted sum of the image intensity inside
// each of its rectangles.
type WeightedRect struct {
	X, Y, W, H int
	Weight     float64
}

// SoftFeature is one Haar-like feature: a small set of [WeightedRect]s whose
// weighted area sums are combined into a single scalar feature value.
type SoftFeature struct {
	Rects []WeightedRect
}

// SoftStump is a depth-1 decision stump over a [SoftFeature]. When the feature
// value is below Threshold (scaled by the window's intensity normalisation) it
// emits Left, otherwise Right.
type SoftStump struct {
	Feature     int
	Threshold   float64
	Left, Right float64
}

// SoftCascade is a soft-cascade detector (Bourdev & Brandt, 2005): a single
// additive ensemble of [SoftStump]s evaluated with a running score that is
// tested against a per-stump rejection threshold after every stump. As soon as
// the accumulated score drops below the rejection threshold for the current
// stump the window is rejected without evaluating the rest of the ensemble —
// this early-exit "reject trace" is what makes a soft cascade fast while keeping
// a single monotone score.
//
// Unlike a classic staged Viola–Jones cascade (see [CascadeClassifier]) there
// are no stage boundaries: rejection can happen after any weak classifier.
// Build one directly by populating the exported fields, or convert a loaded Haar
// cascade with [CascadeClassifier.ToSoftCascade].
type SoftCascade struct {
	// WinW, WinH are the base detection-window dimensions in pixels.
	WinW, WinH int
	// Features and Stumps define the ensemble; each stump references a feature
	// by index into Features.
	Features []SoftFeature
	Stumps   []SoftStump
	// Reject holds one running-score rejection threshold per stump (same length
	// as Stumps). A window is rejected the first time its cumulative score falls
	// below Reject[i] after stump i. A nil Reject disables early exit (the whole
	// ensemble is always evaluated and the final score is compared to 0).
	Reject []float64
	// ScaleFactor is the ratio between successive window sizes in the scan
	// pyramid. Values <= 1 default to 1.1.
	ScaleFactor float64
	// MinNeighbors, when positive, groups overlapping detections and keeps only
	// clusters with at least this many members.
	MinNeighbors int
}

// ToSoftCascade converts a loaded Haar [CascadeClassifier] into an equivalent
// [SoftCascade]. The staged structure is flattened into a single ensemble and a
// rejection trace is synthesised from the stage thresholds: the running score
// must, by the end of each original stage, still clear that stage's threshold,
// so the rejection threshold at the last stump of stage k is the cumulative sum
// of stage thresholds up to and including k (and -inf elsewhere, allowing the
// score to dip mid-stage). It panics if the classifier is unloaded.
func (c *CascadeClassifier) ToSoftCascade() *SoftCascade {
	if !c.loaded {
		panic("objdetect: ToSoftCascade on unloaded CascadeClassifier")
	}
	sc := &SoftCascade{WinW: c.windowW, WinH: c.windowH, ScaleFactor: c.ScaleFactor, MinNeighbors: c.MinNeighbors}
	sc.Features = make([]SoftFeature, len(c.features))
	for i, f := range c.features {
		rs := make([]WeightedRect, len(f.rects))
		for j, r := range f.rects {
			rs[j] = WeightedRect{X: r.x, Y: r.y, W: r.w, H: r.h, Weight: r.weight}
		}
		sc.Features[i] = SoftFeature{Rects: rs}
	}
	negInf := math.Inf(-1)
	cumulative := 0.0
	for si := range c.stages {
		st := &c.stages[si]
		cumulative += st.threshold
		for wi := range st.weaks {
			stump := st.weaks[wi]
			sc.Stumps = append(sc.Stumps, SoftStump{
				Feature:   stump.featureIdx,
				Threshold: stump.threshold,
				Left:      stump.left,
				Right:     stump.right,
			})
			// Only the last stump of a stage carries the cumulative gate.
			if wi == len(st.weaks)-1 {
				sc.Reject = append(sc.Reject, cumulative)
			} else {
				sc.Reject = append(sc.Reject, negInf)
			}
		}
	}
	return sc
}

// DetectMultiScale scans img at growing window sizes and returns every window
// the soft cascade accepts. The window is grown by ScaleFactor between pyramid
// levels (the image is never resampled) and evaluated over an integral image. If
// MinNeighbors is positive the raw detections are grouped. It panics if the
// cascade has no stumps.
func (s *SoftCascade) DetectMultiScale(img *cv.Mat) []cv.Rect {
	if len(s.Stumps) == 0 {
		panic("objdetect: SoftCascade.DetectMultiScale with no stumps")
	}
	if s.Reject != nil && len(s.Reject) != len(s.Stumps) {
		panic("objdetect: SoftCascade Reject length must equal Stumps length")
	}
	ii := NewIntegralImage(img)
	sf := s.ScaleFactor
	if sf <= 1 {
		sf = 1.1
	}

	var raw []cv.Rect
	scale := 1.0
	for {
		sw := int(float64(s.WinW)*scale + 0.5)
		sh := int(float64(s.WinH)*scale + 0.5)
		if sw > ii.W || sh > ii.H {
			break
		}
		step := int(scale + 0.5)
		if step < 1 {
			step = 1
		}
		for y := 0; y+sh <= ii.H; y += step {
			for x := 0; x+sw <= ii.W; x += step {
				if ok, _ := s.evalWindow(ii, x, y, scale); ok {
					raw = append(raw, cv.Rect{X: x, Y: y, Width: sw, Height: sh})
				}
			}
		}
		scale *= sf
	}

	if s.MinNeighbors > 0 {
		return GroupRectangles(raw, s.MinNeighbors, 0.2)
	}
	return raw
}

// evalWindow runs the soft cascade at window (wx,wy) scaled by scale and returns
// whether it is accepted along with the number of stumps actually evaluated
// (which is < len(Stumps) when early exit fired).
func (s *SoftCascade) evalWindow(ii *IntegralImage, wx, wy int, scale float64) (bool, int) {
	sw := int(float64(s.WinW)*scale + 0.5)
	sh := int(float64(s.WinH)*scale + 0.5)
	area := float64(sw * sh)
	sum := ii.Sum(wx, wy, sw, sh)
	sq := ii.SqSum(wx, wy, sw, sh)
	nf := area*sq - sum*sum
	if nf > 0 {
		nf = math.Sqrt(nf)
	} else {
		nf = 1
	}

	running := 0.0
	for i := range s.Stumps {
		stump := &s.Stumps[i]
		f := &s.Features[stump.Feature]
		var fsum float64
		for _, r := range f.Rects {
			rx := wx + int(float64(r.X)*scale+0.5)
			ry := wy + int(float64(r.Y)*scale+0.5)
			rw := int(float64(r.W)*scale + 0.5)
			rh := int(float64(r.H)*scale + 0.5)
			if rw <= 0 || rh <= 0 {
				continue
			}
			fsum += r.Weight * ii.Sum(rx, ry, rw, rh)
		}
		if fsum < stump.Threshold*nf {
			running += stump.Left
		} else {
			running += stump.Right
		}
		if s.Reject != nil {
			if running < s.Reject[i] {
				return false, i + 1
			}
		}
	}
	if s.Reject == nil {
		return running >= 0, len(s.Stumps)
	}
	return true, len(s.Stumps)
}

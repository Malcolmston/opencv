package xobjdetect

import (
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// Feature is one integral-channel probe: the mean value of channel Channel over
// the axis-aligned rectangle (X, Y, W, H) measured in detection-window
// coordinates. Its fields are exported so a [FeaturePool] can be serialised
// with encoding/gob.
type Feature struct {
	// Channel selects the feature channel, in [0, NumChannels).
	Channel int
	// X, Y are the rectangle's top-left corner within the detection window.
	X, Y int
	// W, H are the rectangle's width and height in pixels.
	W, H int
}

// FeaturePool is a bank of [Feature] probes covering a fixed WinW x WinH
// detection window. A pool is the fixed feature vocabulary that both training
// and detection evaluate; keep it stable across a model's lifetime.
type FeaturePool struct {
	// WinW, WinH are the detection-window dimensions the features address.
	WinW, WinH int
	// Features is the ordered list of probes; a feature vector produced by the
	// pool has one entry per element, in this order.
	Features []Feature
}

// NewFeaturePool builds a pool of count random features covering a WinW x WinH
// window. Rectangles are sampled with a minimum side of 1 pixel using rng, so
// passing a seeded *rand.Rand makes the pool deterministic. It panics if the
// window is not positive or count is not positive.
func NewFeaturePool(winW, winH, count int, rng *rand.Rand) *FeaturePool {
	if winW <= 0 || winH <= 0 {
		panic("xobjdetect: NewFeaturePool requires a positive window")
	}
	if count <= 0 {
		panic("xobjdetect: NewFeaturePool requires count > 0")
	}
	fp := &FeaturePool{WinW: winW, WinH: winH, Features: make([]Feature, 0, count)}
	for len(fp.Features) < count {
		ch := rng.Intn(NumChannels)
		w := 1 + rng.Intn(winW)
		h := 1 + rng.Intn(winH)
		if w > winW {
			w = winW
		}
		if h > winH {
			h = winH
		}
		x := rng.Intn(winW - w + 1)
		y := rng.Intn(winH - h + 1)
		fp.Features = append(fp.Features, Feature{Channel: ch, X: x, Y: y, W: w, H: h})
	}
	return fp
}

// Len reports the number of features in the pool, which is the length of every
// feature vector it produces.
func (fp *FeaturePool) Len() int { return len(fp.Features) }

// evaluateWindow fills out with the value of every pooled feature for the window
// whose top-left corner is (ox, oy) in the channel image ic. out must have
// length fp.Len().
func (fp *FeaturePool) evaluateWindow(ic *integralChannels, ox, oy int, out []float64) {
	for i, f := range fp.Features {
		out[i] = ic.rectMean(f.Channel, ox+f.X, oy+f.Y, f.W, f.H)
	}
}

// ACFFeatureEvaluator computes aggregate-channel-feature (ACF) vectors for a
// [FeaturePool], mirroring OpenCV's cv::xobjdetect::ACFFeatureEvaluator. Call
// [ACFFeatureEvaluator.SetImage] once per image to precompute its integral
// channels, then [ACFFeatureEvaluator.EvaluateWindow] for each window origin.
// For the common case of scoring a whole patch, [ACFFeatureEvaluator.Sample]
// resizes an image to the window and returns its feature vector in one call.
type ACFFeatureEvaluator struct {
	pool *FeaturePool
	ic   *integralChannels
}

// NewACFFeatureEvaluator returns an evaluator bound to pool. It panics if pool
// is nil or empty.
func NewACFFeatureEvaluator(pool *FeaturePool) *ACFFeatureEvaluator {
	if pool == nil || pool.Len() == 0 {
		panic("xobjdetect: NewACFFeatureEvaluator requires a non-empty pool")
	}
	return &ACFFeatureEvaluator{pool: pool}
}

// Pool returns the feature pool the evaluator was built with.
func (e *ACFFeatureEvaluator) Pool() *FeaturePool { return e.pool }

// SetImage precomputes the integral feature channels of img so subsequent
// [ACFFeatureEvaluator.EvaluateWindow] calls are cheap. img is used as-is (not
// resized); its size must be at least the detection window.
func (e *ACFFeatureEvaluator) SetImage(img *cv.Mat) {
	e.ic = newIntegralChannels(img)
}

// EvaluateWindow returns the feature vector for the detection window whose
// top-left corner is (ox, oy) in the image most recently passed to
// [ACFFeatureEvaluator.SetImage]. It panics if SetImage has not been called.
func (e *ACFFeatureEvaluator) EvaluateWindow(ox, oy int) []float64 {
	if e.ic == nil {
		panic("xobjdetect: EvaluateWindow before SetImage")
	}
	out := make([]float64, e.pool.Len())
	e.pool.evaluateWindow(e.ic, ox, oy, out)
	return out
}

// Sample resizes img to the pool's detection window and returns its feature
// vector. This is the routine used to turn a training patch into features.
func (e *ACFFeatureEvaluator) Sample(img *cv.Mat) []float64 {
	win := resizeToWindow(img, e.pool.WinW, e.pool.WinH)
	e.SetImage(win)
	return e.EvaluateWindow(0, 0)
}

// resizeToWindow returns img scaled to exactly winW x winH, reusing it when it
// already matches.
func resizeToWindow(img *cv.Mat, winW, winH int) *cv.Mat {
	if img.Cols == winW && img.Rows == winH {
		return img
	}
	return cv.Resize(img, winW, winH, cv.InterLinear)
}

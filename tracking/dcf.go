package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TrackerDCF is a faithful Discriminative Correlation Filter tracker: a
// kernelised ridge-regression correlation filter (KCF) with a Gaussian kernel,
// learned and evaluated entirely in the Fourier domain via [FFT2] / [IFFT2].
// Unlike the online-NCC [TrackerKCF] approximation, this is the real algorithm
// of Henriques et al. (2015): it trains coefficients α so that
//
//	αf = FFT(y) / (FFT(k_xx) + λ)
//
// on a Gaussian regression target y, and at detection correlates the model with
// the new window through the same Gaussian kernel to obtain a response map. It
// searches several scales each frame ([TrackerDCF.Scales]) and adapts the box
// size, so it handles both translation and scale change.
//
// Construct it with [NewTrackerDCF].
type TrackerDCF struct {
	// ModelSize is the power-of-two side length the window is resized to.
	ModelSize int
	// Padding scales the search window relative to the object box (KCF context).
	Padding float64
	// Lambda is the ridge-regression regularisation.
	Lambda float64
	// KernelSigma is the Gaussian-kernel bandwidth.
	KernelSigma float64
	// OutputSigmaFactor sets the Gaussian target width as a fraction of ModelSize.
	OutputSigmaFactor float64
	// LearnRate in [0,1] blends the new model into the old each frame.
	LearnRate float64
	// Scales are the multiplicative scale hypotheses tried each frame (1.0 must be
	// present for the no-scale-change case).
	Scales []float64
	// ScalePenalty (in (0,1]) multiplies the peak response of non-unit scales so
	// scale changes must clearly win; 1 disables the penalty.
	ScalePenalty float64
	// MinResponse is the peak response below which Update reports low confidence.
	MinResponse float64

	modelXF []*ComplexMat // learned appearance spectrum (one plane here)
	alphaF  *ComplexMat
	hann    []float64
	cx, cy  float64
	w, h    float64
	inited  bool
}

// NewTrackerDCF returns a TrackerDCF with sensible defaults (ModelSize 64,
// Padding 2.0, Lambda 1e-4, KernelSigma 0.5, OutputSigmaFactor 1/16, LearnRate
// 0.075, Scales {0.97,1.0,1.03}, ScalePenalty 0.985, MinResponse 0.15).
func NewTrackerDCF() *TrackerDCF {
	return &TrackerDCF{
		ModelSize:         64,
		Padding:           2.0,
		Lambda:            1e-4,
		KernelSigma:       0.5,
		OutputSigmaFactor: 1.0 / 16.0,
		LearnRate:         0.075,
		Scales:            []float64{0.95, 1.0, 1.05},
		ScalePenalty:      0.99,
		MinResponse:       0.15,
	}
}

// features extracts the single-channel preprocessed feature spectrum of the
// window of size win centred on (cx, cy).
func (t *TrackerDCF) features(gray *cv.Mat, cx, cy float64, win int) []*ComplexMat {
	patch := cropResizeGray(gray, cx, cy, win, t.ModelSize)
	pre := preprocessFilter(toFloat(patch), t.hann)
	return []*ComplexMat{FFT2(RealToComplex(pre, t.ModelSize, t.ModelSize))}
}

func (t *TrackerDCF) winSize(scale float64) int {
	w := int(math.Round(t.w * t.Padding * scale))
	if w < 2 {
		w = 2
	}
	return w
}

// train computes αf from a feature spectrum and returns it together with that
// spectrum (the new model).
func (t *TrackerDCF) train(xf []*ComplexMat, yf *ComplexMat) *ComplexMat {
	kf := gaussianCorrelation(xf, xf, t.KernelSigma)
	alpha := NewComplexMat(t.ModelSize, t.ModelSize)
	lam := complex(t.Lambda, 0)
	for i := range alpha.Data {
		alpha.Data[i] = yf.Data[i] / (kf.Data[i] + lam)
	}
	return alpha
}

// Init trains the initial KCF from the object inside bbox.
func (t *TrackerDCF) Init(frame *cv.Mat, bbox cv.Rect) {
	if !isPow2(t.ModelSize) {
		t.ModelSize = NextPow2(t.ModelSize)
	}
	gray := toGray(frame)
	b := clampRect(bbox, gray.Rows, gray.Cols)
	t.w, t.h = float64(b.Width), float64(b.Height)
	t.cx, t.cy = rectCenter(b)
	t.hann = HannWindow2D(t.ModelSize, t.ModelSize)

	sigma := float64(t.ModelSize) * t.OutputSigmaFactor
	y := gaussianResponseOrigin(t.ModelSize, t.ModelSize, sigma)
	yf := FFT2(RealToComplex(y, t.ModelSize, t.ModelSize))

	xf := t.features(gray, t.cx, t.cy, t.winSize(1.0))
	t.modelXF = xf
	t.alphaF = t.train(xf, yf)
	t.inited = true
}

// detect correlates the current model with the window spectrum zf and returns
// the response map.
func (t *TrackerDCF) detect(zf []*ComplexMat) []float64 {
	kzf := gaussianCorrelation(zf, t.modelXF, t.KernelSigma)
	spec := NewComplexMat(t.ModelSize, t.ModelSize)
	for i := range spec.Data {
		spec.Data[i] = t.alphaF.Data[i] * kzf.Data[i]
	}
	return IFFT2(spec).Real()
}

// UpdateConfidence locates the object across the configured scales and returns
// the new box and the peak KCF response as confidence. It panics before Init.
func (t *TrackerDCF) UpdateConfidence(frame *cv.Mat) (cv.Rect, float64) {
	if !t.inited {
		panic("tracking: TrackerDCF.Update called before Init")
	}
	gray := toGray(frame)
	m := t.ModelSize

	bestVal := math.Inf(-1)
	var bestResp []float64
	bestScale := 1.0
	for _, s := range t.Scales {
		zf := t.features(gray, t.cx, t.cy, t.winSize(s))
		resp := t.detect(zf)
		_, _, val := peakLoc(resp, m, m)
		if s != 1.0 {
			val *= t.ScalePenalty
		}
		if val > bestVal {
			bestVal = val
			bestResp = resp
			bestScale = s
		}
	}

	px, py, _ := peakLoc(bestResp, m, m)
	xl := bestResp[py*m+(px-1+m)%m]
	xr := bestResp[py*m+(px+1)%m]
	yt := bestResp[((py-1+m)%m)*m+px]
	yb := bestResp[((py+1)%m)*m+px]
	dx := wrapCoord(px, m) + subPixel(xl, bestResp[py*m+px], xr)
	dy := wrapCoord(py, m) + subPixel(yt, bestResp[py*m+px], yb)

	scaleBack := float64(t.winSize(bestScale)) / float64(m)
	t.cx += dx * scaleBack
	t.cy += dy * scaleBack
	if bestScale != 1.0 {
		// Damp the scale update slightly for stability.
		t.w *= 1 + (bestScale-1)*0.75
		t.h *= 1 + (bestScale-1)*0.75
	}
	t.cx = math.Max(0, math.Min(t.cx, float64(gray.Cols-1)))
	t.cy = math.Max(0, math.Min(t.cy, float64(gray.Rows-1)))

	if bestVal >= t.MinResponse {
		sigma := float64(m) * t.OutputSigmaFactor
		yf := FFT2(RealToComplex(gaussianResponseOrigin(m, m, sigma), m, m))
		xf := t.features(gray, t.cx, t.cy, t.winSize(1.0))
		newAlpha := t.train(xf, yf)
		lr := complex(t.LearnRate, 0)
		ilr := complex(1-t.LearnRate, 0)
		for i := range t.alphaF.Data {
			t.alphaF.Data[i] = ilr*t.alphaF.Data[i] + lr*newAlpha.Data[i]
			t.modelXF[0].Data[i] = ilr*t.modelXF[0].Data[i] + lr*xf[0].Data[i]
		}
	}
	return t.box(gray), bestVal
}

// Update satisfies [Tracker]; the flag is true when the peak response reaches
// MinResponse.
func (t *TrackerDCF) Update(frame *cv.Mat) (cv.Rect, bool) {
	box, conf := t.UpdateConfidence(frame)
	return box, conf >= t.MinResponse
}

func (t *TrackerDCF) box(gray *cv.Mat) cv.Rect {
	r := cv.Rect{
		X:      int(math.Round(t.cx - t.w/2)),
		Y:      int(math.Round(t.cy - t.h/2)),
		Width:  int(math.Round(t.w)),
		Height: int(math.Round(t.h)),
	}
	return clampRect(r, gray.Rows, gray.Cols)
}

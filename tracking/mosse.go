package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TrackerMOSSE implements the MOSSE (Minimum Output Sum of Squared Error)
// correlation filter of Bolme et al. (2010). It is a genuine Fourier-domain
// tracker: the object window is normalised to a fixed power-of-two model size,
// preprocessed (log, zero-mean, unit-norm, Hann-windowed) and transformed with
// [FFT2]; a filter H is learned in closed form so that the filter's correlation
// with the object peaks as a Gaussian at the object centre.
//
// The filter is stored as the numerator A = G⊙F* and denominator B = F⊙F* of
// H* = A/B and both are adapted online each frame ([TrackerMOSSE.LearnRate]).
// Detection correlates the current window's spectrum with H* via an inverse FFT
// and takes the response peak; the peak-to-sidelobe ratio ([psr]) is reported as
// confidence. MOSSE tracks translation at a fixed scale.
//
// Construct it with [NewTrackerMOSSE].
type TrackerMOSSE struct {
	// ModelSize is the power-of-two side length the object window is resized to
	// before the FFT. Larger is more precise but slower.
	ModelSize int
	// Padding scales the search window relative to the object box; values above 1
	// include surrounding context and give the sidelobe room for a stable PSR.
	Padding float64
	// LearnRate in [0,1] is the online adaptation rate of the filter each frame.
	LearnRate float64
	// Sigma is the standard deviation (in model pixels) of the Gaussian target.
	Sigma float64
	// MinPSR is the peak-to-sidelobe ratio below which Update reports low
	// confidence and skips the online update.
	MinPSR float64

	a, b   *ComplexMat // filter numerator/denominator
	g      *ComplexMat // FFT of the Gaussian target (fixed)
	hann   []float64
	cx, cy float64
	w, h   int
	inited bool
}

// NewTrackerMOSSE returns a TrackerMOSSE with sensible defaults (ModelSize 64,
// Padding 2.0, LearnRate 0.125, Sigma 2.0, MinPSR 5.0).
func NewTrackerMOSSE() *TrackerMOSSE {
	return &TrackerMOSSE{ModelSize: 64, Padding: 2.0, LearnRate: 0.125, Sigma: 2.0, MinPSR: 5.0}
}

func (t *TrackerMOSSE) winSize() int {
	return int(math.Round(float64(t.w) * t.Padding))
}

// spectrum returns FFT2 of the preprocessed window centred on the current
// object position of the given gray frame.
func (t *TrackerMOSSE) spectrum(gray *cv.Mat) *ComplexMat {
	patch := cropResizeGray(gray, t.cx, t.cy, t.winSize(), t.ModelSize)
	pre := preprocessFilter(toFloat(patch), t.hann)
	return FFT2(RealToComplex(pre, t.ModelSize, t.ModelSize))
}

// Init learns the initial MOSSE filter from the object inside bbox.
func (t *TrackerMOSSE) Init(frame *cv.Mat, bbox cv.Rect) {
	if !isPow2(t.ModelSize) {
		t.ModelSize = NextPow2(t.ModelSize)
	}
	gray := toGray(frame)
	b := clampRect(bbox, gray.Rows, gray.Cols)
	t.w, t.h = b.Width, b.Height
	t.cx, t.cy = rectCenter(b)
	t.hann = HannWindow2D(t.ModelSize, t.ModelSize)
	t.g = FFT2(RealToComplex(GaussianResponse(t.ModelSize, t.ModelSize, t.Sigma), t.ModelSize, t.ModelSize))

	f := t.spectrum(gray)
	t.a = mulConj(t.g, f)
	t.b = mulConj(f, f)
	for i := range t.b.Data {
		t.b.Data[i] += complex(1e-3, 0) // regularisation
	}
	t.inited = true
}

// filter returns the current filter H* = A/B.
func (t *TrackerMOSSE) filter() *ComplexMat {
	h := NewComplexMat(t.ModelSize, t.ModelSize)
	for i := range h.Data {
		h.Data[i] = t.a.Data[i] / t.b.Data[i]
	}
	return h
}

// response correlates the window spectrum f with the current filter and returns
// the real spatial response map.
func (t *TrackerMOSSE) response(f *ComplexMat) []float64 {
	hstar := t.filter()
	spec := NewComplexMat(t.ModelSize, t.ModelSize)
	for i := range spec.Data {
		spec.Data[i] = f.Data[i] * hstar.Data[i]
	}
	return IFFT2(spec).Real()
}

// UpdateConfidence locates the object in frame and returns the new box together
// with the response's peak-to-sidelobe ratio as a continuous confidence. It
// panics if called before Init.
func (t *TrackerMOSSE) UpdateConfidence(frame *cv.Mat) (cv.Rect, float64) {
	if !t.inited {
		panic("tracking: TrackerMOSSE.Update called before Init")
	}
	gray := toGray(frame)
	f := t.spectrum(gray)
	resp := t.response(f)
	m := t.ModelSize
	px, py, _ := peakLoc(resp, m, m)

	// Sub-pixel refinement about the (Gaussian-centred) peak.
	xl := resp[py*m+(px-1+m)%m]
	xr := resp[py*m+(px+1)%m]
	yt := resp[((py-1+m)%m)*m+px]
	yb := resp[((py+1)%m)*m+px]
	fx := float64(px) + subPixel(xl, resp[py*m+px], xr)
	fy := float64(py) + subPixel(yt, resp[py*m+px], yb)

	// The Gaussian target peaks at the model centre; convert the peak offset back
	// to image pixels through the window/model scale.
	scaleBack := float64(t.winSize()) / float64(m)
	center := float64(m-1) / 2
	t.cx += (fx - center) * scaleBack
	t.cy += (fy - center) * scaleBack

	conf := psr(resp, m, m, px, py, m/16+1)
	if conf >= t.MinPSR {
		nf := t.spectrum(gray)
		na := mulConj(t.g, nf)
		nb := mulConj(nf, nf)
		lr := t.LearnRate
		for i := range t.a.Data {
			t.a.Data[i] = complex(lr, 0)*na.Data[i] + complex(1-lr, 0)*t.a.Data[i]
			t.b.Data[i] = complex(lr, 0)*nb.Data[i] + complex(1-lr, 0)*t.b.Data[i]
		}
	}
	return t.box(gray), conf
}

// Update satisfies [Tracker]; the confidence flag is true when the PSR reaches
// MinPSR.
func (t *TrackerMOSSE) Update(frame *cv.Mat) (cv.Rect, bool) {
	box, conf := t.UpdateConfidence(frame)
	return box, conf >= t.MinPSR
}

// box returns the current bounding box, clamped to the frame.
func (t *TrackerMOSSE) box(gray *cv.Mat) cv.Rect {
	r := cv.Rect{
		X:      int(math.Round(t.cx - float64(t.w)/2)),
		Y:      int(math.Round(t.cy - float64(t.h)/2)),
		Width:  t.w,
		Height: t.h,
	}
	return clampRect(r, gray.Rows, gray.Cols)
}

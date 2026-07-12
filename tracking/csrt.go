package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TrackerCSRT is a lightweight Channel and Spatial Reliability Tracker (CSRT /
// CSR-DCF, Lukežič et al. 2017). Like the full algorithm it combines three
// ideas on top of a Fourier-domain correlation filter ([FFT2] / [IFFT2]):
//
//   - Multiple feature channels (here normalised intensity and gradient
//     magnitude), each with its own MOSSE-style filter.
//   - A spatial reliability map that restricts every filter to the pixels that
//     belong to the object, estimated from foreground/background intensity
//     histograms and a central prior, so background context cannot corrupt the
//     filter.
//   - Per-channel reliability weights (from each channel's learned response
//     sharpness) that weight the channels' contributions when the responses are
//     fused into the final detection map.
//
// It tracks translation at a fixed scale and reports the fused response's
// peak-to-sidelobe ratio as confidence. Construct it with [NewTrackerCSRT].
type TrackerCSRT struct {
	// ModelSize is the power-of-two side length the window is resized to.
	ModelSize int
	// Padding scales the search window relative to the object box.
	Padding float64
	// LearnRate blends new filters into the old each frame.
	LearnRate float64
	// Sigma is the Gaussian target width in model pixels.
	Sigma float64
	// MinPSR is the fused-response PSR below which Update reports low confidence.
	MinPSR float64

	a, b    []*ComplexMat // per-channel filter numerator/denominator
	weights []float64     // per-channel reliability
	g       *ComplexMat
	hann    []float64
	mask    []float64
	cx, cy  float64
	w, h    int
	inited  bool
}

const csrtChannels = 2

// NewTrackerCSRT returns a TrackerCSRT with sensible defaults (ModelSize 64,
// Padding 2.0, LearnRate 0.02, Sigma 2.0, MinPSR 4.0).
func NewTrackerCSRT() *TrackerCSRT {
	return &TrackerCSRT{ModelSize: 64, Padding: 2.0, LearnRate: 0.02, Sigma: 2.0, MinPSR: 4.0}
}

func (t *TrackerCSRT) winSize() int { return int(math.Round(float64(t.w) * t.Padding)) }

// channels returns the csrtChannels preprocessed, Hann-windowed and
// spatial-reliability-masked feature grids of the window centred on (cx, cy).
func (t *TrackerCSRT) channels(gray *cv.Mat) [][]float64 {
	patch := cropResizeGray(gray, t.cx, t.cy, t.winSize(), t.ModelSize)
	m := t.ModelSize
	vals := toFloat(patch)

	// Channel 0: log-normalised intensity.
	c0 := preprocessFilter(vals, t.hann)

	// Channel 1: gradient magnitude, zero-mean/unit-norm, Hann-windowed.
	grad := make([]float64, m*m)
	at := func(y, x int) float64 {
		y = clampInt(y, 0, m-1)
		x = clampInt(x, 0, m-1)
		return vals[y*m+x]
	}
	for y := 0; y < m; y++ {
		for x := 0; x < m; x++ {
			gx := at(y, x+1) - at(y, x-1)
			gy := at(y+1, x) - at(y-1, x)
			grad[y*m+x] = math.Hypot(gx, gy)
		}
	}
	var gm float64
	for _, v := range grad {
		gm += v
	}
	gm /= float64(len(grad))
	var gss float64
	for i := range grad {
		grad[i] -= gm
		gss += grad[i] * grad[i]
	}
	gnorm := math.Sqrt(gss) + 1e-5
	for i := range grad {
		grad[i] = grad[i] / gnorm * t.hann[i]
	}

	out := [][]float64{c0, grad}
	for c := range out {
		for i := range out[c] {
			out[c][i] *= t.mask[i]
		}
	}
	return out
}

// buildMask estimates the spatial reliability map from the model-sized window:
// a foreground/background intensity-histogram posterior modulated by a central
// Gaussian prior, normalised to [0,1].
func (t *TrackerCSRT) buildMask(patch *cv.Mat) []float64 {
	m := t.ModelSize
	// Object occupies the central 1/Padding fraction of the padded window.
	objHalf := int(float64(m) / (2 * t.Padding))
	cx, cy := m/2, m/2
	const nb = 16
	fg := make([]float64, nb)
	bg := make([]float64, nb)
	for y := 0; y < m; y++ {
		for x := 0; x < m; x++ {
			bin := int(patch.Data[y*m+x]) * nb / 256
			if bin >= nb {
				bin = nb - 1
			}
			if abs(x-cx) <= objHalf && abs(y-cy) <= objHalf {
				fg[bin]++
			} else {
				bg[bin]++
			}
		}
	}
	mask := make([]float64, m*m)
	prior := 2 * float64(objHalf)
	var maxv float64
	for y := 0; y < m; y++ {
		for x := 0; x < m; x++ {
			bin := int(patch.Data[y*m+x]) * nb / 256
			if bin >= nb {
				bin = nb - 1
			}
			post := 0.0
			if fg[bin]+bg[bin] > 0 {
				post = fg[bin] / (fg[bin] + bg[bin])
			}
			dx := float64(x - cx)
			dy := float64(y - cy)
			pr := math.Exp(-(dx*dx + dy*dy) / (2 * prior * prior))
			v := post * pr
			mask[y*m+x] = v
			if v > maxv {
				maxv = v
			}
		}
	}
	if maxv < 1e-6 {
		// Degenerate: fall back to the central prior alone.
		for i := range mask {
			mask[i] = 1
		}
		return mask
	}
	for i := range mask {
		mask[i] /= maxv
		if mask[i] < 0.1 {
			mask[i] = 0
		}
	}
	return mask
}

// Init builds the mask and per-channel filters from the object inside bbox.
func (t *TrackerCSRT) Init(frame *cv.Mat, bbox cv.Rect) {
	if !isPow2(t.ModelSize) {
		t.ModelSize = NextPow2(t.ModelSize)
	}
	gray := toGray(frame)
	b := clampRect(bbox, gray.Rows, gray.Cols)
	t.w, t.h = b.Width, b.Height
	t.cx, t.cy = rectCenter(b)
	m := t.ModelSize
	t.hann = HannWindow2D(m, m)
	t.g = FFT2(RealToComplex(GaussianResponse(m, m, t.Sigma), m, m))

	patch := cropResizeGray(gray, t.cx, t.cy, t.winSize(), m)
	t.mask = t.buildMask(patch)

	ch := t.channels(gray)
	t.a = make([]*ComplexMat, csrtChannels)
	t.b = make([]*ComplexMat, csrtChannels)
	t.weights = make([]float64, csrtChannels)
	for c := 0; c < csrtChannels; c++ {
		f := FFT2(RealToComplex(ch[c], m, m))
		t.a[c] = mulConj(t.g, f)
		t.b[c] = mulConj(f, f)
		for i := range t.b[c].Data {
			t.b[c].Data[i] += complex(1e-3, 0)
		}
		// Channel reliability from the sharpness of its self-response.
		resp := t.channelResponse(c, f)
		px, py, _ := peakLoc(resp, m, m)
		t.weights[c] = math.Max(0, psr(resp, m, m, px, py, m/16+1))
	}
	t.normalizeWeights()
	t.inited = true
}

func (t *TrackerCSRT) normalizeWeights() {
	var sum float64
	for _, w := range t.weights {
		sum += w
	}
	if sum < 1e-9 {
		for c := range t.weights {
			t.weights[c] = 1.0 / float64(len(t.weights))
		}
		return
	}
	for c := range t.weights {
		t.weights[c] /= sum
	}
}

// channelResponse correlates channel c's filter with spectrum f.
func (t *TrackerCSRT) channelResponse(c int, f *ComplexMat) []float64 {
	m := t.ModelSize
	spec := NewComplexMat(m, m)
	for i := range spec.Data {
		spec.Data[i] = f.Data[i] * (t.a[c].Data[i] / t.b[c].Data[i])
	}
	return IFFT2(spec).Real()
}

// UpdateConfidence fuses the per-channel responses, moves the box to the peak,
// and returns the box with the fused-response PSR. It panics before Init.
func (t *TrackerCSRT) UpdateConfidence(frame *cv.Mat) (cv.Rect, float64) {
	if !t.inited {
		panic("tracking: TrackerCSRT.Update called before Init")
	}
	gray := toGray(frame)
	m := t.ModelSize
	ch := t.channels(gray)

	fused := make([]float64, m*m)
	specs := make([]*ComplexMat, csrtChannels)
	for c := 0; c < csrtChannels; c++ {
		f := FFT2(RealToComplex(ch[c], m, m))
		specs[c] = f
		resp := t.channelResponse(c, f)
		for i := range fused {
			fused[i] += t.weights[c] * resp[i]
		}
	}

	px, py, _ := peakLoc(fused, m, m)
	xl := fused[py*m+(px-1+m)%m]
	xr := fused[py*m+(px+1)%m]
	yt := fused[((py-1+m)%m)*m+px]
	yb := fused[((py+1)%m)*m+px]
	fx := float64(px) + subPixel(xl, fused[py*m+px], xr)
	fy := float64(py) + subPixel(yt, fused[py*m+px], yb)

	scaleBack := float64(t.winSize()) / float64(m)
	center := float64(m-1) / 2
	t.cx += (fx - center) * scaleBack
	t.cy += (fy - center) * scaleBack

	conf := psr(fused, m, m, px, py, m/16+1)
	if conf >= t.MinPSR {
		lr := t.LearnRate
		// Recompute channels at the refined position for the online update.
		ch2 := t.channels(gray)
		for c := 0; c < csrtChannels; c++ {
			nf := FFT2(RealToComplex(ch2[c], m, m))
			na := mulConj(t.g, nf)
			nb := mulConj(nf, nf)
			for i := range t.a[c].Data {
				t.a[c].Data[i] = complex(lr, 0)*na.Data[i] + complex(1-lr, 0)*t.a[c].Data[i]
				t.b[c].Data[i] = complex(lr, 0)*nb.Data[i] + complex(1-lr, 0)*t.b[c].Data[i]
			}
		}
	}
	return t.box(gray), conf
}

// Update satisfies [Tracker]; the flag is true when the fused PSR reaches MinPSR.
func (t *TrackerCSRT) Update(frame *cv.Mat) (cv.Rect, bool) {
	box, conf := t.UpdateConfidence(frame)
	return box, conf >= t.MinPSR
}

// ChannelWeights returns a copy of the current per-channel reliability weights
// (intensity then gradient), which sum to 1.
func (t *TrackerCSRT) ChannelWeights() []float64 {
	out := make([]float64, len(t.weights))
	copy(out, t.weights)
	return out
}

func (t *TrackerCSRT) box(gray *cv.Mat) cv.Rect {
	r := cv.Rect{
		X:      int(math.Round(t.cx - float64(t.w)/2)),
		Y:      int(math.Round(t.cy - float64(t.h)/2)),
		Width:  t.w,
		Height: t.h,
	}
	return clampRect(r, gray.Rows, gray.Cols)
}

package bioinspired

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// eps guards divisions in the Naka-Rushton compression against zero denominators.
const eps = 1e-6

// maxSample is the maximum 8-bit sample value the model normalises against.
const maxSample = 255.0

// frame is an internal single-channel image of float64 samples in row-major
// order. The retina model works entirely in floating point to preserve the
// small signals produced by band-pass filtering and temporal differencing;
// results are quantised back to [cv.Mat] only at the output boundary.
type frame struct {
	rows, cols int
	data       []float64
}

// newFrame allocates a zero-filled frame.
func newFrame(rows, cols int) *frame {
	return &frame{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

// clone returns an independent copy of f.
func (f *frame) clone() *frame {
	out := &frame{rows: f.rows, cols: f.cols, data: make([]float64, len(f.data))}
	copy(out.data, f.data)
	return out
}

// zero resets every sample to 0 in place.
func (f *frame) zero() {
	for i := range f.data {
		f.data[i] = 0
	}
}

// spatialConstantToCoeff maps a spatial "constant" (a length scale, in pixels)
// to the leak coefficient of a first-order recursive filter. A constant of 0
// disables smoothing (coefficient 0); larger constants push the coefficient
// towards 1 (stronger smoothing). The mapping a = exp(-1/constant) mirrors the
// exponential impulse response of the cellular low-pass filters in the retina.
func spatialConstantToCoeff(constant float64) float64 {
	if constant <= 0 {
		return 0
	}
	a := math.Exp(-1.0 / constant)
	if a < 0 {
		return 0
	}
	if a > 0.999 {
		return 0.999
	}
	return a
}

// spatialLowPass applies a separable, zero-phase, first-order recursive
// low-pass filter with leak coefficient a in [0,1) and returns a new frame. The
// filter is run causally then anti-causally along rows and then columns, which
// approximates an isotropic Gaussian blur with a two-sided exponential kernel.
// Because each 1-D pass has unit DC gain, the overall filter preserves the mean
// of a constant region — it removes high-frequency noise without darkening or
// brightening flat areas. This models the space-invariant diffusion performed
// by horizontal and amacrine cells.
func spatialLowPass(f *frame, a float64) *frame {
	out := f.clone()
	if a <= 0 {
		return out
	}
	b := 1 - a
	rows, cols := f.rows, f.cols
	d := out.data
	// Rows: causal (left to right) then anti-causal (right to left).
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 1; x < cols; x++ {
			d[base+x] = a*d[base+x-1] + b*d[base+x]
		}
		for x := cols - 2; x >= 0; x-- {
			d[base+x] = a*d[base+x+1] + b*d[base+x]
		}
	}
	// Columns: causal (top to bottom) then anti-causal (bottom to top).
	for x := 0; x < cols; x++ {
		for y := 1; y < rows; y++ {
			d[y*cols+x] = a*d[(y-1)*cols+x] + b*d[y*cols+x]
		}
		for y := rows - 2; y >= 0; y-- {
			d[y*cols+x] = a*d[(y+1)*cols+x] + b*d[y*cols+x]
		}
	}
	return out
}

// temporalUpdate advances an exponential temporal low-pass state in place:
//
//	state = k*state + (1-k)*in
//
// k in [0,1) is the retention factor; it is derived from a temporal constant so
// that a static input eventually drives state to that input. The pre-update
// value of state is the model's estimate of the temporally averaged signal,
// which callers subtract from the instantaneous input to obtain a transient
// (temporal high-pass) response.
func temporalUpdate(state, in *frame, k float64) {
	b := 1 - k
	for i := range state.data {
		state.data[i] = k*state.data[i] + b*in.data[i]
	}
}

// temporalConstantToRetention maps a temporal constant (in frames) to the
// retention factor k of the recursive temporal filter. A constant of 0 means no
// memory (k=0); larger constants approach k=1 (long memory, slow adaptation).
func temporalConstantToRetention(constant float64) float64 {
	if constant <= 0 {
		return 0
	}
	k := math.Exp(-1.0 / constant)
	if k > 0.999 {
		return 0.999
	}
	return k
}

// nakaRushton applies the Michaelis-Menten / Naka-Rushton local luminance
// adaptation used by photoreceptor and ganglion cells:
//
//	out = (maxV + X0) * in / (in + X0)
//	X0  = sensitivity*local + maxV*(1 - sensitivity)
//
// local is a spatially/temporally low-passed reference luminance. Where the
// local reference is dark, X0 is small and the response gain (maxV+X0)/X0 is
// large, so faint dark detail is amplified; where it is bright, the response
// saturates towards maxV. This compresses dynamic range while enhancing local
// contrast — the core of both retinal adaptation and fast tone mapping. Inputs
// are clamped to be non-negative before compression.
func nakaRushton(in, local *frame, sensitivity, maxV float64) *frame {
	out := newFrame(in.rows, in.cols)
	for i := range in.data {
		v := in.data[i]
		if v < 0 {
			v = 0
		}
		x0 := sensitivity*local.data[i] + maxV*(1-sensitivity)
		if x0 < 0 {
			x0 = 0
		}
		out.data[i] = (maxV + x0) * v / (v + x0 + eps)
	}
	return out
}

// matToFrames converts a [cv.Mat] into one frame per channel (values 0..255 as
// float64). It panics if the Mat is empty.
func matToFrames(m *cv.Mat) []*frame {
	if m.Empty() {
		panic("bioinspired: input Mat is empty")
	}
	chans := make([]*frame, m.Channels)
	for c := 0; c < m.Channels; c++ {
		chans[c] = newFrame(m.Rows, m.Cols)
	}
	n := m.Rows * m.Cols
	for p := 0; p < n; p++ {
		base := p * m.Channels
		for c := 0; c < m.Channels; c++ {
			chans[c].data[p] = float64(m.Data[base+c])
		}
	}
	return chans
}

// luminance returns the BT.601 luma of the channels: the single channel itself
// when len(chans)==1, otherwise 0.299R+0.587G+0.114B. It panics for unsupported
// channel counts.
func luminance(chans []*frame) *frame {
	switch len(chans) {
	case 1:
		return chans[0].clone()
	case 3:
		out := newFrame(chans[0].rows, chans[0].cols)
		for i := range out.data {
			out.data[i] = 0.299*chans[0].data[i] + 0.587*chans[1].data[i] + 0.114*chans[2].data[i]
		}
		return out
	default:
		// Fall back to a plain channel average.
		out := newFrame(chans[0].rows, chans[0].cols)
		inv := 1.0 / float64(len(chans))
		for i := range out.data {
			var s float64
			for _, ch := range chans {
				s += ch.data[i]
			}
			out.data[i] = s * inv
		}
		return out
	}
}

// framesToMat quantises float channels into an 8-bit [cv.Mat], clamping to
// [0,255] and rounding to nearest.
func framesToMat(chans []*frame) *cv.Mat {
	rows, cols, nc := chans[0].rows, chans[0].cols, len(chans)
	out := cv.NewMat(rows, cols, nc)
	n := rows * cols
	for p := 0; p < n; p++ {
		base := p * nc
		for c := 0; c < nc; c++ {
			out.Data[base+c] = clampRound(chans[c].data[p])
		}
	}
	return out
}

// frameToFloatMat copies a frame into a [cv.FloatMat], preserving the raw
// floating-point response.
func frameToFloatMat(f *frame) *cv.FloatMat {
	out := cv.NewFloatMat(f.rows, f.cols)
	copy(out.Data, f.data)
	return out
}

// clampRound rounds v to the nearest integer and clamps it into [0,255].
func clampRound(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// absFrame returns the element-wise absolute value of f.
func absFrame(f *frame) *frame {
	out := newFrame(f.rows, f.cols)
	for i := range f.data {
		out.data[i] = math.Abs(f.data[i])
	}
	return out
}

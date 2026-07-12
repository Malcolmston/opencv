package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// clampInt clamps v to [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// cropClamped extracts a w×h single-channel patch whose top-left sample is at
// image coordinate (x0, y0), replicating the border for samples that fall
// outside the image (OpenCV's BORDER_REPLICATE). Unlike [cv.Mat.Region] it never
// panics on an out-of-range rectangle, which lets the correlation trackers use a
// padded search window that overhangs the frame. m must be single-channel.
func cropClamped(m *cv.Mat, x0, y0, w, h int) *cv.Mat {
	out := cv.NewMat(h, w, 1)
	for j := 0; j < h; j++ {
		yy := clampInt(y0+j, 0, m.Rows-1)
		base := yy * m.Cols
		for i := 0; i < w; i++ {
			xx := clampInt(x0+i, 0, m.Cols-1)
			out.Data[j*w+i] = m.Data[base+xx]
		}
	}
	return out
}

// cropResizeGray crops a padded window of size win×win centred on the fractional
// point (cx, cy) of the single-channel image gray and resizes it to model×model
// with bilinear interpolation. It returns the resized patch. This is the common
// front end of every Fourier-domain tracker: it normalises the object to a fixed
// power-of-two model size so [FFT2] applies regardless of the object's pixel size.
func cropResizeGray(gray *cv.Mat, cx, cy float64, win, model int) *cv.Mat {
	x0 := int(math.Round(cx - float64(win)/2))
	y0 := int(math.Round(cy - float64(win)/2))
	patch := cropClamped(gray, x0, y0, win, win)
	if win == model {
		return patch
	}
	return cv.Resize(patch, model, model, cv.InterLinear)
}

// toFloat returns the samples of a single-channel Mat as float64 in row-major
// order.
func toFloat(m *cv.Mat) []float64 {
	out := make([]float64, len(m.Data))
	for i, v := range m.Data {
		out[i] = float64(v)
	}
	return out
}

// preprocessFilter applies the standard correlation-filter preprocessing to an
// n-length patch: a log transform to compress illumination, normalisation to
// zero mean and unit norm, and multiplication by the Hann window (which must be
// the same length). It returns a fresh slice and does not modify its inputs.
func preprocessFilter(vals, hann []float64) []float64 {
	out := make([]float64, len(vals))
	var mean float64
	for i, v := range vals {
		lv := math.Log(v + 1)
		out[i] = lv
		mean += lv
	}
	mean /= float64(len(vals))
	var ss float64
	for i := range out {
		out[i] -= mean
		ss += out[i] * out[i]
	}
	norm := math.Sqrt(ss) + 1e-5
	for i := range out {
		out[i] = out[i] / norm * hann[i]
	}
	return out
}

// mulConj returns the element-wise product a·conj(b) of two equally sized
// spectra.
func mulConj(a, b *ComplexMat) *ComplexMat {
	out := NewComplexMat(a.Rows, a.Cols)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * conj(b.Data[i])
	}
	return out
}

// conj returns the complex conjugate of z.
func conj(z complex128) complex128 { return complex(real(z), -imag(z)) }

// peakLoc returns the location and value of the maximum of a real row-major
// response grid.
func peakLoc(resp []float64, rows, cols int) (px, py int, val float64) {
	val = math.Inf(-1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := resp[y*cols+x]
			if v > val {
				val = v
				px, py = x, y
			}
		}
	}
	return
}

// subPixel refines an integer peak coordinate p (along one axis of length n,
// circular) to sub-sample accuracy by fitting a parabola through the responses
// at p-1, p and p+1. It returns the fractional offset in [-0.5, 0.5] to add to p.
func subPixel(prev, cur, next float64) float64 {
	den := prev - 2*cur + next
	if den == 0 {
		return 0
	}
	d := 0.5 * (prev - next) / den
	if d > 0.5 {
		d = 0.5
	} else if d < -0.5 {
		d = -0.5
	}
	return d
}

// wrapCoord maps a circular response index i (0..n-1) to a signed shift about
// zero: indices past the midpoint wrap to negative displacements.
func wrapCoord(i, n int) float64 {
	if i > n/2 {
		return float64(i - n)
	}
	return float64(i)
}

// psr computes the peak-to-sidelobe ratio of a response grid: the peak minus the
// mean of the samples outside an exclusion window around the peak, divided by
// their standard deviation. A high PSR (typically > 7) indicates a confident,
// well-localised detection; MOSSE uses it as its confidence measure.
func psr(resp []float64, rows, cols, px, py, exclude int) float64 {
	var peak float64 = resp[py*cols+px]
	var sum, sumSq float64
	var n int
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if abs(x-px) <= exclude && abs(y-py) <= exclude {
				continue
			}
			v := resp[y*cols+x]
			sum += v
			sumSq += v * v
			n++
		}
	}
	if n == 0 {
		return 0
	}
	mean := sum / float64(n)
	varc := sumSq/float64(n) - mean*mean
	if varc < 1e-12 {
		return 0
	}
	return (peak - mean) / math.Sqrt(varc)
}

// abs returns the absolute value of an int.
func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// gaussianCorrelation returns the FFT of the Gaussian kernel-correlation between
// two feature stacks (one spectrum per channel) with bandwidth sigma, following
// Henriques et al.'s KCF formulation. xf and yf must be equally sized slices of
// equally shaped spectra; numel is rows*cols. The spatial kernel is
//
//	k = exp(-1/sigma² · max(0, (‖x‖²+‖y‖²-2·xy) / (numel·channels)))
//
// where xy is the channel-summed cross-correlation recovered by an inverse FFT.
// The returned value is FFT2(k), ready to multiply the learned coefficients.
func gaussianCorrelation(xf, yf []*ComplexMat, sigma float64) *ComplexMat {
	rows, cols := xf[0].Rows, xf[0].Cols
	numel := float64(rows * cols)
	channels := float64(len(xf))

	var xx, yy float64
	xy := make([]float64, rows*cols)
	for c := range xf {
		for i := range xf[c].Data {
			xv := xf[c].Data[i]
			yv := yf[c].Data[i]
			xx += real(xv)*real(xv) + imag(xv)*imag(xv)
			yy += real(yv)*real(yv) + imag(yv)*imag(yv)
		}
		prod := mulConj(xf[c], yf[c])
		sp := IFFT2(prod)
		for i := range sp.Data {
			xy[i] += real(sp.Data[i])
		}
	}
	xx /= numel
	yy /= numel

	k := make([]float64, rows*cols)
	scale := 1 / (numel * channels)
	for i := range k {
		d := (xx + yy - 2*xy[i]) * scale
		if d < 0 {
			d = 0
		}
		k[i] = math.Exp(-d / (sigma * sigma))
	}
	return FFT2(RealToComplex(k, rows, cols))
}

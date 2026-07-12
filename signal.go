package cv

import (
	"fmt"
	"math"
)

// dft1d evaluates the (inverse) discrete Fourier transform of a single complex
// signal by direct summation. It does not apply any normalisation.
func dft1d(re, im []float64, inverse bool) (outRe, outIm []float64) {
	n := len(re)
	outRe = make([]float64, n)
	outIm = make([]float64, n)
	sgn := -1.0
	if inverse {
		sgn = 1.0
	}
	for k := 0; k < n; k++ {
		var sr, si float64
		for t := 0; t < n; t++ {
			ang := sgn * 2 * math.Pi * float64(k) * float64(t) / float64(n)
			c := math.Cos(ang)
			s := math.Sin(ang)
			sr += re[t]*c - im[t]*s
			si += re[t]*s + im[t]*c
		}
		outRe[k] = sr
		outIm[k] = si
	}
	return outRe, outIm
}

// dft2d applies a separable 2-D DFT (rows then columns) to the complex plane
// (re, im), returning the transformed plane. No normalisation is applied.
func dft2d(re, im *FloatMat, inverse bool) (*FloatMat, *FloatMat) {
	rows, cols := re.Rows, re.Cols
	outRe := fclone(re)
	outIm := fclone(im)
	rr := make([]float64, cols)
	ri := make([]float64, cols)
	for y := 0; y < rows; y++ {
		copy(rr, outRe.Data[y*cols:(y+1)*cols])
		copy(ri, outIm.Data[y*cols:(y+1)*cols])
		tr, ti := dft1d(rr, ri, inverse)
		copy(outRe.Data[y*cols:(y+1)*cols], tr)
		copy(outIm.Data[y*cols:(y+1)*cols], ti)
	}
	cr := make([]float64, rows)
	ci := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			cr[y] = outRe.Data[y*cols+x]
			ci[y] = outIm.Data[y*cols+x]
		}
		tr, ti := dft1d(cr, ci, inverse)
		for y := 0; y < rows; y++ {
			outRe.Data[y*cols+x] = tr[y]
			outIm.Data[y*cols+x] = ti[y]
		}
	}
	return outRe, outIm
}

// DFT computes the forward 2-D discrete Fourier transform of a complex image
// given as separate real and imaginary planes, returning the complex spectrum
// as (real, imaginary) planes. For a real input pass a zero-filled imaginary
// plane. It panics on a size mismatch.
func DFT(re, im *FloatMat) (outRe, outIm *FloatMat) {
	requireSameFloatShape(re, im, "DFT")
	return dft2d(re, im, false)
}

// IDFT computes the inverse 2-D discrete Fourier transform of a complex
// spectrum given as (real, imaginary) planes. When scale is true the result is
// divided by the number of elements, so that IDFT(DFT(x)) recovers x. It panics
// on a size mismatch.
func IDFT(re, im *FloatMat, scale bool) (outRe, outIm *FloatMat) {
	requireSameFloatShape(re, im, "IDFT")
	outRe, outIm = dft2d(re, im, true)
	if scale {
		n := float64(re.Rows * re.Cols)
		for i := range outRe.Data {
			outRe.Data[i] /= n
			outIm.Data[i] /= n
		}
	}
	return outRe, outIm
}

// MulSpectrums multiplies two complex spectra element-wise, optionally
// conjugating the second operand (as needed for correlation). Each spectrum is
// a pair of (real, imaginary) planes of matching size. It panics on a size
// mismatch.
func MulSpectrums(aRe, aIm, bRe, bIm *FloatMat, conjB bool) (re, im *FloatMat) {
	requireSameFloatShape(aRe, bRe, "MulSpectrums")
	re = NewFloatMat(aRe.Rows, aRe.Cols)
	im = NewFloatMat(aRe.Rows, aRe.Cols)
	for i := range aRe.Data {
		ar, ai := aRe.Data[i], aIm.Data[i]
		br, bi := bRe.Data[i], bIm.Data[i]
		if conjB {
			bi = -bi
		}
		re.Data[i] = ar*br - ai*bi
		im.Data[i] = ar*bi + ai*br
	}
	return re, im
}

// dct1d applies the orthonormal DCT-II (forward) or DCT-III (inverse) to a
// single signal.
func dct1d(x []float64, inverse bool) []float64 {
	n := len(x)
	out := make([]float64, n)
	fn := float64(n)
	if !inverse {
		for k := 0; k < n; k++ {
			var s float64
			for t := 0; t < n; t++ {
				s += x[t] * math.Cos(math.Pi*(float64(t)+0.5)*float64(k)/fn)
			}
			alpha := math.Sqrt(2 / fn)
			if k == 0 {
				alpha = math.Sqrt(1 / fn)
			}
			out[k] = alpha * s
		}
		return out
	}
	for t := 0; t < n; t++ {
		var s float64
		for k := 0; k < n; k++ {
			alpha := math.Sqrt(2 / fn)
			if k == 0 {
				alpha = math.Sqrt(1 / fn)
			}
			s += alpha * x[k] * math.Cos(math.Pi*(float64(t)+0.5)*float64(k)/fn)
		}
		out[t] = s
	}
	return out
}

// dct2d applies a separable 2-D DCT (rows then columns).
func dct2d(src *FloatMat, inverse bool) *FloatMat {
	rows, cols := src.Rows, src.Cols
	out := fclone(src)
	row := make([]float64, cols)
	for y := 0; y < rows; y++ {
		copy(row, out.Data[y*cols:(y+1)*cols])
		tr := dct1d(row, inverse)
		copy(out.Data[y*cols:(y+1)*cols], tr)
	}
	col := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			col[y] = out.Data[y*cols+x]
		}
		tc := dct1d(col, inverse)
		for y := 0; y < rows; y++ {
			out.Data[y*cols+x] = tc[y]
		}
	}
	return out
}

// DCT computes the 2-D discrete cosine transform (type-II, orthonormal) of a
// real matrix. Use [IDCT] to invert it.
func DCT(src *FloatMat) *FloatMat {
	return dct2d(src, false)
}

// IDCT computes the inverse 2-D discrete cosine transform (type-III,
// orthonormal), so that IDCT(DCT(x)) recovers x.
func IDCT(src *FloatMat) *FloatMat {
	return dct2d(src, true)
}

// CreateHanningWindow builds a rows×cols Hann (raised-cosine) window, the 2-D
// separable product of 1-D Hann windows. It is typically multiplied into an
// image before [PhaseCorrelate] to suppress edge effects. It panics unless both
// dimensions are positive.
func CreateHanningWindow(rows, cols int) *FloatMat {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("cv: CreateHanningWindow requires positive size, got %dx%d", rows, cols))
	}
	wr := make([]float64, rows)
	for i := 0; i < rows; i++ {
		if rows == 1 {
			wr[i] = 1
		} else {
			wr[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(rows-1)))
		}
	}
	wc := make([]float64, cols)
	for j := 0; j < cols; j++ {
		if cols == 1 {
			wc[j] = 1
		} else {
			wc[j] = 0.5 * (1 - math.Cos(2*math.Pi*float64(j)/float64(cols-1)))
		}
	}
	out := NewFloatMat(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out.Data[i*cols+j] = wr[i] * wc[j]
		}
	}
	return out
}

// PhaseCorrelate estimates the translational shift between two real images a
// and b using the phase-correlation method: it multiplies the cross-power
// spectrum, normalises it and locates the peak of the inverse transform. It
// returns the sub-pixel shift (shiftX, shiftY) that maps a onto b and the peak
// response in [0,1]. Shifts larger than half a dimension are reported as
// negative. It panics on a size mismatch.
func PhaseCorrelate(a, b *FloatMat) (shiftX, shiftY, response float64) {
	requireSameFloatShape(a, b, "PhaseCorrelate")
	rows, cols := a.Rows, a.Cols
	zeros := NewFloatMat(rows, cols)
	aRe, aIm := dft2d(a, zeros, false)
	bRe, bIm := dft2d(b, zeros, false)
	// Cross-power spectrum B·conj(A); its inverse transform peaks at the
	// (positive) shift that maps a onto b.
	cRe, cIm := MulSpectrums(bRe, bIm, aRe, aIm, true)
	for i := range cRe.Data {
		mag := math.Hypot(cRe.Data[i], cIm.Data[i])
		if mag > 1e-12 {
			cRe.Data[i] /= mag
			cIm.Data[i] /= mag
		}
	}
	rRe, _ := dft2d(cRe, cIm, true)
	n := float64(rows * cols)
	peak := math.Inf(-1)
	peakX, peakY := 0, 0
	var total float64
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := rRe.Data[y*cols+x] / n
			total += math.Abs(v)
			if v > peak {
				peak = v
				peakX, peakY = x, y
			}
		}
	}
	sx := peakX
	if sx > cols/2 {
		sx -= cols
	}
	sy := peakY
	if sy > rows/2 {
		sy -= rows
	}
	if total > 0 {
		response = peak * n / total
		if response > 1 {
			response = 1
		}
	}
	return float64(sx), float64(sy), response
}

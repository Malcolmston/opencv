package freqdomain

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// IsPowerOfTwo reports whether n is a positive power of two.
func IsPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0
}

// NextPowerOfTwo returns the smallest power of two that is greater than or
// equal to n. For n <= 1 it returns 1.
func NextPowerOfTwo(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// dft1dDirect evaluates the length-n (inverse) DFT by direct summation. It is
// the exact O(n²) fallback used when n is not a power of two. No normalisation
// is applied.
func dft1dDirect(re, im []float64, inverse bool) (outRe, outIm []float64) {
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

// fftRadix2 performs an in-place iterative Cooley-Tukey FFT on a power-of-two
// length signal. When inverse is true the conjugate kernel is used; no
// normalisation is applied.
func fftRadix2(re, im []float64, inverse bool) {
	n := len(re)
	// Bit-reversal permutation.
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := 2 * math.Pi / float64(length)
		if !inverse {
			ang = -ang
		}
		wr, wi := math.Cos(ang), math.Sin(ang)
		half := length >> 1
		for i := 0; i < n; i += length {
			cr, ci := 1.0, 0.0
			for k := 0; k < half; k++ {
				a := i + k
				b := a + half
				vr := re[b]*cr - im[b]*ci
				vi := re[b]*ci + im[b]*cr
				re[b] = re[a] - vr
				im[b] = im[a] - vi
				re[a] = re[a] + vr
				im[a] = im[a] + vi
				cr, ci = cr*wr-ci*wi, cr*wi+ci*wr
			}
		}
	}
}

// fft1d transforms a single complex signal in place, dispatching to the
// radix-2 kernel for power-of-two lengths and to the direct DFT otherwise. No
// normalisation is applied.
func fft1d(re, im []float64, inverse bool) {
	n := len(re)
	if n <= 1 {
		return
	}
	if n&(n-1) == 0 {
		fftRadix2(re, im, inverse)
		return
	}
	r, ii := dft1dDirect(re, im, inverse)
	copy(re, r)
	copy(im, ii)
}

// FFT1D returns the forward discrete Fourier transform of a single complex
// signal given as parallel real and imaginary slices. The inputs are not
// modified. It panics if the two slices differ in length.
func FFT1D(re, im []float64) (outRe, outIm []float64) {
	if len(re) != len(im) {
		panic("freqdomain: FFT1D length mismatch")
	}
	outRe = append([]float64(nil), re...)
	outIm = append([]float64(nil), im...)
	fft1d(outRe, outIm, false)
	return outRe, outIm
}

// IFFT1D returns the inverse discrete Fourier transform of a single complex
// signal, normalised by the signal length so that IFFT1D(FFT1D(x)) recovers x.
// The inputs are not modified. It panics if the two slices differ in length.
func IFFT1D(re, im []float64) (outRe, outIm []float64) {
	if len(re) != len(im) {
		panic("freqdomain: IFFT1D length mismatch")
	}
	outRe = append([]float64(nil), re...)
	outIm = append([]float64(nil), im...)
	fft1d(outRe, outIm, true)
	n := float64(len(outRe))
	for i := range outRe {
		outRe[i] /= n
		outIm[i] /= n
	}
	return outRe, outIm
}

// fft2dInPlace transforms the complex planes (re, im) of a rows×cols image in
// place: a 1-D transform along every row followed by one along every column. No
// normalisation is applied.
func fft2dInPlace(rows, cols int, re, im []float64, inverse bool) {
	rr := make([]float64, cols)
	ri := make([]float64, cols)
	for y := 0; y < rows; y++ {
		off := y * cols
		copy(rr, re[off:off+cols])
		copy(ri, im[off:off+cols])
		fft1d(rr, ri, inverse)
		copy(re[off:off+cols], rr)
		copy(im[off:off+cols], ri)
	}
	cr := make([]float64, rows)
	ci := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			cr[y] = re[y*cols+x]
			ci[y] = im[y*cols+x]
		}
		fft1d(cr, ci, inverse)
		for y := 0; y < rows; y++ {
			re[y*cols+x] = cr[y]
			im[y*cols+x] = ci[y]
		}
	}
}

// FFT2DComplex returns the forward 2-D discrete Fourier transform of a complex
// image. The input spectrum is not modified.
func FFT2DComplex(s *Spectrum) *Spectrum {
	out := s.Clone()
	fft2dInPlace(out.Rows, out.Cols, out.Re, out.Im, false)
	return out
}

// IFFT2DComplex returns the inverse 2-D discrete Fourier transform of a complex
// spectrum, normalised by the number of elements so that
// IFFT2DComplex(FFT2DComplex(x)) recovers x. The input is not modified.
func IFFT2DComplex(s *Spectrum) *Spectrum {
	out := s.Clone()
	fft2dInPlace(out.Rows, out.Cols, out.Re, out.Im, true)
	n := float64(out.Rows * out.Cols)
	for i := range out.Re {
		out.Re[i] /= n
		out.Im[i] /= n
	}
	return out
}

// FFT2D returns the forward 2-D discrete Fourier transform of a real image as a
// complex [Spectrum].
func FFT2D(f *cv.FloatMat) *Spectrum {
	return FFT2DComplex(SpectrumFromFloat(f))
}

// IFFT2D returns the real part of the inverse 2-D discrete Fourier transform of
// a spectrum, normalised so that IFFT2D(FFT2D(x)) recovers x. The imaginary
// part, which is numerically zero for a spectrum produced from a real image, is
// discarded.
func IFFT2D(s *Spectrum) *cv.FloatMat {
	inv := IFFT2DComplex(s)
	return inv.RealPlane()
}

// DFT2D returns the forward 2-D transform of a real image computed by direct
// summation. It is numerically equivalent to [FFT2D] and exists mainly for
// cross-validation; prefer FFT2D for speed.
func DFT2D(f *cv.FloatMat) *Spectrum {
	out := SpectrumFromFloat(f)
	rows, cols := out.Rows, out.Cols
	for y := 0; y < rows; y++ {
		off := y * cols
		r, i := dft1dDirect(out.Re[off:off+cols], out.Im[off:off+cols], false)
		copy(out.Re[off:off+cols], r)
		copy(out.Im[off:off+cols], i)
	}
	cr := make([]float64, rows)
	ci := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			cr[y] = out.Re[y*cols+x]
			ci[y] = out.Im[y*cols+x]
		}
		r, i := dft1dDirect(cr, ci, false)
		for y := 0; y < rows; y++ {
			out.Re[y*cols+x] = r[y]
			out.Im[y*cols+x] = i[y]
		}
	}
	return out
}

// IDFT2D returns the real part of the inverse 2-D transform of a spectrum
// computed by direct summation, normalised by the number of elements. It is the
// direct-DFT counterpart of [IFFT2D].
func IDFT2D(s *Spectrum) *cv.FloatMat {
	tmp := s.Clone()
	rows, cols := tmp.Rows, tmp.Cols
	for y := 0; y < rows; y++ {
		off := y * cols
		r, i := dft1dDirect(tmp.Re[off:off+cols], tmp.Im[off:off+cols], true)
		copy(tmp.Re[off:off+cols], r)
		copy(tmp.Im[off:off+cols], i)
	}
	cr := make([]float64, rows)
	ci := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			cr[y] = tmp.Re[y*cols+x]
			ci[y] = tmp.Im[y*cols+x]
		}
		r, i := dft1dDirect(cr, ci, true)
		for y := 0; y < rows; y++ {
			tmp.Re[y*cols+x] = r[y]
			tmp.Im[y*cols+x] = i[y]
		}
	}
	n := float64(rows * cols)
	out := cv.NewFloatMat(rows, cols)
	for i := range out.Data {
		out.Data[i] = tmp.Re[i] / n
	}
	return out
}

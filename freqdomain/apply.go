package freqdomain

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// ApplyFilterSpectrum applies a centred real transfer function to a spectrum
// and returns the filtered spectrum. The filter must be given in centred form
// (DC at the middle, as produced by the filter constructors); it is
// ifftshifted internally to align with the un-shifted spectrum before
// element-wise multiplication. It panics on a size mismatch.
func ApplyFilterSpectrum(s *Spectrum, filter *cv.FloatMat) *Spectrum {
	if s.Rows != filter.Rows || s.Cols != filter.Cols {
		panic(fmt.Sprintf("freqdomain: ApplyFilterSpectrum shape mismatch %dx%d vs %dx%d", s.Rows, s.Cols, filter.Rows, filter.Cols))
	}
	h := IFFTShift(filter)
	return s.MulReal(h)
}

// ApplyFilter applies a centred real transfer function to a real image in the
// frequency domain and returns the real filtered image. It performs the forward
// transform, multiplies by the (centred) filter, inverts the transform and
// returns the real part. It panics on a size mismatch.
func ApplyFilter(f *cv.FloatMat, filter *cv.FloatMat) *cv.FloatMat {
	if f.Rows != filter.Rows || f.Cols != filter.Cols {
		panic(fmt.Sprintf("freqdomain: ApplyFilter shape mismatch %dx%d vs %dx%d", f.Rows, f.Cols, filter.Rows, filter.Cols))
	}
	return IFFT2D(ApplyFilterSpectrum(FFT2D(f), filter))
}

// FilterImage is a convenience wrapper that filters an 8-bit image with a
// centred transfer function and returns an 8-bit image. It converts the input
// to float with [MatToFloat], applies filter with [ApplyFilter] and rescales
// the result to [0,255] with [FloatToMatScaled]. It panics on a size mismatch.
func FilterImage(m *cv.Mat, filter *cv.FloatMat) *cv.Mat {
	f := MatToFloat(m)
	return FloatToMatScaled(ApplyFilter(f, filter))
}

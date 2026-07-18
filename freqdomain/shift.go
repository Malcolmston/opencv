package freqdomain

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// shiftIndex returns the source index that maps to destination index i under a
// circular shift by half the length. When inverse is false it performs the
// forward fftshift (moving the DC term to the centre); when inverse is true it
// performs the ifftshift (the exact inverse, which differs from fftshift only
// for odd lengths).
func shiftAmount(n int, inverse bool) int {
	if inverse {
		return (n + 1) / 2
	}
	return n / 2
}

// circularShift2D returns a new slice with the rows×cols data shifted circularly
// by (dy, dx).
func circularShift2D(rows, cols int, data []float64, dy, dx int) []float64 {
	out := make([]float64, rows*cols)
	dy = ((dy % rows) + rows) % rows
	dx = ((dx % cols) + cols) % cols
	for y := 0; y < rows; y++ {
		sy := y - dy
		if sy < 0 {
			sy += rows
		}
		for x := 0; x < cols; x++ {
			sx := x - dx
			if sx < 0 {
				sx += cols
			}
			out[y*cols+x] = data[sy*cols+sx]
		}
	}
	return out
}

// FFTShift rearranges a spectrum plane so the zero-frequency component moves to
// the centre of the image (OpenCV / NumPy fftshift). It returns a new
// cv.FloatMat and does not modify the input.
func FFTShift(f *cv.FloatMat) *cv.FloatMat {
	out := cv.NewFloatMat(f.Rows, f.Cols)
	out.Data = circularShift2D(f.Rows, f.Cols, f.Data, shiftAmount(f.Rows, false), shiftAmount(f.Cols, false))
	return out
}

// IFFTShift is the exact inverse of [FFTShift], moving a centred
// zero-frequency component back to the top-left corner. For even dimensions it
// is identical to FFTShift; for odd dimensions it differs by one sample.
func IFFTShift(f *cv.FloatMat) *cv.FloatMat {
	out := cv.NewFloatMat(f.Rows, f.Cols)
	out.Data = circularShift2D(f.Rows, f.Cols, f.Data, shiftAmount(f.Rows, true), shiftAmount(f.Cols, true))
	return out
}

// FFTShiftSpectrum applies [FFTShift] to both planes of a complex spectrum,
// returning a new spectrum with the DC term centred.
func FFTShiftSpectrum(s *Spectrum) *Spectrum {
	out := NewSpectrum(s.Rows, s.Cols)
	dy, dx := shiftAmount(s.Rows, false), shiftAmount(s.Cols, false)
	out.Re = circularShift2D(s.Rows, s.Cols, s.Re, dy, dx)
	out.Im = circularShift2D(s.Rows, s.Cols, s.Im, dy, dx)
	return out
}

// IFFTShiftSpectrum is the exact inverse of [FFTShiftSpectrum], moving a centred
// DC term back to the corner of a complex spectrum.
func IFFTShiftSpectrum(s *Spectrum) *Spectrum {
	out := NewSpectrum(s.Rows, s.Cols)
	dy, dx := shiftAmount(s.Rows, true), shiftAmount(s.Cols, true)
	out.Re = circularShift2D(s.Rows, s.Cols, s.Re, dy, dx)
	out.Im = circularShift2D(s.Rows, s.Cols, s.Im, dy, dx)
	return out
}

// LogMagnitude returns log(1+|F|) of a spectrum, the standard compression used
// to visualise a Fourier magnitude whose dynamic range spans many orders of
// magnitude. The result is not shifted; wrap it in [FFTShift] to centre the DC
// term for display.
func LogMagnitude(s *Spectrum) *cv.FloatMat {
	out := cv.NewFloatMat(s.Rows, s.Cols)
	for i := range s.Re {
		out.Data[i] = math.Log1p(math.Hypot(s.Re[i], s.Im[i]))
	}
	return out
}

// MagnitudeSpectrum returns a display-ready magnitude spectrum of a real image:
// the forward transform, log(1+|F|) compression, an fftshift to centre the DC
// term and a linear rescale to the 8-bit range [0,255]. The result is a
// single-channel cv.Mat suitable for visualisation.
func MagnitudeSpectrum(f *cv.FloatMat) *cv.Mat {
	logMag := FFTShift(LogMagnitude(FFT2D(f)))
	return normalizeToMat(logMag)
}

// PhaseSpectrum returns a display-ready phase spectrum of a real image: the
// forward transform, an fftshift to centre the DC term and a linear rescale of
// the phase angle in (-π, π] to the 8-bit range [0,255]. The result is a
// single-channel cv.Mat.
func PhaseSpectrum(f *cv.FloatMat) *cv.Mat {
	ph := FFTShift(FFT2D(f).Phase())
	return normalizeToMat(ph)
}

// normalizeToMat linearly rescales a FloatMat to [0,255] and rounds to a
// single-channel cv.Mat. A constant image maps to all zeros.
func normalizeToMat(f *cv.FloatMat) *cv.Mat {
	minV, maxV := math.Inf(1), math.Inf(-1)
	for _, v := range f.Data {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	out := cv.NewMat(f.Rows, f.Cols, 1)
	rng := maxV - minV
	if rng <= 0 {
		return out
	}
	for i, v := range f.Data {
		out.Data[i] = uint8(math.Round((v - minV) / rng * 255))
	}
	return out
}

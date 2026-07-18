package freqdomain

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Spectrum is a complex-valued frequency-domain image stored as two parallel
// planes of float64 samples in row-major order. Re holds the real parts and Im
// the imaginary parts; the value at row y, column x lives at index y*Cols+x in
// each slice. The zero value is not usable; build a Spectrum with [NewSpectrum],
// [SpectrumFromFloat], [SpectrumFromComplex] or the transform functions such as
// [FFT2D].
type Spectrum struct {
	// Rows is the number of rows (image height).
	Rows int
	// Cols is the number of columns (image width).
	Cols int
	// Re holds the real parts, length Rows*Cols.
	Re []float64
	// Im holds the imaginary parts, length Rows*Cols.
	Im []float64
}

// NewSpectrum allocates a zero-filled Spectrum with the given dimensions. It
// panics if either dimension is not positive.
func NewSpectrum(rows, cols int) *Spectrum {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("freqdomain: NewSpectrum requires positive size, got %dx%d", rows, cols))
	}
	return &Spectrum{Rows: rows, Cols: cols, Re: make([]float64, rows*cols), Im: make([]float64, rows*cols)}
}

// Size returns the spectrum dimensions as (rows, cols).
func (s *Spectrum) Size() (rows, cols int) {
	return s.Rows, s.Cols
}

// At returns the complex sample at row y, column x as its real and imaginary
// parts.
func (s *Spectrum) At(y, x int) (re, im float64) {
	i := y*s.Cols + x
	return s.Re[i], s.Im[i]
}

// Set stores the complex sample re+im·i at row y, column x.
func (s *Spectrum) Set(y, x int, re, im float64) {
	i := y*s.Cols + x
	s.Re[i] = re
	s.Im[i] = im
}

// Clone returns a deep copy of the spectrum.
func (s *Spectrum) Clone() *Spectrum {
	out := NewSpectrum(s.Rows, s.Cols)
	copy(out.Re, s.Re)
	copy(out.Im, s.Im)
	return out
}

// RealPlane returns the real part of the spectrum as a fresh cv.FloatMat.
func (s *Spectrum) RealPlane() *cv.FloatMat {
	out := cv.NewFloatMat(s.Rows, s.Cols)
	copy(out.Data, s.Re)
	return out
}

// ImagPlane returns the imaginary part of the spectrum as a fresh cv.FloatMat.
func (s *Spectrum) ImagPlane() *cv.FloatMat {
	out := cv.NewFloatMat(s.Rows, s.Cols)
	copy(out.Data, s.Im)
	return out
}

// Magnitude returns the per-element magnitude sqrt(re²+im²) of the spectrum as
// a cv.FloatMat.
func (s *Spectrum) Magnitude() *cv.FloatMat {
	out := cv.NewFloatMat(s.Rows, s.Cols)
	for i := range s.Re {
		out.Data[i] = math.Hypot(s.Re[i], s.Im[i])
	}
	return out
}

// Phase returns the per-element phase angle atan2(im, re) of the spectrum in
// radians (range (-π, π]) as a cv.FloatMat.
func (s *Spectrum) Phase() *cv.FloatMat {
	out := cv.NewFloatMat(s.Rows, s.Cols)
	for i := range s.Re {
		out.Data[i] = math.Atan2(s.Im[i], s.Re[i])
	}
	return out
}

// PowerSpectrum returns the per-element power |F|² = re²+im² of the spectrum as
// a cv.FloatMat.
func (s *Spectrum) PowerSpectrum() *cv.FloatMat {
	out := cv.NewFloatMat(s.Rows, s.Cols)
	for i := range s.Re {
		out.Data[i] = s.Re[i]*s.Re[i] + s.Im[i]*s.Im[i]
	}
	return out
}

// Conjugate returns a new spectrum with every sample complex-conjugated (the
// imaginary part negated).
func (s *Spectrum) Conjugate() *Spectrum {
	out := NewSpectrum(s.Rows, s.Cols)
	copy(out.Re, s.Re)
	for i := range s.Im {
		out.Im[i] = -s.Im[i]
	}
	return out
}

// Scale returns a new spectrum with every sample multiplied by the real scalar
// factor.
func (s *Spectrum) Scale(factor float64) *Spectrum {
	out := NewSpectrum(s.Rows, s.Cols)
	for i := range s.Re {
		out.Re[i] = s.Re[i] * factor
		out.Im[i] = s.Im[i] * factor
	}
	return out
}

// Add returns the element-wise complex sum of s and other. It panics on a size
// mismatch.
func (s *Spectrum) Add(other *Spectrum) *Spectrum {
	requireSameSpectrum(s, other, "Add")
	out := NewSpectrum(s.Rows, s.Cols)
	for i := range s.Re {
		out.Re[i] = s.Re[i] + other.Re[i]
		out.Im[i] = s.Im[i] + other.Im[i]
	}
	return out
}

// Mul returns the element-wise complex product of s and other. It panics on a
// size mismatch.
func (s *Spectrum) Mul(other *Spectrum) *Spectrum {
	requireSameSpectrum(s, other, "Mul")
	out := NewSpectrum(s.Rows, s.Cols)
	for i := range s.Re {
		ar, ai := s.Re[i], s.Im[i]
		br, bi := other.Re[i], other.Im[i]
		out.Re[i] = ar*br - ai*bi
		out.Im[i] = ar*bi + ai*br
	}
	return out
}

// MulReal returns a new spectrum with every complex sample multiplied by the
// corresponding real gain from mask, a real transfer function of matching size.
// It panics on a size mismatch.
func (s *Spectrum) MulReal(mask *cv.FloatMat) *Spectrum {
	if s.Rows != mask.Rows || s.Cols != mask.Cols {
		panic(fmt.Sprintf("freqdomain: MulReal shape mismatch %dx%d vs %dx%d", s.Rows, s.Cols, mask.Rows, mask.Cols))
	}
	out := NewSpectrum(s.Rows, s.Cols)
	for i := range s.Re {
		out.Re[i] = s.Re[i] * mask.Data[i]
		out.Im[i] = s.Im[i] * mask.Data[i]
	}
	return out
}

// SpectrumFromFloat builds a Spectrum from a real image, setting the imaginary
// plane to zero.
func SpectrumFromFloat(f *cv.FloatMat) *Spectrum {
	out := NewSpectrum(f.Rows, f.Cols)
	copy(out.Re, f.Data)
	return out
}

// SpectrumFromComplex builds a Spectrum from separate real and imaginary planes
// of matching size. It panics on a size mismatch.
func SpectrumFromComplex(re, im *cv.FloatMat) *Spectrum {
	if re.Rows != im.Rows || re.Cols != im.Cols {
		panic(fmt.Sprintf("freqdomain: SpectrumFromComplex shape mismatch %dx%d vs %dx%d", re.Rows, re.Cols, im.Rows, im.Cols))
	}
	out := NewSpectrum(re.Rows, re.Cols)
	copy(out.Re, re.Data)
	copy(out.Im, im.Data)
	return out
}

// requireSameSpectrum panics unless a and b share the same dimensions.
func requireSameSpectrum(a, b *Spectrum, name string) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("freqdomain: %s shape mismatch %dx%d vs %dx%d", name, a.Rows, a.Cols, b.Rows, b.Cols))
	}
}

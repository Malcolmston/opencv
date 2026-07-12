package structured_light

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Params configures a [SinusoidalPattern] (N-step phase-shifting profilometry).
type Params struct {
	// Width and Height are the projected image dimensions in pixels.
	Width, Height int
	// NumOfPatternImages is the number of phase-shifted fringe images (N). It
	// must be at least 3; three is the minimum for an unambiguous three-unknown
	// (offset, amplitude, phase) solution.
	NumOfPatternImages int
	// Shift is the phase step, in radians, between consecutive pattern images.
	// If zero, the canonical uniform step 2π/N is used.
	Shift float64
	// Frequency is the number of full sinusoid periods (fringes) across the
	// varying direction of the image. It must be at least 1.
	Frequency int
	// Horizontal selects the fringe orientation. When false (the default) the
	// fringes are vertical and the phase varies along x (columns); when true the
	// fringes are horizontal and the phase varies along y (rows).
	Horizontal bool
}

// SinusoidalPattern generates phase-shifted sinusoidal fringe patterns and
// decodes them into a wrapped phase map. Construct it with
// [NewSinusoidalPattern]. The zero value is not usable.
type SinusoidalPattern struct {
	p     Params
	shift float64 // effective phase step (Shift or 2π/N)
}

// NewSinusoidalPattern validates params and returns a [SinusoidalPattern]. It
// panics if the dimensions are non-positive, NumOfPatternImages < 3, or
// Frequency < 1.
func NewSinusoidalPattern(params Params) *SinusoidalPattern {
	if params.Width <= 0 || params.Height <= 0 {
		panic(fmt.Sprintf("structured_light: Sinusoidal requires positive dimensions, got %dx%d", params.Width, params.Height))
	}
	if params.NumOfPatternImages < 3 {
		panic(fmt.Sprintf("structured_light: Sinusoidal requires NumOfPatternImages>=3, got %d", params.NumOfPatternImages))
	}
	if params.Frequency < 1 {
		panic(fmt.Sprintf("structured_light: Sinusoidal requires Frequency>=1, got %d", params.Frequency))
	}
	s := params.Shift
	if s == 0 {
		s = 2 * math.Pi / float64(params.NumOfPatternImages)
	}
	return &SinusoidalPattern{p: params, shift: s}
}

// Params returns a copy of the configuration the pattern was built with.
func (s *SinusoidalPattern) Params() Params { return s.p }

// PhaseShift returns the effective phase step in radians between consecutive
// pattern images.
func (s *SinusoidalPattern) PhaseShift() float64 { return s.shift }

// referencePhase returns the ideal projected phase at coordinate along the
// varying direction (x for vertical fringes, y for horizontal), i.e.
// 2π·Frequency·coord/extent.
func (s *SinusoidalPattern) referencePhase(coord int) float64 {
	extent := s.p.Width
	if s.p.Horizontal {
		extent = s.p.Height
	}
	return 2 * math.Pi * float64(s.p.Frequency) * float64(coord) / float64(extent)
}

// Generate returns NumOfPatternImages single-channel fringe images of size
// Height×Width. Image i has intensity 127.5·(1 + cos(φ + i·Shift)), where φ is
// the reference phase along the varying direction. Samples are rounded to the
// 0..255 range.
func (s *SinusoidalPattern) Generate() []*cv.Mat {
	imgs := make([]*cv.Mat, s.p.NumOfPatternImages)
	for i := 0; i < s.p.NumOfPatternImages; i++ {
		m := cv.NewMat(s.p.Height, s.p.Width, 1)
		delta := float64(i) * s.shift
		for y := 0; y < s.p.Height; y++ {
			for x := 0; x < s.p.Width; x++ {
				coord := x
				if s.p.Horizontal {
					coord = y
				}
				phi := s.referencePhase(coord)
				val := 127.5 * (1 + math.Cos(phi+delta))
				m.Set(y, x, 0, clampRound(val))
			}
		}
		imgs[i] = m
	}
	return imgs
}

// ComputeWrappedPhase recovers the wrapped phase map from a captured stack of N
// phase-shifted images using the standard N-step estimator
//
//	φ = atan2( -Σ Iᵢ·sin(i·Shift), Σ Iᵢ·cos(i·Shift) )
//
// The result is a row-major []float64 of length Rows*Cols with values in
// (-π, π]. captured must hold exactly NumOfPatternImages images of identical
// size; each may be single-channel or convertible to grayscale. It panics if
// the stack size or dimensions are inconsistent.
func (s *SinusoidalPattern) ComputeWrappedPhase(captured []*cv.Mat) []float64 {
	if len(captured) != s.p.NumOfPatternImages {
		panic(fmt.Sprintf("structured_light: ComputeWrappedPhase expects %d images, got %d", s.p.NumOfPatternImages, len(captured)))
	}
	if captured[0] == nil {
		panic("structured_light: ComputeWrappedPhase captured[0] is nil")
	}
	rows, cols := captured[0].Rows, captured[0].Cols
	grays := make([]*cv.Mat, len(captured))
	for i, m := range captured {
		if m == nil {
			panic(fmt.Sprintf("structured_light: ComputeWrappedPhase captured[%d] is nil", i))
		}
		if m.Rows != rows || m.Cols != cols {
			panic(fmt.Sprintf("structured_light: ComputeWrappedPhase captured[%d] is %dx%d, want %dx%d", i, m.Rows, m.Cols, rows, cols))
		}
		grays[i] = toGray(m)
	}

	// Precompute the shift sines/cosines.
	sinD := make([]float64, len(grays))
	cosD := make([]float64, len(grays))
	for i := range grays {
		d := float64(i) * s.shift
		sinD[i] = math.Sin(d)
		cosD[i] = math.Cos(d)
	}

	n := rows * cols
	phase := make([]float64, n)
	for p := 0; p < n; p++ {
		var num, den float64
		for i := range grays {
			v := float64(grays[i].Data[p])
			num += v * sinD[i]
			den += v * cosD[i]
		}
		phase[p] = math.Atan2(-num, den)
	}
	return phase
}

// UnwrapPhaseMap removes the 2π discontinuities of a wrapped phase map by a
// simple line-by-line spatial unwrap along the fringe direction. For vertical
// fringes (horizontal == false) each image row is unwrapped left-to-right; for
// horizontal fringes each image column is unwrapped top-to-bottom. The input is
// a row-major []float64 of length rows*cols in (-π, π]; the output is a new
// slice of continuous absolute phase.
//
// This is exact for a clean monotonic phase ramp but is not quality-guided; see
// the package Deferred list.
func UnwrapPhaseMap(wrapped []float64, rows, cols int, horizontal bool) []float64 {
	if len(wrapped) != rows*cols {
		panic(fmt.Sprintf("structured_light: UnwrapPhaseMap length %d != rows*cols %d", len(wrapped), rows*cols))
	}
	out := make([]float64, len(wrapped))
	if !horizontal {
		for y := 0; y < rows; y++ {
			base := y * cols
			out[base] = wrapped[base]
			for x := 1; x < cols; x++ {
				out[base+x] = out[base+x-1] + wrapDelta(wrapped[base+x]-wrapped[base+x-1])
			}
		}
	} else {
		for x := 0; x < cols; x++ {
			out[x] = wrapped[x]
			for y := 1; y < rows; y++ {
				cur := y*cols + x
				prev := (y-1)*cols + x
				out[cur] = out[prev] + wrapDelta(wrapped[cur]-wrapped[prev])
			}
		}
	}
	return out
}

// wrapDelta brings a phase difference into (-π, π] by subtracting the nearest
// multiple of 2π.
func wrapDelta(d float64) float64 {
	return d - 2*math.Pi*math.Round(d/(2*math.Pi))
}

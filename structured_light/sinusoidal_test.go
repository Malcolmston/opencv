package structured_light

import (
	"math"
	"testing"
)

func TestSinusoidalWrappedPhaseRamp(t *testing.T) {
	params := Params{Width: 128, Height: 4, NumOfPatternImages: 4, Frequency: 3}
	s := NewSinusoidalPattern(params)
	patterns := s.Generate()
	if len(patterns) != 4 {
		t.Fatalf("Generate produced %d, want 4", len(patterns))
	}

	wrapped := s.ComputeWrappedPhase(patterns)
	// The wrapped phase must equal wrap(reference phase) at every pixel.
	for y := 0; y < params.Height; y++ {
		for x := 0; x < params.Width; x++ {
			want := wrapToPi(s.referencePhase(x))
			got := wrapped[y*params.Width+x]
			if d := angleDiff(got, want); d > 0.02 {
				t.Fatalf("wrapped phase at (%d,%d) = %.4f, want %.4f (diff %.4f)", x, y, got, want, d)
			}
		}
	}
}

func TestSinusoidalUnwrapMatchesRamp(t *testing.T) {
	params := Params{Width: 96, Height: 3, NumOfPatternImages: 5, Frequency: 2}
	s := NewSinusoidalPattern(params)
	patterns := s.Generate()
	wrapped := s.ComputeWrappedPhase(patterns)
	abs := UnwrapPhaseMap(wrapped, params.Height, params.Width, params.Horizontal)

	// Absolute phase should match the reference ramp up to a per-row constant.
	// Row start is out[base] = wrapped[base] = referencePhase(0) = 0, so it
	// matches exactly.
	maxErr := 0.0
	for y := 0; y < params.Height; y++ {
		for x := 0; x < params.Width; x++ {
			want := s.referencePhase(x)
			got := abs[y*params.Width+x]
			if e := math.Abs(got - want); e > maxErr {
				maxErr = e
			}
		}
	}
	if maxErr > 0.05 {
		t.Fatalf("unwrapped phase max error %.4f exceeds tolerance", maxErr)
	}
}

func TestSinusoidalHorizontal(t *testing.T) {
	params := Params{Width: 4, Height: 96, NumOfPatternImages: 4, Frequency: 2, Horizontal: true}
	s := NewSinusoidalPattern(params)
	patterns := s.Generate()
	wrapped := s.ComputeWrappedPhase(patterns)
	abs := UnwrapPhaseMap(wrapped, params.Height, params.Width, true)

	maxErr := 0.0
	for y := 0; y < params.Height; y++ {
		for x := 0; x < params.Width; x++ {
			want := s.referencePhase(y)
			got := abs[y*params.Width+x]
			if e := math.Abs(got - want); e > maxErr {
				maxErr = e
			}
		}
	}
	if maxErr > 0.05 {
		t.Fatalf("horizontal unwrapped phase max error %.4f exceeds tolerance", maxErr)
	}
}

func TestPhaseShiftDefault(t *testing.T) {
	s := NewSinusoidalPattern(Params{Width: 8, Height: 8, NumOfPatternImages: 4, Frequency: 1})
	if got, want := s.PhaseShift(), math.Pi/2; math.Abs(got-want) > 1e-9 {
		t.Fatalf("default PhaseShift = %.6f, want %.6f", got, want)
	}
	custom := NewSinusoidalPattern(Params{Width: 8, Height: 8, NumOfPatternImages: 4, Frequency: 1, Shift: 1.0})
	if custom.PhaseShift() != 1.0 {
		t.Fatalf("custom PhaseShift = %.6f, want 1.0", custom.PhaseShift())
	}
	if custom.Params().Frequency != 1 {
		t.Fatal("Params() did not round-trip")
	}
}

func TestPhaseMapToMat(t *testing.T) {
	phase := []float64{0, 1, 2, 3}
	m := PhaseMapToMat(phase, 2, 2)
	if m.Data[0] != 0 {
		t.Fatalf("min should map to 0, got %d", m.Data[0])
	}
	if m.Data[3] != 255 {
		t.Fatalf("max should map to 255, got %d", m.Data[3])
	}
	// Constant map -> all zeros.
	cst := PhaseMapToMat([]float64{2, 2, 2, 2}, 2, 2)
	for _, v := range cst.Data {
		if v != 0 {
			t.Fatal("constant phase map should be all zeros")
		}
	}
}

func TestSinusoidalPanics(t *testing.T) {
	cases := []Params{
		{Width: 0, Height: 8, NumOfPatternImages: 4, Frequency: 1},
		{Width: 8, Height: 8, NumOfPatternImages: 2, Frequency: 1},
		{Width: 8, Height: 8, NumOfPatternImages: 4, Frequency: 0},
	}
	for i, p := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("case %d: expected panic", i)
				}
			}()
			NewSinusoidalPattern(p)
		}()
	}
}

// wrapToPi maps an angle into (-π, π].
func wrapToPi(a float64) float64 {
	for a > math.Pi {
		a -= 2 * math.Pi
	}
	for a <= -math.Pi {
		a += 2 * math.Pi
	}
	return a
}

// angleDiff returns the smallest absolute difference between two angles.
func angleDiff(a, b float64) float64 {
	d := math.Abs(a - b)
	for d > math.Pi {
		d = math.Abs(d - 2*math.Pi)
	}
	return d
}

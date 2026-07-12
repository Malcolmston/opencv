package structured_light

import (
	"math"
	"testing"
)

func TestWrapPhase(t *testing.T) {
	cases := []struct{ in, want float64 }{
		{0, 0},
		{math.Pi, math.Pi},
		{-math.Pi, math.Pi},
		{3 * math.Pi, math.Pi},
		{2 * math.Pi, 0},
		{math.Pi / 2, math.Pi / 2},
	}
	for _, c := range cases {
		if got := WrapPhase(c.in); math.Abs(got-c.want) > 1e-12 {
			t.Fatalf("WrapPhase(%.4f) = %.4f, want %.4f", c.in, got, c.want)
		}
	}
}

func TestPhaseToCoord(t *testing.T) {
	extent, freq := 128, 4.0
	for _, coord := range []int{0, 17, 63, 127} {
		abs := 2 * math.Pi * freq * float64(coord) / float64(extent)
		got := PhaseToCoord([]float64{abs}, freq, extent)
		if math.Abs(got[0]-float64(coord)) > 1e-9 {
			t.Fatalf("PhaseToCoord recovered %.4f, want %d", got[0], coord)
		}
	}
}

func TestNStepMatchesSinusoidal(t *testing.T) {
	s := NewSinusoidalPattern(Params{Width: 64, Height: 4, NumOfPatternImages: 5, Frequency: 3})
	patterns := s.Generate()
	ref := s.ComputeWrappedPhase(patterns)
	got := NStepWrappedPhase(patterns, 0) // 0 -> canonical 2π/N step
	for i := range ref {
		if angleDiff(ref[i], got[i]) > 1e-9 {
			t.Fatalf("NStepWrappedPhase differs from ComputeWrappedPhase at %d", i)
		}
	}
}

func TestCombineGrayAndPhase(t *testing.T) {
	// Absolute phase 2π·order + θ recovered from a coarse order and fine wrap.
	order := []int{0, 1, 2, 5}
	theta := []float64{0.3, -0.4, 1.1, -2.0}
	wrapped := make([]float64, len(theta))
	for i := range theta {
		wrapped[i] = WrapPhase(theta[i])
	}
	abs := CombineGrayAndPhase(order, wrapped)
	for i := range abs {
		want := 2*math.Pi*float64(order[i]) + wrapped[i]
		if math.Abs(abs[i]-want) > 1e-9 {
			t.Fatalf("CombineGrayAndPhase[%d] = %.4f, want %.4f", i, abs[i], want)
		}
	}
}

func TestPhaseGradientQuality(t *testing.T) {
	// A smooth field scores higher than one with a sharp jump.
	rows, cols := 2, 4
	smooth := []float64{0, 0.1, 0.2, 0.3, 0, 0.1, 0.2, 0.3}
	q := PhaseGradientQuality(smooth, rows, cols)
	for _, v := range q {
		if v <= 0 || v > 1 {
			t.Fatalf("quality out of range: %.4f", v)
		}
	}
	jump := []float64{0, 3.0, 0, 3.0, 0, 3.0, 0, 3.0}
	qj := PhaseGradientQuality(jump, rows, cols)
	if qj[0] >= q[0] {
		t.Fatalf("jumpy field should score lower: %.4f vs %.4f", qj[0], q[0])
	}
}

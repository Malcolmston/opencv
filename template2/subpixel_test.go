package template2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestParabolicPeakSymmetric(t *testing.T) {
	// Symmetric samples: peak exactly at the center, offset 0.
	if got := ParabolicPeak(1, 2, 1); got != 0 {
		t.Fatalf("symmetric peak offset should be 0, got %g", got)
	}
	// Peak biased toward the right sample.
	// left=1, center=3, right=2 => offset = 0.5*(1-2)/(1-6+2) = 0.5*(-1)/(-3)=1/6.
	if got := ParabolicPeak(1, 3, 2); math.Abs(got-1.0/6.0) > 1e-12 {
		t.Fatalf("expected offset 1/6, got %g", got)
	}
}

func TestParabolicPeakKnownVertex(t *testing.T) {
	// f(x) = -(x-0.25)^2 + 10 sampled at x=-1,0,1.
	f := func(x float64) float64 { return -(x-0.25)*(x-0.25) + 10 }
	off := ParabolicPeak(f(-1), f(0), f(1))
	if math.Abs(off-0.25) > 1e-9 {
		t.Fatalf("expected vertex offset 0.25, got %g", off)
	}
}

func TestRefinePeakOnQuadraticSurface(t *testing.T) {
	// Build a 5x5 score surface with a known peak at (2.3, 1.8).
	peakX, peakY := 2.3, 1.8
	s := cv.NewFloatMat(5, 5)
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			dx := float64(x) - peakX
			dy := float64(y) - peakY
			s.Data[y*5+x] = 100 - (dx*dx + dy*dy)
		}
	}
	// Integer maximum is nearest grid point (2,2).
	ix, iy, _, _ := LocateExtremum(s, true)
	sp := RefinePeakQuadratic(s, ix, iy)
	if math.Abs(sp.X-peakX) > 1e-6 || math.Abs(sp.Y-peakY) > 1e-6 {
		t.Fatalf("refined peak (%g,%g) far from (%g,%g)", sp.X, sp.Y, peakX, peakY)
	}
	if math.Abs(sp.Score-100) > 1e-6 {
		t.Fatalf("refined score should be ~100, got %g", sp.Score)
	}
}

func TestRefinePeakSeparable(t *testing.T) {
	// Separable refinement should recover a purely axis-aligned offset.
	peakX, peakY := 3.0, 2.4
	s := cv.NewFloatMat(6, 6)
	for y := 0; y < 6; y++ {
		for x := 0; x < 6; x++ {
			dx := float64(x) - peakX
			dy := float64(y) - peakY
			s.Data[y*6+x] = 50 - (dx*dx + dy*dy)
		}
	}
	ix, iy, _, _ := LocateExtremum(s, true)
	sp := RefinePeak(s, ix, iy)
	if math.Abs(sp.X-peakX) > 1e-6 || math.Abs(sp.Y-peakY) > 1e-6 {
		t.Fatalf("separable refined peak (%g,%g) far from (%g,%g)", sp.X, sp.Y, peakX, peakY)
	}
}

func TestRefinePeakBorder(t *testing.T) {
	// A peak on the border is returned unrefined along the missing axis.
	s := cv.NewFloatMat(3, 3)
	s.Data[0] = 10 // top-left corner is the max
	sp := RefinePeak(s, 0, 0)
	if sp.X != 0 || sp.Y != 0 {
		t.Fatalf("border peak should stay at (0,0), got (%g,%g)", sp.X, sp.Y)
	}
}

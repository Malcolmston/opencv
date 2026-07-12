package structured_light

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestComputeDataModulation(t *testing.T) {
	// A full-contrast sinusoidal stack has background ≈ amplitude, so the data
	// modulation γ = B/A is ≈ 1.
	s := NewSinusoidalPattern(Params{Width: 64, Height: 4, NumOfPatternImages: 4, Frequency: 3})
	patterns := s.Generate()

	mod := ComputeDataModulation(patterns, 0)
	bg := ComputeBackground(patterns, 0)
	amp := ComputeAmplitude(patterns, 0)
	for i := range mod {
		if math.Abs(mod[i]-1) > 0.05 {
			t.Fatalf("data modulation[%d] = %.4f, want ~1", i, mod[i])
		}
		if math.Abs(bg[i]-127.5) > 1.0 {
			t.Fatalf("background[%d] = %.4f, want ~127.5", i, bg[i])
		}
		if math.Abs(amp[i]-127.5) > 1.5 {
			t.Fatalf("amplitude[%d] = %.4f, want ~127.5", i, amp[i])
		}
	}
}

func TestShadowAndModulationMasks(t *testing.T) {
	rows, cols := 2, 2
	white := cv.NewMat(rows, cols, 1)
	black := cv.NewMat(rows, cols, 1)
	// pixel 0 lit, pixel 3 shadow.
	white.Data = []uint8{200, 100, 60, 45}
	black.Data = []uint8{10, 10, 10, 40}
	mask := ShadowMask(white, black, 40)
	want := []bool{true, true, true, false}
	for i := range mask {
		if mask[i] != want[i] {
			t.Fatalf("ShadowMask[%d] = %v, want %v", i, mask[i], want[i])
		}
	}

	mod := []float64{0.9, 0.2, 0.5, 0.05}
	mm := ModulationMask(mod, 0.4)
	wantMM := []bool{true, false, true, false}
	for i := range mm {
		if mm[i] != wantMM[i] {
			t.Fatalf("ModulationMask[%d] = %v, want %v", i, mm[i], wantMM[i])
		}
	}

	combined := CombineMasks(mask, mm)
	wantC := []bool{true, false, true, false}
	for i := range combined {
		if combined[i] != wantC[i] {
			t.Fatalf("CombineMasks[%d] = %v, want %v", i, combined[i], wantC[i])
		}
	}
	if CombineMasks() != nil {
		t.Fatal("CombineMasks() with no args should be nil")
	}
}

func TestOverexposureMask(t *testing.T) {
	rows, cols := 1, 3
	a := cv.NewMat(rows, cols, 1)
	b := cv.NewMat(rows, cols, 1)
	a.Data = []uint8{255, 100, 10}
	b.Data = []uint8{100, 100, 250}
	mask := OverexposureMask([]*cv.Mat{a, b}, 250)
	want := []bool{true, false, true}
	for i := range mask {
		if mask[i] != want[i] {
			t.Fatalf("OverexposureMask[%d] = %v, want %v", i, mask[i], want[i])
		}
	}
}

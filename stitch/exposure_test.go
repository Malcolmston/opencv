package stitch

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestApplyGain(t *testing.T) {
	img := cv.NewMat(1, 3, 1)
	copy(img.Data, []uint8{10, 100, 200})
	out := ApplyGain(img, 1.5)
	want := []uint8{15, 150, 255} // 300 clamps to 255
	for i, w := range want {
		if out.Data[i] != w {
			t.Fatalf("gain[%d] = %d, want %d", i, out.Data[i], w)
		}
	}
}

func TestMeanIntensityMasked(t *testing.T) {
	img := cv.NewMat(1, 4, 1)
	copy(img.Data, []uint8{10, 20, 30, 40})
	w := cv.NewFloatMat(1, 4)
	copy(w.Data, []float64{1, 1, 0, 0})
	mean, count := MeanIntensityMasked(img, w)
	if count != 2 || mean != 15 {
		t.Fatalf("mean=%g count=%d, want 15,2", mean, count)
	}
}

func TestEstimateGainsIdentical(t *testing.T) {
	// Two images of equal brightness overlapping fully → gains ≈ 1.
	means := [][]float64{{100, 100}, {100, 100}}
	counts := [][]float64{{50, 50}, {50, 50}}
	g := EstimateGains(means, counts, 10, 0.1)
	for i, v := range g {
		if v < 0.95 || v > 1.05 {
			t.Fatalf("gain[%d] = %g, want ≈1", i, v)
		}
	}
}

func TestEstimateGainsAsymmetric(t *testing.T) {
	// Image 1 is twice as bright in the overlap; its gain must be smaller than
	// image 0's, and the compensated overlaps must move closer together.
	means := [][]float64{{100, 100}, {200, 200}}
	counts := [][]float64{{50, 50}, {50, 50}}
	g := EstimateGains(means, counts, 1, 100)
	if g[0] <= g[1] {
		t.Fatalf("expected g0 (%g) > g1 (%g)", g[0], g[1])
	}
	before := 200.0 - 100.0
	after := g[1]*200 - g[0]*100
	if after < 0 {
		after = -after
	}
	if after >= before {
		t.Fatalf("compensation did not reduce brightness gap: before %g after %g", before, after)
	}
}

func TestGainCompensatorFeed(t *testing.T) {
	rows, cols := 4, 8
	// Two fully-overlapping layers, layer 1 twice as bright.
	mk := func(val uint8) Layer {
		img := cv.NewMat(rows, cols, 1)
		w := cv.NewFloatMat(rows, cols)
		for p := range img.Data {
			img.Data[p] = val
			w.Data[p] = 1
		}
		return Layer{Image: img, Weight: w}
	}
	gc := NewGainCompensator()
	gc.SigmaG = 1e6 // very weak prior → gains equalise brightness
	gains := gc.Feed([]Layer{mk(80), mk(160)})
	if len(gains) != 2 {
		t.Fatalf("gains len = %d", len(gains))
	}
	// After compensation the brightnesses should nearly match.
	c0 := gains[0] * 80
	c1 := gains[1] * 160
	if diff := c0 - c1; diff > 5 || diff < -5 {
		t.Fatalf("compensated brightness mismatch: %g vs %g", c0, c1)
	}
	applied := gc.Apply(1, mk(160))
	if applied.Image.Data[0] == 160 && gains[1] != 1 {
		t.Fatal("Apply did not scale layer")
	}
}

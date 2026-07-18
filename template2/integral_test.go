package template2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestIntegralRegionSum(t *testing.T) {
	// 3x3 image with values 1..9.
	m := newGray(t, 3, 3, []uint8{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	in := NewIntegral(m)
	// Whole image sum = 45.
	if got := in.RegionSum(0, 0, 3, 3); got != 45 {
		t.Fatalf("whole sum: expected 45, got %g", got)
	}
	// Bottom-right 2x2 block: 5+6+8+9 = 28.
	if got := in.RegionSum(1, 1, 3, 3); got != 28 {
		t.Fatalf("2x2 sum: expected 28, got %g", got)
	}
	// Squared sum of whole = 1+4+9+...+81 = 285.
	if got := in.RegionSqSum(0, 0, 3, 3); got != 285 {
		t.Fatalf("sqsum: expected 285, got %g", got)
	}
	// Mean of whole = 5.
	if got := in.RegionMean(0, 0, 3, 3); got != 5 {
		t.Fatalf("mean: expected 5, got %g", got)
	}
	// Population variance of 1..9 = 60/9 = 6.6667.
	if got := in.RegionVariance(0, 0, 3, 3); math.Abs(got-60.0/9.0) > 1e-9 {
		t.Fatalf("variance: expected 6.6667, got %g", got)
	}
}

func TestFastZNCCMatchesDirect(t *testing.T) {
	src, templ := embeddedScene(t)
	fast, err := FastZNCC(src, templ)
	if err != nil {
		t.Fatal(err)
	}
	direct, err := MatchZNCC(src, templ)
	if err != nil {
		t.Fatal(err)
	}
	if fast.Rows != direct.Rows || fast.Cols != direct.Cols {
		t.Fatalf("shape mismatch fast %dx%d vs direct %dx%d", fast.Rows, fast.Cols, direct.Rows, direct.Cols)
	}
	for i := range fast.Data {
		if math.Abs(fast.Data[i]-direct.Data[i]) > 1e-9 {
			t.Fatalf("FastZNCC vs MatchZNCC differ at %d: %g vs %g", i, fast.Data[i], direct.Data[i])
		}
	}
}

func TestFastNCCMatchesDirect(t *testing.T) {
	src, templ := embeddedScene(t)
	fast, err := FastNCC(src, templ)
	if err != nil {
		t.Fatal(err)
	}
	direct, err := MatchNCC(src, templ)
	if err != nil {
		t.Fatal(err)
	}
	for i := range fast.Data {
		if math.Abs(fast.Data[i]-direct.Data[i]) > 1e-9 {
			t.Fatalf("FastNCC vs MatchNCC differ at %d: %g vs %g", i, fast.Data[i], direct.Data[i])
		}
	}
}

func TestFastZNCCPerfectPeak(t *testing.T) {
	src, templ := embeddedScene(t)
	fast, err := FastZNCC(src, templ)
	if err != nil {
		t.Fatal(err)
	}
	x, y, v, ok := LocateExtremum(fast, true)
	if !ok || x != 2 || y != 1 || math.Abs(v-1.0) > 1e-9 {
		t.Fatalf("FastZNCC peak wrong: (%d,%d) v=%g ok=%v", x, y, v, ok)
	}
}

func TestToGrayscale(t *testing.T) {
	rgb := cv.NewMat(1, 1, 3)
	rgb.Data[0], rgb.Data[1], rgb.Data[2] = 255, 255, 255
	g := ToGrayscale(rgb)
	if g.Channels != 1 || g.Data[0] != 255 {
		t.Fatalf("white RGB should map to 255 gray, got %d channels=%d", g.Data[0], g.Channels)
	}
}

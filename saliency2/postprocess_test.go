package saliency2_test

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/saliency2"
)

func TestNormalizeRange(t *testing.T) {
	m := buildMap(1, 3, []float64{0, 5, 10})
	n := saliency2.NormalizeRange(m, -1, 1)
	want := []float64{-1, 0, 1}
	for i, w := range want {
		if math.Abs(n.Data[i]-w) > 1e-9 {
			t.Fatalf("NormalizeRange[%d] = %v, want %v", i, n.Data[i], w)
		}
	}
}

// TestIttiNormalizePromotesSinglePeak checks Itti's N(.) operator: a map with a
// single dominant peak keeps most of its energy, while a map with several equal
// peaks is strongly suppressed.
func TestIttiNormalizePromotesSinglePeak(t *testing.T) {
	single := saliency2.NewSaliencyMap(7, 7)
	single.Set(3, 3, 1.0)

	multi := saliency2.NewSaliencyMap(7, 7)
	multi.Set(1, 1, 1.0)
	multi.Set(1, 5, 1.0)
	multi.Set(5, 1, 1.0)
	multi.Set(5, 5, 1.0)

	sPeak := saliency2.IttiNormalize(single).At(3, 3)
	mPeak := saliency2.IttiNormalize(multi).At(1, 1)
	if !(sPeak > mPeak) {
		t.Fatalf("single-peak response %.4f should exceed multi-peak %.4f", sPeak, mPeak)
	}
}

// TestCenterPriorDampsBorder checks the centre prior leaves the centre untouched
// and reduces the border.
func TestCenterPriorDampsBorder(t *testing.T) {
	m := saliency2.NewSaliencyMap(9, 9)
	for i := range m.Data {
		m.Data[i] = 1
	}
	p := saliency2.CenterPrior(m, 0.5)
	center := p.At(4, 4)
	corner := p.At(0, 0)
	if math.Abs(center-1) > 1e-9 {
		t.Fatalf("centre = %v, want ~1", center)
	}
	if !(corner < center) {
		t.Fatalf("corner %v not damped below centre %v", corner, center)
	}
}

func TestCombineMaps(t *testing.T) {
	a := buildMap(1, 3, []float64{0, 5, 10}) // normalises to 0,0.5,1
	b := buildMap(1, 3, []float64{10, 5, 0}) // normalises to 1,0.5,0
	c := saliency2.CombineMaps(a, b)
	for i, v := range c.Data {
		if math.Abs(v-0.5) > 1e-9 {
			t.Fatalf("CombineMaps[%d] = %v, want 0.5", i, v)
		}
	}
}

func TestGammaCorrect(t *testing.T) {
	m := buildMap(1, 3, []float64{0, 5, 10})
	g := saliency2.GammaCorrect(m, 2)
	// midpoint 0.5 -> 0.25 under gamma 2.
	if math.Abs(g.Data[1]-0.25) > 1e-9 {
		t.Fatalf("GammaCorrect mid = %v, want 0.25", g.Data[1])
	}
}

func TestSalientMask(t *testing.T) {
	m := buildMap(1, 4, []float64{0, 0, 0, 8}) // mean 2, threshold 4
	mask := saliency2.SalientMask(m)
	if mask.Data[3] != 255 || mask.Data[0] != 0 {
		t.Fatalf("SalientMask = %v", mask.Data)
	}
}

func TestHeatmapOverlayShape(t *testing.T) {
	m := buildMap(4, 4, make([]float64, 16))
	m.Set(1, 1, 1)
	img := squareImage(4, 0, 0, 1, 0, 0) // 1-channel, all zero
	out := saliency2.HeatmapOverlay(img, m, 1.0)
	if out.Channels != 3 || out.Rows != 4 || out.Cols != 4 {
		t.Fatalf("overlay shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
}

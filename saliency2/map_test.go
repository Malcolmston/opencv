package saliency2_test

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/saliency2"
)

// buildMap fills a fresh map from a row-major slice of values.
func buildMap(rows, cols int, vals []float64) *saliency2.SaliencyMap {
	m := saliency2.NewSaliencyMap(rows, cols)
	copy(m.Data, vals)
	return m
}

func TestSaliencyMapStats(t *testing.T) {
	m := buildMap(2, 2, []float64{0, 2, 4, 6})
	if lo, hi := m.MinMax(); lo != 0 || hi != 6 {
		t.Fatalf("MinMax = (%v,%v), want (0,6)", lo, hi)
	}
	if m.Sum() != 12 {
		t.Fatalf("Sum = %v, want 12", m.Sum())
	}
	if m.Mean() != 3 {
		t.Fatalf("Mean = %v, want 3", m.Mean())
	}
	// population variance of {0,2,4,6} = 5, std = sqrt(5).
	if got := m.StdDev(); math.Abs(got-math.Sqrt(5)) > 1e-9 {
		t.Fatalf("StdDev = %v, want %v", got, math.Sqrt(5))
	}
}

func TestSaliencyMapNormalize(t *testing.T) {
	m := buildMap(1, 3, []float64{2, 4, 6})
	n := m.Normalize()
	want := []float64{0, 0.5, 1}
	for i, w := range want {
		if math.Abs(n.Data[i]-w) > 1e-9 {
			t.Fatalf("Normalize[%d] = %v, want %v", i, n.Data[i], w)
		}
	}
	// Constant map normalises to zeros.
	c := buildMap(1, 2, []float64{7, 7})
	for _, v := range c.Normalize().Data {
		if v != 0 {
			t.Fatalf("constant Normalize = %v, want 0", v)
		}
	}
}

func TestSaliencyMapToMat(t *testing.T) {
	m := buildMap(1, 3, []float64{2, 4, 6})
	mat := m.ToMat()
	if mat.Data[0] != 0 || mat.Data[2] != 255 {
		t.Fatalf("ToMat endpoints = %d,%d want 0,255", mat.Data[0], mat.Data[2])
	}
	if mat.Data[1] != 128 { // 0.5*255 = 127.5 rounds to 128
		t.Fatalf("ToMat midpoint = %d, want 128", mat.Data[1])
	}
}

func TestOtsuThresholdBimodal(t *testing.T) {
	// Two tight clusters at 1 and 9 separate cleanly at a threshold in between.
	m := buildMap(1, 8, []float64{1, 1, 1, 1, 9, 9, 9, 9})
	mask, thr := m.OtsuThreshold()
	if thr <= 1 || thr >= 9 {
		t.Fatalf("Otsu threshold = %v, want between 1 and 9", thr)
	}
	want := []uint8{0, 0, 0, 0, 255, 255, 255, 255}
	for i, w := range want {
		if mask.Data[i] != w {
			t.Fatalf("Otsu mask[%d] = %d, want %d", i, mask.Data[i], w)
		}
	}
}

func TestMeanThreshold(t *testing.T) {
	m := buildMap(1, 4, []float64{0, 0, 0, 8}) // mean 2, 2*mean = 4
	mask, thr := m.MeanThreshold(2.0)
	if thr != 4 {
		t.Fatalf("threshold = %v, want 4", thr)
	}
	if mask.Data[3] != 255 || mask.Data[0] != 0 {
		t.Fatalf("mask = %v, want only last set", mask.Data)
	}
}

func TestPercentileThreshold(t *testing.T) {
	m := buildMap(1, 10, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	mask, _ := m.PercentileThreshold(0.2) // keep top ~20%
	var count int
	for _, v := range mask.Data {
		if v == 255 {
			count++
		}
	}
	if count < 2 || count > 3 {
		t.Fatalf("percentile kept %d, want about 2", count)
	}
}

func TestCenterOfMass(t *testing.T) {
	// Single hot pixel at (x=3, y=1).
	m := saliency2.NewSaliencyMap(3, 5)
	m.Set(1, 3, 10)
	x, y := m.CenterOfMass()
	if math.Abs(x-3) > 1e-9 || math.Abs(y-1) > 1e-9 {
		t.Fatalf("CenterOfMass = (%v,%v), want (3,1)", x, y)
	}
}

func TestBoundingBox(t *testing.T) {
	m := saliency2.NewSaliencyMap(6, 6)
	m.Set(2, 1, 5)
	m.Set(4, 3, 5)
	rect, ok := m.BoundingBox(1)
	if !ok {
		t.Fatal("BoundingBox reported no pixels")
	}
	if rect.Min.X != 1 || rect.Min.Y != 2 || rect.Max.X != 4 || rect.Max.Y != 5 {
		t.Fatalf("BoundingBox = %v, want (1,2)-(4,5)", rect)
	}
	if _, ok := m.BoundingBox(100); ok {
		t.Fatal("BoundingBox should be empty above all values")
	}
}

func TestAddMultiplyMap(t *testing.T) {
	a := buildMap(1, 3, []float64{1, 2, 3})
	b := buildMap(1, 3, []float64{4, 5, 6})
	sum := a.AddMap(b)
	prod := a.MultiplyMap(b)
	if sum.Data[2] != 9 || prod.Data[2] != 18 {
		t.Fatalf("Add/Multiply = %v/%v, want 9/18", sum.Data[2], prod.Data[2])
	}
}

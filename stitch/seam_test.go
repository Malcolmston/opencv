package stitch

import (
	"reflect"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func floatMat(rows, cols int, data []float64) *cv.FloatMat {
	m := cv.NewFloatMat(rows, cols)
	copy(m.Data, data)
	return m
}

func TestFindVerticalSeamDPStraight(t *testing.T) {
	cost := floatMat(3, 3, []float64{
		9, 0, 9,
		9, 0, 9,
		9, 0, 9,
	})
	got := FindVerticalSeamDP(cost)
	if !reflect.DeepEqual(got, []int{1, 1, 1}) {
		t.Fatalf("seam = %v, want [1 1 1]", got)
	}
}

func TestFindVerticalSeamDPDiagonal(t *testing.T) {
	cost := floatMat(3, 3, []float64{
		0, 9, 9,
		9, 0, 9,
		9, 9, 0,
	})
	got := FindVerticalSeamDP(cost)
	if !reflect.DeepEqual(got, []int{0, 1, 2}) {
		t.Fatalf("seam = %v, want [0 1 2]", got)
	}
}

func TestFindHorizontalSeamDP(t *testing.T) {
	cost := floatMat(3, 3, []float64{
		0, 9, 9,
		9, 0, 9,
		9, 9, 0,
	})
	got := FindHorizontalSeamDP(cost)
	if !reflect.DeepEqual(got, []int{0, 1, 2}) {
		t.Fatalf("seam = %v, want [0 1 2]", got)
	}
}

func TestSeamMaskFromColumns(t *testing.T) {
	mask := SeamMaskFromColumns([]int{0, 1, 2}, 3, 3)
	want := []uint8{
		255, 0, 0,
		255, 255, 0,
		255, 255, 255,
	}
	if !reflect.DeepEqual(mask.Data, want) {
		t.Fatalf("mask = %v, want %v", mask.Data, want)
	}
}

func TestDPSeamFinder(t *testing.T) {
	// Build two 3×3 single-channel images whose squared difference reproduces
	// the diagonal cost map (0 on the diagonal, 9 elsewhere).
	a := cv.NewMat(3, 3, 1)
	for i := range a.Data {
		a.Data[i] = 10
	}
	b := cv.NewMat(3, 3, 1)
	diff := []uint8{0, 3, 3, 3, 0, 3, 3, 3, 0} // squared → 0/9
	for i := range b.Data {
		b.Data[i] = 10 + diff[i]
	}
	mask := DPSeamFinder{}.Find(a, b)
	want := []uint8{
		255, 0, 0,
		255, 255, 0,
		255, 255, 255,
	}
	if !reflect.DeepEqual(mask.Data, want) {
		t.Fatalf("seam mask = %v, want %v", mask.Data, want)
	}
}

func TestSeamCostMap(t *testing.T) {
	a := cv.NewMat(1, 2, 3)
	b := cv.NewMat(1, 2, 3)
	copy(a.Data, []uint8{10, 10, 10, 0, 0, 0})
	copy(b.Data, []uint8{10, 10, 10, 3, 4, 0})
	cost := SeamCostMap(a, b)
	if cost.Data[0] != 0 {
		t.Fatalf("cost[0] = %g, want 0", cost.Data[0])
	}
	if cost.Data[1] != 25 { // 9 + 16 + 0
		t.Fatalf("cost[1] = %g, want 25", cost.Data[1])
	}
}

package matching2

import (
	"math"
	"testing"
)

func TestKDTreeNearestMatchesBruteForce(t *testing.T) {
	pts := [][]float64{
		{1, 1}, {2, 5}, {9, 3}, {4, 8}, {7, 7}, {0, 6}, {5, 2}, {8, 9},
	}
	tree := BuildKDTree(pts)
	if tree.Size() != len(pts) || tree.Dim() != 2 {
		t.Fatalf("tree size/dim = %d/%d", tree.Size(), tree.Dim())
	}
	queries := [][]float64{{3, 3}, {8, 8}, {0, 0}, {5, 5}}
	for _, q := range queries {
		gotIdx, gotD2 := tree.Nearest(q)
		// Brute force reference distance (ties in index are acceptable).
		bd := math.Inf(1)
		for _, p := range pts {
			if d := sqDist(p, q); d < bd {
				bd = d
			}
		}
		if !approx(gotD2, bd, 1e-12) {
			t.Fatalf("Nearest(%v) distance = %v, want %v", q, gotD2, bd)
		}
		if !approx(sqDist(pts[gotIdx], q), bd, 1e-12) {
			t.Fatalf("Nearest(%v) idx %d is not an argmin", q, gotIdx)
		}
	}
}

func TestKDTreeKNearest(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0}, {0, 1}, {5, 5}, {10, 10}}
	tree := BuildKDTree(pts)
	idx, d2 := tree.KNearest([]float64{0, 0}, 3)
	if len(idx) != 3 {
		t.Fatalf("KNearest len = %d, want 3", len(idx))
	}
	if idx[0] != 0 {
		t.Fatalf("nearest idx = %d, want 0", idx[0])
	}
	// Distances must be non-decreasing.
	for i := 1; i < len(d2); i++ {
		if d2[i] < d2[i-1] {
			t.Fatalf("KNearest distances not sorted: %v", d2)
		}
	}
}

func TestKDTreeRadiusSearch(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0}, {0, 1}, {5, 5}}
	tree := BuildKDTree(pts)
	idx, _ := tree.RadiusSearch([]float64{0, 0}, 1.5)
	if len(idx) != 3 {
		t.Fatalf("RadiusSearch found %d, want 3", len(idx))
	}
}

func TestFLANNMatcherAgreesWithBruteForce(t *testing.T) {
	train := [][]float64{
		{0, 0, 0}, {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, {5, 5, 5}, {9, 9, 9},
	}
	query := [][]float64{
		{0.2, 0.1, 0.0}, {9.8, 0.1, 0.2}, {5.1, 4.9, 5.2},
	}
	flann := NewFLANNMatcher(train)
	bf := NewBFMatcher(NormL2)
	fm := flann.Match(query)
	bm := bf.Match(query, train)
	for i := range fm {
		if fm[i].TrainIdx != bm[i].TrainIdx {
			t.Fatalf("FLANN vs BF differ at %d: %d vs %d", i, fm[i].TrainIdx, bm[i].TrainIdx)
		}
		if !approx(fm[i].Distance, bm[i].Distance, 1e-9) {
			t.Fatalf("FLANN vs BF distance differ at %d: %v vs %v", i, fm[i].Distance, bm[i].Distance)
		}
	}
}

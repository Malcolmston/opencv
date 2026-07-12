package features2d

import (
	"math"
	"math/rand"
	"testing"
)

// randomFloatDescriptors builds n well-separated random float descriptors of the
// given dimension.
func randomFloatDescriptors(n, dim int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	rows := make([][]float64, n)
	for i := range rows {
		row := make([]float64, dim)
		for j := range row {
			row[j] = rng.Float64()
		}
		rows[i] = row
	}
	return rows
}

func TestFlannMatchesExactOnCleanData(t *testing.T) {
	train := randomFloatDescriptors(60, 16, 1)
	// Query = train permuted with tiny noise, so each query's true nearest is
	// its source row.
	perm := rand.New(rand.NewSource(2)).Perm(len(train))
	query := make([][]float64, len(train))
	truth := make([]int, len(train))
	for qi, ti := range perm {
		row := make([]float64, len(train[ti]))
		for j := range row {
			row[j] = train[ti][j] + 1e-4
		}
		query[qi] = row
		truth[qi] = ti
	}

	flann := NewFlannBasedMatcher(4, 64)
	matches := flann.Match(NewFloatDescriptors(query), NewFloatDescriptors(train))
	correct := 0
	for _, m := range matches {
		if m.TrainIdx == truth[m.QueryIdx] {
			correct++
		}
	}
	if correct < len(query)*9/10 {
		t.Fatalf("FLANN recovered only %d/%d nearest neighbours", correct, len(query))
	}
}

func TestFlannAgreesWithBruteForce(t *testing.T) {
	train := randomFloatDescriptors(80, 8, 3)
	query := randomFloatDescriptors(20, 8, 4)
	// Exhaustive (Checks<0) FLANN must equal brute force exactly.
	flann := NewFlannBasedMatcher(1, -1)
	fm := flann.Match(NewFloatDescriptors(query), NewFloatDescriptors(train))
	bf := NewBFMatcher(NormL2)
	bm := bf.Match(NewFloatDescriptors(query), NewFloatDescriptors(train))
	if len(fm) != len(bm) {
		t.Fatalf("match count differs: flann=%d bf=%d", len(fm), len(bm))
	}
	for i := range fm {
		if fm[i].TrainIdx != bm[i].TrainIdx {
			t.Fatalf("query %d: flann->%d bf->%d", i, fm[i].TrainIdx, bm[i].TrainIdx)
		}
		if math.Abs(fm[i].Distance-bm[i].Distance) > 1e-9 {
			t.Fatalf("query %d distance differs", i)
		}
	}
}

func TestFlannKnnAndRatioTest(t *testing.T) {
	train := randomFloatDescriptors(50, 12, 5)
	query := randomFloatDescriptors(10, 12, 6)
	knn := NewFlannBasedMatcher(0, 0).KnnMatch(NewFloatDescriptors(query), NewFloatDescriptors(train), 2)
	if len(knn) != len(query) {
		t.Fatalf("expected %d rows, got %d", len(query), len(knn))
	}
	for _, row := range knn {
		if len(row) > 0 && len(row) >= 2 && row[0].Distance > row[1].Distance {
			t.Fatalf("knn row not sorted by distance")
		}
	}
	_ = RatioTest(knn, 0.8) // must not panic
}

func TestFlannRejectsBinary(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on binary descriptors")
		}
	}()
	NewFlannBasedMatcher(0, 0).Match(
		NewBinaryDescriptors([][]byte{{1}}),
		NewBinaryDescriptors([][]byte{{1}}))
}

package matching2

import "testing"

// A small deterministic descriptor set where each query has an obvious nearest
// train row.
var (
	bfTrain = [][]float64{
		{0, 0},
		{10, 0},
		{0, 10},
		{10, 10},
	}
	bfQuery = [][]float64{
		{0.1, 0.1},  // -> train 0
		{9.9, 0.2},  // -> train 1
		{0.0, 9.7},  // -> train 2
		{10.2, 9.8}, // -> train 3
	}
)

func TestBFMatcherMatch(t *testing.T) {
	m := NewBFMatcher(NormL2)
	got := m.Match(bfQuery, bfTrain)
	want := []int{0, 1, 2, 3}
	if len(got) != len(want) {
		t.Fatalf("got %d matches, want %d", len(got), len(want))
	}
	for i, dm := range got {
		if dm.QueryIdx != i || dm.TrainIdx != want[i] {
			t.Fatalf("match %d = (q=%d,t=%d), want (q=%d,t=%d)", i, dm.QueryIdx, dm.TrainIdx, i, want[i])
		}
	}
}

func TestBFMatcherCrossCheck(t *testing.T) {
	m := NewBFMatcher(NormL2)
	m.CrossCheck = true
	got := m.Match(bfQuery, bfTrain)
	if len(got) != 4 {
		t.Fatalf("cross-check got %d matches, want 4", len(got))
	}
	// Sorted ascending by distance and all mutually best.
	for i := 1; i < len(got); i++ {
		if got[i].Distance < got[i-1].Distance {
			t.Fatalf("cross-check matches not sorted by distance")
		}
	}
}

func TestBFMatcherKnnAndRatioTest(t *testing.T) {
	m := NewBFMatcher(NormL2)
	knn := m.KnnMatch(bfQuery, bfTrain, 2)
	if len(knn) != 4 {
		t.Fatalf("knn outer len = %d, want 4", len(knn))
	}
	for i, nbrs := range knn {
		if len(nbrs) != 2 {
			t.Fatalf("knn[%d] len = %d, want 2", i, len(nbrs))
		}
		if nbrs[0].Distance > nbrs[1].Distance {
			t.Fatalf("knn[%d] not sorted", i)
		}
	}
	// All neighbours are unambiguous, so the ratio test keeps every query.
	good := RatioTest(knn, 0.75)
	if len(good) != 4 {
		t.Fatalf("ratio test kept %d, want 4", len(good))
	}
}

func TestBFMatcherRadiusMatch(t *testing.T) {
	m := NewBFMatcher(NormL2)
	got := m.RadiusMatch(bfQuery, bfTrain, 1.0)
	for i, nbrs := range got {
		if len(nbrs) != 1 {
			t.Fatalf("radius[%d] = %d matches, want 1", i, len(nbrs))
		}
	}
}

func TestHammingMatcher(t *testing.T) {
	train := [][]byte{
		{0b0000_0000},
		{0b1111_1111},
		{0b0000_1111},
	}
	query := [][]byte{
		{0b0000_0001}, // closest to train 0 (1 bit)
		{0b1111_1110}, // closest to train 1 (1 bit)
	}
	m := NewHammingMatcher()
	got := m.Match(query, train)
	if got[0].TrainIdx != 0 || got[1].TrainIdx != 1 {
		t.Fatalf("hamming match = %d,%d want 0,1", got[0].TrainIdx, got[1].TrainIdx)
	}
	if got[0].Distance != 1 || got[1].Distance != 1 {
		t.Fatalf("hamming distances = %v,%v want 1,1", got[0].Distance, got[1].Distance)
	}
}

func TestCrossCheckFunction(t *testing.T) {
	m := NewBFMatcher(NormL2)
	fwd := m.Match(bfQuery, bfTrain)
	bwd := m.Match(bfTrain, bfQuery)
	got := CrossCheck(fwd, bwd)
	if len(got) != 4 {
		t.Fatalf("CrossCheck kept %d, want 4", len(got))
	}
}

func TestFilterAndMinMax(t *testing.T) {
	matches := []DMatch{{Distance: 1}, {Distance: 5}, {Distance: 3}}
	min, max := MinMaxDistance(matches)
	if min != 1 || max != 5 {
		t.Fatalf("MinMax = %v,%v want 1,5", min, max)
	}
	kept := FilterMatchesByDistance(matches, 3)
	if len(kept) != 2 {
		t.Fatalf("filter kept %d want 2", len(kept))
	}
	sorted := SortMatchesByDistance(matches)
	if sorted[0].Distance != 1 || sorted[2].Distance != 5 {
		t.Fatalf("sort wrong: %v", sorted)
	}
}

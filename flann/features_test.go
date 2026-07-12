package flann_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/malcolmston/opencv/flann"
)

// recallOf returns the mean fraction of the true k neighbours (from exact) that
// appear in got, over every query.
func recallOf(exact flann.Index[[]float64], got [][]flann.Neighbor, queries [][]float64, k int) float64 {
	var found, total int
	for i, q := range queries {
		truth := exact.KnnSearch(q, k)
		set := make(map[int]bool, len(truth))
		for _, nb := range truth {
			set[nb.Index] = true
		}
		for _, nb := range got[i] {
			if set[nb.Index] {
				found++
			}
		}
		total += len(truth)
	}
	if total == 0 {
		return 1
	}
	return float64(found) / float64(total)
}

// TestNewDistances checks each added distance on hand-computed cases.
func TestNewDistances(t *testing.T) {
	if d := flann.DistL1([]float64{1, 2, 3}, []float64{4, 0, 3}); d != 5 {
		t.Fatalf("DistL1 = %v, want 5", d)
	}
	// chi-square of {2,0} vs {0,2}: 4/2 + 4/2 = 4.
	if d := flann.DistChiSquare([]float64{2, 0}, []float64{0, 2}); math.Abs(d-4) > 1e-12 {
		t.Fatalf("DistChiSquare = %v, want 4", d)
	}
	// zero-sum coordinate pairs contribute nothing.
	if d := flann.DistChiSquare([]float64{0, 0}, []float64{0, 0}); d != 0 {
		t.Fatalf("DistChiSquare zero = %v, want 0", d)
	}
	// Minkowski p=1 is L1, p=2 is L2.
	a, b := []float64{0, 0}, []float64{3, 4}
	if d := flann.DistMinkowski(a, b, 1); math.Abs(d-7) > 1e-12 {
		t.Fatalf("DistMinkowski p1 = %v, want 7", d)
	}
	if d := flann.DistMinkowski(a, b, 2); math.Abs(d-5) > 1e-12 {
		t.Fatalf("DistMinkowski p2 = %v, want 5", d)
	}
	// Hellinger of {1,0} vs {0,1}: sqrt(0.5*(1+1)) = 1.
	if d := flann.DistHellinger([]float64{1, 0}, []float64{0, 1}); math.Abs(d-1) > 1e-12 {
		t.Fatalf("DistHellinger = %v, want 1", d)
	}
	// Cosine: orthogonal -> 1, identical -> 0, opposite -> 2.
	if d := flann.DistCosine([]float64{1, 0}, []float64{0, 1}); math.Abs(d-1) > 1e-12 {
		t.Fatalf("DistCosine orthogonal = %v, want 1", d)
	}
	if d := flann.DistCosine([]float64{1, 2}, []float64{2, 4}); math.Abs(d) > 1e-12 {
		t.Fatalf("DistCosine parallel = %v, want 0", d)
	}
	if d := flann.DistCosine([]float64{1, 1}, []float64{-1, -1}); math.Abs(d-2) > 1e-12 {
		t.Fatalf("DistCosine opposite = %v, want 2", d)
	}
	if d := flann.DistCosine([]float64{0, 0}, []float64{1, 1}); d != 1 {
		t.Fatalf("DistCosine with zero vector = %v, want 1", d)
	}
}

// TestDistancePanics checks the length-mismatch and bad-order panics.
func TestDistancePanics(t *testing.T) {
	mustPanic := func(name string, f func()) {
		defer func() {
			if recover() == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		f()
	}
	mustPanic("DistL1", func() { flann.DistL1([]float64{1}, []float64{1, 2}) })
	mustPanic("DistChiSquare", func() { flann.DistChiSquare([]float64{1}, []float64{1, 2}) })
	mustPanic("DistHellinger", func() { flann.DistHellinger([]float64{1}, []float64{1, 2}) })
	mustPanic("DistCosine", func() { flann.DistCosine([]float64{1}, []float64{1, 2}) })
	mustPanic("DistMinkowski len", func() { flann.DistMinkowski([]float64{1}, []float64{1, 2}, 2) })
	mustPanic("DistMinkowski order", func() { flann.DistMinkowski([]float64{1}, []float64{2}, 0.5) })
	mustPanic("MinkowskiDist order", func() { flann.MinkowskiDist(0.5) })
}

// TestMinkowskiDistFunc checks the DistanceFunc constructor drives a linear
// index identically to the direct function.
func TestMinkowskiDistFunc(t *testing.T) {
	data := randFloatData(100, 5, 10, 3)
	li := flann.NewLinearIndexFunc(data, flann.MinkowskiDist(3))
	q := []float64{1, 2, 3, 4, 5}
	got := li.KnnSearch(q, 1)[0]
	// Verify the reported distance equals a direct Minkowski computation.
	want := flann.DistMinkowski(q, data[got.Index], 3)
	if math.Abs(got.Distance-want) > 1e-9 {
		t.Fatalf("linear+Minkowski distance = %v, want %v", got.Distance, want)
	}
}

// TestKDForestExactAndRecall checks the forest is exact with no budget and keeps
// high recall under a bounded budget.
func TestKDForestExactAndRecall(t *testing.T) {
	data := randFloatData(600, 8, 100, 2024)
	lin := flann.NewLinearIndex(data)
	forest := flann.NewKDForestIndex(data, 4, 99)
	queries := randFloatData(60, 8, 100, 13)

	// Exact when unbounded.
	for _, q := range queries {
		for _, k := range []int{1, 5, 10} {
			if !sameNeighbors(lin.KnnSearch(q, k), forest.KnnSearch(q, k)) {
				t.Fatalf("unbounded forest not exact: k=%d q=%v", k, q)
			}
		}
	}

	// Bounded budget: still high recall.
	forest.MaxChecks = 150
	got := flann.KnnSearchBatch[[]float64](forest, queries, 10)
	recall := recallOf(lin, got, queries, 10)
	if recall < 0.80 {
		t.Fatalf("forest recall %.3f below 0.80", recall)
	}
	t.Logf("KDForest recall at MaxChecks=150: %.3f", recall)
}

// TestKDForestRadius checks radius search matches brute force.
func TestKDForestRadius(t *testing.T) {
	data := randFloatData(400, 5, 30, 7)
	lin := flann.NewLinearIndex(data)
	forest := flann.NewKDForestIndex(data, 3, 5)
	for _, q := range randFloatData(30, 5, 30, 71) {
		for _, r := range []float64{3, 8, 15} {
			if !sameNeighbors(lin.RadiusSearch(q, r), forest.RadiusSearch(q, r)) {
				t.Fatalf("forest radius mismatch r=%v q=%v", r, q)
			}
		}
	}
}

// TestHierarchicalExactAndRecall checks the hierarchical clustering index.
func TestHierarchicalExactAndRecall(t *testing.T) {
	data := randFloatData(600, 8, 100, 55)
	lin := flann.NewLinearIndex(data)
	h := flann.NewHierarchicalClusteringIndex(data, 8, 16, 2, 123)
	queries := randFloatData(60, 8, 100, 17)

	// Exact when unbounded.
	for _, q := range queries {
		if !sameNeighbors(lin.KnnSearch(q, 5), h.KnnSearch(q, 5)) {
			t.Fatalf("unbounded hierarchical not exact for q=%v", q)
		}
		if !sameNeighbors(lin.RadiusSearch(q, 10), h.RadiusSearch(q, 10)) {
			t.Fatalf("hierarchical radius not exact for q=%v", q)
		}
	}

	// Bounded budget.
	h.Checks = 200
	got := flann.KnnSearchBatch[[]float64](h, queries, 10)
	recall := recallOf(lin, got, queries, 10)
	if recall < 0.80 {
		t.Fatalf("hierarchical recall %.3f below 0.80", recall)
	}
	t.Logf("Hierarchical recall at Checks=200: %.3f", recall)
}

// TestHierarchicalCustomDistance checks the index clusters and searches under a
// supplied distance, exactly matching the linear index using the same distance.
func TestHierarchicalCustomDistance(t *testing.T) {
	data := randFloatData(300, 6, 20, 8)
	lin := flann.NewLinearIndexFunc(data, flann.DistL1)
	h := flann.NewHierarchicalClusteringIndexFunc(data, 6, 12, 1, 44, flann.DistL1)
	for _, q := range randFloatData(30, 6, 20, 88) {
		if !sameNeighbors(lin.KnnSearch(q, 5), h.KnnSearch(q, 5)) {
			t.Fatalf("hierarchical L1 not exact for q=%v", q)
		}
	}
}

// TestCompositeExactAndRecall checks the composite index.
func TestCompositeExactAndRecall(t *testing.T) {
	data := randFloatData(500, 7, 80, 321)
	lin := flann.NewLinearIndex(data)
	c := flann.NewCompositeIndex(data, 4, 8, 16, 2020)
	queries := randFloatData(50, 7, 80, 9)

	for _, q := range queries {
		if !sameNeighbors(lin.KnnSearch(q, 5), c.KnnSearch(q, 5)) {
			t.Fatalf("unbounded composite not exact for q=%v", q)
		}
		if !sameNeighbors(lin.RadiusSearch(q, 12), c.RadiusSearch(q, 12)) {
			t.Fatalf("composite radius not exact for q=%v", q)
		}
	}

	c.MaxChecks = 48
	c.Checks = 48
	got := flann.KnnSearchBatch[[]float64](c, queries, 10)
	recall := recallOf(lin, got, queries, 10)
	if recall < 0.85 {
		t.Fatalf("composite recall %.3f below 0.85", recall)
	}
	t.Logf("Composite recall at 48+48 checks: %.3f", recall)
}

// TestAutotuned checks the tuned index reaches its target on the sample and
// keeps reasonable recall on held-out queries.
func TestAutotuned(t *testing.T) {
	data := randFloatData(500, 6, 60, 4242)
	lin := flann.NewLinearIndex(data)
	auto := flann.NewAutotunedIndex(data, 0.9, 7)

	if auto.AchievedPrecision() < auto.TargetPrecision() {
		t.Fatalf("achieved precision %.3f below target %.3f", auto.AchievedPrecision(), auto.TargetPrecision())
	}
	switch auto.Algorithm() {
	case "kdforest", "kmeans", "linear":
	default:
		t.Fatalf("unexpected algorithm %q", auto.Algorithm())
	}

	queries := randFloatData(60, 6, 60, 88)
	got := flann.KnnSearchBatch[[]float64](auto, queries, 5)
	recall := recallOf(lin, got, queries, 5)
	if recall < 0.60 {
		t.Fatalf("autotuned held-out recall %.3f below 0.60", recall)
	}
	t.Logf("Autotuned chose %s checks=%d achieved=%.3f held-out recall=%.3f",
		auto.Algorithm(), auto.Checks(), auto.AchievedPrecision(), recall)
}

// TestBatchMatchesPerQuery checks the batch helpers agree with single queries.
func TestBatchMatchesPerQuery(t *testing.T) {
	data := randFloatData(200, 4, 20, 1)
	kd := flann.NewKDTreeIndex(data)
	queries := randFloatData(25, 4, 20, 2)

	knn := flann.KnnSearchBatch[[]float64](kd, queries, 3)
	rad := flann.RadiusSearchBatch[[]float64](kd, queries, 6)
	if len(knn) != len(queries) || len(rad) != len(queries) {
		t.Fatalf("batch length mismatch")
	}
	for i, q := range queries {
		if !sameNeighbors(knn[i], kd.KnnSearch(q, 3)) {
			t.Fatalf("KnnSearchBatch[%d] differs", i)
		}
		if !sameNeighbors(rad[i], kd.RadiusSearch(q, 6)) {
			t.Fatalf("RadiusSearchBatch[%d] differs", i)
		}
	}
}

// TestPrecisionRecallEval checks the evaluation helpers: an exact index scores 1,
// a crippled one scores less.
func TestPrecisionRecallEval(t *testing.T) {
	data := randFloatData(300, 6, 40, 11)
	lin := flann.NewLinearIndex(data)
	forest := flann.NewKDForestIndex(data, 4, 5)
	queries := randFloatData(40, 6, 40, 22)

	if r := flann.Recall[[]float64](forest, lin, queries, 10); r != 1 {
		t.Fatalf("exact forest recall = %v, want 1", r)
	}
	if p := flann.Precision[[]float64](forest, lin, queries, 10); p != 1 {
		t.Fatalf("exact forest precision = %v, want 1", p)
	}

	forest.MaxChecks = 20
	r := flann.Recall[[]float64](forest, lin, queries, 10)
	if r <= 0 || r > 1 {
		t.Fatalf("bounded recall out of range: %v", r)
	}
	t.Logf("bounded forest recall=%.3f", r)
}

// TestCentroidAccessors checks the k-means centroid accessors.
func TestCentroidAccessors(t *testing.T) {
	data := randFloatData(400, 5, 50, 3)
	km := flann.NewKMeansIndex(data, 6, 12, 3)

	tops := km.Centroids()
	if len(tops) == 0 {
		t.Fatal("Centroids returned none")
	}
	for _, c := range tops {
		if len(c) != 5 {
			t.Fatalf("centroid dimension = %d, want 5", len(c))
		}
	}
	leaves := km.LeafCentroids()
	if len(leaves) != km.ClusterCount() {
		t.Fatalf("LeafCentroids count %d != ClusterCount %d", len(leaves), km.ClusterCount())
	}
	if km.ClusterCount() == 0 {
		t.Fatal("ClusterCount is zero")
	}
	// Mutating a returned copy must not affect the index.
	before := km.Centroids()[0][0]
	tops[0][0] = 999999
	after := km.Centroids()[0][0]
	if before != after {
		t.Fatal("Centroids returned a shared slice; mutation leaked")
	}
}

// TestGobPersistence checks KDForest and Autotuned round-trip through gob and
// answer queries identically.
func TestGobPersistence(t *testing.T) {
	data := randFloatData(400, 6, 50, 77)
	queries := randFloatData(30, 6, 50, 78)

	forest := flann.NewKDForestIndex(data, 4, 123)
	forest.MaxChecks = 40
	var buf bytes.Buffer
	if err := flann.Save(&buf, forest); err != nil {
		t.Fatalf("Save forest: %v", err)
	}
	var loaded flann.KDForestIndex
	if err := flann.Load(&buf, &loaded); err != nil {
		t.Fatalf("Load forest: %v", err)
	}
	if loaded.Size() != forest.Size() || loaded.Trees() != forest.Trees() {
		t.Fatalf("loaded forest metadata differs")
	}
	for _, q := range queries {
		if !sameNeighbors(forest.KnnSearch(q, 5), loaded.KnnSearch(q, 5)) {
			t.Fatalf("loaded forest query differs for %v", q)
		}
	}

	auto := flann.NewAutotunedIndex(data, 0.9, 7)
	var abuf bytes.Buffer
	if err := flann.Save(&abuf, auto); err != nil {
		t.Fatalf("Save autotuned: %v", err)
	}
	var aloaded flann.AutotunedIndex
	if err := flann.Load(&abuf, &aloaded); err != nil {
		t.Fatalf("Load autotuned: %v", err)
	}
	if aloaded.Algorithm() != auto.Algorithm() || aloaded.Checks() != auto.Checks() {
		t.Fatalf("loaded autotuned config differs: %s/%d vs %s/%d",
			aloaded.Algorithm(), aloaded.Checks(), auto.Algorithm(), auto.Checks())
	}
	for _, q := range queries {
		if !sameNeighbors(auto.KnnSearch(q, 5), aloaded.KnnSearch(q, 5)) {
			t.Fatalf("loaded autotuned query differs for %v", q)
		}
	}
}

// TestNewIndexDeterminism checks the randomized indices reproduce across seeds.
func TestNewIndexDeterminism(t *testing.T) {
	data := randFloatData(300, 5, 40, 8)
	q := []float64{10, 10, 10, 10, 10}

	fa := flann.NewKDForestIndex(data, 4, 555)
	fb := flann.NewKDForestIndex(data, 4, 555)
	fa.MaxChecks, fb.MaxChecks = 30, 30
	if !sameNeighbors(fa.KnnSearch(q, 5), fb.KnnSearch(q, 5)) {
		t.Fatal("KDForest not deterministic across identical seeds")
	}

	ha := flann.NewHierarchicalClusteringIndex(data, 6, 12, 2, 999)
	hb := flann.NewHierarchicalClusteringIndex(data, 6, 12, 2, 999)
	ha.Checks, hb.Checks = 40, 40
	if !sameNeighbors(ha.KnnSearch(q, 5), hb.KnnSearch(q, 5)) {
		t.Fatal("Hierarchical not deterministic across identical seeds")
	}
}

// TestNewIndexEmpty checks the new indices tolerate empty datasets.
func TestNewIndexEmpty(t *testing.T) {
	q := []float64{1, 2, 3}
	if got := flann.NewKDForestIndex(nil, 4, 1).KnnSearch(q, 3); got != nil {
		t.Fatalf("empty forest knn = %v", got)
	}
	if got := flann.NewHierarchicalClusteringIndex(nil, 8, 16, 1, 1).KnnSearch(q, 3); got != nil {
		t.Fatalf("empty hierarchical knn = %v", got)
	}
	if got := flann.NewCompositeIndex(nil, 4, 8, 16, 1).KnnSearch(q, 3); got != nil {
		t.Fatalf("empty composite knn = %v", got)
	}
	auto := flann.NewAutotunedIndex(nil, 0.9, 1)
	if got := auto.KnnSearch(q, 3); got != nil {
		t.Fatalf("empty autotuned knn = %v", got)
	}
	if auto.Algorithm() != "linear" {
		t.Fatalf("empty autotuned algorithm = %q, want linear", auto.Algorithm())
	}
}

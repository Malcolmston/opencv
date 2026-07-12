package flann_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/malcolmston/opencv/flann"
)

// randFloatData returns n points of the given dimension with coordinates in
// [0, scale), drawn from a seeded generator so tests are reproducible.
func randFloatData(n, dim int, scale float64, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]float64, n)
	for i := range data {
		row := make([]float64, dim)
		for d := range row {
			row[d] = rng.Float64() * scale
		}
		data[i] = row
	}
	return data
}

// randBinaryData returns n binary descriptors of the given byte length.
func randBinaryData(n, dim int, seed int64) [][]byte {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]byte, n)
	for i := range data {
		row := make([]byte, dim)
		for d := range row {
			row[d] = byte(rng.Intn(256))
		}
		data[i] = row
	}
	return data
}

func sameNeighbors(a, b []flann.Neighbor) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Index != b[i].Index || a[i].Distance != b[i].Distance {
			return false
		}
	}
	return true
}

// TestDistL2 checks the Euclidean distance and its length-mismatch panic.
func TestDistL2(t *testing.T) {
	got := flann.DistL2([]float64{0, 0}, []float64{3, 4})
	if got != 5 {
		t.Fatalf("DistL2 = %v, want 5", got)
	}
	if d := flann.DistL2([]float64{1, 2, 3}, []float64{1, 2, 3}); d != 0 {
		t.Fatalf("DistL2 of equal vectors = %v, want 0", d)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("DistL2 did not panic on length mismatch")
			}
		}()
		flann.DistL2([]float64{1}, []float64{1, 2})
	}()
}

// TestDistHamming checks the bit-difference count and its panic.
func TestDistHamming(t *testing.T) {
	got := flann.DistHamming([]byte{0x00, 0xFF}, []byte{0x0F, 0xF0})
	if got != 8 {
		t.Fatalf("DistHamming = %v, want 8", got)
	}
	if d := flann.DistHamming([]byte{0xAB}, []byte{0xAB}); d != 0 {
		t.Fatalf("DistHamming of equal descriptors = %v, want 0", d)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("DistHamming did not panic on length mismatch")
			}
		}()
		flann.DistHamming([]byte{1}, []byte{1, 2})
	}()
}

// TestKDTreeMatchesLinearWellSeparated is the headline correctness check: on a
// set of well-separated clusters an exact k-d tree returns exactly the true
// nearest neighbours, identical to the brute-force baseline.
func TestKDTreeMatchesLinearWellSeparated(t *testing.T) {
	// Five tight clusters far apart in 3-D.
	var data [][]float64
	centres := [][]float64{{0, 0, 0}, {100, 0, 0}, {0, 100, 0}, {0, 0, 100}, {100, 100, 100}}
	rng := rand.New(rand.NewSource(1))
	for _, c := range centres {
		for j := 0; j < 20; j++ {
			data = append(data, []float64{
				c[0] + rng.Float64(),
				c[1] + rng.Float64(),
				c[2] + rng.Float64(),
			})
		}
	}
	lin := flann.NewLinearIndex(data)
	kd := flann.NewKDTreeIndex(data)

	for _, c := range centres {
		q := []float64{c[0] + 0.5, c[1] + 0.5, c[2] + 0.5}
		want := lin.KnnSearch(q, 3)
		got := kd.KnnSearch(q, 3)
		if !sameNeighbors(want, got) {
			t.Fatalf("query %v: kd = %v, linear = %v", q, got, want)
		}
		// Every returned neighbour must belong to the query's own cluster.
		for _, nb := range got {
			if flann.DistL2(q, data[nb.Index]) > 2 {
				t.Fatalf("query %v: neighbour %d is not in the near cluster", q, nb.Index)
			}
		}
	}
}

// TestKDTreeMatchesLinearRandom fuzzes the exact k-d tree against brute force on
// random data across many queries.
func TestKDTreeMatchesLinearRandom(t *testing.T) {
	data := randFloatData(500, 6, 50, 42)
	lin := flann.NewLinearIndex(data)
	kd := flann.NewKDTreeIndex(data)

	queries := randFloatData(50, 6, 50, 7)
	for _, q := range queries {
		for _, k := range []int{1, 3, 10} {
			want := lin.KnnSearch(q, k)
			got := kd.KnnSearch(q, k)
			if !sameNeighbors(want, got) {
				t.Fatalf("k=%d query=%v mismatch:\n kd=%v\nlin=%v", k, q, got, want)
			}
		}
	}
}

// TestRadiusSearch checks that RadiusSearch returns exactly the points within
// the radius, matching a brute-force scan, for both the k-d tree and the linear
// index.
func TestRadiusSearch(t *testing.T) {
	data := randFloatData(400, 4, 20, 99)
	lin := flann.NewLinearIndex(data)
	kd := flann.NewKDTreeIndex(data)

	queries := randFloatData(30, 4, 20, 5)
	for _, q := range queries {
		for _, r := range []float64{2, 5, 12} {
			want := lin.RadiusSearch(q, r)
			got := kd.RadiusSearch(q, r)
			if !sameNeighbors(want, got) {
				t.Fatalf("r=%v query=%v: kd returned %d, linear %d", r, q, len(got), len(want))
			}
			// Independently verify the invariant: all and only points <= r.
			var brute int
			for i := range data {
				if flann.DistL2(q, data[i]) <= r {
					brute++
				}
			}
			if brute != len(got) {
				t.Fatalf("r=%v query=%v: got %d neighbours, brute force counts %d", r, q, len(got), brute)
			}
			for i := 1; i < len(got); i++ {
				if got[i].Distance < got[i-1].Distance {
					t.Fatalf("RadiusSearch result not sorted: %v", got)
				}
			}
		}
	}
}

// TestKMeansRecall checks that the hierarchical k-means tree, with a bounded
// number of checks, recovers most of the true nearest neighbours, and that with
// unlimited checks it is exact.
func TestKMeansRecall(t *testing.T) {
	data := randFloatData(600, 8, 100, 2024)
	lin := flann.NewLinearIndex(data)
	km := flann.NewKMeansIndex(data, 8, 16, 2024)

	queries := randFloatData(60, 8, 100, 13)
	const k = 10

	// Exhaustive search (Checks == 0) must be exact.
	km.Checks = 0
	for _, q := range queries {
		if !sameNeighbors(lin.KnnSearch(q, k), km.KnnSearch(q, k)) {
			t.Fatalf("exhaustive k-means search is not exact for query %v", q)
		}
	}

	// Bounded search should still recover most neighbours.
	km.Checks = 96
	var total, found int
	for _, q := range queries {
		truth := lin.KnnSearch(q, k)
		truthSet := make(map[int]bool, len(truth))
		for _, nb := range truth {
			truthSet[nb.Index] = true
		}
		got := km.KnnSearch(q, k)
		for _, nb := range got {
			if truthSet[nb.Index] {
				found++
			}
		}
		total += len(truth)
	}
	recall := float64(found) / float64(total)
	if recall < 0.80 {
		t.Fatalf("k-means recall %.3f below threshold 0.80", recall)
	}
	t.Logf("k-means recall at Checks=96: %.3f", recall)
}

// TestLSHFindsExactMatch checks that LSH always retrieves an exact binary match
// hidden among many distractors, and ranks it first with distance 0.
func TestLSHFindsExactMatch(t *testing.T) {
	data := randBinaryData(1000, 32, 777)
	// Seeds chosen for reproducibility; the exact-match guarantee holds for any.
	for _, seed := range []int64{1, 2, 3, 4, 5} {
		lsh := flann.NewLSHIndex(data, 8, 16, seed)
		for _, target := range []int{0, 250, 500, 999} {
			query := make([]byte, len(data[target]))
			copy(query, data[target])
			got := lsh.KnnSearch(query, 1)
			if len(got) == 0 {
				t.Fatalf("seed=%d target=%d: no neighbour returned", seed, target)
			}
			if got[0].Distance != 0 {
				t.Fatalf("seed=%d target=%d: nearest distance %v, want 0", seed, target, got[0].Distance)
			}
			if flann.DistHamming(query, data[got[0].Index]) != 0 {
				t.Fatalf("seed=%d target=%d: returned index %d is not an exact match", seed, target, got[0].Index)
			}
		}
	}
}

// TestLSHNearMatchRecall checks that, with several tables, LSH usually finds the
// true nearest neighbour of a lightly perturbed query, matching the brute-force
// answer with high probability.
func TestLSHNearMatchRecall(t *testing.T) {
	data := randBinaryData(800, 32, 555)
	lin := flann.NewLinearBinaryIndex(data)
	lsh := flann.NewLSHIndex(data, 12, 14, 20240712)

	rng := rand.New(rand.NewSource(31))
	trials, hits := 0, 0
	for target := 0; target < 200; target++ {
		query := make([]byte, len(data[target]))
		copy(query, data[target])
		// Flip three random bits.
		for f := 0; f < 3; f++ {
			p := rng.Intn(len(query) * 8)
			query[p>>3] ^= 1 << uint(p&7)
		}
		truth := lin.KnnSearch(query, 1)
		got := lsh.KnnSearch(query, 1)
		trials++
		if len(got) > 0 && got[0].Index == truth[0].Index {
			hits++
		}
	}
	recall := float64(hits) / float64(trials)
	if recall < 0.90 {
		t.Fatalf("LSH near-match recall %.3f below threshold 0.90", recall)
	}
	t.Logf("LSH near-match recall: %.3f", recall)
}

// TestLinearBinaryBaseline checks the binary brute-force index reports exact
// Hamming neighbours in sorted order.
func TestLinearBinaryBaseline(t *testing.T) {
	data := [][]byte{
		{0x00, 0x00},
		{0xFF, 0xFF},
		{0x0F, 0x00},
		{0x00, 0x0F},
	}
	lin := flann.NewLinearBinaryIndex(data)
	got := lin.KnnSearch([]byte{0x00, 0x00}, 3)
	want := []flann.Neighbor{{Index: 0, Distance: 0}, {Index: 2, Distance: 4}, {Index: 3, Distance: 4}}
	if !sameNeighbors(got, want) {
		t.Fatalf("binary knn = %v, want %v", got, want)
	}
}

// TestDeterminism checks that rebuilding an index with the same seed produces
// identical query results.
func TestDeterminism(t *testing.T) {
	data := randFloatData(300, 5, 40, 8)
	q := []float64{10, 10, 10, 10, 10}
	a := flann.NewKMeansIndex(data, 6, 12, 123)
	b := flann.NewKMeansIndex(data, 6, 12, 123)
	a.Checks, b.Checks = 40, 40
	if !sameNeighbors(a.KnnSearch(q, 5), b.KnnSearch(q, 5)) {
		t.Fatal("k-means results differ across identical seeds")
	}

	bin := randBinaryData(300, 16, 3)
	query := bin[10]
	la := flann.NewLSHIndex(bin, 6, 12, 55)
	lb := flann.NewLSHIndex(bin, 6, 12, 55)
	if !sameNeighbors(la.KnnSearch(query, 4), lb.KnnSearch(query, 4)) {
		t.Fatal("LSH results differ across identical seeds")
	}
}

// TestEdgeCases exercises empty datasets, non-positive k and ragged input.
func TestEdgeCases(t *testing.T) {
	empty := flann.NewKDTreeIndex(nil)
	if got := empty.KnnSearch([]float64{1, 2}, 3); got != nil {
		t.Fatalf("empty KnnSearch = %v, want nil", got)
	}
	if got := empty.RadiusSearch([]float64{1, 2}, 5); got != nil {
		t.Fatalf("empty RadiusSearch = %v, want nil", got)
	}

	data := randFloatData(10, 3, 5, 1)
	kd := flann.NewKDTreeIndex(data)
	if got := kd.KnnSearch([]float64{1, 1, 1}, 0); got != nil {
		t.Fatalf("k=0 KnnSearch = %v, want nil", got)
	}
	// Requesting more than available returns everything.
	if got := kd.KnnSearch([]float64{1, 1, 1}, 100); len(got) != 10 {
		t.Fatalf("k>n returned %d neighbours, want 10", len(got))
	}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("ragged dataset did not panic")
			}
		}()
		flann.NewLinearIndex([][]float64{{1, 2}, {3}})
	}()

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("oversized LSH key did not panic")
			}
		}()
		flann.NewLSHIndex(randBinaryData(4, 2, 1), 4, 65, 1)
	}()
}

// TestNeighborDistancesAreEuclidean confirms reported distances are true
// Euclidean magnitudes, not squared.
func TestNeighborDistancesAreEuclidean(t *testing.T) {
	data := [][]float64{{0, 0}, {3, 4}, {6, 8}}
	kd := flann.NewKDTreeIndex(data)
	got := kd.KnnSearch([]float64{0, 0}, 3)
	wantDist := []float64{0, 5, 10}
	for i, nb := range got {
		if math.Abs(nb.Distance-wantDist[i]) > 1e-9 {
			t.Fatalf("neighbour %d distance = %v, want %v", i, nb.Distance, wantDist[i])
		}
	}
}

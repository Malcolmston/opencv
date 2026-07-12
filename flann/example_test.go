package flann_test

import (
	"fmt"

	"github.com/malcolmston/opencv/flann"
)

// ExampleKDTreeIndex_KnnSearch finds the two nearest points to a query with an
// exact k-d tree.
func ExampleKDTreeIndex_KnnSearch() {
	data := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, // near the origin
		{10, 10}, {10, 11}, {11, 10}, // a distant cluster
	}
	kd := flann.NewKDTreeIndex(data)

	for _, nb := range kd.KnnSearch([]float64{0.2, 0.1}, 2) {
		fmt.Printf("index %d at distance %.3f\n", nb.Index, nb.Distance)
	}
	// Output:
	// index 0 at distance 0.224
	// index 2 at distance 0.806
}

// ExampleKDTreeIndex_RadiusSearch returns every point within a fixed radius.
func ExampleKDTreeIndex_RadiusSearch() {
	data := [][]float64{{0, 0}, {1, 0}, {2, 0}, {5, 0}, {9, 0}}
	kd := flann.NewKDTreeIndex(data)

	for _, nb := range kd.RadiusSearch([]float64{0, 0}, 2.5) {
		fmt.Printf("index %d at distance %.1f\n", nb.Index, nb.Distance)
	}
	// Output:
	// index 0 at distance 0.0
	// index 1 at distance 1.0
	// index 2 at distance 2.0
}

// ExampleLinearIndex shows the exact brute-force baseline.
func ExampleLinearIndex() {
	data := [][]float64{{1, 1}, {2, 2}, {8, 8}, {9, 9}}
	lin := flann.NewLinearIndex(data)

	nb := lin.KnnSearch([]float64{8.4, 8.4}, 1)[0]
	fmt.Printf("nearest is index %d\n", nb.Index)
	// Output:
	// nearest is index 2
}

// ExampleKMeansIndex builds a hierarchical k-means tree and finds the nearest
// neighbour of a query.
func ExampleKMeansIndex() {
	data := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{20, 20}, {20, 21}, {21, 20}, {21, 21},
	}
	km := flann.NewKMeansIndex(data, 2, 2, 1)

	nb := km.KnnSearch([]float64{20.4, 20.4}, 1)[0]
	fmt.Printf("nearest is index %d\n", nb.Index)
	// Output:
	// nearest is index 4
}

// ExampleLSHIndex locates an exact binary match among distractors.
func ExampleLSHIndex() {
	data := [][]byte{
		{0x00, 0x00},
		{0xFF, 0xFF},
		{0x0F, 0xF0},
		{0xAA, 0x55},
	}
	lsh := flann.NewLSHIndex(data, 8, 12, 1)

	nb := lsh.KnnSearch([]byte{0x0F, 0xF0}, 1)[0]
	fmt.Printf("index %d at Hamming distance %.0f\n", nb.Index, nb.Distance)
	// Output:
	// index 2 at Hamming distance 0
}

// ExampleDistL2 computes a Euclidean distance.
func ExampleDistL2() {
	fmt.Println(flann.DistL2([]float64{0, 0}, []float64{3, 4}))
	// Output: 5
}

// ExampleDistHamming counts differing bits.
func ExampleDistHamming() {
	fmt.Println(flann.DistHamming([]byte{0x00, 0xFF}, []byte{0x0F, 0xF0}))
	// Output: 8
}

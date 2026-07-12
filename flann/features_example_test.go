package flann_test

import (
	"bytes"
	"fmt"

	"github.com/malcolmston/opencv/flann"
)

// ExampleKDForestIndex finds the nearest neighbour with a randomized k-d forest.
func ExampleKDForestIndex() {
	data := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{10, 10}, {10, 11}, {11, 10}, {11, 11},
	}
	forest := flann.NewKDForestIndex(data, 4, 1)

	nb := forest.KnnSearch([]float64{10.2, 10.1}, 1)[0]
	fmt.Printf("nearest is index %d\n", nb.Index)
	// Output:
	// nearest is index 4
}

// ExampleHierarchicalClusteringIndex searches a hierarchical clustering tree.
func ExampleHierarchicalClusteringIndex() {
	data := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{20, 20}, {20, 21}, {21, 20}, {21, 21},
	}
	h := flann.NewHierarchicalClusteringIndex(data, 2, 2, 1, 1)

	nb := h.KnnSearch([]float64{20.4, 20.4}, 1)[0]
	fmt.Printf("nearest is index %d\n", nb.Index)
	// Output:
	// nearest is index 4
}

// ExampleCompositeIndex combines a k-d forest and a k-means tree.
func ExampleCompositeIndex() {
	data := [][]float64{{0, 0}, {1, 1}, {2, 2}, {30, 30}, {31, 31}, {32, 32}}
	c := flann.NewCompositeIndex(data, 3, 2, 4, 1)

	nb := c.KnnSearch([]float64{30.4, 30.4}, 1)[0]
	fmt.Printf("nearest is index %d\n", nb.Index)
	// Output:
	// nearest is index 3
}

// ExampleAutotunedIndex tunes an index to a target precision.
func ExampleAutotunedIndex() {
	data := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{50, 50}, {50, 51}, {51, 50}, {51, 51},
	}
	auto := flann.NewAutotunedIndex(data, 1.0, 1)

	nb := auto.KnnSearch([]float64{50.2, 50.2}, 1)[0]
	fmt.Printf("nearest is index %d\n", nb.Index)
	// Output:
	// nearest is index 4
}

// ExampleDistL1 computes a Manhattan distance.
func ExampleDistL1() {
	fmt.Println(flann.DistL1([]float64{1, 2, 3}, []float64{4, 0, 3}))
	// Output: 5
}

// ExampleDistCosine shows orthogonal vectors are at cosine distance 1.
func ExampleDistCosine() {
	fmt.Println(flann.DistCosine([]float64{1, 0}, []float64{0, 2}))
	// Output: 1
}

// ExampleKnnSearchBatch answers several queries in one call.
func ExampleKnnSearchBatch() {
	data := [][]float64{{0, 0}, {5, 5}, {10, 10}}
	kd := flann.NewKDTreeIndex(data)

	results := flann.KnnSearchBatch[[]float64](kd, [][]float64{{0.1, 0.1}, {9.9, 9.9}}, 1)
	for i, r := range results {
		fmt.Printf("query %d -> index %d\n", i, r[0].Index)
	}
	// Output:
	// query 0 -> index 0
	// query 1 -> index 2
}

// ExampleRecall measures an approximate index against brute force.
func ExampleRecall() {
	data := [][]float64{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}
	exact := flann.NewLinearIndex(data)
	forest := flann.NewKDForestIndex(data, 4, 1) // unbounded: exact

	queries := [][]float64{{0.1, 0.1}, {3.9, 3.9}}
	fmt.Printf("%.1f\n", flann.Recall[[]float64](forest, exact, queries, 2))
	// Output: 1.0
}

// ExampleSave round-trips a forest through gob and queries the reloaded copy.
func ExampleSave() {
	data := [][]float64{{0, 0}, {5, 5}, {10, 10}}
	forest := flann.NewKDForestIndex(data, 4, 1)

	var buf bytes.Buffer
	if err := flann.Save(&buf, forest); err != nil {
		panic(err)
	}
	var loaded flann.KDForestIndex
	if err := flann.Load(&buf, &loaded); err != nil {
		panic(err)
	}
	nb := loaded.KnnSearch([]float64{9.5, 9.5}, 1)[0]
	fmt.Printf("nearest is index %d\n", nb.Index)
	// Output:
	// nearest is index 2
}

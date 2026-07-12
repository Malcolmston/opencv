package ml_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/ml"
)

// ExampleKNearest classifies a point using a memorised training set.
func ExampleKNearest() {
	samples := [][]float64{
		{1, 1}, {1, 2}, {2, 1}, // class 0
		{8, 8}, {8, 9}, {9, 8}, // class 1
	}
	labels := []int{0, 0, 0, 1, 1, 1}

	knn := ml.NewKNearest(3)
	_ = knn.Train(samples, labels)

	fmt.Println(knn.Predict([]float64{1.5, 1.5}))
	fmt.Println(knn.Predict([]float64{8.5, 8.5}))
	// Output:
	// 0
	// 1
}

// ExampleLogisticRegression trains a softmax classifier and reports its
// training accuracy.
func ExampleLogisticRegression() {
	samples := [][]float64{
		{0, 0}, {0.2, 0.1}, {0.1, 0.3},
		{5, 5}, {5.2, 4.9}, {4.8, 5.1},
	}
	labels := []int{0, 0, 0, 1, 1, 1}

	lr := ml.NewLogisticRegression()
	_ = lr.Train(samples, labels)

	pred := lr.PredictBatch(samples)
	fmt.Printf("accuracy=%.2f\n", ml.Accuracy(pred, labels))
	// Output:
	// accuracy=1.00
}

// ExampleKMeans recovers two clusters from unlabelled data.
func ExampleKMeans() {
	data := [][]float64{
		{1, 1}, {1, 2}, {2, 1},
		{9, 9}, {9, 8}, {8, 9},
	}
	labels, centers := ml.KMeans(data, 2, 100, 1)

	// The two clusters must be split into two distinct groups.
	fmt.Println(labels[0] == labels[1] && labels[1] == labels[2])
	fmt.Println(labels[3] == labels[4] && labels[4] == labels[5])
	fmt.Println(labels[0] != labels[3])
	fmt.Println(len(centers))
	// Output:
	// true
	// true
	// true
	// 2
}

// ExampleMatToSamples converts a single-channel Mat into a feature matrix, one
// row per sample.
func ExampleMatToSamples() {
	m := cv.NewMat(2, 3, 1) // 2 samples, 3 features each
	m.Data[0], m.Data[1], m.Data[2] = 10, 20, 30
	m.Data[3], m.Data[4], m.Data[5] = 40, 50, 60

	samples := ml.MatToSamples(m)
	fmt.Println(samples[0])
	fmt.Println(samples[1])
	// Output:
	// [10 20 30]
	// [40 50 60]
}

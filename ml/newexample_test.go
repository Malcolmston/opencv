package ml_test

import (
	"fmt"

	"github.com/malcolmston/opencv/ml"
)

// ExampleRTrees trains a random forest and reports its out-of-bag error.
func ExampleRTrees() {
	samples := [][]float64{
		{0, 0}, {0.3, 0.1}, {0.1, 0.4}, {0.2, 0.2},
		{5, 5}, {5.3, 4.9}, {4.8, 5.2}, {5.1, 4.7},
	}
	labels := []int{0, 0, 0, 0, 1, 1, 1, 1}

	rf := ml.NewRTrees(50)
	_ = rf.Train(samples, labels)

	fmt.Println(rf.Predict([]float64{0.1, 0.1}))
	fmt.Println(rf.Predict([]float64{5.0, 5.0}))
	// Output:
	// 0
	// 1
}

// ExampleBoost classifies with an AdaBoost ensemble of decision stumps.
func ExampleBoost() {
	samples := [][]float64{
		{1, 1}, {1.2, 0.8}, {0.9, 1.1},
		{5, 5}, {5.2, 4.8}, {4.9, 5.1},
	}
	labels := []int{0, 0, 0, 1, 1, 1}

	b := ml.NewBoost(20)
	_ = b.Train(samples, labels)

	pred := b.PredictBatch(samples)
	fmt.Printf("accuracy=%.2f\n", ml.Accuracy(pred, labels))
	// Output:
	// accuracy=1.00
}

// ExampleANNMLP trains a multilayer perceptron to solve the XOR problem, which
// no linear model can.
func ExampleANNMLP() {
	samples := [][]float64{{0, 0}, {1, 1}, {0, 1}, {1, 0}}
	labels := []int{1, 1, 0, 0}

	net := ml.NewANNMLP(8)
	net.Activation = ml.Tanh
	net.LearningRate = 0.5
	net.Epochs = 4000
	_ = net.Train(samples, labels)

	fmt.Println(ml.Accuracy(net.PredictBatch(samples), labels))
	// Output:
	// 1
}

// ExampleGaussianMixture fits a two-component mixture and assigns points to
// clusters.
func ExampleGaussianMixture() {
	data := [][]float64{
		{0, 0}, {0.2, 0.1}, {0.1, 0.2},
		{5, 5}, {5.2, 4.9}, {4.8, 5.1},
	}
	gmm := ml.NewGaussianMixture(2)
	_ = gmm.Fit(data)

	// Points in the same blob share a component; the two blobs differ.
	a := gmm.Predict(data[0])
	b := gmm.Predict(data[3])
	fmt.Println(gmm.Predict(data[1]) == a)
	fmt.Println(a != b)
	// Output:
	// true
	// true
}

// ExampleKernelSVM separates a non-linearly-separable XOR layout with the RBF
// kernel.
func ExampleKernelSVM() {
	samples := [][]float64{
		{0, 0}, {1, 1}, // class 1 (main diagonal)
		{0, 1}, {1, 0}, // class 0 (anti-diagonal)
	}
	labels := []int{1, 1, 0, 0}

	svm := ml.NewKernelSVM(ml.RBFKernel)
	svm.Gamma = 1.0
	svm.Epochs = 200
	_ = svm.Train(samples, labels)

	fmt.Println(ml.Accuracy(svm.PredictBatch(samples), labels))
	// Output:
	// 1
}

// ExampleCrossValScore reports five-fold cross-validated accuracy for a forest.
func ExampleCrossValScore() {
	var samples [][]float64
	var labels []int
	for i := 0; i < 20; i++ {
		samples = append(samples, []float64{float64(i%5) * 0.1, float64(i%3) * 0.1})
		labels = append(labels, 0)
		samples = append(samples, []float64{5 + float64(i%5)*0.1, 5 + float64(i%3)*0.1})
		labels = append(labels, 1)
	}
	td := ml.NewTrainData(samples, labels)

	scores := ml.CrossValScore(ml.NewRTrees(20), td, 5, 1)
	fmt.Printf("mean=%.2f\n", ml.MeanScore(scores))
	// Output:
	// mean=1.00
}

// ExampleAUC scores a ranking with the area under the ROC curve.
func ExampleAUC() {
	scores := []float64{0.9, 0.6, 0.4, 0.1}
	actual := []int{1, 1, 0, 0}
	fmt.Printf("%.2f\n", ml.AUC(scores, actual, 1))
	// Output:
	// 1.00
}

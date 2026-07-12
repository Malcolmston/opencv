package dnn_test

import (
	"fmt"

	"github.com/malcolmston/opencv/dnn"
)

// ExampleConvTranspose2D upsamples a 2x2 map with a diagonal 2x2 kernel.
func ExampleConvTranspose2D() {
	in := dnn.NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 2, 3, 4})
	w := dnn.NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 0, 0, 1})
	out := dnn.NewConvTranspose2D(w, nil, 1, 0, 1).Forward([]*dnn.Tensor{in})[0]
	fmt.Println(out.Shape, out.Data)
	// Output:
	// [1 1 3 3] [1 2 0 3 5 2 0 3 4]
}

// ExampleGlobalAvgPool averages each channel over its spatial extent.
func ExampleGlobalAvgPool() {
	in := dnn.NewTensorFrom([]int{1, 2, 2, 2}, []float64{1, 2, 3, 4, 10, 20, 30, 40})
	out := (&dnn.GlobalAvgPool{}).Forward([]*dnn.Tensor{in})[0]
	fmt.Println(out.Shape, out.Data)
	// Output:
	// [1 2 1 1] [2.5 25]
}

// ExampleUpsample nearest-neighbour doubles a 2x2 feature map.
func ExampleUpsample() {
	in := dnn.NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 2, 3, 4})
	out := dnn.NewUpsampleNearest(2).Forward([]*dnn.Tensor{in})[0]
	fmt.Println(out.Data)
	// Output:
	// [1 1 2 2 1 1 2 2 3 3 4 4 3 3 4 4]
}

// ExampleEltwise takes the elementwise maximum of two tensors.
func ExampleEltwise() {
	a := dnn.NewTensorFrom([]int{1, 3}, []float64{1, 8, 2})
	b := dnn.NewTensorFrom([]int{1, 3}, []float64{5, 3, 9})
	out := dnn.NewEltwise(dnn.EltwiseMax).Forward([]*dnn.Tensor{a, b})[0]
	fmt.Println(out.Data)
	// Output:
	// [5 8 9]
}

// ExampleArgMax reports the winning class per row.
func ExampleArgMax() {
	in := dnn.NewTensorFrom([]int{2, 3}, []float64{1, 5, 2, 9, 0, 3})
	out := (&dnn.ArgMax{Axis: -1, KeepDims: false}).Forward([]*dnn.Tensor{in})[0]
	fmt.Println(out.Data)
	// Output:
	// [1 0]
}

// ExampleNMSBoxes filters overlapping detections, keeping the strongest.
func ExampleNMSBoxes() {
	boxes := []dnn.Box{
		{X: 0, Y: 0, W: 10, H: 10},
		{X: 1, Y: 1, W: 10, H: 10},
		{X: 50, Y: 50, W: 10, H: 10},
	}
	scores := []float64{0.9, 0.8, 0.7}
	fmt.Println(dnn.NMSBoxes(boxes, scores, 0.5, 0.5))
	// Output:
	// [0 2]
}

// ExampleClassifyTopK ranks the top classes of a score vector.
func ExampleClassifyTopK() {
	scores := dnn.NewTensorFrom([]int{1, 4}, []float64{0.1, 0.7, 0.15, 0.05})
	for _, c := range dnn.ClassifyTopK(scores, 2) {
		fmt.Printf("class %d: %.2f\n", c.Index, c.Score)
	}
	// Output:
	// class 1: 0.70
	// class 2: 0.15
}

package dnn_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/dnn"
)

// ExampleBlobFromImage converts a single RGB pixel into an NCHW blob.
func ExampleBlobFromImage() {
	m := cv.NewMat(1, 1, 3)
	m.Set(0, 0, 0, 100) // R
	m.Set(0, 0, 1, 150) // G
	m.Set(0, 0, 2, 200) // B

	blob := dnn.BlobFromImage(m, 1.0, nil, false)
	fmt.Println(blob.Shape, blob.Data)
	// Output:
	// [1 3 1 1] [100 150 200]
}

// ExampleConv2D convolves a 2x2 kernel over a 3x3 image.
func ExampleConv2D() {
	in := dnn.NewTensorFrom([]int{1, 1, 3, 3}, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	w := dnn.NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 0, 0, -1})
	bias := dnn.NewTensorFrom([]int{1}, []float64{1})

	out := dnn.NewConv2D(w, bias, 1, 0, 1).Forward([]*dnn.Tensor{in})[0]
	fmt.Println(out.Data)
	// Output:
	// [-3 -3 -3 -3]
}

// ExampleSoftmax turns logits into a probability distribution.
func ExampleSoftmax() {
	in := dnn.NewTensorFrom([]int{1, 3}, []float64{1, 2, 3})
	out := dnn.NewSoftmax().Forward([]*dnn.Tensor{in})[0]
	fmt.Printf("%.3f %.3f %.3f\n", out.Data[0], out.Data[1], out.Data[2])
	// Output:
	// 0.090 0.245 0.665
}

// ExampleSequential assembles a dense classifier and reports the predicted
// class for an input.
func ExampleSequential() {
	w := dnn.NewTensorFrom([]int{2, 3}, []float64{
		2, 0, 0,
		0, 0, 2,
	})
	net := dnn.NewSequential().
		Dense(dnn.NewFullyConnected(w, nil)).
		Softmax().
		Build()

	out := net.Forward(dnn.NewTensorFrom([]int{1, 3}, []float64{1, 2, 3}))
	class := 0
	if out.Data[1] > out.Data[0] {
		class = 1
	}
	fmt.Printf("class %d\n", class)
	// Output:
	// class 1
}

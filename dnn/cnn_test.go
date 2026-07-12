package dnn

import "testing"

// argmax returns the index of the largest element.
func argmax(v []float64) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// buildTinyCNN hand-wires a two-class classifier:
//
//	Conv2D(2 filters, 2x2) -> ReLU -> MaxPool(3x3 global) -> Flatten -> Dense -> Softmax
//
// Filter 0 detects a top-heavy vertical gradient, filter 1 a bottom-heavy one.
// The dense layer is an identity map from the two pooled features to two class
// logits, so class 0 = "top-heavy" and class 1 = "bottom-heavy".
func buildTinyCNN() *Net {
	convW := NewTensorFrom([]int{2, 1, 2, 2}, []float64{
		1, 1, -1, -1, // filter 0: top row +, bottom row -
		-1, -1, 1, 1, // filter 1: bottom row +, top row -
	})
	denseW := NewTensorFrom([]int{2, 2}, []float64{
		1, 0,
		0, 1,
	})
	return NewSequential().
		Conv2D(NewConv2D(convW, nil, 1, 0, 1)).
		ReLU().
		MaxPool2D(NewMaxPool2D(3, 1)).
		Flatten().
		Dense(NewFullyConnected(denseW, nil)).
		Softmax().
		Build()
}

// topHeavy builds a 4x4 grayscale blob whose rows decrease in brightness from
// top to bottom (multiplied by amp), i.e. class 0.
func topHeavy(amp float64) *Tensor {
	data := make([]float64, 16)
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			data[r*4+c] = amp * float64(4-r)
		}
	}
	return NewTensorFrom([]int{1, 1, 4, 4}, data)
}

// bottomHeavy is the vertical mirror of topHeavy, i.e. class 1.
func bottomHeavy(amp float64) *Tensor {
	data := make([]float64, 16)
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			data[r*4+c] = amp * float64(r+1)
		}
	}
	return NewTensorFrom([]int{1, 1, 4, 4}, data)
}

func TestTinyCNNClassifies(t *testing.T) {
	net := buildTinyCNN()

	type sample struct {
		in   *Tensor
		want int
	}
	// A small synthetic set: several amplitudes per class, all deterministic.
	samples := []sample{
		{topHeavy(1), 0},
		{topHeavy(3), 0},
		{topHeavy(10), 0},
		{bottomHeavy(1), 1},
		{bottomHeavy(3), 1},
		{bottomHeavy(10), 1},
	}

	for i, s := range samples {
		out := net.Forward(s.in)
		if !sameShape(out.Shape, []int{1, 2}) {
			t.Fatalf("sample %d output shape = %v want [1 2]", i, out.Shape)
		}
		// Probabilities must sum to 1.
		if !almostEqual(out.Data[0]+out.Data[1], 1, 1e-12) {
			t.Fatalf("sample %d probs %v do not sum to 1", i, out.Data)
		}
		if got := argmax(out.Data); got != s.want {
			t.Fatalf("sample %d classified as %d (probs %v), want %d", i, got, out.Data, s.want)
		}
	}
}

// TestTinyCNNConfidentAndDeterministic checks that the winning probability is
// clearly above 0.5 and that repeated inference is bit-identical.
func TestTinyCNNConfidentAndDeterministic(t *testing.T) {
	net := buildTinyCNN()
	in := topHeavy(5)

	first := net.Forward(in).Clone()
	if first.Data[0] <= 0.5 {
		t.Fatalf("class-0 confidence %v not > 0.5", first.Data[0])
	}
	for i := 0; i < 3; i++ {
		again := net.Forward(in)
		if !again.Equal(first) {
			t.Fatalf("inference not deterministic: %v vs %v", again.Data, first.Data)
		}
	}
}

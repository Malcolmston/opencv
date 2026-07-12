package dnn

import (
	"math"
	"testing"
)

func TestActivations(t *testing.T) {
	in := NewTensorFrom([]int{1, 4}, []float64{-2, -0.5, 0, 3})

	relu := (&ReLU{}).Forward([]*Tensor{in})[0]
	if !relu.Equal(NewTensorFrom([]int{1, 4}, []float64{0, 0, 0, 3})) {
		t.Fatalf("ReLU = %v", relu.Data)
	}

	leaky := (&LeakyReLU{Alpha: 0.1}).Forward([]*Tensor{in})[0]
	wantLeaky := []float64{-0.2, -0.05, 0, 3}
	for i, w := range wantLeaky {
		if !almostEqual(leaky.Data[i], w, 1e-12) {
			t.Fatalf("LeakyReLU[%d] = %v want %v", i, leaky.Data[i], w)
		}
	}

	sig := (&Sigmoid{}).Forward([]*Tensor{in})[0]
	if !almostEqual(sig.Data[2], 0.5, 1e-12) {
		t.Fatalf("Sigmoid(0) = %v want 0.5", sig.Data[2])
	}
	for i := 0; i < 4; i++ {
		if sig.Data[i] <= 0 || sig.Data[i] >= 1 {
			t.Fatalf("Sigmoid[%d] = %v out of (0,1)", i, sig.Data[i])
		}
	}

	th := (&Tanh{}).Forward([]*Tensor{in})[0]
	if !almostEqual(th.Data[2], 0, 1e-12) {
		t.Fatalf("Tanh(0) = %v want 0", th.Data[2])
	}
}

func TestFullyConnected(t *testing.T) {
	in := NewTensorFrom([]int{1, 3}, []float64{1, 2, 3})
	w := NewTensorFrom([]int{2, 3}, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	b := NewTensorFrom([]int{2}, []float64{0.5, -1})
	out := NewFullyConnected(w, b).Forward([]*Tensor{in})[0]
	want := NewTensorFrom([]int{1, 2}, []float64{1.5, 4})
	if !out.Equal(want) {
		t.Fatalf("FullyConnected = %v, want %v", out.Data, want.Data)
	}
}

func TestSoftmaxSumsToOne(t *testing.T) {
	in := NewTensorFrom([]int{2, 3}, []float64{
		1, 2, 3,
		0, 0, 0,
	})
	out := NewSoftmax().Forward([]*Tensor{in})[0]

	for row := 0; row < 2; row++ {
		var sum float64
		for c := 0; c < 3; c++ {
			v := out.At(row, c)
			if v <= 0 || v >= 1 {
				t.Fatalf("softmax prob %v not in (0,1)", v)
			}
			sum += v
		}
		if !almostEqual(sum, 1, 1e-12) {
			t.Fatalf("softmax row %d sums to %v, want 1", row, sum)
		}
	}
	// Uniform logits give a uniform distribution.
	for c := 0; c < 3; c++ {
		if !almostEqual(out.At(1, c), 1.0/3.0, 1e-12) {
			t.Fatalf("uniform softmax = %v want 1/3", out.At(1, c))
		}
	}
	// Larger logit -> larger probability.
	if !(out.At(0, 0) < out.At(0, 1) && out.At(0, 1) < out.At(0, 2)) {
		t.Fatalf("softmax not monotonic with logits: %v", out.Data)
	}
}

func TestSoftmaxStableWithLargeLogits(t *testing.T) {
	in := NewTensorFrom([]int{1, 2}, []float64{1000, 1000})
	out := NewSoftmax().Forward([]*Tensor{in})[0]
	for _, v := range out.Data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("softmax overflowed: %v", out.Data)
		}
		if !almostEqual(v, 0.5, 1e-12) {
			t.Fatalf("softmax = %v want 0.5", v)
		}
	}
}

func TestBatchNorm(t *testing.T) {
	// Two channels, 2x2 spatial. Channel 0 mean 0 var 1; channel 1 mean 10 var 4.
	in := NewTensorFrom([]int{1, 2, 2, 2}, []float64{
		-1, 1, -1, 1, // channel 0
		8, 12, 8, 12, // channel 1
	})
	mean := NewTensorFrom([]int{2}, []float64{0, 10})
	variance := NewTensorFrom([]int{2}, []float64{1, 4})
	bn := NewBatchNorm(nil, nil, mean, variance, 0)
	out := bn.Forward([]*Tensor{in})[0]

	// channel 0: (x-0)/1 = x ; channel 1: (x-10)/2 = {-1,1,-1,1}
	want := NewTensorFrom([]int{1, 2, 2, 2}, []float64{
		-1, 1, -1, 1,
		-1, 1, -1, 1,
	})
	for i := range want.Data {
		if !almostEqual(out.Data[i], want.Data[i], 1e-12) {
			t.Fatalf("BatchNorm[%d] = %v want %v", i, out.Data[i], want.Data[i])
		}
	}
}

func TestBatchNormGammaBeta(t *testing.T) {
	in := NewTensorFrom([]int{1, 1}, []float64{4})
	mean := NewTensorFrom([]int{1}, []float64{2})
	variance := NewTensorFrom([]int{1}, []float64{4}) // sqrt = 2
	gamma := NewTensorFrom([]int{1}, []float64{3})
	beta := NewTensorFrom([]int{1}, []float64{1})
	// y = 3*(4-2)/2 + 1 = 3*1 + 1 = 4
	out := NewBatchNorm(gamma, beta, mean, variance, 0).Forward([]*Tensor{in})[0]
	if !almostEqual(out.Data[0], 4, 1e-12) {
		t.Fatalf("BatchNorm gamma/beta = %v want 4", out.Data[0])
	}
}

func TestFlatten(t *testing.T) {
	in := NewTensor(2, 3, 4, 5)
	out := (&Flatten{}).Forward([]*Tensor{in})[0]
	if !sameShape(out.Shape, []int{2, 60}) {
		t.Fatalf("Flatten shape = %v want [2 60]", out.Shape)
	}
}

func TestConcat(t *testing.T) {
	a := NewTensorFrom([]int{1, 1, 1, 2}, []float64{1, 2})
	b := NewTensorFrom([]int{1, 2, 1, 2}, []float64{3, 4, 5, 6})
	out := NewConcat(1).Forward([]*Tensor{a, b})[0]
	if !sameShape(out.Shape, []int{1, 3, 1, 2}) {
		t.Fatalf("Concat shape = %v want [1 3 1 2]", out.Shape)
	}
	want := []float64{1, 2, 3, 4, 5, 6}
	for i, w := range want {
		if out.Data[i] != w {
			t.Fatalf("Concat[%d] = %v want %v", i, out.Data[i], w)
		}
	}
}

func TestAdd(t *testing.T) {
	a := NewTensorFrom([]int{2, 2}, []float64{1, 2, 3, 4})
	b := NewTensorFrom([]int{2, 2}, []float64{10, 20, 30, 40})
	out := (&Add{}).Forward([]*Tensor{a, b})[0]
	want := NewTensorFrom([]int{2, 2}, []float64{11, 22, 33, 44})
	if !out.Equal(want) {
		t.Fatalf("Add = %v want %v", out.Data, want.Data)
	}
}

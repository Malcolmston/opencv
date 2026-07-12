package dnn

import "testing"

var pool4x4 = NewTensorFrom([]int{1, 1, 4, 4}, []float64{
	1, 2, 3, 4,
	5, 6, 7, 8,
	9, 10, 11, 12,
	13, 14, 15, 16,
})

func TestMaxPool2DDownsamples(t *testing.T) {
	out := NewMaxPool2D(2, 2).Forward([]*Tensor{pool4x4})[0]
	want := NewTensorFrom([]int{1, 1, 2, 2}, []float64{6, 8, 14, 16})
	if !out.Equal(want) {
		t.Fatalf("MaxPool2D = %v, want %v", out.Data, want.Data)
	}
}

func TestAvgPool2DDownsamples(t *testing.T) {
	out := NewAvgPool2D(2, 2).Forward([]*Tensor{pool4x4})[0]
	want := NewTensorFrom([]int{1, 1, 2, 2}, []float64{3.5, 5.5, 11.5, 13.5})
	if !out.Equal(want) {
		t.Fatalf("AvgPool2D = %v, want %v", out.Data, want.Data)
	}
}

// TestMaxPoolPreservesChannels confirms each channel is pooled independently.
func TestMaxPoolPreservesChannels(t *testing.T) {
	in := NewTensorFrom([]int{1, 2, 2, 2}, []float64{
		1, 2, 3, 4, // channel 0 -> max 4
		9, 8, 7, 6, // channel 1 -> max 9
	})
	out := NewMaxPool2D(2, 2).Forward([]*Tensor{in})[0]
	want := NewTensorFrom([]int{1, 2, 1, 1}, []float64{4, 9})
	if !out.Equal(want) {
		t.Fatalf("channel-wise MaxPool = %v, want %v", out.Data, want.Data)
	}
}

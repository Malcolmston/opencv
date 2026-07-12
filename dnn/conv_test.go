package dnn

import "testing"

// TestConv2DHandComputed checks a 2x2 kernel over a 3x3 image against a result
// worked out by hand.
//
//	image:            kernel:
//	1 2 3             1  0
//	4 5 6             0 -1
//	7 8 9
//
// Cross-correlation, stride 1, no padding, bias 1:
//
//	out[0,0] = 1*1 + 5*(-1) + 1 = -3
//	out[0,1] = 2*1 + 6*(-1) + 1 = -3
//	out[1,0] = 4*1 + 8*(-1) + 1 = -3
//	out[1,1] = 5*1 + 9*(-1) + 1 = -3
func TestConv2DHandComputed(t *testing.T) {
	in := NewTensorFrom([]int{1, 1, 3, 3}, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	w := NewTensorFrom([]int{1, 1, 2, 2}, []float64{
		1, 0,
		0, -1,
	})
	bias := NewTensorFrom([]int{1}, []float64{1})

	conv := NewConv2D(w, bias, 1, 0, 1)
	out := conv.Forward([]*Tensor{in})[0]

	want := NewTensorFrom([]int{1, 1, 2, 2}, []float64{-3, -3, -3, -3})
	if !out.Equal(want) {
		t.Fatalf("Conv2D output = %v, want %v", out.Data, want.Data)
	}
}

// TestConv2DPaddingStride exercises padding and stride against a hand result.
// A 3x3 identity-ish kernel with a single 1 in the centre and stride 2, pad 1
// over a 3x3 input simply samples the input at (0,0),(0,2),(2,0),(2,2).
func TestConv2DPaddingStride(t *testing.T) {
	in := NewTensorFrom([]int{1, 1, 3, 3}, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	w := NewTensorFrom([]int{1, 1, 3, 3}, []float64{
		0, 0, 0,
		0, 1, 0,
		0, 0, 0,
	})
	conv := NewConv2D(w, nil, 2, 1, 1)
	out := conv.Forward([]*Tensor{in})[0]

	// Output is 2x2, sampling the corners of the input.
	want := NewTensorFrom([]int{1, 1, 2, 2}, []float64{1, 3, 7, 9})
	if !out.Equal(want) {
		t.Fatalf("Conv2D padded output = %v, want %v", out.Data, want.Data)
	}
}

// TestConv2DMultiChannel sums contributions across two input channels and two
// output channels.
func TestConv2DMultiChannel(t *testing.T) {
	// 1x2x2x2 input: channel0 all 1, channel1 all 2.
	in := NewTensorFrom([]int{1, 2, 2, 2}, []float64{
		1, 1, 1, 1, // channel 0
		2, 2, 2, 2, // channel 1
	})
	// 2 output channels, 2 input channels, 1x1 kernels.
	// oc0 weights: (ic0=1, ic1=1) -> 1*1 + 2*1 = 3
	// oc1 weights: (ic0=2, ic1=0) -> 1*2 + 2*0 = 2
	w := NewTensorFrom([]int{2, 2, 1, 1}, []float64{1, 1, 2, 0})
	conv := NewConv2D(w, nil, 1, 0, 1)
	out := conv.Forward([]*Tensor{in})[0]

	if !sameShape(out.Shape, []int{1, 2, 2, 2}) {
		t.Fatalf("shape = %v want [1 2 2 2]", out.Shape)
	}
	for i := 0; i < 4; i++ {
		if out.Data[i] != 3 {
			t.Fatalf("oc0[%d] = %v want 3", i, out.Data[i])
		}
	}
	for i := 4; i < 8; i++ {
		if out.Data[i] != 2 {
			t.Fatalf("oc1[%d] = %v want 2", i, out.Data[i])
		}
	}
}

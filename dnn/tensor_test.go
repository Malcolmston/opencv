package dnn

import (
	"math"
	"testing"
)

// almostEqual reports whether a and b are within tol.
func almostEqual(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestTensorOffsetAndAt(t *testing.T) {
	// A 2x3 tensor laid out row-major: rows [0,1,2] and [3,4,5].
	x := NewTensorFrom([]int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	if got := x.At(0, 0); got != 0 {
		t.Fatalf("At(0,0)=%v want 0", got)
	}
	if got := x.At(1, 2); got != 5 {
		t.Fatalf("At(1,2)=%v want 5", got)
	}
	if off := x.Offset(1, 1); off != 4 {
		t.Fatalf("Offset(1,1)=%d want 4", off)
	}
	x.Set(9, 0, 1)
	if x.Data[1] != 9 {
		t.Fatalf("Set did not write flat index 1, data=%v", x.Data)
	}
}

func TestTensorReshapeInfer(t *testing.T) {
	x := NewTensor(2, 3, 4)
	r := x.Reshape(2, -1)
	if r.Dims() != 2 || r.Shape[0] != 2 || r.Shape[1] != 12 {
		t.Fatalf("Reshape inferred shape = %v, want [2 12]", r.Shape)
	}
	// Reshape shares storage.
	r.Data[0] = 7
	if x.Data[0] != 7 {
		t.Fatalf("Reshape should share backing storage")
	}
}

func TestNewTensorPanicsOnBadShape(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on non-positive axis")
		}
	}()
	NewTensor(2, 0)
}

func TestTensorEqual(t *testing.T) {
	a := NewTensorFrom([]int{2, 2}, []float64{1, 2, 3, 4})
	b := NewTensorFrom([]int{2, 2}, []float64{1, 2, 3, 4})
	c := NewTensorFrom([]int{4}, []float64{1, 2, 3, 4})
	if !a.Equal(b) {
		t.Fatalf("equal tensors reported unequal")
	}
	if a.Equal(c) {
		t.Fatalf("differently-shaped tensors reported equal")
	}
}

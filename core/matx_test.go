package core

import (
	"math"
	"testing"
)

func TestMatx22Determinant(t *testing.T) {
	m := NewMatx22d(1, 2, 3, 4)
	if got := m.Determinant(); got != -2 {
		t.Errorf("Determinant = %v, want -2", got)
	}
	inv, ok := m.Inverse()
	if !ok {
		t.Fatal("Inverse reported singular")
	}
	want := NewMatx22d(-2, 1, 1.5, -0.5)
	for i := range want {
		if math.Abs(inv[i]-want[i]) > 1e-12 {
			t.Errorf("Inverse[%d] = %v, want %v", i, inv[i], want[i])
		}
	}
}

func TestMatx33IdentityAndMul(t *testing.T) {
	id := Matx33dIdentity()
	if id.Determinant() != 1 {
		t.Errorf("identity determinant = %v", id.Determinant())
	}
	m := NewMatx33d(1, 2, 3, 4, 5, 6, 7, 8, 10)
	if got := id.Mul(m); got != m {
		t.Errorf("I*M = %v, want %v", got, m)
	}
	// M * M^{-1} == I
	inv, ok := m.Inverse()
	if !ok {
		t.Fatal("singular")
	}
	prod := m.Mul(inv)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			exp := 0.0
			if i == j {
				exp = 1
			}
			if math.Abs(float64(prod[i*3+j])-exp) > 1e-9 {
				t.Errorf("M*Minv[%d,%d] = %v", i, j, prod[i*3+j])
			}
		}
	}
}

func TestMatx33MulVec(t *testing.T) {
	m := NewMatx33d(2, 0, 0, 0, 3, 0, 0, 0, 4)
	v := NewVec3d(1, 1, 1)
	if got := m.MulVec(v); !got.Equals(NewVec3d(2, 3, 4)) {
		t.Errorf("MulVec = %v", got)
	}
}

func TestMatxTranspose(t *testing.T) {
	m := NewMatx23d(1, 2, 3, 4, 5, 6)
	tr := m.Transpose()
	if tr.Rows() != 3 || tr.Cols() != 2 {
		t.Errorf("transpose dims = %dx%d", tr.Rows(), tr.Cols())
	}
	if tr.At(0, 1) != 4 || tr.At(2, 0) != 3 {
		t.Errorf("transpose = %v", tr)
	}
}

func TestMatx33Trace(t *testing.T) {
	m := NewMatx33d(1, 0, 0, 0, 2, 0, 0, 0, 3)
	if got := m.Trace(); got != 6 {
		t.Errorf("Trace = %v, want 6", got)
	}
}

func BenchmarkMatx33Inverse(b *testing.B) {
	m := NewMatx33d(4, 7, 2, 3, 6, 1, 2, 5, 8)
	for i := 0; i < b.N; i++ {
		m.Inverse()
	}
}

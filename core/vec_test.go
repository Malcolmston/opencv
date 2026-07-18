package core

import (
	"math"
	"testing"
)

func TestVec3dArithmetic(t *testing.T) {
	a := NewVec3d(1, 2, 3)
	b := NewVec3d(4, 5, 6)
	if got := a.Add(b); !got.Equals(NewVec3d(5, 7, 9)) {
		t.Errorf("Add = %v", got)
	}
	if got := b.Sub(a); !got.Equals(NewVec3d(3, 3, 3)) {
		t.Errorf("Sub = %v", got)
	}
	if got := a.Mul(2); !got.Equals(NewVec3d(2, 4, 6)) {
		t.Errorf("Mul = %v", got)
	}
	if got := a.Dot(b); got != 32 {
		t.Errorf("Dot = %v, want 32", got)
	}
	if got := a.MulElem(b); !got.Equals(NewVec3d(4, 10, 18)) {
		t.Errorf("MulElem = %v", got)
	}
}

func TestVec3dNormAndCross(t *testing.T) {
	v := NewVec3d(1, 2, 2)
	if got := v.Norm(); got != 3 {
		t.Errorf("Norm = %v, want 3", got)
	}
	if got := v.NormSq(); got != 9 {
		t.Errorf("NormSq = %v, want 9", got)
	}
	if got := v.NormL1(); got != 5 {
		t.Errorf("NormL1 = %v, want 5", got)
	}
	x := NewVec3d(1, 0, 0)
	y := NewVec3d(0, 1, 0)
	if got := x.Cross(y); !got.Equals(NewVec3d(0, 0, 1)) {
		t.Errorf("Cross = %v, want (0,0,1)", got)
	}
	if got := NewVec3d(3, 0, 4).Normalize(); math.Abs(got[0]-0.6) > 1e-12 || math.Abs(got[2]-0.8) > 1e-12 {
		t.Errorf("Normalize = %v", got)
	}
}

func TestVecConversions(t *testing.T) {
	if got := NewVec3b(1, 2, 3).ToVec3d(); !got.Equals(NewVec3d(1, 2, 3)) {
		t.Errorf("ToVec3d = %v", got)
	}
	if got := NewVec2i(5, 6).ToVec2f(); !got.Equals(NewVec2f(5, 6)) {
		t.Errorf("ToVec2f = %v", got)
	}
	s := NewVec4i(1, 2, 3, 4).ToSlice()
	if len(s) != 4 || s[3] != 4 {
		t.Errorf("ToSlice = %v", s)
	}
}

func TestVec4fDot(t *testing.T) {
	a := NewVec4f(1, 2, 3, 4)
	b := NewVec4f(4, 3, 2, 1)
	if got := a.Dot(b); got != 20 {
		t.Errorf("Dot = %v, want 20", got)
	}
}

func BenchmarkVec3dDot(b *testing.B) {
	v := NewVec3d(1, 2, 3)
	w := NewVec3d(4, 5, 6)
	var s float64
	for i := 0; i < b.N; i++ {
		s += v.Dot(w)
	}
	_ = s
}

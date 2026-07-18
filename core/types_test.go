package core

import (
	"math"
	"testing"
)

func TestPoint2iOps(t *testing.T) {
	a := Pt2i(3, 4)
	if a.Norm() != 5 {
		t.Errorf("Norm = %v, want 5", a.Norm())
	}
	if got := a.Add(Pt2i(1, 1)); !got.Equals(Pt2i(4, 5)) {
		t.Errorf("Add = %v", got)
	}
	if got := a.Dot(Pt2i(1, 2)); got != 11 {
		t.Errorf("Dot = %v, want 11", got)
	}
	if !a.Inside(Rc2i(0, 0, 10, 10)) {
		t.Error("point should be inside")
	}
	if a.Inside(Rc2i(0, 0, 2, 2)) {
		t.Error("point should be outside")
	}
}

func TestPoint3dCross(t *testing.T) {
	x := Pt3d(1, 0, 0)
	y := Pt3d(0, 1, 0)
	if got := x.Cross(y); !got.Equals(Pt3d(0, 0, 1)) {
		t.Errorf("Cross = %v", got)
	}
}

func TestSizeAndRect(t *testing.T) {
	s := Sz2i(4, 5)
	if s.Area() != 20 {
		t.Errorf("Area = %d", s.Area())
	}
	r := Rc2i(0, 0, 4, 4)
	o := Rc2i(2, 2, 4, 4)
	if got := r.And(o); !got.Equals(Rc2i(2, 2, 2, 2)) {
		t.Errorf("And = %v", got)
	}
	if got := r.Or(o); !got.Equals(Rc2i(0, 0, 6, 6)) {
		t.Errorf("Or = %v", got)
	}
	disjoint := Rc2i(100, 100, 2, 2)
	if !r.And(disjoint).Empty() {
		t.Error("disjoint intersection should be empty")
	}
	if !r.Contains(Pt2i(1, 1)) || r.Contains(Pt2i(4, 4)) {
		t.Error("Contains wrong")
	}
}

func TestComplexAndRange(t *testing.T) {
	c := NewComplexd(1, 2).Mul(NewComplexd(3, 4))
	if c.Re != -5 || c.Im != 10 {
		t.Errorf("complex mul = %v", c)
	}
	if math.Abs(NewComplexd(3, 4).Abs()-5) > 1e-12 {
		t.Errorf("abs = %v", NewComplexd(3, 4).Abs())
	}
	rg := NewRange(2, 7)
	if rg.Size() != 5 || !rg.Contains(3) || rg.Contains(7) {
		t.Errorf("range wrong: %v", rg)
	}
	if got := rg.Intersect(NewRange(5, 10)); !got.Equals(NewRange(5, 7)) {
		t.Errorf("intersect = %v", got)
	}
}

func TestScalarAndTermCriteria(t *testing.T) {
	s := ScalarAll(2).Add(NewScalar(1, 1, 1, 1))
	if !s.Equals(NewScalar(3, 3, 3, 3)) {
		t.Errorf("scalar add = %v", s)
	}
	tc := NewTermCriteria(TermCount|TermEps, 30, 0.01)
	if !tc.IsValid() {
		t.Error("term criteria should be valid")
	}
	if NewTermCriteria(TermCount, 0, 0).IsValid() {
		t.Error("zero-count criteria should be invalid")
	}
}

func TestRotatedRectBounding(t *testing.T) {
	r := NewRotatedRect(Pt2f(10, 10), Sz2f(4, 2), 0)
	br := r.BoundingRect()
	if br.Width != 4 || br.Height != 2 {
		t.Errorf("bounding = %v", br)
	}
	if math.Abs(r.Area()-8) > 1e-9 {
		t.Errorf("area = %v", r.Area())
	}
}

package tracking

import (
	"math"
	"testing"
)

func TestRectIoU(t *testing.T) {
	a := NewRect(0, 0, 10, 10)
	b := NewRect(5, 5, 10, 10)
	// Intersection is 5x5=25, union is 100+100-25=175.
	got := a.IoU(b)
	want := 25.0 / 175.0
	if !approx(got, want, 1e-9) {
		t.Fatalf("IoU = %v, want %v", got, want)
	}
	if IoU(a, b) != got {
		t.Fatalf("free IoU disagrees with method")
	}
	// Identical boxes.
	if got := a.IoU(a); !approx(got, 1, 1e-9) {
		t.Fatalf("self IoU = %v, want 1", got)
	}
	// Disjoint boxes.
	if got := a.IoU(NewRect(100, 100, 5, 5)); got != 0 {
		t.Fatalf("disjoint IoU = %v, want 0", got)
	}
}

func TestRectIntersectUnion(t *testing.T) {
	a := NewRect(0, 0, 4, 4)
	b := NewRect(2, 2, 4, 4)
	in := a.Intersect(b)
	if in != (Rect{2, 2, 2, 2}) {
		t.Fatalf("Intersect = %v, want [2x2 from (2,2)]", in)
	}
	un := a.Union(b)
	if un != (Rect{0, 0, 6, 6}) {
		t.Fatalf("Union = %v, want [6x6 from (0,0)]", un)
	}
	if !a.Intersect(NewRect(50, 50, 2, 2)).Empty() {
		t.Fatalf("expected empty intersection")
	}
}

func TestRectCenterContains(t *testing.T) {
	r := NewRect(2, 4, 6, 8)
	c := r.Center()
	if !approx(c.X, 5, 1e-9) || !approx(c.Y, 8, 1e-9) {
		t.Fatalf("Center = %v, want (5, 8)", c)
	}
	if !r.Contains(2, 4) || !r.Contains(7, 11) {
		t.Fatalf("Contains should include corners inside")
	}
	if r.Contains(8, 4) || r.Contains(2, 12) {
		t.Fatalf("Contains should exclude the right/bottom edge")
	}
}

func TestPoint2f(t *testing.T) {
	p := Pt2f(3, 4)
	if !approx(p.Norm(), 5, 1e-9) {
		t.Fatalf("Norm = %v, want 5", p.Norm())
	}
	if d := p.Distance(Pt2f(0, 0)); !approx(d, 5, 1e-9) {
		t.Fatalf("Distance = %v, want 5", d)
	}
	s := p.Add(Pt2f(1, 1)).Sub(Pt2f(1, 1)).Scale(2)
	if !approx(s.X, 6, 1e-9) || !approx(s.Y, 8, 1e-9) {
		t.Fatalf("vector ops = %v, want (6, 8)", s)
	}
}

func TestRotatedRectBoundingRect(t *testing.T) {
	// A 10x10 box rotated 45 degrees about (0,0) has a half-diagonal of
	// 10/sqrt(2) ~= 7.07 in each axis-aligned direction.
	rr := RotatedRect{Center: Pt2f(0, 0), Width: 10, Height: 10, Angle: 45}
	b := rr.BoundingRect()
	half := 10.0 / math.Sqrt2
	if !approx(float64(b.Width)/2, half, 1.0) {
		t.Fatalf("bounding half-width = %v, want ~%v", float64(b.Width)/2, half)
	}
}

func TestTermCriteria(t *testing.T) {
	tc := NewTermCriteria(5, 0.1)
	if !tc.reached(4, 1.0) {
		t.Fatalf("should stop at iteration cap")
	}
	if !tc.reached(0, 0.05) {
		t.Fatalf("should stop on small step")
	}
	if tc.reached(0, 1.0) {
		t.Fatalf("should not stop early")
	}
	if tc.iterCap(99) != 5 {
		t.Fatalf("iterCap should honour MaxCount")
	}
	if NewTermCriteria(0, 0).iterCap(7) != 7 {
		t.Fatalf("iterCap fallback failed")
	}
}

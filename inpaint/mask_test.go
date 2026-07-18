package inpaint

import (
	"image"
	"testing"
)

func TestMaskBasics(t *testing.T) {
	m := NewMask(5, 5)
	if m.Count() != 0 || !m.Empty() {
		t.Fatalf("new mask should be empty, count=%d", m.Count())
	}
	m.Set(2, 2, true)
	if m.Count() != 1 || m.Empty() {
		t.Fatalf("count after one set = %d, want 1", m.Count())
	}
	if !m.At(2, 2) || m.At(0, 0) {
		t.Fatalf("At mismatch")
	}
}

func TestMaskDilateErode(t *testing.T) {
	m := NewMask(7, 7)
	m.Set(3, 3, true)
	d := m.Dilate(1)
	if d.Count() != 9 {
		t.Fatalf("dilate(1) count = %d, want 9", d.Count())
	}
	e := d.Erode(1)
	if e.Count() != 1 || !e.At(3, 3) {
		t.Fatalf("erode should recover single centre, count=%d", e.Count())
	}
}

func TestMaskSetOps(t *testing.T) {
	a := NewMask(4, 4)
	b := NewMask(4, 4)
	a.Set(0, 0, true)
	a.Set(1, 1, true)
	b.Set(1, 1, true)
	b.Set(2, 2, true)
	if u := a.Union(b); u.Count() != 3 {
		t.Fatalf("union count = %d, want 3", u.Count())
	}
	if in := a.Intersect(b); in.Count() != 1 || !in.At(1, 1) {
		t.Fatalf("intersect count = %d, want 1 at (1,1)", in.Count())
	}
	if s := a.Subtract(b); s.Count() != 1 || !s.At(0, 0) {
		t.Fatalf("subtract count = %d, want 1 at (0,0)", s.Count())
	}
	if inv := a.Invert(); inv.Count() != 16-2 {
		t.Fatalf("invert count = %d, want 14", inv.Count())
	}
}

func TestMaskBoundingBoxAndBoundary(t *testing.T) {
	m := centerHoleMask(8, 8, 2, 3, 2, 2) // rows 2..3, cols 3..4
	bb, ok := m.BoundingBox()
	if !ok || bb != image.Rect(3, 2, 5, 4) {
		t.Fatalf("bbox = %v ok=%v, want (3,2)-(5,4)", bb, ok)
	}
	// All four pixels of a 2x2 block are on the boundary.
	if len(m.Boundary()) != 4 {
		t.Fatalf("boundary len = %d, want 4", len(m.Boundary()))
	}
}

func TestMaskMatRoundTrip(t *testing.T) {
	m := NewMask(3, 4)
	m.Set(1, 2, true)
	mat := m.ToMat()
	if mat.At(1, 2, 0) != 255 || mat.At(0, 0, 0) != 0 {
		t.Fatalf("ToMat values wrong")
	}
	back := MaskFromMat(mat, 0)
	if !back.At(1, 2) || back.Count() != 1 {
		t.Fatalf("MaskFromMat round trip failed")
	}
}

func TestMaskFillRect(t *testing.T) {
	m := NewMask(5, 5)
	m.FillRect(image.Rect(1, 1, 3, 4), true) // cols 1..2, rows 1..3
	if m.Count() != 2*3 {
		t.Fatalf("fillrect count = %d, want 6", m.Count())
	}
}

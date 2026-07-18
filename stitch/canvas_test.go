package stitch

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestBoundsBasics(t *testing.T) {
	b := Bounds{MinX: 2, MinY: 3, MaxX: 12, MaxY: 8}
	if b.Width() != 10 || b.Height() != 5 {
		t.Fatalf("size = %dx%d, want 10x5", b.Width(), b.Height())
	}
	if b.Empty() {
		t.Fatal("expected non-empty")
	}
	if !b.Contains(2, 3) || b.Contains(12, 8) {
		t.Fatal("contains boundary handling wrong")
	}
	if (Bounds{}).Width() != 0 || !(Bounds{}).Empty() {
		t.Fatal("zero bounds must be empty")
	}
}

func TestBoundsUnionIntersect(t *testing.T) {
	a := Bounds{0, 0, 10, 10}
	b := Bounds{5, 5, 20, 8}
	u := a.Union(b)
	if u != (Bounds{0, 0, 20, 10}) {
		t.Fatalf("union = %+v", u)
	}
	in := a.Intersect(b)
	if in != (Bounds{5, 5, 10, 8}) {
		t.Fatalf("intersect = %+v", in)
	}
	if !a.Intersect(Bounds{100, 100, 110, 110}).Empty() {
		t.Fatal("disjoint intersect must be empty")
	}
	if u2 := (Bounds{}).Union(a); u2 != a {
		t.Fatalf("union with empty = %+v", u2)
	}
}

func TestUnionBounds(t *testing.T) {
	got := UnionBounds([]Bounds{{0, 0, 4, 4}, {-2, 1, 3, 9}, {}})
	if got != (Bounds{-2, 0, 4, 9}) {
		t.Fatalf("UnionBounds = %+v", got)
	}
}

func TestWarpedBoundsTranslation(t *testing.T) {
	h := TranslationHomography(3, -2)
	b := WarpedBounds(10, 6, h)
	if b != (Bounds{3, -2, 13, 4}) {
		t.Fatalf("WarpedBounds = %+v", b)
	}
}

func TestPlaceImage(t *testing.T) {
	dst := cv.NewMat(5, 5, 1)
	src := cv.NewMat(2, 2, 1)
	for i := range src.Data {
		src.Data[i] = 200
	}
	PlaceImage(dst, src, 1, 1)
	if dst.At(1, 1, 0) != 200 || dst.At(2, 2, 0) != 200 {
		t.Fatal("placed pixels not written")
	}
	if dst.At(0, 0, 0) != 0 {
		t.Fatal("outside placement should stay zero")
	}
	// Clipping: place partly out of bounds must not panic.
	PlaceImage(dst, src, 4, 4)
	if dst.At(4, 4, 0) != 200 {
		t.Fatal("clipped placement failed")
	}
}

package stitch

import (
	"math"
	"testing"
)

func approxPoint(a, b PointF, tol float64) bool {
	return math.Abs(a.X-b.X) <= tol && math.Abs(a.Y-b.Y) <= tol
}

func TestChainHomographies(t *testing.T) {
	hs := []Homography{TranslationHomography(2, 0), TranslationHomography(3, 1)}
	c := ChainHomographies(hs)
	got := c.Apply(PointF{0, 0})
	if !approxPoint(got, PointF{5, 1}, 1e-9) {
		t.Fatalf("chain apply = %v, want {5 1}", got)
	}
	if ChainHomographies(nil).Apply(PointF{4, 4}) != (PointF{4, 4}) {
		t.Fatal("empty chain must be identity")
	}
}

func TestGlobalTransformsReferenceZero(t *testing.T) {
	pairwise := []Homography{TranslationHomography(10, 0), TranslationHomography(5, 0)}
	g := GlobalTransforms(pairwise, 0)
	if len(g) != 3 {
		t.Fatalf("len = %d, want 3", len(g))
	}
	if !approxPoint(g[0].Apply(PointF{0, 0}), PointF{0, 0}, 1e-9) {
		t.Fatal("reference should be identity")
	}
	if !approxPoint(g[1].Apply(PointF{0, 0}), PointF{10, 0}, 1e-9) {
		t.Fatalf("g[1] = %v", g[1].Apply(PointF{0, 0}))
	}
	if !approxPoint(g[2].Apply(PointF{0, 0}), PointF{15, 0}, 1e-9) {
		t.Fatalf("g[2] = %v", g[2].Apply(PointF{0, 0}))
	}
}

func TestGlobalTransformsReferenceMiddle(t *testing.T) {
	pairwise := []Homography{TranslationHomography(10, 0), TranslationHomography(5, 0)}
	g := GlobalTransforms(pairwise, 1)
	if !approxPoint(g[1].Apply(PointF{0, 0}), PointF{0, 0}, 1e-9) {
		t.Fatal("reference (index 1) should be identity")
	}
	if !approxPoint(g[0].Apply(PointF{0, 0}), PointF{-10, 0}, 1e-9) {
		t.Fatalf("g[0] = %v, want {-10 0}", g[0].Apply(PointF{0, 0}))
	}
	if !approxPoint(g[2].Apply(PointF{0, 0}), PointF{5, 0}, 1e-9) {
		t.Fatalf("g[2] = %v, want {5 0}", g[2].Apply(PointF{0, 0}))
	}
}

func TestIncrementalAligner(t *testing.T) {
	a := NewIncrementalAligner()
	if g := a.Add(IdentityHomography()); g != IdentityHomography() {
		t.Fatal("first add must be identity")
	}
	a.Add(TranslationHomography(10, 0))
	a.Add(TranslationHomography(5, 0))
	if a.Count() != 3 {
		t.Fatalf("count = %d, want 3", a.Count())
	}
	g2, ok := a.Transform(2)
	if !ok || !approxPoint(g2.Apply(PointF{0, 0}), PointF{15, 0}, 1e-9) {
		t.Fatalf("transform(2) = %v", g2.Apply(PointF{0, 0}))
	}
	if _, ok := a.Transform(9); ok {
		t.Fatal("out-of-range transform must return false")
	}
	if len(a.Transforms()) != 3 {
		t.Fatal("Transforms length wrong")
	}
}

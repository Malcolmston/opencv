package template2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// newGray builds a single-channel Mat from row-major data.
func newGray(t *testing.T, rows, cols int, data []uint8) *cv.Mat {
	t.Helper()
	if len(data) != rows*cols {
		t.Fatalf("newGray: expected %d samples, got %d", rows*cols, len(data))
	}
	m := cv.NewMat(rows, cols, 1)
	copy(m.Data, data)
	return m
}

// embeddedScene returns a 6x6 background of value 5 with a 3x3 gradient template
// placed at top-left (x=2, y=1), plus the template itself.
func embeddedScene(t *testing.T) (src, templ *cv.Mat) {
	t.Helper()
	templ = newGray(t, 3, 3, []uint8{
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
	})
	src = cv.NewMat(6, 6, 1)
	for i := range src.Data {
		src.Data[i] = 5
	}
	// Copy the template to (row=1, col=2).
	for ty := 0; ty < 3; ty++ {
		for tx := 0; tx < 3; tx++ {
			src.Data[(1+ty)*6+(2+tx)] = templ.Data[ty*3+tx]
		}
	}
	return src, templ
}

func TestMatchShape(t *testing.T) {
	src, templ := embeddedScene(t)
	scores, err := MatchSSD(src, templ)
	if err != nil {
		t.Fatal(err)
	}
	if scores.Rows != 4 || scores.Cols != 4 {
		t.Fatalf("expected 4x4 score map, got %dx%d", scores.Rows, scores.Cols)
	}
}

func TestPerfectMatchSSDAndSAD(t *testing.T) {
	src, templ := embeddedScene(t)
	for _, m := range []Method{MethodSAD, MethodSSD} {
		best, err := BestMatch(src, templ, m)
		if err != nil {
			t.Fatal(err)
		}
		if best.X != 2 || best.Y != 1 {
			t.Fatalf("%s: expected match at (2,1), got (%d,%d)", m, best.X, best.Y)
		}
		if best.Score != 0 {
			t.Fatalf("%s: expected perfect score 0, got %g", m, best.Score)
		}
	}
}

func TestPerfectMatchNormalized(t *testing.T) {
	src, templ := embeddedScene(t)
	for _, m := range []Method{MethodNCC, MethodZNCC} {
		best, err := BestMatch(src, templ, m)
		if err != nil {
			t.Fatal(err)
		}
		if best.X != 2 || best.Y != 1 {
			t.Fatalf("%s: expected match at (2,1), got (%d,%d)", m, best.X, best.Y)
		}
		if math.Abs(best.Score-1.0) > 1e-9 {
			t.Fatalf("%s: expected score 1.0, got %g", m, best.Score)
		}
	}
}

func TestZNCCContrastInvariance(t *testing.T) {
	// ZNCC must be invariant to affine intensity change T' = a*T + b.
	_, templ := embeddedScene(t)
	src := cv.NewMat(3, 3, 1)
	for i := range src.Data {
		v := 2*int(templ.Data[i]) + 20 // a=2, b=20
		if v > 255 {
			v = 255
		}
		src.Data[i] = uint8(v)
	}
	best, err := BestMatch(src, templ, MethodZNCC)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(best.Score-1.0) > 1e-9 {
		t.Fatalf("ZNCC not contrast invariant: got %g", best.Score)
	}
}

func TestKnownSSDValue(t *testing.T) {
	// Two 2x2 patches differing by a known amount.
	src := newGray(t, 2, 2, []uint8{10, 12, 14, 16})
	templ := newGray(t, 2, 2, []uint8{10, 10, 10, 10})
	scores, err := MatchSSD(src, templ)
	if err != nil {
		t.Fatal(err)
	}
	// SSD = 0 + 4 + 16 + 36 = 56.
	if scores.At(0, 0) != 56 {
		t.Fatalf("expected SSD 56, got %g", scores.At(0, 0))
	}
	sad, _ := MatchSAD(src, templ)
	// SAD = 0 + 2 + 4 + 6 = 12.
	if sad.At(0, 0) != 12 {
		t.Fatalf("expected SAD 12, got %g", sad.At(0, 0))
	}
}

func TestFindMatchesThreshold(t *testing.T) {
	src, templ := embeddedScene(t)
	matches, err := FindMatches(src, templ, MethodZNCC, 0.99)
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 {
		t.Fatalf("expected exactly 1 strong match, got %d", len(matches))
	}
	if matches[0].X != 2 || matches[0].Y != 1 {
		t.Fatalf("expected match at (2,1), got (%d,%d)", matches[0].X, matches[0].Y)
	}
	if matches[0].Width != 3 || matches[0].Height != 3 {
		t.Fatalf("expected 3x3 match box, got %dx%d", matches[0].Width, matches[0].Height)
	}
}

func TestMatchGeometry(t *testing.T) {
	m := Match{X: 2, Y: 3, Width: 4, Height: 6}
	if m.Area() != 24 {
		t.Fatalf("expected area 24, got %d", m.Area())
	}
	if c := m.Center(); c.X != 4 || c.Y != 6 {
		t.Fatalf("expected center (4,6), got (%d,%d)", c.X, c.Y)
	}
	r := m.Rect()
	if r.Min.X != 2 || r.Min.Y != 3 || r.Max.X != 6 || r.Max.Y != 9 {
		t.Fatalf("unexpected rect %v", r)
	}
}

func TestMatchIoU(t *testing.T) {
	a := Match{X: 0, Y: 0, Width: 2, Height: 2}
	b := Match{X: 1, Y: 1, Width: 2, Height: 2}
	// Intersection = 1, union = 4+4-1 = 7.
	if got := a.IoU(b); math.Abs(got-1.0/7.0) > 1e-12 {
		t.Fatalf("expected IoU 1/7, got %g", got)
	}
	// Disjoint.
	c := Match{X: 10, Y: 10, Width: 2, Height: 2}
	if got := a.IoU(c); got != 0 {
		t.Fatalf("expected IoU 0 for disjoint, got %g", got)
	}
	// Identical.
	if got := a.IoU(a); math.Abs(got-1.0) > 1e-12 {
		t.Fatalf("expected IoU 1 for identical, got %g", got)
	}
}

func TestMethodPolarity(t *testing.T) {
	if !MethodZNCC.HigherIsBetter() || MethodSSD.HigherIsBetter() {
		t.Fatal("method polarity wrong")
	}
	if MethodZNCC.String() != "ZNCC" || MethodSAD.String() != "SAD" {
		t.Fatal("method string wrong")
	}
	if Method(99).Valid() {
		t.Fatal("invalid method reported valid")
	}
}

func TestMatchErrors(t *testing.T) {
	src := cv.NewMat(3, 3, 1)
	big := cv.NewMat(5, 5, 1)
	if _, err := MatchTemplate(src, big, MethodSSD); err != ErrTemplateLarger {
		t.Fatalf("expected ErrTemplateLarger, got %v", err)
	}
	three := cv.NewMat(3, 3, 3)
	if _, err := MatchTemplate(src, three, MethodSSD); err != ErrChannelMismatch {
		t.Fatalf("expected ErrChannelMismatch, got %v", err)
	}
	if _, err := MatchTemplate(src, src, Method(99)); err != ErrInvalidMethod {
		t.Fatalf("expected ErrInvalidMethod, got %v", err)
	}
}

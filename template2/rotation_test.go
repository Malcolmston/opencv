package template2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestBuildAngles(t *testing.T) {
	a := BuildAngles(0, 90, 30)
	if len(a) != 4 || a[0] != 0 || a[3] != 90 {
		t.Fatalf("expected 0,30,60,90, got %v", a)
	}
	// Reversed inputs are normalised.
	if r := BuildAngles(90, 0, 45); r[0] != 0 || r[len(r)-1] != 90 {
		t.Fatalf("reversed angles not normalised: %v", r)
	}
	// Non-positive step yields the midpoint.
	if r := BuildAngles(0, 90, 0); len(r) != 1 || r[0] != 45 {
		t.Fatalf("expected single midpoint 45, got %v", r)
	}
}

func TestRotateTemplateZero(t *testing.T) {
	_, templ := embeddedScene(t)
	r := RotateTemplate(templ, 0)
	if r.Rows != templ.Rows || r.Cols != templ.Cols {
		t.Fatalf("zero rotation changed size: %dx%d", r.Rows, r.Cols)
	}
	for i := range r.Data {
		if r.Data[i] != templ.Data[i] {
			t.Fatalf("zero rotation altered data at %d", i)
		}
	}
}

func TestRotateTemplateExpandsCanvas(t *testing.T) {
	// A non-square template rotated 90 degrees swaps its dimensions.
	templ := newGray(t, 4, 2, []uint8{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	r := RotateTemplate(templ, 90)
	if r.Rows != 2 || r.Cols != 4 {
		t.Fatalf("expected 2x4 after 90 deg rotation, got %dx%d", r.Rows, r.Cols)
	}
}

func TestMatchRotationInvariant(t *testing.T) {
	// Build an asymmetric template, rotate it a known amount, embed it, and
	// confirm the search recovers the correct angle and location.
	templ := newGray(t, 5, 5, []uint8{
		200, 200, 200, 200, 200,
		200, 40, 40, 40, 200,
		200, 40, 255, 40, 200,
		10, 10, 10, 10, 10,
		10, 10, 10, 10, 10,
	})
	rot := RotateTemplate(templ, 90)
	src := cv.NewMat(rot.Rows+6, rot.Cols+6, 1)
	for i := range src.Data {
		src.Data[i] = 128
	}
	offX, offY := 3, 2
	for ty := 0; ty < rot.Rows; ty++ {
		for tx := 0; tx < rot.Cols; tx++ {
			src.Data[(offY+ty)*src.Cols+(offX+tx)] = rot.Data[ty*rot.Cols+tx]
		}
	}

	params := DefaultRotationParams()
	params.AngleStep = 30
	params.Threshold = 0.6
	params.NMSIoU = 0.3
	matches, err := MatchRotationInvariant(src, templ, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("expected a rotated detection")
	}
	best := matches[0]
	if math.Abs(best.Angle-90) > 1e-9 {
		t.Fatalf("expected recovered angle 90, got %g", best.Angle)
	}
	if best.X != offX || best.Y != offY {
		t.Fatalf("expected detection at (%d,%d), got (%d,%d)", offX, offY, best.X, best.Y)
	}
	if best.Score < 0.9 {
		t.Fatalf("expected strong score at correct angle, got %g", best.Score)
	}
}

func TestBestMatchRotatedIdentity(t *testing.T) {
	// The template embedded unrotated must be found best at angle 0.
	_, templ := embeddedScene(t)
	src := cv.NewMat(9, 9, 1)
	for i := range src.Data {
		src.Data[i] = 5
	}
	for ty := 0; ty < 3; ty++ {
		for tx := 0; tx < 3; tx++ {
			src.Data[(2+ty)*9+(3+tx)] = templ.Data[ty*3+tx]
		}
	}
	best, err := BestMatchRotated(src, templ, MethodZNCC)
	if err != nil {
		t.Fatal(err)
	}
	if best.Angle != 0 {
		t.Fatalf("expected best angle 0, got %g", best.Angle)
	}
	if best.X != 3 || best.Y != 2 {
		t.Fatalf("expected (3,2), got (%d,%d)", best.X, best.Y)
	}
}

func BenchmarkMatchMultiScale(b *testing.B) {
	// The heaviest routine: multi-scale ZNCC over an 11-step scale sweep.
	src := cv.NewMat(96, 96, 1)
	for i := range src.Data {
		src.Data[i] = uint8((i * 7) % 251)
	}
	templ := cv.NewMat(16, 16, 1)
	for i := range templ.Data {
		templ.Data[i] = uint8((i * 13) % 251)
	}
	params := DefaultMultiScaleParams()
	params.Threshold = 0.5
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatchMultiScale(src, templ, params); err != nil {
			b.Fatal(err)
		}
	}
}

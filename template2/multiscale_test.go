package template2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestBuildScales(t *testing.T) {
	s := BuildScales(0.5, 1.5, 11)
	if len(s) != 11 {
		t.Fatalf("expected 11 scales, got %d", len(s))
	}
	if math.Abs(s[0]-0.5) > 1e-12 || math.Abs(s[10]-1.5) > 1e-12 {
		t.Fatalf("scale endpoints wrong: %g..%g", s[0], s[10])
	}
	if math.Abs(s[5]-1.0) > 1e-12 {
		t.Fatalf("expected middle scale 1.0, got %g", s[5])
	}
	// Reversed inputs are normalised.
	if r := BuildScales(1.5, 0.5, 3); r[0] != 0.5 || r[2] != 1.5 {
		t.Fatalf("reversed scales not normalised: %v", r)
	}
	// Single level returns the midpoint.
	if r := BuildScales(0.5, 1.5, 1); len(r) != 1 || r[0] != 1.0 {
		t.Fatalf("single level should be midpoint 1.0, got %v", r)
	}
}

func TestBuildPyramid(t *testing.T) {
	src := cv.NewMat(16, 16, 1)
	p := BuildPyramid(src, 3, 0.5)
	if p.NumLevels() != 3 {
		t.Fatalf("expected 3 levels, got %d", p.NumLevels())
	}
	if p.Level(0).Cols != 16 || p.Level(1).Cols != 8 || p.Level(2).Cols != 4 {
		t.Fatalf("unexpected pyramid widths: %d %d %d",
			p.Level(0).Cols, p.Level(1).Cols, p.Level(2).Cols)
	}
	if p.Scales[1] != 0.5 {
		t.Fatalf("expected level-1 scale 0.5, got %g", p.Scales[1])
	}
}

// texturedTemplate returns a high-frequency 4x4 template whose sub-windows and
// rescalings do not spuriously correlate perfectly (unlike a linear ramp).
func texturedTemplate(t *testing.T) *cv.Mat {
	t.Helper()
	return newGray(t, 4, 4, []uint8{
		10, 200, 30, 190,
		220, 15, 205, 25,
		35, 195, 12, 210,
		200, 20, 215, 18,
	})
}

func TestMatchMultiScaleNativeScale(t *testing.T) {
	templ := texturedTemplate(t)
	src := cv.NewMat(9, 9, 1)
	for i := range src.Data {
		src.Data[i] = 100
	}
	offX, offY := 2, 1
	for ty := 0; ty < 4; ty++ {
		for tx := 0; tx < 4; tx++ {
			src.Data[(offY+ty)*9+(offX+tx)] = templ.Data[ty*4+tx]
		}
	}
	params := DefaultMultiScaleParams()
	// Search native scale and larger, so the perfect hit is unique at scale 1.
	params.MinScale = 1.0
	params.MaxScale = 1.5
	params.Threshold = 0.99
	params.NMSIoU = 0.3
	matches, err := MatchMultiScale(src, templ, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("expected at least one match")
	}
	best := matches[0]
	if best.X != offX || best.Y != offY {
		t.Fatalf("expected match at (%d,%d), got (%d,%d)", offX, offY, best.X, best.Y)
	}
	if math.Abs(best.Scale-1.0) > 1e-9 {
		t.Fatalf("expected native scale 1.0, got %g", best.Scale)
	}
	if math.Abs(best.Score-1.0) > 1e-9 {
		t.Fatalf("expected perfect score at native scale, got %g", best.Score)
	}
}

func TestMatchMultiScaleEnlarged(t *testing.T) {
	// Embed a 2x-enlarged textured template and confirm it is found at scale 2.
	templ := texturedTemplate(t)
	big := cv.Resize(templ, 8, 8, cv.InterLinear) // 2x
	src := cv.NewMat(14, 14, 1)
	for i := range src.Data {
		src.Data[i] = 100
	}
	offX, offY := 3, 2
	for ty := 0; ty < 8; ty++ {
		for tx := 0; tx < 8; tx++ {
			src.Data[(offY+ty)*14+(offX+tx)] = big.Data[ty*8+tx]
		}
	}
	params := DefaultMultiScaleParams()
	params.MinScale = 1.0
	params.MaxScale = 2.0
	params.Levels = 11 // includes scale 2.0
	params.Threshold = 0.9
	params.NMSIoU = 0.2
	matches, err := MatchMultiScale(src, templ, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("expected a multi-scale detection")
	}
	best := matches[0]
	if best.X != offX || best.Y != offY {
		t.Fatalf("expected detection at (%d,%d), got (%d,%d)", offX, offY, best.X, best.Y)
	}
	// The 4px template resized to the matched 8px box corresponds to scale ~2.
	// (Scales 1.9 and 2.0 both round to an 8px template, so allow the band.)
	if best.Width != 8 || best.Height != 8 {
		t.Fatalf("expected 8x8 matched box, got %dx%d", best.Width, best.Height)
	}
	if best.Scale < 1.85 || best.Scale > 2.0 {
		t.Fatalf("expected scale near 2.0, got %g", best.Scale)
	}
	if math.Abs(best.Score-1.0) > 1e-9 {
		t.Fatalf("expected perfect score at correct scale, got %g", best.Score)
	}
}

package texture_test

import (
	"testing"

	"github.com/malcolmston/opencv/texture"
)

// A 2x2 anti-diagonal pattern quantised to two levels gives a fully-known
// horizontal GLCM: exactly the pairs (0,1) and (1,0), each once.
func checker2() *texture.GLCM {
	img := makeGray([][]uint8{{0, 200}, {200, 0}})
	return texture.NewGLCM(img, 2, 1, 0, false)
}

func TestGLCMCounts(t *testing.T) {
	g := checker2()
	if g.Levels() != 2 {
		t.Fatalf("Levels = %d, want 2", g.Levels())
	}
	if g.Sum() != 2 {
		t.Fatalf("Sum = %v, want 2", g.Sum())
	}
	if g.At(0, 1) != 1 || g.At(1, 0) != 1 {
		t.Fatalf("off-diagonal counts = %v,%v want 1,1", g.At(0, 1), g.At(1, 0))
	}
	if g.At(0, 0) != 0 || g.At(1, 1) != 0 {
		t.Fatalf("diagonal counts should be 0")
	}
}

func TestGLCMHaralickKnown(t *testing.T) {
	g := checker2()
	cases := []struct {
		name string
		got  float64
		want float64
	}{
		{"Contrast", g.Contrast(), 1.0},
		{"Dissimilarity", g.Dissimilarity(), 1.0},
		{"Homogeneity", g.Homogeneity(), 0.5},
		{"ASM", g.ASM(), 0.5},
		{"Energy", g.Energy(), 0.70710678},
		{"Entropy", g.Entropy(), 0.69314718},
		{"MaximumProbability", g.MaximumProbability(), 0.5},
		{"Correlation", g.Correlation(), -1.0},
		{"SumAverage", g.SumAverage(), 1.0},
		{"DifferenceAverage", g.DifferenceAverage(), 1.0},
	}
	for _, c := range cases {
		if !approx(c.got, c.want, 1e-6) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestGLCMConstantImage(t *testing.T) {
	g := texture.NewGLCM(fill(4, 4, 90), 4, 1, 0, true)
	// A flat image co-occurs only (0,0) after quantisation: pure order.
	if !approx(g.Contrast(), 0, 1e-12) {
		t.Errorf("Contrast on flat image = %v, want 0", g.Contrast())
	}
	if !approx(g.ASM(), 1, 1e-12) {
		t.Errorf("ASM on flat image = %v, want 1", g.ASM())
	}
	if !approx(g.Entropy(), 0, 1e-12) {
		t.Errorf("Entropy on flat image = %v, want 0", g.Entropy())
	}
	// Correlation is defined as 0 when a marginal has zero variance.
	if g.Correlation() != 0 {
		t.Errorf("Correlation on flat image = %v, want 0", g.Correlation())
	}
}

func TestGLCMSymmetry(t *testing.T) {
	img := makeGray([][]uint8{{0, 64, 128}, {192, 128, 64}, {0, 64, 128}})
	g := texture.NewGLCM(img, 4, 1, 0, true)
	L := g.Levels()
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			if g.At(i, j) != g.At(j, i) {
				t.Fatalf("symmetric GLCM not symmetric at (%d,%d)", i, j)
			}
		}
	}
}

func TestGLCMHaralickBundle(t *testing.T) {
	g := checker2()
	h := g.Haralick()
	if !approx(h.Contrast, 1.0, 1e-9) || !approx(h.Energy, 0.70710678, 1e-6) {
		t.Errorf("bundle mismatch: %+v", h)
	}
}

func TestComputeGLCMDirections(t *testing.T) {
	img := makeGray([][]uint8{{0, 0}, {200, 200}})
	// Vertical neighbours differ (0 above 200) -> non-zero contrast.
	v := texture.ComputeGLCM(img, 2, texture.Vertical, 1, true)
	if v.Contrast() == 0 {
		t.Errorf("vertical contrast should be > 0 for a horizontal edge")
	}
	// Horizontal neighbours are equal -> zero contrast.
	h := texture.ComputeGLCM(img, 2, texture.Horizontal, 1, true)
	if !approx(h.Contrast(), 0, 1e-12) {
		t.Errorf("horizontal contrast = %v, want 0", h.Contrast())
	}
}

func TestDirectionOffset(t *testing.T) {
	cases := []struct {
		dir    texture.Direction
		dx, dy int
	}{
		{texture.Horizontal, 2, 0},
		{texture.Diagonal45, 2, -2},
		{texture.Vertical, 0, -2},
		{texture.Diagonal135, -2, -2},
	}
	for _, c := range cases {
		dx, dy := c.dir.Offset(2)
		if dx != c.dx || dy != c.dy {
			t.Errorf("%v.Offset(2) = (%d,%d), want (%d,%d)", c.dir, dx, dy, c.dx, c.dy)
		}
	}
}

func TestGLCMPanics(t *testing.T) {
	assertPanic(t, "levels<2", func() { texture.NewGLCM(fill(2, 2, 0), 1, 1, 0, false) })
	assertPanic(t, "zero offset", func() { texture.NewGLCM(fill(2, 2, 0), 2, 0, 0, false) })
}

func assertPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic", name)
		}
	}()
	fn()
}

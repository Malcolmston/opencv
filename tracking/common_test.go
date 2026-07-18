package tracking

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// noiseField is a deterministic, smooth, non-periodic 2-D field: independent
// random values on a coarse lattice, bilinearly interpolated. It yields rich
// gradients in both directions at every scale without the aliasing that a
// periodic pattern causes at coarse pyramid levels, making it a good probe for
// optical-flow algorithms.
type noiseField struct {
	step float64
	g    [][]float64
}

// newNoiseField builds a field large enough to cover a rows*cols image with a
// margin, using the given seed for reproducibility.
func newNoiseField(rows, cols int, seed int64) *noiseField {
	step := 6.0
	gr := int(float64(rows)/step) + 8
	gc := int(float64(cols)/step) + 8
	r := rand.New(rand.NewSource(seed))
	g := make([][]float64, gr)
	for i := range g {
		g[i] = make([]float64, gc)
		for j := range g[i] {
			g[i][j] = 40 + r.Float64()*180
		}
	}
	return &noiseField{step: step, g: g}
}

// at samples the field at continuous coordinate (fx, fy) with a positive margin.
func (n *noiseField) at(fx, fy float64) float64 {
	gx := fx/n.step + 4
	gy := fy/n.step + 4
	x0 := int(math.Floor(gx))
	y0 := int(math.Floor(gy))
	dx := gx - float64(x0)
	dy := gy - float64(y0)
	get := func(j, i int) float64 {
		if j < 0 {
			j = 0
		} else if j >= len(n.g) {
			j = len(n.g) - 1
		}
		if i < 0 {
			i = 0
		} else if i >= len(n.g[0]) {
			i = len(n.g[0]) - 1
		}
		return n.g[j][i]
	}
	top := get(y0, x0)*(1-dx) + get(y0, x0+1)*dx
	bot := get(y0+1, x0)*(1-dx) + get(y0+1, x0+1)*dx
	return top*(1-dy) + bot*dy
}

// synthTexture builds a deterministic single-channel image whose content is a
// smooth non-periodic texture sampled with the content shifted by (shiftX,
// shiftY). A feature at (x, y) in the unshifted image appears at (x+shiftX,
// y+shiftY) here, so optical flow from the unshifted image to this one is
// (shiftX, shiftY). The underlying field depends only on (rows, cols), so the
// prev and next frames of a pair sample the same texture.
func synthTexture(rows, cols int, shiftX, shiftY float64) *cv.Mat {
	field := newNoiseField(rows, cols, int64(rows*1000+cols))
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := field.at(float64(x)-shiftX, float64(y)-shiftY)
			m.Data[y*cols+x] = clampU8(v)
		}
	}
	return m
}

// approx reports whether a and b are within tol of each other.
func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func requireTrue(t *testing.T, cond bool, format string, args ...any) {
	t.Helper()
	if !cond {
		t.Fatalf(format, args...)
	}
}

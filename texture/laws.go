package texture

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// The five canonical Laws 1-D masks of length 5. Each captures a distinct
// micro-structure: Level (local average), Edge, Spot, Wave and Ripple. Their
// outer products form the 25 separable 5x5 Laws convolution kernels.

// LawsL5 returns the length-5 Level mask [1 4 6 4 1], a smoothing (local
// average) filter.
func LawsL5() []float64 { return []float64{1, 4, 6, 4, 1} }

// LawsE5 returns the length-5 Edge mask [-1 -2 0 2 1], a first-derivative
// (edge) filter.
func LawsE5() []float64 { return []float64{-1, -2, 0, 2, 1} }

// LawsS5 returns the length-5 Spot mask [-1 0 2 0 -1], a second-derivative
// (spot) filter.
func LawsS5() []float64 { return []float64{-1, 0, 2, 0, -1} }

// LawsW5 returns the length-5 Wave mask [-1 2 0 -2 1].
func LawsW5() []float64 { return []float64{-1, 2, 0, -2, 1} }

// LawsR5 returns the length-5 Ripple mask [1 -4 6 -4 1].
func LawsR5() []float64 { return []float64{1, -4, 6, -4, 1} }

// LawsKernel2D forms the separable 2-D Laws kernel as the outer product of a
// vertical mask v and a horizontal mask h (each typically length 5), returning
// it in row-major order with len(v) rows and len(h) columns.
func LawsKernel2D(v, h []float64) []float64 {
	if len(v) == 0 || len(h) == 0 {
		panic("texture: LawsKernel2D requires non-empty masks")
	}
	out := make([]float64, len(v)*len(h))
	idx := 0
	for i := range v {
		for j := range h {
			out[idx] = v[i] * h[j]
			idx++
		}
	}
	return out
}

// lawsMasks returns the five masks in canonical order with their names.
func lawsMasks() ([]string, [][]float64) {
	names := []string{"L5", "E5", "S5", "W5", "R5"}
	masks := [][]float64{LawsL5(), LawsE5(), LawsS5(), LawsW5(), LawsR5()}
	return names, masks
}

// LawsEnergy holds the texture-energy maps produced by [LawsEnergyMaps]. Maps
// are keyed by the pair of mask names (for example "E5L5"), each value a
// rows-by-cols grid of local energy. Rotation-invariant combined maps (for
// example "E5L5" averaged with "L5E5") are provided by [LawsEnergy.Combined].
type LawsEnergy struct {
	// Rows and Cols are the map dimensions.
	Rows, Cols int
	// Maps holds every ordered mask-pair energy map, keyed "<v><h>".
	Maps map[string][]float64
}

// Map returns the energy map for the ordered mask pair name (for example
// "S5S5"), or nil if that pair was not computed.
func (le *LawsEnergy) Map(name string) []float64 { return le.Maps[name] }

// Combined returns the rotation-invariant energy map for an unordered mask pair
// by averaging the two ordered maps (for example "E5L5" and "L5E5"). For a
// symmetric pair such as "S5S5" it simply returns that map. The result is a
// rows-by-cols grid. It panics if either constituent map is missing.
func (le *LawsEnergy) Combined(a, b string) []float64 {
	m1 := le.Maps[a+b]
	if a == b {
		if m1 == nil {
			panic(fmt.Sprintf("texture: LawsEnergy.Combined missing map %s%s", a, b))
		}
		out := make([]float64, len(m1))
		copy(out, m1)
		return out
	}
	m2 := le.Maps[b+a]
	if m1 == nil || m2 == nil {
		panic(fmt.Sprintf("texture: LawsEnergy.Combined missing map for %s,%s", a, b))
	}
	out := make([]float64, len(m1))
	for i := range out {
		out[i] = (m1[i] + m2[i]) / 2
	}
	return out
}

// LawsEnergyMaps computes Laws texture-energy maps for img. The image is first
// (optionally) contrast-normalised by subtracting a local mean over a window of
// side `window`, then convolved with each of the 25 ordered 5x5 Laws kernels;
// the texture-energy map for a kernel is the local average of the absolute
// filter response over a `window`-by-`window` neighbourhood. window must be a
// positive odd number. The returned [LawsEnergy] contains all 25 ordered maps.
func LawsEnergyMaps(img *cv.Mat, window int) *LawsEnergy {
	textureRequire(img, "LawsEnergyMaps")
	if window < 1 || window%2 == 0 {
		panic(fmt.Sprintf("texture: LawsEnergyMaps requires positive odd window, got %d", window))
	}
	rows, cols := img.Rows, img.Cols
	luma := textureLumaFloat(img)
	// Local-mean contrast normalisation removes illumination bias so the L5L5
	// response measures brightness only, per Laws' method.
	norm := textureSubtractLocalMean(luma, rows, cols, window)

	names, masks := lawsMasks()
	le := &LawsEnergy{Rows: rows, Cols: cols, Maps: make(map[string][]float64, 25)}
	for vi := range masks {
		for hi := range masks {
			kernel := LawsKernel2D(masks[vi], masks[hi])
			resp := textureConvolve(norm, rows, cols, kernel, 5)
			// energy = local mean of |response|
			for i := range resp {
				resp[i] = math.Abs(resp[i])
			}
			energy := textureBoxMean(resp, rows, cols, window)
			le.Maps[names[vi]+names[hi]] = energy
		}
	}
	return le
}

// LawsFeatures returns a compact rotation-invariant Laws texture feature
// vector: the image-wide mean of each of the nine standard combined energy maps
// (L5E5, L5S5, L5R5, L5W5, E5E5, E5S5, S5S5, R5R5, E5L5-style combinations),
// computed with the given window. The features are returned in a fixed,
// documented order suitable for direct use in a classifier.
func LawsFeatures(img *cv.Mat, window int) []float64 {
	le := LawsEnergyMaps(img, window)
	// Nine widely used combinations (rotation-invariant averages).
	pairs := [][2]string{
		{"L5", "E5"}, {"L5", "S5"}, {"L5", "R5"}, {"L5", "W5"},
		{"E5", "E5"}, {"E5", "S5"}, {"S5", "S5"}, {"R5", "R5"}, {"E5", "R5"},
	}
	out := make([]float64, len(pairs))
	for i, p := range pairs {
		m := le.Combined(p[0], p[1])
		var s float64
		for _, v := range m {
			s += v
		}
		out[i] = s / float64(len(m))
	}
	return out
}

// textureConvolve convolves a row-major plane with a square kernel of odd side
// using reflect-101 borders, returning a new plane.
func textureConvolve(src []float64, rows, cols int, kernel []float64, side int) []float64 {
	half := side / 2
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			ki := 0
			for ky := -half; ky <= half; ky++ {
				yy := textureReflect(y+ky, rows)
				base := yy * cols
				for kx := -half; kx <= half; kx++ {
					xx := textureReflect(x+kx, cols)
					acc += src[base+xx] * kernel[ki]
					ki++
				}
			}
			out[y*cols+x] = acc
		}
	}
	return out
}

// textureBoxMean returns the local mean over a window-by-window neighbourhood
// (reflect-101 borders).
func textureBoxMean(src []float64, rows, cols, window int) []float64 {
	half := window / 2
	inv := 1.0 / float64(window*window)
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for wy := -half; wy <= half; wy++ {
				yy := textureReflect(y+wy, rows)
				base := yy * cols
				for wx := -half; wx <= half; wx++ {
					xx := textureReflect(x+wx, cols)
					acc += src[base+xx]
				}
			}
			out[y*cols+x] = acc * inv
		}
	}
	return out
}

// textureSubtractLocalMean returns src minus its local box mean.
func textureSubtractLocalMean(src []float64, rows, cols, window int) []float64 {
	mean := textureBoxMean(src, rows, cols, window)
	out := make([]float64, len(src))
	for i := range src {
		out[i] = src[i] - mean[i]
	}
	return out
}

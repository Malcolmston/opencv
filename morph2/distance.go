package morph2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FloatGrid is a dense row-major grid of float64 values used to return
// real-valued morphological results such as distance maps. It is a thin helper
// container; convert to the parent package's image type with [FloatGrid.ToMat].
type FloatGrid struct {
	// Rows is the grid height.
	Rows int
	// Cols is the grid width.
	Cols int
	// Data holds Rows*Cols values in row-major order.
	Data []float64
}

// NewFloatGrid allocates a zero-filled grid of the given size. It panics on a
// non-positive dimension.
func NewFloatGrid(rows, cols int) *FloatGrid {
	if rows <= 0 || cols <= 0 {
		panic("morph2: NewFloatGrid requires positive size")
	}
	return &FloatGrid{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// At returns the value at (y, x). It panics on out-of-range coordinates.
func (g *FloatGrid) At(y, x int) float64 {
	if y < 0 || y >= g.Rows || x < 0 || x >= g.Cols {
		panic("morph2: FloatGrid.At out of range")
	}
	return g.Data[y*g.Cols+x]
}

// Set stores value at (y, x). It panics on out-of-range coordinates.
func (g *FloatGrid) Set(y, x int, value float64) {
	if y < 0 || y >= g.Rows || x < 0 || x >= g.Cols {
		panic("morph2: FloatGrid.Set out of range")
	}
	g.Data[y*g.Cols+x] = value
}

// Max returns the largest value in the grid, or 0 for an empty grid.
func (g *FloatGrid) Max() float64 {
	m := math.Inf(-1)
	for _, v := range g.Data {
		if v > m {
			m = v
		}
	}
	if math.IsInf(m, -1) {
		return 0
	}
	return m
}

// Min returns the smallest value in the grid, or 0 for an empty grid.
func (g *FloatGrid) Min() float64 {
	m := math.Inf(1)
	for _, v := range g.Data {
		if v < m {
			m = v
		}
	}
	if math.IsInf(m, 1) {
		return 0
	}
	return m
}

// ToMat converts the grid to a single-channel [cv.Mat]. When normalize is true
// the values are linearly scaled so the maximum maps to 255; otherwise each
// value is rounded and clamped to 0..255.
func (g *FloatGrid) ToMat(normalize bool) *cv.Mat {
	out := cv.NewMat(g.Rows, g.Cols, 1)
	scale := 1.0
	if normalize {
		m := g.Max()
		if m > 0 {
			scale = 255.0 / m
		}
	}
	for i, v := range g.Data {
		s := math.Round(v * scale)
		if s < 0 {
			s = 0
		} else if s > 255 {
			s = 255
		}
		out.Data[i] = uint8(s)
	}
	return out
}

// DistanceType selects the metric approximated by a distance transform.
type DistanceType int

const (
	// DistL1 is the city-block (Manhattan) metric.
	DistL1 DistanceType = iota
	// DistL2 is the Euclidean metric.
	DistL2
	// DistC is the Chebyshev (chessboard) metric.
	DistC
)

// DistanceMask selects the neighbourhood mask used to approximate a distance
// transform.
type DistanceMask int

const (
	// Mask3 uses a 3x3 chamfer mask.
	Mask3 DistanceMask = iota
	// Mask5 uses a 5x5 chamfer mask (more accurate for the Euclidean metric).
	Mask5
	// MaskPrecise computes an exact result; it is only valid with [DistL2],
	// where it uses the Felzenszwalb-Huttenlocher exact Euclidean transform.
	MaskPrecise
)

// DistanceTransform computes, for every foreground (non-zero) pixel, the
// distance to the nearest background (zero) pixel under the chosen metric and
// mask; background pixels receive 0. The result is returned as a [FloatGrid].
//
// For DistL1 and DistC the 3x3 chamfer is exact. For DistL2 the chamfer masks
// approximate the true Euclidean distance, while MaskPrecise yields the exact
// Euclidean distance. It panics on multi-channel input or an invalid
// metric/mask combination.
func DistanceTransform(src *cv.Mat, dt DistanceType, mask DistanceMask) *FloatGrid {
	requireGray(src)
	if mask == MaskPrecise {
		if dt != DistL2 {
			panic("morph2: MaskPrecise is only valid with DistL2")
		}
		return exactEuclidean(src)
	}
	var a, b, c float64
	five := mask == Mask5
	switch dt {
	case DistL1:
		a, b, c = 1, 2, 3
	case DistC:
		a, b, c = 1, 1, 2
	case DistL2:
		if five {
			a, b, c = 1, 1.4, 2.1969
		} else {
			a, b = 0.955, 1.3693
		}
	default:
		panic("morph2: unknown distance type")
	}
	return chamfer(src, a, b, c, five)
}

// DistanceTransformL1 is shorthand for the exact city-block distance transform.
func DistanceTransformL1(src *cv.Mat) *FloatGrid {
	return DistanceTransform(src, DistL1, Mask3)
}

// DistanceTransformChebyshev is shorthand for the exact chessboard distance
// transform.
func DistanceTransformChebyshev(src *cv.Mat) *FloatGrid {
	return DistanceTransform(src, DistC, Mask3)
}

// DistanceTransformExact is shorthand for the exact Euclidean distance
// transform (Felzenszwalb-Huttenlocher).
func DistanceTransformExact(src *cv.Mat) *FloatGrid {
	return exactEuclidean(src)
}

// ChamferDistance computes a chamfer distance transform with caller-supplied
// 3x3 weights: a for orthogonal steps and b for diagonal steps. It panics on
// multi-channel input.
func ChamferDistance(src *cv.Mat, a, b float64) *FloatGrid {
	requireGray(src)
	return chamfer(src, a, b, 0, false)
}

// chamfer runs a two-pass chamfer distance transform. a, b, c are the
// orthogonal, diagonal and knight-move weights; c and the knight moves are used
// only when five is true.
func chamfer(src *cv.Mat, a, b, c float64, five bool) *FloatGrid {
	rows, cols := src.Rows, src.Cols
	const inf = 1e18
	g := NewFloatGrid(rows, cols)
	for i, v := range src.Data {
		if v == 0 {
			g.Data[i] = 0
		} else {
			g.Data[i] = inf
		}
	}
	d := g.Data
	relax := func(y, x int, dy, dx int, w float64) {
		yy, xx := y+dy, x+dx
		if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
			return
		}
		nv := d[yy*cols+xx] + w
		if nv < d[y*cols+x] {
			d[y*cols+x] = nv
		}
	}
	// Forward pass.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if d[y*cols+x] == 0 {
				continue
			}
			relax(y, x, -1, -1, b)
			relax(y, x, -1, 0, a)
			relax(y, x, -1, 1, b)
			relax(y, x, 0, -1, a)
			if five {
				relax(y, x, -2, -1, c)
				relax(y, x, -2, 1, c)
				relax(y, x, -1, -2, c)
				relax(y, x, -1, 2, c)
			}
		}
	}
	// Backward pass.
	for y := rows - 1; y >= 0; y-- {
		for x := cols - 1; x >= 0; x-- {
			if d[y*cols+x] == 0 {
				continue
			}
			relax(y, x, 1, 1, b)
			relax(y, x, 1, 0, a)
			relax(y, x, 1, -1, b)
			relax(y, x, 0, 1, a)
			if five {
				relax(y, x, 2, 1, c)
				relax(y, x, 2, -1, c)
				relax(y, x, 1, 2, c)
				relax(y, x, 1, -2, c)
			}
		}
	}
	return g
}

// exactEuclidean computes the exact Euclidean distance transform using the
// Felzenszwalb-Huttenlocher separable squared-distance algorithm, then takes
// square roots.
func exactEuclidean(src *cv.Mat) *FloatGrid {
	rows, cols := src.Rows, src.Cols
	const inf = 1e18
	f := make([]float64, rows*cols)
	for i, v := range src.Data {
		if v == 0 {
			f[i] = 0
		} else {
			f[i] = inf
		}
	}
	// Transform along columns.
	col := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			col[y] = f[y*cols+x]
		}
		out := dt1d(col)
		for y := 0; y < rows; y++ {
			f[y*cols+x] = out[y]
		}
	}
	// Transform along rows.
	row := make([]float64, cols)
	for y := 0; y < rows; y++ {
		copy(row, f[y*cols:y*cols+cols])
		out := dt1d(row)
		copy(f[y*cols:y*cols+cols], out)
	}
	g := NewFloatGrid(rows, cols)
	for i, v := range f {
		g.Data[i] = math.Sqrt(v)
	}
	return g
}

// dt1d computes the 1D squared distance transform d[q] = min_p (q-p)^2 + f[p]
// via the lower envelope of parabolas.
func dt1d(f []float64) []float64 {
	n := len(f)
	d := make([]float64, n)
	v := make([]int, n)       // locations of parabolas in the lower envelope
	z := make([]float64, n+1) // boundaries between parabolas
	k := 0
	v[0] = 0
	z[0] = math.Inf(-1)
	z[1] = math.Inf(1)
	for q := 1; q < n; q++ {
		s := ((f[q] + float64(q)*float64(q)) - (f[v[k]] + float64(v[k])*float64(v[k]))) / (2*float64(q) - 2*float64(v[k]))
		for s <= z[k] {
			k--
			s = ((f[q] + float64(q)*float64(q)) - (f[v[k]] + float64(v[k])*float64(v[k]))) / (2*float64(q) - 2*float64(v[k]))
		}
		k++
		v[k] = q
		z[k] = s
		z[k+1] = math.Inf(1)
	}
	k = 0
	for q := 0; q < n; q++ {
		for z[k+1] < float64(q) {
			k++
		}
		dq := float64(q) - float64(v[k])
		d[q] = dq*dq + f[v[k]]
	}
	return d
}

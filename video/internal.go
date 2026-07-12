package video

import (
	cv "github.com/malcolmston/opencv"
)

// grid is a single-channel float64 image used internally for sub-pixel sampling
// of intensities and gradients. Data is stored row-major, length Rows*Cols.
type grid struct {
	Rows int
	Cols int
	Data []float64
}

// newGrid allocates a zero-filled grid.
func newGrid(rows, cols int) *grid {
	return &grid{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// clampInt clamps v to the inclusive range [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// atClamp returns the sample at integer coordinates (x, y), replicating the
// border for out-of-range coordinates (BORDER_REPLICATE).
func (g *grid) atClamp(x, y int) float64 {
	x = clampInt(x, 0, g.Cols-1)
	y = clampInt(y, 0, g.Rows-1)
	return g.Data[y*g.Cols+x]
}

// bilinear samples the grid at fractional coordinates (x, y) using bilinear
// interpolation, replicating the border for out-of-range neighbours.
func (g *grid) bilinear(x, y float64) float64 {
	x0 := int(floor(x))
	y0 := int(floor(y))
	dx := x - float64(x0)
	dy := y - float64(y0)
	v00 := g.atClamp(x0, y0)
	v01 := g.atClamp(x0+1, y0)
	v10 := g.atClamp(x0, y0+1)
	v11 := g.atClamp(x0+1, y0+1)
	top := v00*(1-dx) + v01*dx
	bot := v10*(1-dx) + v11*dx
	return top*(1-dy) + bot*dy
}

// gridFromMat converts a single-channel cv.Mat to a float grid.
func gridFromMat(m *cv.Mat) *grid {
	g := newGrid(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		g.Data[i] = float64(m.Data[i])
	}
	return g
}

// toGray returns a single-channel cv.Mat. A one-channel input is cloned; a
// three-channel input is converted with cv.CvtColor; any other channel count
// falls back to the first channel.
func toGray(m *cv.Mat) *cv.Mat {
	switch m.Channels {
	case 1:
		return m.Clone()
	case 3:
		return cv.CvtColor(m, cv.ColorRGB2Gray)
	default:
		out := cv.NewMat(m.Rows, m.Cols, 1)
		for p := 0; p < m.Total(); p++ {
			out.Data[p] = m.Data[p*m.Channels]
		}
		return out
	}
}

// sobelScale3 normalises a 3x3 Sobel response to a true first derivative. The
// separable Sobel kernel is the outer product of the derivative kernel
// [-1,0,1] (which yields twice the central difference) and the smoothing kernel
// [1,2,1] (which sums to four), so the raw response is 8x the per-pixel
// gradient.
const sobelScale3 = 1.0 / 8.0

// gradients returns the normalised x- and y-derivative grids of a
// single-channel cv.Mat, computed with the reused cv.SobelFloat operator.
func gradients(m *cv.Mat) (gx, gy *grid) {
	sx := cv.SobelFloat(m, 1, 0, 3)[0]
	sy := cv.SobelFloat(m, 0, 1, 3)[0]
	gx = newGrid(m.Rows, m.Cols)
	gy = newGrid(m.Rows, m.Cols)
	for i := range sx {
		gx.Data[i] = sx[i] * sobelScale3
		gy.Data[i] = sy[i] * sobelScale3
	}
	return gx, gy
}

// --- Small dense linear-algebra helpers (used by KalmanFilter) ---

// zeros allocates an r x c zero matrix.
func zeros(r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := range m {
		m[i] = make([]float64, c)
	}
	return m
}

// identity returns the n x n identity matrix.
func identity(n int) [][]float64 {
	m := zeros(n, n)
	for i := 0; i < n; i++ {
		m[i][i] = 1
	}
	return m
}

// cloneMat returns a deep copy of a matrix.
func cloneMat(a [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(a[i]))
		copy(out[i], a[i])
	}
	return out
}

// matMul returns the product a*b. It panics on a dimension mismatch.
func matMul(a, b [][]float64) [][]float64 {
	ar, ac := len(a), len(a[0])
	br, bc := len(b), len(b[0])
	if ac != br {
		panic("video: matMul dimension mismatch")
	}
	out := zeros(ar, bc)
	for i := 0; i < ar; i++ {
		for k := 0; k < ac; k++ {
			aik := a[i][k]
			if aik == 0 {
				continue
			}
			for j := 0; j < bc; j++ {
				out[i][j] += aik * b[k][j]
			}
		}
	}
	return out
}

// matVec returns the product a*v of a matrix and a column vector.
func matVec(a [][]float64, v []float64) []float64 {
	if len(a[0]) != len(v) {
		panic("video: matVec dimension mismatch")
	}
	out := make([]float64, len(a))
	for i := range a {
		var s float64
		for j := range v {
			s += a[i][j] * v[j]
		}
		out[i] = s
	}
	return out
}

// transpose returns the transpose of a.
func transpose(a [][]float64) [][]float64 {
	r, c := len(a), len(a[0])
	out := zeros(c, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j][i] = a[i][j]
		}
	}
	return out
}

// matAdd returns a+b.
func matAdd(a, b [][]float64) [][]float64 {
	out := zeros(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			out[i][j] = a[i][j] + b[i][j]
		}
	}
	return out
}

// matSub returns a-b.
func matSub(a, b [][]float64) [][]float64 {
	out := zeros(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			out[i][j] = a[i][j] - b[i][j]
		}
	}
	return out
}

// vecSub returns a-b for vectors.
func vecSub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

// vecAdd returns a+b for vectors.
func vecAdd(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// matInverse returns the inverse of a square matrix via Gauss-Jordan
// elimination with partial pivoting. The second result is false when the matrix
// is singular (to within a small tolerance).
func matInverse(a [][]float64) ([][]float64, bool) {
	n := len(a)
	// Augment [a | I].
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		copy(aug[i], a[i])
		aug[i][n+i] = 1
	}
	for col := 0; col < n; col++ {
		// Partial pivot: find the row with the largest absolute value.
		pivot := col
		best := abs(aug[col][col])
		for r := col + 1; r < n; r++ {
			if v := abs(aug[r][col]); v > best {
				best = v
				pivot = r
			}
		}
		if best < 1e-12 {
			return nil, false
		}
		aug[col], aug[pivot] = aug[pivot], aug[col]
		// Normalise the pivot row.
		pv := aug[col][col]
		for j := 0; j < 2*n; j++ {
			aug[col][j] /= pv
		}
		// Eliminate the column from every other row.
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := aug[r][col]
			if f == 0 {
				continue
			}
			for j := 0; j < 2*n; j++ {
				aug[r][j] -= f * aug[col][j]
			}
		}
	}
	inv := make([][]float64, n)
	for i := 0; i < n; i++ {
		inv[i] = make([]float64, n)
		copy(inv[i], aug[i][n:])
	}
	return inv, true
}

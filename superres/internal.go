package superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// superresClampInt clamps v into the inclusive range [lo, hi].
func superresClampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// superresClamp8 rounds v to the nearest integer and clamps it to the valid
// uint8 range [0, 255].
func superresClamp8(v float64) uint8 {
	if math.IsNaN(v) {
		return 0
	}
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// superresAt returns channel c of pixel (y, x) as a float64, replicating the
// nearest border sample for out-of-range coordinates.
func superresAt(m *cv.Mat, y, x, c int) float64 {
	y = superresClampInt(y, 0, m.Rows-1)
	x = superresClampInt(x, 0, m.Cols-1)
	return float64(m.Data[(y*m.Cols+x)*m.Channels+c])
}

// superresPlane holds a single-channel image of float64 samples used as the
// working representation for the iterative and estimation routines. It keeps
// full precision between stages so repeated resampling does not accumulate
// quantisation error.
type superresPlane struct {
	rows, cols int
	data       []float64
}

// newSuperresPlane allocates a zero-filled plane.
func newSuperresPlane(rows, cols int) *superresPlane {
	return &superresPlane{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

// at returns the sample at (y, x) with border replication.
func (p *superresPlane) at(y, x int) float64 {
	y = superresClampInt(y, 0, p.rows-1)
	x = superresClampInt(x, 0, p.cols-1)
	return p.data[y*p.cols+x]
}

// atRaw returns the sample at (y, x) without bounds checking.
func (p *superresPlane) atRaw(y, x int) float64 { return p.data[y*p.cols+x] }

// set stores value at (y, x) without bounds checking.
func (p *superresPlane) set(y, x int, value float64) { p.data[y*p.cols+x] = value }

// superresSplitPlanes decomposes a Mat into one float64 plane per channel.
func superresSplitPlanes(m *cv.Mat) []*superresPlane {
	planes := make([]*superresPlane, m.Channels)
	for c := 0; c < m.Channels; c++ {
		p := newSuperresPlane(m.Rows, m.Cols)
		for y := 0; y < m.Rows; y++ {
			for x := 0; x < m.Cols; x++ {
				p.data[y*m.Cols+x] = float64(m.Data[(y*m.Cols+x)*m.Channels+c])
			}
		}
		planes[c] = p
	}
	return planes
}

// superresMergePlanes recombines per-channel float64 planes into a uint8 Mat,
// rounding and clamping each sample. All planes must share dimensions.
func superresMergePlanes(planes []*superresPlane) *cv.Mat {
	ch := len(planes)
	rows, cols := planes[0].rows, planes[0].cols
	out := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		p := planes[c]
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				out.Data[(y*cols+x)*ch+c] = superresClamp8(p.data[y*cols+x])
			}
		}
	}
	return out
}

// superresSolve solves the linear system a*x = b for a small n×n matrix using
// Gaussian elimination with partial pivoting. a is a flat row-major slice of
// length n*n; it is modified in place. It reports ok=false if the matrix is
// singular (or nearly so).
func superresSolve(a []float64, b []float64, n int) (x []float64, ok bool) {
	// Work on copies so callers may reuse their buffers.
	m := make([]float64, len(a))
	copy(m, a)
	rhs := make([]float64, n)
	copy(rhs, b)
	for col := 0; col < n; col++ {
		// Partial pivot.
		piv := col
		best := math.Abs(m[col*n+col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r*n+col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-12 {
			return nil, false
		}
		if piv != col {
			for k := 0; k < n; k++ {
				m[col*n+k], m[piv*n+k] = m[piv*n+k], m[col*n+k]
			}
			rhs[col], rhs[piv] = rhs[piv], rhs[col]
		}
		// Eliminate below.
		pivVal := m[col*n+col]
		for r := col + 1; r < n; r++ {
			factor := m[r*n+col] / pivVal
			if factor == 0 {
				continue
			}
			for k := col; k < n; k++ {
				m[r*n+k] -= factor * m[col*n+k]
			}
			rhs[r] -= factor * rhs[col]
		}
	}
	// Back-substitute.
	x = make([]float64, n)
	for r := n - 1; r >= 0; r-- {
		sum := rhs[r]
		for k := r + 1; k < n; k++ {
			sum -= m[r*n+k] * x[k]
		}
		x[r] = sum / m[r*n+r]
	}
	return x, true
}

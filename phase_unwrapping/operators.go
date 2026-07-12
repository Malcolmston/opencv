package phase_unwrapping

import (
	"errors"
	"math"
)

// ErrShapeMismatch is returned when two or more grids that must share a shape
// (for example a wrapped map and its weight or mask) have differing dimensions.
var ErrShapeMismatch = errors.New("phase_unwrapping: input grids have mismatched shapes")

// WrapToPi maps an arbitrary phase value into the closed interval [-pi, pi]. It
// is the canonical scalar wrapping operator computed as atan2(sin(x), cos(x)),
// so it agrees with [Wrap] on the open interval and differs only in the treated
// endpoint (WrapToPi may return -pi where [Wrap] returns +pi). It is provided as
// the conventionally named building block used when re-wrapping unwrapped
// surfaces or checking congruence.
func WrapToPi(x float64) float64 {
	return math.Atan2(math.Sin(x), math.Cos(x))
}

// Rewrap re-wraps an unwrapped surface back into (-pi, pi], returning a new grid
// of the same shape. It is the inverse-model companion to unwrapping: for any
// correct unwrapped result u derived from a wrapped map psi, Rewrap(u) reproduces
// psi exactly (they are congruent modulo 2*pi). It is therefore the standard way
// to verify that a method — least squares in particular, which need not be
// congruent — reproduces its input. The argument is not modified.
func Rewrap(unwrapped [][]float64) [][]float64 {
	return WrapPhaseMap(unwrapped)
}

// Congruence projects a real-valued phase estimate onto the surface that is
// closest to it while still being congruent to a wrapped map, i.e. it returns
//
//	result = wrapped + 2*pi * round((estimate - wrapped) / (2*pi))
//
// per pixel. This is Ghiglia and Pritt's congruence operation: least-squares and
// other minimum-norm solvers return a smooth surface whose wrapped values need
// not match the measured wrapped phase, and applying Congruence snaps the result
// so that Rewrap(result) == wrapped exactly while changing each value by less
// than pi. Both grids must have identical shape; otherwise it returns
// [ErrShapeMismatch]. Inputs are wrapped defensively where required and neither
// argument is modified.
func Congruence(estimate, wrapped [][]float64) ([][]float64, error) {
	er, ec, ok := gridDims(estimate)
	if !ok {
		return nil, ErrEmptyInput
	}
	wr, wc, ok := gridDims(wrapped)
	if !ok {
		return nil, ErrEmptyInput
	}
	if er != wr || ec != wc {
		return nil, ErrShapeMismatch
	}
	out := make([][]float64, er)
	for i := 0; i < er; i++ {
		out[i] = make([]float64, ec)
		for j := 0; j < ec; j++ {
			w := Wrap(wrapped[i][j])
			k := math.Round((estimate[i][j] - w) / twoPi)
			out[i][j] = w + twoPi*k
		}
	}
	return out, nil
}

// gridDims validates that g is a non-empty rectangular grid and returns its
// dimensions. ok is false for a nil, empty or ragged grid.
func gridDims(g [][]float64) (rows, cols int, ok bool) {
	rows = len(g)
	if rows == 0 {
		return 0, 0, false
	}
	cols = len(g[0])
	if cols == 0 {
		return 0, 0, false
	}
	for i := 1; i < rows; i++ {
		if len(g[i]) != cols {
			return 0, 0, false
		}
	}
	return rows, cols, true
}

// flatten copies a rectangular grid into a row-major slice, wrapping every value
// into (-pi, pi] so downstream code can assume a valid wrapped input.
func flatten(g [][]float64, rows, cols int) []float64 {
	f := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			f[i*cols+j] = Wrap(g[i][j])
		}
	}
	return f
}

// unflatten reshapes a row-major slice into a [row][col] grid sharing the
// backing array for each row (the slice must outlive the grid).
func unflatten(f []float64, rows, cols int) [][]float64 {
	g := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		g[i] = f[i*cols : (i+1)*cols : (i+1)*cols]
	}
	return g
}

// wrappedDiffs computes the wrapped forward differences of a flat phase map.
// dx[i*cols+j] holds Wrap(psi(i,j+1)-psi(i,j)) for j < cols-1 (0 elsewhere) and
// dy[i*cols+j] holds Wrap(psi(i+1,j)-psi(i,j)) for i < rows-1 (0 elsewhere).
// These are the discrete estimates of the true phase gradient; on a residue-free
// map they equal the true gradient exactly.
func wrappedDiffs(phase []float64, rows, cols int) (dx, dy []float64) {
	dx = make([]float64, rows*cols)
	dy = make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a := i*cols + j
			if j+1 < cols {
				dx[a] = Wrap(phase[a+1] - phase[a])
			}
			if i+1 < rows {
				dy[a] = Wrap(phase[a+cols] - phase[a])
			}
		}
	}
	return dx, dy
}

package phase_unwrapping

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Wrap maps an arbitrary phase value into the principal interval (-pi, pi]. It
// is the scalar wrapping operator that phase unwrapping inverts: for any real x,
// Wrap(x) == atan2(sin(x), cos(x)) up to floating-point rounding, and exactly
// -pi maps to +pi so the interval is half-open on the left.
func Wrap(x float64) float64 {
	r := math.Mod(x+math.Pi, twoPi)
	if r <= 0 {
		r += twoPi
	}
	return r - math.Pi
}

// WrapPhaseMap wraps every value of a continuous [row][col] surface into
// (-pi, pi], returning a new grid of the same shape. It is the forward model
// that produces a wrapped phase map from a known absolute surface and is handy
// for constructing inputs and tests. The input is not modified.
func WrapPhaseMap(continuous [][]float64) [][]float64 {
	out := make([][]float64, len(continuous))
	for i := range continuous {
		row := make([]float64, len(continuous[i]))
		for j := range continuous[i] {
			row[j] = Wrap(continuous[i][j])
		}
		out[i] = row
	}
	return out
}

// ReliabilityMap returns the inverse-reliability image of a wrapped phase map:
// for each pixel, the sum of squared second differences of the wrapped phase
// over its 3x3 neighbourhood (horizontal, vertical and both diagonals). Larger
// values indicate less reliable pixels, more likely to sit on a discontinuity.
// Border pixels are set to the maximum interior value.
//
// This exposes exactly the quality measure that [HistogramPhaseUnwrapping] uses
// to guide its path, but as a standalone helper on a [][]float64 grid. Input
// values are wrapped defensively first. It returns nil for an empty input.
func ReliabilityMap(wrapped [][]float64) [][]float64 {
	rows := len(wrapped)
	if rows == 0 || len(wrapped[0]) == 0 {
		return nil
	}
	cols := len(wrapped[0])
	flat := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols && j < len(wrapped[i]); j++ {
			flat[i*cols+j] = Wrap(wrapped[i][j])
		}
	}
	rel := computeInverseReliability(flat, rows, cols)
	out := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		out[i] = rel[i*cols : (i+1)*cols : (i+1)*cols]
	}
	return out
}

// ReliabilityMapMat is the FloatMat-valued counterpart of [ReliabilityMap]. It
// returns [ErrEmptyInput] for a nil or empty matrix.
func ReliabilityMapMat(wrapped *cv.FloatMat) (*cv.FloatMat, error) {
	if wrapped == nil || wrapped.Rows <= 0 || wrapped.Cols <= 0 || len(wrapped.Data) == 0 {
		return nil, ErrEmptyInput
	}
	rows, cols := wrapped.Rows, wrapped.Cols
	flat := make([]float64, rows*cols)
	for i := range flat {
		flat[i] = Wrap(wrapped.Data[i])
	}
	rel := computeInverseReliability(flat, rows, cols)
	res := cv.NewFloatMat(rows, cols)
	copy(res.Data, rel)
	return res, nil
}

// Residues detects phase residues using the classic Goldstein loop-integration
// test. For each 2x2 loop of pixels whose top-left corner is (i, j) with
// 0 <= i < rows-1 and 0 <= j < cols-1, it sums the wrapped phase differences
// around the loop; the sum is always an integer multiple of 2*pi, and the
// returned charge is that integer: +1 for a positive residue, -1 for a negative
// residue and 0 for a balanced (residue-free) loop.
//
// The returned grid has dimensions (rows-1) x (cols-1); entry [i][j] is the
// charge of the loop anchored at (i, j). A residue-free map (all zeros) is
// guaranteed to unwrap exactly along any spanning path; non-zero charges mark
// genuinely ambiguous regions (under-sampling, noise or true discontinuities)
// that no path-following method can resolve unambiguously. Input values are
// wrapped defensively first. It returns nil if the map is smaller than 2x2.
func Residues(wrapped [][]float64) [][]int {
	rows := len(wrapped)
	if rows < 2 || len(wrapped[0]) < 2 {
		return nil
	}
	cols := len(wrapped[0])
	out := make([][]int, rows-1)
	for i := 0; i < rows-1; i++ {
		out[i] = make([]int, cols-1)
		for j := 0; j < cols-1; j++ {
			// Sum wrapped gradients clockwise around the 2x2 loop:
			// (i,j) -> (i,j+1) -> (i+1,j+1) -> (i+1,j) -> (i,j).
			p00 := Wrap(wrapped[i][j])
			p01 := Wrap(wrapped[i][j+1])
			p11 := Wrap(wrapped[i+1][j+1])
			p10 := Wrap(wrapped[i+1][j])
			s := Wrap(p01-p00) + Wrap(p11-p01) + Wrap(p10-p11) + Wrap(p00-p10)
			out[i][j] = int(math.Round(s / twoPi))
		}
	}
	return out
}

// CountResidues returns the number of non-zero residues reported by [Residues]
// for a wrapped phase map, i.e. the count of loops around which the wrapped
// gradient does not integrate to zero. Zero means the map is residue-free and
// can be unwrapped exactly by [HistogramPhaseUnwrapping] up to a global 2*pi
// constant.
func CountResidues(wrapped [][]float64) int {
	res := Residues(wrapped)
	count := 0
	for i := range res {
		for j := range res[i] {
			if res[i][j] != 0 {
				count++
			}
		}
	}
	return count
}

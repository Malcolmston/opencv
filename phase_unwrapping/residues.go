package phase_unwrapping

import "math"

// Residue is a detected phase residue: the loop with top-left corner (Row, Col)
// carries the integer Charge (+1 or -1, occasionally larger in magnitude) around
// which the wrapped gradient fails to integrate to zero.
type Residue struct {
	Row, Col, Charge int
}

// ResidueList returns every non-zero residue of a wrapped phase map as an
// explicit list in row-major order, complementing the dense grid produced by
// [Residues] and the total from [CountResidues]. Residues occur in opposite-sign
// pairs (plus any that pair with the border) and mark the genuinely ambiguous
// locations that force branch-cut, least-squares or minimum-discontinuity methods
// to disagree. Input values are wrapped defensively first; the argument is not
// modified. It returns nil for a map smaller than 2x2.
func ResidueList(wrapped [][]float64) []Residue {
	res := Residues(wrapped)
	if res == nil {
		return nil
	}
	var out []Residue
	for i := range res {
		for j := range res[i] {
			if res[i][j] != 0 {
				out = append(out, Residue{Row: i, Col: j, Charge: res[i][j]})
			}
		}
	}
	return out
}

// TotalDiscontinuity counts the 2*pi discontinuities remaining in an unwrapped
// surface: the sum over all 4-connected edges of |round((u_b-u_a)/(2*pi))|. A
// perfectly unwrapped smooth surface has none (every neighbouring difference is
// below pi in magnitude); the count rises along branch cuts and wherever a method
// has been forced to leave a jump. It is the objective that
// [FlynnMinimumDiscontinuity] reduces and a convenient scalar quality measure for
// comparing results. It returns 0 for a grid smaller than 2 in either dimension.
func TotalDiscontinuity(unwrapped [][]float64) int {
	rows, cols, ok := gridDims(unwrapped)
	if !ok {
		return 0
	}
	total := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if j+1 < cols {
				total += absInt(int(math.Round((unwrapped[i][j+1] - unwrapped[i][j]) / twoPi)))
			}
			if i+1 < rows {
				total += absInt(int(math.Round((unwrapped[i+1][j] - unwrapped[i][j]) / twoPi)))
			}
		}
	}
	return total
}

// BranchCuts computes the Goldstein-style branch-cut mask of a wrapped phase map
// without unwrapping it, exposing exactly the barrier that
// [GoldsteinBranchCut] integrates around. The returned boolean grid has the same
// shape as wrapped and is true on the pixels that lie on a cut. Cuts join
// residues of opposite polarity (preferring the nearest partner within
// maxBoxRadius) and connect any residue left unbalanced to the nearest image
// border, so that no admissible integration path can encircle a lone residue. A
// non-positive maxBoxRadius selects a default of the image diagonal (no distance
// limit). Input values are wrapped defensively first; the argument is not
// modified. It returns nil for a map smaller than 2x2.
func BranchCuts(wrapped [][]float64, maxBoxRadius int) [][]bool {
	rows, cols, ok := gridDims(wrapped)
	if !ok || rows < 2 || cols < 2 {
		return nil
	}
	phase := flatten(wrapped, rows, cols)
	cuts := placeBranchCuts(phase, rows, cols, maxBoxRadius)
	out := make([][]bool, rows)
	for i := 0; i < rows; i++ {
		out[i] = cuts[i*cols : (i+1)*cols : (i+1)*cols]
	}
	return out
}

// absInt returns the absolute value of an int.
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

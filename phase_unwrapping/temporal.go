package phase_unwrapping

import "math"

// TemporalUnwrap performs temporal (multi-wavelength / multi-frequency) phase
// unwrapping. Given the same scene measured at several spatial frequencies whose
// relative scale factors are given by scales, it unwraps hierarchically from the
// coarsest map — assumed unambiguous, i.e. its absolute phase already lies within
// (-pi, pi] — to the finest. At each step the coarser unwrapped phase predicts
// the fringe order of the next map:
//
//	Phi_k = psi_k + 2*pi * round( (Phi_{k-1} * scales_k/scales_{k-1} - psi_k) / (2*pi) )
//
// so that a single noisy pixel is unwrapped independently of its neighbours —
// there is no spatial path and therefore no spatial error propagation, the
// defining advantage of temporal unwrapping.
//
// wrapped holds one wrapped map per frequency, all of identical shape and ordered
// coarsest first; scales holds the matching positive scale factors in strictly
// increasing order (scales[0] is typically 1). The returned map is the unwrapped
// absolute phase at the finest frequency (the last entry). Provided each
// prediction error stays below pi the recovery is exact. Input values are wrapped
// defensively first and no argument is modified. It returns [ErrEmptyInput] if
// wrapped or a map is empty, and [ErrShapeMismatch] if the maps disagree in shape
// or len(scales) != len(wrapped).
func TemporalUnwrap(wrapped [][][]float64, scales []float64) ([][]float64, error) {
	if len(wrapped) == 0 || len(scales) != len(wrapped) {
		return nil, ErrShapeMismatch
	}
	rows, cols, ok := gridDims(wrapped[0])
	if !ok {
		return nil, ErrEmptyInput
	}
	for _, m := range wrapped {
		r, c, mok := gridDims(m)
		if !mok {
			return nil, ErrEmptyInput
		}
		if r != rows || c != cols {
			return nil, ErrShapeMismatch
		}
	}
	for _, s := range scales {
		if !(s > 0) {
			return nil, ErrShapeMismatch
		}
	}

	// Start from the coarsest map, taken as already unwrapped.
	phi := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		phi[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			phi[i][j] = Wrap(wrapped[0][i][j])
		}
	}

	for k := 1; k < len(wrapped); k++ {
		ratio := scales[k] / scales[k-1]
		next := make([][]float64, rows)
		for i := 0; i < rows; i++ {
			next[i] = make([]float64, cols)
			for j := 0; j < cols; j++ {
				psi := Wrap(wrapped[k][i][j])
				predicted := phi[i][j] * ratio
				order := math.Round((predicted - psi) / twoPi)
				next[i][j] = psi + twoPi*order
			}
		}
		phi = next
	}
	return phi, nil
}

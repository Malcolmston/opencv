package phase_unwrapping

import "math"

// FlynnMinimumDiscontinuity unwraps a wrapped phase map by minimising the total
// weighted 2*pi discontinuity of the result, in the spirit of Flynn's
// minimum-discontinuity algorithm. The unwrapped surface is written as
// psi + 2*pi*k for an integer field k; the objective is the weighted count of
// edges whose neighbouring difference still exceeds pi,
//
//	Σ w^x |Δk^x + p^x| + Σ w^y |Δk^y + p^y|,
//
// where p is the fixed integer wrap of each wrapped difference. Starting from a
// path-integrated estimate (already optimal, with zero discontinuity, whenever
// the map is residue-free) the k-field is refined by iterated local 2*pi moves
// that each strictly reduce the objective, converging to a local minimum.
//
// The result is always congruent to the input (Rewrap(result) == wrapped). weights
// is an optional per-pixel reliability map (nil means uniform); when supplied it
// must match wrapped's shape and biases the remaining discontinuities toward the
// least reliable edges. maxIter caps the refinement sweeps (non-positive selects a
// default). On a residue-free map the true surface is recovered exactly up to a
// global constant. Input values are wrapped defensively first and no argument is
// modified. It returns [ErrEmptyInput] for an empty grid and [ErrShapeMismatch]
// if weights has the wrong shape.
func FlynnMinimumDiscontinuity(wrapped, weights [][]float64, maxIter int) ([][]float64, error) {
	rows, cols, ok := gridDims(wrapped)
	if !ok {
		return nil, ErrEmptyInput
	}
	if weights != nil {
		wr, wc, wok := gridDims(weights)
		if !wok || wr != rows || wc != cols {
			return nil, ErrShapeMismatch
		}
	}
	if maxIter <= 0 {
		maxIter = 100
	}
	phase := flatten(wrapped, rows, cols)

	// Edge weights (squared min of endpoint weights, or 1 when uniform).
	wx := make([]float64, rows*cols)
	wy := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a := i*cols + j
			if j+1 < cols {
				wx[a] = edgeWeight(weights, i, j, i, j+1)
			}
			if i+1 < rows {
				wy[a] = edgeWeight(weights, i, j, i+1, j)
			}
		}
	}

	// p^x, p^y: fixed integer part relating the raw wrapped difference to its
	// principal value: (psi_b - psi_a) = Δ + 2*pi*p.
	px := make([]int, rows*cols)
	py := make([]int, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a := i*cols + j
			if j+1 < cols {
				d := phase[a+1] - phase[a]
				px[a] = int(math.Round((d - Wrap(d)) / twoPi))
			}
			if i+1 < rows {
				d := phase[a+cols] - phase[a]
				py[a] = int(math.Round((d - Wrap(d)) / twoPi))
			}
		}
	}

	// Initialise k from a plain flood-fill integration (exact for residue-free
	// inputs, so no refinement is then needed).
	nocuts := make([]bool, rows*cols)
	u0 := floodFillIntegrate(phase, nocuts, rows, cols)
	k := make([]int, rows*cols)
	for a := range k {
		k[a] = int(math.Round((u0[a] - phase[a]) / twoPi))
	}

	// Horizontal discontinuity across edge a->a+1: k[a+1]-k[a]+px[a].
	jx := func(a int) int { return k[a+1] - k[a] + px[a] }
	jy := func(a int) int { return k[a+cols] - k[a] + py[a] }

	for iter := 0; iter < maxIter; iter++ {
		changed := false
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				a := i*cols + j
				// Apply at most one improving move per pixel per sweep so a move
				// and its inverse cannot oscillate within a single visit.
				for _, delta := range [2]int{1, -1} {
					if flynnGain(a, delta, i, j, rows, cols, wx, wy, jx, jy) < 0 {
						k[a] += delta
						changed = true
						break
					}
				}
			}
		}
		if !changed {
			break
		}
	}

	u := make([]float64, rows*cols)
	for a := range u {
		u[a] = phase[a] + twoPi*float64(k[a])
	}
	return unflatten(u, rows, cols), nil
}

// flynnGain returns the change in the weighted discontinuity objective caused by
// adding delta to k[a]; a negative value means the move reduces the objective.
func flynnGain(a, delta, i, j, rows, cols int, wx, wy []float64, jx, jy func(int) int) float64 {
	var g float64
	if j+1 < cols { // edge a -> a+1, changes by +delta
		cur := jx(a)
		g += wx[a] * float64(absInt(cur+delta)-absInt(cur))
	}
	if j > 0 { // edge (a-1) -> a, changes by -delta
		cur := jx(a - 1)
		g += wx[a-1] * float64(absInt(cur-delta)-absInt(cur))
	}
	if i+1 < rows { // edge a -> a+cols, changes by +delta
		cur := jy(a)
		g += wy[a] * float64(absInt(cur+delta)-absInt(cur))
	}
	if i > 0 { // edge (a-cols) -> a, changes by -delta
		cur := jy(a - cols)
		g += wy[a-cols] * float64(absInt(cur-delta)-absInt(cur))
	}
	return g
}

// edgeWeight returns the squared minimum of two endpoint weights, or 1 when
// weights is nil (uniform).
func edgeWeight(weights [][]float64, r0, c0, r1, c1 int) float64 {
	if weights == nil {
		return 1
	}
	m := math.Min(weights[r0][c0], weights[r1][c1])
	return m * m
}

package phase_unwrapping

import "math"

// PhaseDerivativeVariance returns the phase-derivative-variance quality map of a
// wrapped phase map (Ghiglia and Pritt). For each pixel it measures, over a 3x3
// window, the sum of the sample standard deviations of the wrapped x- and
// y-derivatives:
//
//	PDV = ( sqrt(Σ(Δx-mean Δx)²) + sqrt(Σ(Δy-mean Δy)²) ) / n
//
// Small values indicate a smooth, reliable neighbourhood; large values flag
// noise, discontinuities and residues. This is a cost map: LOWER is better, so
// negate it (or use [QualityGuidedUnwrap] which accepts a higher-is-better map)
// when a reliability map is wanted. The window is clamped at the borders. Input
// values are wrapped defensively first; the argument is not modified. It returns
// nil for an empty or ragged grid.
func PhaseDerivativeVariance(wrapped [][]float64) [][]float64 {
	rows, cols, ok := gridDims(wrapped)
	if !ok {
		return nil
	}
	phase := flatten(wrapped, rows, cols)
	// Per-pixel wrapped derivatives (forward difference, replicated at the far
	// edge so every pixel has a value).
	gx := make([]float64, rows*cols)
	gy := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a := i*cols + j
			if j+1 < cols {
				gx[a] = Wrap(phase[a+1] - phase[a])
			} else if j > 0 {
				gx[a] = Wrap(phase[a] - phase[a-1])
			}
			if i+1 < rows {
				gy[a] = Wrap(phase[a+cols] - phase[a])
			} else if i > 0 {
				gy[a] = Wrap(phase[a] - phase[a-cols])
			}
		}
	}
	out := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		out[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			sx, sxx, sy, syy, n := 0.0, 0.0, 0.0, 0.0, 0.0
			for di := -1; di <= 1; di++ {
				ii := i + di
				if ii < 0 || ii >= rows {
					continue
				}
				for dj := -1; dj <= 1; dj++ {
					jj := j + dj
					if jj < 0 || jj >= cols {
						continue
					}
					a := ii*cols + jj
					sx += gx[a]
					sxx += gx[a] * gx[a]
					sy += gy[a]
					syy += gy[a] * gy[a]
					n++
				}
			}
			varX := sxx/n - (sx/n)*(sx/n)
			varY := syy/n - (sy/n)*(sy/n)
			if varX < 0 {
				varX = 0
			}
			if varY < 0 {
				varY = 0
			}
			out[i][j] = (math.Sqrt(varX) + math.Sqrt(varY)) / n
		}
	}
	return out
}

// MaximumPhaseGradient returns the maximum-phase-gradient quality map: for each
// pixel, the largest absolute wrapped derivative (in x or y) found in its 3x3
// neighbourhood. Steep wrapped gradients coincide with noise and discontinuities,
// so this is a cost map where LOWER is better. The window is clamped at the
// borders. Input values are wrapped defensively first; the argument is not
// modified. It returns nil for an empty or ragged grid.
func MaximumPhaseGradient(wrapped [][]float64) [][]float64 {
	rows, cols, ok := gridDims(wrapped)
	if !ok {
		return nil
	}
	phase := flatten(wrapped, rows, cols)
	dx, dy := wrappedDiffs(phase, rows, cols)
	mag := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a := i*cols + j
			m := math.Abs(dx[a])
			if v := math.Abs(dy[a]); v > m {
				m = v
			}
			mag[a] = m
		}
	}
	out := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		out[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			best := 0.0
			for di := -1; di <= 1; di++ {
				ii := i + di
				if ii < 0 || ii >= rows {
					continue
				}
				for dj := -1; dj <= 1; dj++ {
					jj := j + dj
					if jj < 0 || jj >= cols {
						continue
					}
					if v := mag[ii*cols+jj]; v > best {
						best = v
					}
				}
			}
			out[i][j] = best
		}
	}
	return out
}

// PseudoCorrelation returns the pseudocorrelation quality map (Ghiglia and
// Pritt): for each pixel, over a 3x3 window,
//
//	gamma = sqrt( (Σ cos psi)² + (Σ sin psi)² ) / n
//
// which approaches 1 where the wrapped phase is locally coherent and drops toward
// 0 where it is noisy. Unlike the derivative-based maps this is a reliability map
// where HIGHER is better, so it can be fed directly to [QualityGuidedUnwrap] and
// [MaskedUnwrap]. The window is clamped at the borders. Input values are wrapped
// defensively first; the argument is not modified. It returns nil for an empty or
// ragged grid.
func PseudoCorrelation(wrapped [][]float64) [][]float64 {
	rows, cols, ok := gridDims(wrapped)
	if !ok {
		return nil
	}
	phase := flatten(wrapped, rows, cols)
	cosv := make([]float64, rows*cols)
	sinv := make([]float64, rows*cols)
	for i := range phase {
		cosv[i] = math.Cos(phase[i])
		sinv[i] = math.Sin(phase[i])
	}
	out := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		out[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			sc, ss, n := 0.0, 0.0, 0.0
			for di := -1; di <= 1; di++ {
				ii := i + di
				if ii < 0 || ii >= rows {
					continue
				}
				for dj := -1; dj <= 1; dj++ {
					jj := j + dj
					if jj < 0 || jj >= cols {
						continue
					}
					a := ii*cols + jj
					sc += cosv[a]
					ss += sinv[a]
					n++
				}
			}
			out[i][j] = math.Sqrt(sc*sc+ss*ss) / n
		}
	}
	return out
}

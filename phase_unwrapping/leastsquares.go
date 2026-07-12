package phase_unwrapping

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// LeastSquaresUnwrap performs unweighted least-squares (minimum-norm) phase
// unwrapping following Ghiglia and Pritt, "Two-Dimensional Phase Unwrapping:
// Theory, Algorithms, and Software". It finds the surface phi whose discrete
// gradient is closest, in the L2 sense, to the wrapped-phase gradient by solving
// the discrete Poisson equation
//
//	∇² phi = rho
//
// with Neumann (natural) boundary conditions, where rho is the divergence of the
// wrapped phase gradient. The solve is direct and exact via the two-dimensional
// discrete cosine transform, which diagonalises the Neumann Laplacian: the
// gradient field is transformed, divided by the Laplacian eigenvalues and
// transformed back.
//
// Unlike path-following methods the result is global and never traps error in a
// local region, but it is also not congruent to the input in general (its
// wrapped values need not match the measured phase); apply [Congruence] if a
// congruent result is required. For a residue-free map the least-squares surface
// equals the true surface exactly (up to a global 2*pi constant). Input values
// are wrapped defensively first and the argument is not modified. It returns
// [ErrEmptyInput] for an empty or ragged grid.
func LeastSquaresUnwrap(wrapped [][]float64) ([][]float64, error) {
	rows, cols, ok := gridDims(wrapped)
	if !ok {
		return nil, ErrEmptyInput
	}
	phase := flatten(wrapped, rows, cols)
	rhs := poissonRHS(phase, rows, cols)
	sol := solvePoissonNeumann(rhs, rows, cols)
	return unflatten(sol, rows, cols), nil
}

// LeastSquaresUnwrapMat is the [cv.FloatMat] counterpart of [LeastSquaresUnwrap].
func LeastSquaresUnwrapMat(wrapped *cv.FloatMat) (*cv.FloatMat, error) {
	if wrapped == nil || wrapped.Rows <= 0 || wrapped.Cols <= 0 || len(wrapped.Data) == 0 {
		return nil, ErrEmptyInput
	}
	rows, cols := wrapped.Rows, wrapped.Cols
	phase := make([]float64, rows*cols)
	for i := range phase {
		phase[i] = Wrap(wrapped.Data[i])
	}
	rhs := poissonRHS(phase, rows, cols)
	sol := solvePoissonNeumann(rhs, rows, cols)
	res := cv.NewFloatMat(rows, cols)
	copy(res.Data, sol)
	return res, nil
}

// WeightedLeastSquaresUnwrap performs weighted least-squares phase unwrapping
// (Ghiglia and Pritt) by minimising the weighted L2 gradient mismatch
//
//	Σ w^x (phi_{i,j+1}-phi_{i,j}-Δx)² + Σ w^y (phi_{i+1,j}-phi_{i,j}-Δy)²
//
// where the per-edge weights are the squared minimum of the two endpoint
// weights. Down-weighting unreliable pixels (or setting their weight to zero)
// keeps the error they carry from spreading across the map, which the unweighted
// solver cannot do. The resulting weighted-Laplacian normal equations are singular
// and are solved iteratively with preconditioned conjugate gradients using the
// unweighted DCT Poisson solver as preconditioner — the classic and rapidly
// converging combination.
//
// weights must either be nil (all weights 1, equivalent to [LeastSquaresUnwrap])
// or a grid matching wrapped's shape with non-negative entries; larger means more
// reliable. maxIter caps the PCG iterations (a non-positive value selects a
// generous default); tol is the relative residual stopping threshold (a
// non-positive value selects 1e-9). For a residue-free map the surface is
// recovered exactly up to a global constant regardless of the (positive) weights.
// It returns [ErrEmptyInput] for an empty grid and [ErrShapeMismatch] if weights
// has the wrong shape. Neither argument is modified.
func WeightedLeastSquaresUnwrap(wrapped, weights [][]float64, maxIter int, tol float64) ([][]float64, error) {
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
	phase := flatten(wrapped, rows, cols)
	w := make([]float64, rows*cols)
	if weights == nil {
		for i := range w {
			w[i] = 1
		}
	} else {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				v := weights[i][j]
				if v < 0 {
					v = 0
				}
				w[i*cols+j] = v
			}
		}
	}
	if maxIter <= 0 {
		maxIter = 2*(rows+cols) + 50
	}
	if !(tol > 0) {
		tol = 1e-9
	}
	sol := weightedLeastSquares(phase, w, rows, cols, maxIter, tol)
	return unflatten(sol, rows, cols), nil
}

// poissonRHS builds the right-hand side b of the Neumann-boundary normal
// equations (-Laplacian) phi = b for the unweighted problem: the divergence of
// the wrapped gradient field, b_{p,q} = (Δx_{p,q-1}-Δx_{p,q}) + (Δy_{p-1,q}-Δy_{p,q}),
// with out-of-range gradient terms taken as zero (the natural boundary).
func poissonRHS(phase []float64, rows, cols int) []float64 {
	dx, dy := wrappedDiffs(phase, rows, cols)
	b := make([]float64, rows*cols)
	for p := 0; p < rows; p++ {
		for q := 0; q < cols; q++ {
			a := p*cols + q
			var s float64
			if q > 0 {
				s += dx[a-1]
			}
			if q < cols-1 {
				s -= dx[a]
			}
			if p > 0 {
				s += dy[a-cols]
			}
			if p < rows-1 {
				s -= dy[a]
			}
			b[a] = s
		}
	}
	return b
}

// solvePoissonNeumann solves (-Laplacian) x = b with Neumann boundary conditions
// exactly via the 2-D DCT. The Neumann Laplacian is diagonal in the DCT-II basis
// with eigenvalues 4 - 2cos(pi*i/rows) - 2cos(pi*j/cols); the zero eigenvalue at
// mode (0,0) corresponds to the free global constant and is set to zero.
func solvePoissonNeumann(b []float64, rows, cols int) []float64 {
	bt := dct2(b, rows, cols, true)
	for i := 0; i < rows; i++ {
		ei := 2 * math.Cos(math.Pi*float64(i)/float64(rows))
		for j := 0; j < cols; j++ {
			if i == 0 && j == 0 {
				bt[0] = 0
				continue
			}
			denom := 4 - ei - 2*math.Cos(math.Pi*float64(j)/float64(cols))
			bt[i*cols+j] /= denom
		}
	}
	return dct2(bt, rows, cols, false)
}

// weightedLeastSquares solves the weighted normal equations Q phi = c with
// preconditioned conjugate gradients, using solvePoissonNeumann as the
// preconditioner (the exact inverse of the unweighted operator).
func weightedLeastSquares(phase, weight []float64, rows, cols, maxIter int, tol float64) []float64 {
	n := rows * cols
	dx, dy := wrappedDiffs(phase, rows, cols)
	wx := make([]float64, n)
	wy := make([]float64, n)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a := i*cols + j
			if j+1 < cols {
				m := math.Min(weight[a], weight[a+1])
				wx[a] = m * m
			}
			if i+1 < rows {
				m := math.Min(weight[a], weight[a+cols])
				wy[a] = m * m
			}
		}
	}

	// c_{p,q} = wx_{p,q-1} Δx_{p,q-1} - wx_{p,q} Δx_{p,q}
	//         + wy_{p-1,q} Δy_{p-1,q} - wy_{p,q} Δy_{p,q}
	c := make([]float64, n)
	for p := 0; p < rows; p++ {
		for q := 0; q < cols; q++ {
			a := p*cols + q
			var s float64
			if q > 0 {
				s += wx[a-1] * dx[a-1]
			}
			if q < cols-1 {
				s -= wx[a] * dx[a]
			}
			if p > 0 {
				s += wy[a-cols] * dy[a-cols]
			}
			if p < rows-1 {
				s -= wy[a] * dy[a]
			}
			c[a] = s
		}
	}

	applyQ := func(phi []float64) []float64 {
		out := make([]float64, n)
		for p := 0; p < rows; p++ {
			for q := 0; q < cols; q++ {
				a := p*cols + q
				center := phi[a]
				var s float64
				if q > 0 {
					s += wx[a-1] * (center - phi[a-1])
				}
				if q < cols-1 {
					s += wx[a] * (center - phi[a+1])
				}
				if p > 0 {
					s += wy[a-cols] * (center - phi[a-cols])
				}
				if p < rows-1 {
					s += wy[a] * (center - phi[a+cols])
				}
				out[a] = s
			}
		}
		return out
	}

	phi := make([]float64, n)
	r := make([]float64, n)
	copy(r, c)
	cnorm := math.Sqrt(dotf(c, c))
	if cnorm == 0 {
		return phi
	}
	pvec := make([]float64, n)
	var rhoOld float64
	for k := 0; k < maxIter; k++ {
		z := solvePoissonNeumann(r, rows, cols)
		rhoNew := dotf(r, z)
		if k == 0 {
			copy(pvec, z)
		} else {
			beta := rhoNew / rhoOld
			for i := range pvec {
				pvec[i] = z[i] + beta*pvec[i]
			}
		}
		qp := applyQ(pvec)
		denom := dotf(pvec, qp)
		if denom == 0 {
			break
		}
		alpha := rhoNew / denom
		for i := range phi {
			phi[i] += alpha * pvec[i]
			r[i] -= alpha * qp[i]
		}
		rhoOld = rhoNew
		if math.Sqrt(dotf(r, r)) <= tol*cnorm {
			break
		}
	}
	return phi
}

// dotf returns the Euclidean inner product of two equal-length slices.
func dotf(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// dct1DII computes the unnormalised type-II DCT of x:
// X_k = Σ_n x_n cos(pi*(n+0.5)*k/N).
func dct1DII(x []float64) []float64 {
	n := len(x)
	out := make([]float64, n)
	for k := 0; k < n; k++ {
		var s float64
		f := math.Pi * float64(k) / float64(n)
		for m := 0; m < n; m++ {
			s += x[m] * math.Cos(f*(float64(m)+0.5))
		}
		out[k] = s
	}
	return out
}

// idct1DIII inverts dct1DII: the normalised type-III DCT
// x_m = (1/N)[X_0 + 2 Σ_{k>=1} X_k cos(pi*k*(m+0.5)/N)].
func idct1DIII(x []float64) []float64 {
	n := len(x)
	out := make([]float64, n)
	inv := 1.0 / float64(n)
	for m := 0; m < n; m++ {
		s := x[0] * 0.5
		fm := math.Pi * (float64(m) + 0.5) / float64(n)
		for k := 1; k < n; k++ {
			s += x[k] * math.Cos(fm*float64(k))
		}
		out[m] = 2 * s * inv
	}
	return out
}

// dct2 applies a separable 2-D DCT to a row-major grid: the forward (type-II) or
// inverse (type-III) 1-D transform along every row and then every column. The
// forward/inverse pair round-trips to the identity.
func dct2(a []float64, rows, cols int, forward bool) []float64 {
	tf := dct1DII
	if !forward {
		tf = idct1DIII
	}
	tmp := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		copy(tmp[i*cols:(i+1)*cols], tf(a[i*cols:(i+1)*cols]))
	}
	out := make([]float64, rows*cols)
	col := make([]float64, rows)
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			col[i] = tmp[i*cols+j]
		}
		tc := tf(col)
		for i := 0; i < rows; i++ {
			out[i*cols+j] = tc[i]
		}
	}
	return out
}

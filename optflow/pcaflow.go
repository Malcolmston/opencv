package optflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PCAFlowParams configures [CalcOpticalFlowPCAFlow]. The defaults from
// [DefaultPCAFlowParams] fit a smooth global motion model to sparse feature
// matches.
type PCAFlowParams struct {
	// BasisOrder is the maximum cosine order per axis of the low-dimensional
	// flow subspace (the "PCA prior"). The subspace has (BasisOrder+1)² basis
	// fields per flow component. Order 3–5 captures translation, rotation and
	// smooth divergence/shear.
	BasisOrder int
	// GridStep is the spacing (in pixels) of the sparse seed grid whose
	// Lucas-Kanade matches drive the fit.
	GridStep int
	// Ridge is the Tikhonov regularisation added to the normal equations,
	// stabilising the least-squares fit against noisy or degenerate matches.
	Ridge float64
	// WinRadius is the half-size of the Lucas-Kanade window used to establish
	// each sparse match.
	WinRadius int
}

// DefaultPCAFlowParams returns a sensible PCAFlow configuration
// (BasisOrder=4, GridStep=6, Ridge=1e-3, WinRadius=6).
func DefaultPCAFlowParams() PCAFlowParams {
	return PCAFlowParams{
		BasisOrder: 4,
		GridStep:   6,
		Ridge:      1e-3,
		WinRadius:  6,
	}
}

func (p PCAFlowParams) validate() {
	if p.BasisOrder < 1 {
		panic("optflow: PCAFlow requires BasisOrder >= 1")
	}
	if p.GridStep < 1 {
		panic("optflow: PCAFlow requires GridStep >= 1")
	}
	if p.WinRadius < 1 {
		panic("optflow: PCAFlow requires WinRadius >= 1")
	}
	if p.Ridge < 0 {
		panic("optflow: PCAFlow requires Ridge >= 0")
	}
}

// CalcOpticalFlowPCAFlow computes a dense flow field from prev to next with a
// PCAFlow-style estimator in the spirit of Wulff & Black, "Efficient Sparse-to-
// Dense Optical Flow Estimation using a Learned Basis and Layers" (2015).
//
// Rather than learning a basis from training data (which a stdlib port cannot
// ship), the flow is constrained to a fixed, low-dimensional linear subspace
// spanned by separable two-dimensional cosine fields up to order BasisOrder —
// the smooth global-motion prior at the heart of PCAFlow. The pipeline is: (1)
// track a regular grid of seed points with Lucas-Kanade to obtain sparse
// correspondences; (2) solve a ridge-regularised least-squares problem for the
// subspace coefficients that best explain those matches (independently for the
// horizontal and vertical components); (3) reconstruct the dense field by
// evaluating the basis at every pixel. Because the solution lives in a handful
// of dimensions it is inherently smooth and robust to per-seed match noise, and
// it represents translation, rotation and smooth deformation exactly.
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. p is validated (see [PCAFlowParams]). The result
// is deterministic.
func CalcOpticalFlowPCAFlow(prev, next *cv.Mat, p PCAFlowParams) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowPCAFlow")
	p.validate()

	pg := grayGrid(prev)
	ng := grayGrid(next)
	rows, cols := pg.Rows, pg.Cols
	gx, gy := sobelGradients(pg)

	// (1) Sparse Lucas-Kanade matches on a regular grid.
	seeds := gridSeeds(rows, cols, p.GridStep)
	type sample struct {
		x, y int
		u, v float64
	}
	samples := make([]sample, 0, len(seeds))
	for _, pt := range seeds {
		u, v, ok := lucasKanadePoint(pg, ng, gx, gy, pt.X, pt.Y, p.WinRadius, 10)
		if ok {
			samples = append(samples, sample{pt.X, pt.Y, u, v})
		}
	}

	k := (p.BasisOrder + 1) * (p.BasisOrder + 1)
	flow := NewFlowField(rows, cols)
	if len(samples) < k {
		// Too few reliable matches for a stable fit: fall back to the mean
		// translation of whatever matches exist (order-0 model).
		var su, sv float64
		for _, s := range samples {
			su += s.u
			sv += s.v
		}
		if len(samples) > 0 {
			su /= float64(len(samples))
			sv /= float64(len(samples))
		}
		for i := 0; i < rows*cols; i++ {
			flow.Data[i*2] = su
			flow.Data[i*2+1] = sv
		}
		return flow
	}

	// (2) Build and solve the ridge-regularised normal equations A c = b for
	// each flow component. Phi is the k-dim basis evaluated at a sample.
	ata := make([][]float64, k)
	for i := range ata {
		ata[i] = make([]float64, k)
	}
	btu := make([]float64, k)
	btv := make([]float64, k)
	phi := make([]float64, k)
	for _, s := range samples {
		pcaBasis(float64(s.x), float64(s.y), rows, cols, p.BasisOrder, phi)
		for a := 0; a < k; a++ {
			pa := phi[a]
			btu[a] += pa * s.u
			btv[a] += pa * s.v
			row := ata[a]
			for b := 0; b < k; b++ {
				row[b] += pa * phi[b]
			}
		}
	}
	for a := 0; a < k; a++ {
		ata[a][a] += p.Ridge
	}
	cu := solveLinearSystem(cloneMatrix(ata), append([]float64(nil), btu...))
	cvv := solveLinearSystem(ata, btv)

	// (3) Reconstruct the dense field from the fitted coefficients.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			pcaBasis(float64(x), float64(y), rows, cols, p.BasisOrder, phi)
			var uu, vv float64
			for a := 0; a < k; a++ {
				uu += cu[a] * phi[a]
				vv += cvv[a] * phi[a]
			}
			flow.set(y, x, uu, vv)
		}
	}
	return flow
}

// pcaBasis writes the separable 2-D cosine basis of the given order, evaluated
// at (x, y) on a rows×cols grid, into out (length (order+1)²). Basis element
// (i, j) is cos(iπ(x+0.5)/cols)·cos(jπ(y+0.5)/rows); the (0,0) term is the
// constant field, so the order-0 subspace is pure translation.
func pcaBasis(x, y float64, rows, cols, order int, out []float64) {
	cx := make([]float64, order+1)
	cy := make([]float64, order+1)
	for i := 0; i <= order; i++ {
		cx[i] = math.Cos(float64(i) * math.Pi * (x + 0.5) / float64(cols))
		cy[i] = math.Cos(float64(i) * math.Pi * (y + 0.5) / float64(rows))
	}
	idx := 0
	for j := 0; j <= order; j++ {
		for i := 0; i <= order; i++ {
			out[idx] = cx[i] * cy[j]
			idx++
		}
	}
}

// cloneMatrix deep-copies a square coefficient matrix so the two component
// solves do not corrupt each other's system.
func cloneMatrix(a [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = append([]float64(nil), a[i]...)
	}
	return out
}

// solveLinearSystem solves a·x = b for a square, symmetric-positive-definite
// system by Gaussian elimination with partial pivoting. It mutates a and b and
// returns x. A singular system yields a best-effort solution with zeros for the
// unconstrained variables.
func solveLinearSystem(a [][]float64, b []float64) []float64 {
	n := len(b)
	for col := 0; col < n; col++ {
		// Partial pivot.
		piv := col
		maxv := math.Abs(a[col][col])
		for r := col + 1; r < n; r++ {
			if av := math.Abs(a[r][col]); av > maxv {
				maxv = av
				piv = r
			}
		}
		if maxv < 1e-12 {
			continue // singular column; leave x[col] = 0
		}
		if piv != col {
			a[col], a[piv] = a[piv], a[col]
			b[col], b[piv] = b[piv], b[col]
		}
		inv := 1.0 / a[col][col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a[r][col] * inv
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				a[r][c] -= f * a[col][c]
			}
			b[r] -= f * b[col]
		}
	}
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		if math.Abs(a[i][i]) > 1e-12 {
			x[i] = b[i] / a[i][i]
		}
	}
	return x
}

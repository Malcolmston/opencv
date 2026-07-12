package optflow

import (
	cv "github.com/malcolmston/opencv"
)

// CalcOpticalFlowDenseHS computes a dense optical-flow field from prev to next
// using the classic Horn-Schunck variational method.
//
// Horn-Schunck couples the local brightness-constancy constraint
//
//	Ix·u + Iy·v + It = 0
//
// with a global first-order smoothness prior, and minimises their weighted sum
// over the whole image. The regularisation weight alpha controls the trade-off:
// larger alpha yields a smoother, more strongly coupled field (better for
// weakly textured regions), while smaller alpha follows the raw data term more
// closely. The Euler-Lagrange equations are solved by the standard Jacobi
// iteration
//
//	u = ū − Ix·(Ix·ū + Iy·v̄ + It) / (alpha² + Ix² + Iy²)
//	v = v̄ − Iy·(Ix·ū + Iy·v̄ + It) / (alpha² + Ix² + Iy²)
//
// where ū and v̄ are the local averages of the current flow under the classic
// Horn-Schunck weighted-Laplacian neighbourhood. iters gives the number of
// Jacobi sweeps.
//
// The spatial gradients Ix, Iy are the average of the Sobel derivatives of the
// two frames and It is their intensity difference (next − prev). Because the
// data term is linearised around zero displacement, the method assumes small
// motions (roughly a pixel per iteration is well handled; very large
// displacements are outside its regime — prefer [CalcOpticalFlowDIS] there).
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. alpha must be > 0 and iters >= 1. The computation
// is fully deterministic.
func CalcOpticalFlowDenseHS(prev, next *cv.Mat, alpha float64, iters int) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowDenseHS")
	if alpha <= 0 {
		panic("optflow: CalcOpticalFlowDenseHS requires alpha > 0")
	}
	if iters < 1 {
		panic("optflow: CalcOpticalFlowDenseHS requires iters >= 1")
	}
	pg := grayGrid(prev)
	ng := grayGrid(next)
	rows, cols := pg.Rows, pg.Cols

	// Spatial gradients: average the Sobel derivatives of both frames, which is
	// better centred in time than using a single frame. Temporal gradient is the
	// plain intensity difference.
	pgx, pgy := sobelGradients(pg)
	ngx, ngy := sobelGradients(ng)
	ix := make([]float64, rows*cols)
	iy := make([]float64, rows*cols)
	it := make([]float64, rows*cols)
	for i := range ix {
		ix[i] = 0.5 * (pgx.Data[i] + ngx.Data[i])
		iy[i] = 0.5 * (pgy.Data[i] + ngy.Data[i])
		it[i] = ng.Data[i] - pg.Data[i]
	}

	a2 := alpha * alpha
	u := make([]float64, rows*cols)
	v := make([]float64, rows*cols)
	un := make([]float64, rows*cols)
	vn := make([]float64, rows*cols)

	// avg returns the Horn-Schunck weighted-Laplacian average of field f at
	// (x, y): 1/6 for the 4-neighbours and 1/12 for the diagonals, with border
	// replication.
	avg := func(f []float64, x, y int) float64 {
		clampX := func(x int) int { return clampInt(x, 0, cols-1) }
		clampY := func(y int) int { return clampInt(y, 0, rows-1) }
		idx := func(xx, yy int) int { return clampY(yy)*cols + clampX(xx) }
		card := f[idx(x-1, y)] + f[idx(x+1, y)] + f[idx(x, y-1)] + f[idx(x, y+1)]
		diag := f[idx(x-1, y-1)] + f[idx(x+1, y-1)] + f[idx(x-1, y+1)] + f[idx(x+1, y+1)]
		return card/6.0 + diag/12.0
	}

	for iter := 0; iter < iters; iter++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				i := y*cols + x
				ub := avg(u, x, y)
				vb := avg(v, x, y)
				num := ix[i]*ub + iy[i]*vb + it[i]
				den := a2 + ix[i]*ix[i] + iy[i]*iy[i]
				t := num / den
				un[i] = ub - ix[i]*t
				vn[i] = vb - iy[i]*t
			}
		}
		u, un = un, u
		v, vn = vn, v
	}

	flow := NewFlowField(rows, cols)
	for i := 0; i < rows*cols; i++ {
		flow.Data[i*2] = u[i]
		flow.Data[i*2+1] = v[i]
	}
	return flow
}

package tracking

import (
	cv "github.com/malcolmston/opencv"
)

// CalcOpticalFlowHornSchunck computes a dense optical-flow field from prev to
// next using the classic Horn-Schunck variational method.
//
// Horn-Schunck couples the local brightness-constancy constraint
//
//	Ix·u + Iy·v + It = 0
//
// with a global first-order smoothness prior and minimises their weighted sum
// over the whole image. The regularisation weight alpha trades the two off:
// larger alpha yields a smoother, more strongly coupled field (better for weakly
// textured regions) while smaller alpha follows the raw data term more closely.
// The Euler-Lagrange equations are solved by the standard Gauss-Seidel/Jacobi
// iteration
//
//	u = ū − Ix·(Ix·ū + Iy·v̄ + It) / (alpha² + Ix² + Iy²)
//	v = v̄ − Iy·(Ix·ū + Iy·v̄ + It) / (alpha² + Ix² + Iy²)
//
// where ū and v̄ are the 4-neighbour averages of the current flow. iters is the
// number of Jacobi sweeps.
//
// The spatial gradients Ix, Iy are the average of the derivatives of the two
// frames and It is their intensity difference (next − prev). Because the data
// term is linearised around zero displacement, the method assumes small motions.
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. alpha must be > 0 and iters >= 1. The computation
// is fully deterministic.
func CalcOpticalFlowHornSchunck(prev, next *cv.Mat, alpha float64, iters int) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowHornSchunck")
	if alpha <= 0 {
		panic("tracking: CalcOpticalFlowHornSchunck requires alpha > 0")
	}
	if iters < 1 {
		panic("tracking: CalcOpticalFlowHornSchunck requires iters >= 1")
	}
	gp := trackingToGrayF(prev)
	gn := trackingToGrayF(next)
	rows, cols := gp.rows, gp.cols

	ixP, iyP := gp.sobel()
	ixN, iyN := gn.sobel()
	ix := make([]float64, rows*cols)
	iy := make([]float64, rows*cols)
	it := make([]float64, rows*cols)
	for i := range ix {
		ix[i] = (ixP.data[i] + ixN.data[i]) / 2
		iy[i] = (iyP.data[i] + iyN.data[i]) / 2
		it[i] = gn.data[i] - gp.data[i]
	}

	ff := NewFlowField(rows, cols)
	u := ff.u
	v := ff.v
	a2 := alpha * alpha

	avg := func(buf []float64, y, x int) float64 {
		clampY := func(k int) int {
			if k < 0 {
				return 0
			}
			if k >= rows {
				return rows - 1
			}
			return k
		}
		clampX := func(k int) int {
			if k < 0 {
				return 0
			}
			if k >= cols {
				return cols - 1
			}
			return k
		}
		up := buf[clampY(y-1)*cols+x]
		dn := buf[clampY(y+1)*cols+x]
		lf := buf[y*cols+clampX(x-1)]
		rt := buf[y*cols+clampX(x+1)]
		return (up + dn + lf + rt) / 4
	}

	for iter := 0; iter < iters; iter++ {
		newU := make([]float64, rows*cols)
		newV := make([]float64, rows*cols)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				i := y*cols + x
				ubar := avg(u, y, x)
				vbar := avg(v, y, x)
				denom := a2 + ix[i]*ix[i] + iy[i]*iy[i]
				common := (ix[i]*ubar + iy[i]*vbar + it[i]) / denom
				newU[i] = ubar - ix[i]*common
				newV[i] = vbar - iy[i]*common
			}
		}
		copy(u, newU)
		copy(v, newV)
	}
	return ff
}

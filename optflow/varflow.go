package optflow

import "math"

// This file collects small numerical helpers shared by the variational and
// duality-based dense flow methods added to the package (TV-L1, DeepFlow-lite).
// They operate on flat row-major float64 planes of size rows*cols, matching the
// storage used by FlowField's two channels.

// warpGrid warps src by the displacement field (u, v), returning a new grid in
// which out(x, y) = src(x + u, y + v) sampled bilinearly with border
// replication. This is the motion-compensated resampling used to linearise the
// brightness-constancy term around the current flow estimate.
func warpGrid(src *grid, u, v []float64) *grid {
	rows, cols := src.Rows, src.Cols
	out := newGrid(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			out.Data[i] = src.bilinear(float64(x)+u[i], float64(y)+v[i])
		}
	}
	return out
}

// forwardGradient computes the forward-difference gradient of a scalar plane:
// gx(x, y) = f(x+1, y) − f(x, y) and gy(x, y) = f(x, y+1) − f(x, y), with a zero
// derivative on the trailing edge. It is the discrete gradient adjoint to
// divergence below, as used in the TV-L1 dual update.
func forwardGradient(f []float64, rows, cols int) (gx, gy []float64) {
	gx = make([]float64, rows*cols)
	gy = make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if x < cols-1 {
				gx[i] = f[i+1] - f[i]
			}
			if y < rows-1 {
				gy[i] = f[i+cols] - f[i]
			}
		}
	}
	return gx, gy
}

// divergence computes the divergence of the vector field (v1, v2), the negative
// adjoint of forwardGradient, using backward differences with the standard
// TV-L1 boundary handling (Chambolle / Zach et al.).
func divergence(v1, v2 []float64, rows, cols int) []float64 {
	div := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			var d1, d2 float64
			if x > 0 {
				d1 = v1[i] - v1[i-1]
			} else {
				d1 = v1[i]
			}
			if y > 0 {
				d2 = v2[i] - v2[i-cols]
			} else {
				d2 = v2[i]
			}
			div[i] = d1 + d2
		}
	}
	return div
}

// upscaleFlow resamples a coarse flow (cu, cv) of size oldRows×oldCols to a fine
// grid of size newRows×newCols, scaling the displacement magnitudes by the size
// ratio so they are expressed in the fine level's pixel units. It mirrors the
// coarse-to-fine propagation used throughout the pyramidal solvers.
func upscaleFlow(cu, cv []float64, oldRows, oldCols, newRows, newCols int) (u, v []float64) {
	u = make([]float64, newRows*newCols)
	v = make([]float64, newRows*newCols)
	sx := float64(newCols) / float64(oldCols)
	sy := float64(newRows) / float64(oldRows)
	for y := 0; y < newRows; y++ {
		fy := (float64(y)+0.5)/sy - 0.5
		for x := 0; x < newCols; x++ {
			fx := (float64(x)+0.5)/sx - 0.5
			i := y*newCols + x
			u[i] = bilerpPlane(cu, oldRows, oldCols, fx, fy) * sx
			v[i] = bilerpPlane(cv, oldRows, oldCols, fx, fy) * sy
		}
	}
	return u, v
}

// solve2x2 solves the symmetric 2×2 system
//
//	[a11 a12][x]   [b1]
//	[a12 a22][y] = [b2]
//
// returning (0, 0) when the matrix is numerically singular.
func solve2x2(a11, a12, a22, b1, b2 float64) (x, y float64) {
	det := a11*a22 - a12*a12
	if math.Abs(det) < 1e-12 {
		return 0, 0
	}
	inv := 1.0 / det
	x = (a22*b1 - a12*b2) * inv
	y = (a11*b2 - a12*b1) * inv
	return x, y
}

// flowFromPlanes packs two flow planes into a FlowField.
func flowFromPlanes(u, v []float64, rows, cols int) *FlowField {
	f := NewFlowField(rows, cols)
	for i := 0; i < rows*cols; i++ {
		f.Data[i*2] = u[i]
		f.Data[i*2+1] = v[i]
	}
	return f
}

// scalePyramidLevels returns the two frame pyramids trimmed to a common depth,
// coarsest last. It is the shared front end for the pyramidal dense methods.
func scalePyramidLevels(prev, next *grid, levels, minSize int) (pPyr, nPyr []*grid) {
	pPyr = buildPyramid(prev, levels, minSize)
	nPyr = buildPyramid(next, levels, minSize)
	if len(nPyr) < len(pPyr) {
		pPyr = pPyr[:len(nPyr)]
	} else if len(pPyr) < len(nPyr) {
		nPyr = nPyr[:len(pPyr)]
	}
	return pPyr, nPyr
}

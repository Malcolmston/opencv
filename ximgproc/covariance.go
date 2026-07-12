package ximgproc

import cv "github.com/malcolmston/opencv"

// CovarianceEstimation estimates the covariance matrix of the local windows of
// img using the sliding-window formulation of OpenCV's covarianceEstimation. It
// returns a symmetric N×N matrix (N = windowRows·windowCols) as a row-major
// slice of rows.
//
// Every (windowRows×windowCols) window that fits inside the grayscale image is
// flattened, in row-major order, into an N-dimensional observation vector. The
// function subtracts the mean observation and accumulates the outer products,
//
//	C = (1/M) · Σ_m (v_m − μ)·(v_m − μ)ᵀ,
//
// over all M = (rows−windowRows+1)·(cols−windowCols+1) windows, yielding the
// covariance between the N sample positions of the window. Entry C[i][j] is the
// covariance between window offset i and window offset j (both in [0,N)); the
// diagonal holds the per-position variances.
//
// img may be 1- or 3-channel; colour is reduced to luma. It panics if the window
// is non-positive or larger than the image. The estimate is deterministic and
// the returned matrix is symmetric and positive-semidefinite.
func CovarianceEstimation(img *cv.Mat, windowRows, windowCols int) [][]float64 {
	if windowRows <= 0 || windowCols <= 0 {
		panic("ximgproc: CovarianceEstimation requires a positive window")
	}
	rows, cols := img.Rows, img.Cols
	if windowRows > rows || windowCols > cols {
		panic("ximgproc: CovarianceEstimation window larger than image")
	}
	g := channelPlane(toGray(img), 0)
	nDim := windowRows * windowCols
	oy := rows - windowRows + 1
	ox := cols - windowCols + 1
	m := oy * ox

	// Collect observation vectors and their mean.
	obs := make([][]float64, 0, m)
	mean := make([]float64, nDim)
	for wy := 0; wy < oy; wy++ {
		for wx := 0; wx < ox; wx++ {
			v := make([]float64, nDim)
			k := 0
			for dy := 0; dy < windowRows; dy++ {
				row := (wy + dy) * cols
				for dx := 0; dx < windowCols; dx++ {
					val := g[row+wx+dx]
					v[k] = val
					mean[k] += val
					k++
				}
			}
			obs = append(obs, v)
		}
	}
	invM := 1.0 / float64(m)
	for i := range mean {
		mean[i] *= invM
	}

	// Accumulate outer products of the centred observations.
	cov := make([][]float64, nDim)
	for i := range cov {
		cov[i] = make([]float64, nDim)
	}
	for _, v := range obs {
		for i := 0; i < nDim; i++ {
			di := v[i] - mean[i]
			ci := cov[i]
			for j := i; j < nDim; j++ {
				ci[j] += di * (v[j] - mean[j])
			}
		}
	}
	for i := 0; i < nDim; i++ {
		for j := i; j < nDim; j++ {
			cov[i][j] *= invM
			cov[j][i] = cov[i][j]
		}
	}
	return cov
}

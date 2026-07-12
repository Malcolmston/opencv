package ximgproc

import cv "github.com/malcolmston/opencv"

// GuidedFilter applies the edge-preserving guided filter of He, Sun and Tang
// ("Guided Image Filtering", 2013) to src using guide as the guidance image,
// and returns a new Mat with the same shape as src.
//
// The filter models the output q as a locally linear transform of the guidance
// I within each square window ω of radius r:
//
//	q_i = a_k·I_i + b_k   for every pixel i in window ω_k,
//
// where the per-window coefficients minimise the reconstruction error against
// the input p (src) with an L2 penalty eps on a:
//
//	a_k = cov(I,p) / (var(I) + eps),   b_k = mean(p) − a_k·mean(I).
//
// Overlapping windows are averaged, giving the closed form
//
//	q_i = mean(a)_i·I_i + mean(b)_i.
//
// Because a_k → 0 in high-variance (edge) neighbourhoods and a_k → 1 in flat
// ones, the filter smooths noise inside uniform regions while preserving strong
// edges, without the gradient reversal that can afflict the bilateral filter.
//
// radius is the window radius in pixels (window side = 2·radius+1); it must be
// positive. eps is the regularisation term, in squared-intensity units on the
// native [0,255] scale — larger eps smooths more. A common starting point is
// eps = (0.1·255)² ≈ 650 for gentle smoothing.
//
// The guidance is treated as a single scalar channel: a 3-channel guide is
// reduced to luma via [cv.CvtColor]. When src has multiple channels each is
// filtered independently against that scalar guidance. guide and src must have
// the same width and height. It panics on a size mismatch or non-positive
// radius. The matrix-form colour-guidance model is deferred (see the package
// documentation).
func GuidedFilter(src, guide *cv.Mat, radius int, eps float64) *cv.Mat {
	if radius <= 0 {
		panic("ximgproc: GuidedFilter requires a positive radius")
	}
	if src.Rows != guide.Rows || src.Cols != guide.Cols {
		panic("ximgproc: GuidedFilter src and guide must share dimensions")
	}
	rows, cols := src.Rows, src.Cols

	// Scalar guidance I and its window statistics (independent of src).
	I := toGrayFloat(guide)
	meanI := boxMean(I, rows, cols, radius)
	corrI := boxMean(mul(I, I), rows, cols, radius)
	varI := make([]float64, len(I))
	for i := range varI {
		varI[i] = corrI[i] - meanI[i]*meanI[i]
	}

	out := cv.NewMat(rows, cols, src.Channels)
	p := make([]float64, rows*cols)
	for c := 0; c < src.Channels; c++ {
		for i := 0; i < rows*cols; i++ {
			p[i] = float64(src.Data[i*src.Channels+c])
		}
		q := guidedChannel(I, meanI, varI, p, rows, cols, radius, eps)
		for i := 0; i < rows*cols; i++ {
			out.Data[i*src.Channels+c] = clampU8(q[i])
		}
	}
	return out
}

// guidedChannel runs the guided-filter linear model for one input channel p
// against precomputed guidance statistics.
func guidedChannel(I, meanI, varI, p []float64, rows, cols, r int, eps float64) []float64 {
	meanP := boxMean(p, rows, cols, r)
	corrIp := boxMean(mul(I, p), rows, cols, r)

	a := make([]float64, len(p))
	b := make([]float64, len(p))
	for i := range p {
		covIp := corrIp[i] - meanI[i]*meanP[i]
		a[i] = covIp / (varI[i] + eps)
		b[i] = meanP[i] - a[i]*meanI[i]
	}
	meanA := boxMean(a, rows, cols, r)
	meanB := boxMean(b, rows, cols, r)

	q := make([]float64, len(p))
	for i := range q {
		q[i] = meanA[i]*I[i] + meanB[i]
	}
	return q
}

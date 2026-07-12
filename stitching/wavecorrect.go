package stitching

import "math"

// WaveCorrectKind selects the axis a [WaveCorrect] call straightens.
type WaveCorrectKind int

const (
	// WaveCorrectHoriz removes horizontal waviness, keeping the panorama's
	// horizon level. It is the right choice for the common case of a camera swept
	// left-to-right.
	WaveCorrectHoriz WaveCorrectKind = iota
	// WaveCorrectVert removes vertical waviness, keeping vertical structure
	// upright, for a camera swept top-to-bottom.
	WaveCorrectVert
)

// WaveCorrect straightens the global "wave" — the slow up-and-down (or
// side-to-side) drift of the horizon — that remains in a set of camera rotations
// after estimation and bundle adjustment. Chaining rotations around a panorama
// leaves the whole strip free to tilt and undulate; wave correction fixes that
// residual gauge freedom by finding the single world rotation that best flattens
// the cameras' optical axes onto a common plane and pre-multiplying every camera
// by it. The rotations in cams are modified in place.
//
// It works by finding the axis about which the cameras' x-axes vary least (for
// horizontal correction) via the eigenvector of the smallest eigenvalue of their
// covariance, then rebuilding an orthonormal world frame around it.
func WaveCorrect(cams []CameraParams, kind WaveCorrectKind) {
	if len(cams) == 0 {
		return
	}
	// Covariance of the cameras' first columns (their world x-axes).
	var moment mat3
	for _, c := range cams {
		col0 := [3]float64{c.R[0], c.R[3], c.R[6]}
		for r := 0; r < 3; r++ {
			for cc := 0; cc < 3; cc++ {
				moment[r*3+cc] += col0[r] * col0[cc]
			}
		}
	}
	_, vecs := jacobiEigenSym3(moment) // ascending eigenvalues, eigenvectors in columns

	// rg1 is the rotation axis to preserve.
	var rg1 [3]float64
	switch kind {
	case WaveCorrectVert:
		rg1 = [3]float64{vecs[2], vecs[5], vecs[8]} // largest-eigenvalue eigenvector
	default: // WaveCorrectHoriz
		rg1 = [3]float64{vecs[0], vecs[3], vecs[6]} // smallest-eigenvalue eigenvector
	}

	// imgK is the mean optical axis (third column) of the cameras.
	var imgK [3]float64
	for _, c := range cams {
		imgK[0] += c.R[2]
		imgK[1] += c.R[5]
		imgK[2] += c.R[8]
	}

	rg0 := cross3(rg1, imgK)
	if n := norm3(rg0); n > 1e-12 {
		rg0 = [3]float64{rg0[0] / n, rg0[1] / n, rg0[2] / n}
	}
	rg2 := cross3(rg0, rg1)

	// Resolve the sign ambiguity so the correction does not flip the panorama.
	var conf float64
	for _, c := range cams {
		col0 := [3]float64{c.R[0], c.R[3], c.R[6]}
		if kind == WaveCorrectVert {
			conf -= dot3(rg1, col0)
		} else {
			conf += dot3(rg0, col0)
		}
	}
	if conf < 0 {
		rg0 = [3]float64{-rg0[0], -rg0[1], -rg0[2]}
		rg1 = [3]float64{-rg1[0], -rg1[1], -rg1[2]}
		rg2 = cross3(rg0, rg1)
	}

	// Correction rotation with rows rg0, rg1, rg2.
	corr := mat3{
		rg0[0], rg0[1], rg0[2],
		rg1[0], rg1[1], rg1[2],
		rg2[0], rg2[1], rg2[2],
	}
	for i := range cams {
		cams[i].R = [9]float64(corr.mul(cams[i].rot()))
	}
}

// cross3 returns the cross product a × b.
func cross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// dot3 returns the dot product of two 3-vectors.
func dot3(a, b [3]float64) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

// norm3 returns the Euclidean length of a 3-vector.
func norm3(a [3]float64) float64 { return math.Sqrt(dot3(a, a)) }

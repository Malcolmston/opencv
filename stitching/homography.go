package stitching

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// pointF is a floating-point image coordinate used throughout the geometric
// estimation code (x is the column, y is the row).
type pointF struct {
	x, y float64
}

// identityH returns the 3×3 identity projective transform.
func identityH() cv.PerspectiveMatrix {
	return cv.PerspectiveMatrix{1, 0, 0, 0, 1, 0, 0, 0, 1}
}

// translationH returns the projective transform that shifts a point by
// (tx, ty).
func translationH(tx, ty float64) cv.PerspectiveMatrix {
	return cv.PerspectiveMatrix{1, 0, tx, 0, 1, ty, 0, 0, 1}
}

// matMul3 multiplies two 3×3 projective transforms (row-major), returning a*b.
// The composition applies b first, then a: applyH(matMul3(a, b), p) equals
// applyH(a, applyH(b, p)).
func matMul3(a, b cv.PerspectiveMatrix) cv.PerspectiveMatrix {
	var out cv.PerspectiveMatrix
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += a[r*3+k] * b[k*3+c]
			}
			out[r*3+c] = s
		}
	}
	return out
}

// applyH maps the point (x, y) through the projective transform h, returning the
// dehomogenised destination coordinates. The second result is false when the
// homogeneous scale collapses to zero (a point mapped to infinity).
func applyH(h cv.PerspectiveMatrix, x, y float64) (float64, float64, bool) {
	w := h[6]*x + h[7]*y + h[8]
	if w == 0 {
		return 0, 0, false
	}
	return (h[0]*x + h[1]*y + h[2]) / w, (h[3]*x + h[4]*y + h[5]) / w, true
}

// normalizePoints computes the isotropic similarity transform that recentres the
// points on their centroid and scales them so the mean distance to the origin is
// √2, the conditioning step of the normalised DLT. It returns the transform T
// and its inverse. Degenerate (coincident) inputs yield the identity.
func normalizePoints(pts []pointF) (t, tInv cv.PerspectiveMatrix) {
	n := float64(len(pts))
	var cx, cy float64
	for _, p := range pts {
		cx += p.x
		cy += p.y
	}
	cx /= n
	cy /= n
	var meanDist float64
	for _, p := range pts {
		meanDist += math.Hypot(p.x-cx, p.y-cy)
	}
	meanDist /= n
	if meanDist < 1e-12 {
		return identityH(), identityH()
	}
	s := math.Sqrt2 / meanDist
	t = cv.PerspectiveMatrix{s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1}
	tInv = cv.PerspectiveMatrix{1 / s, 0, cx, 0, 1 / s, cy, 0, 0, 1}
	return t, tInv
}

// computeHomographyDLT estimates the homography mapping src points onto dst
// points with the normalised Direct Linear Transform. It accepts four or more
// correspondences and solves a least-squares system with the scale fixed to
// h8 = 1 (valid for the quasi-affine transforms produced by panorama capture).
// The bool result is false when there are too few points or the system is
// singular.
func computeHomographyDLT(src, dst []pointF) (cv.PerspectiveMatrix, bool) {
	if len(src) < 4 || len(src) != len(dst) {
		return cv.PerspectiveMatrix{}, false
	}
	ts, _ := normalizePoints(src)
	td, tdInv := normalizePoints(dst)

	// Accumulate the 8×8 normal equations A^T A h = A^T b over the normalised
	// correspondences, where each pair contributes two rows.
	var ata [8][8]float64
	var atb [8]float64
	addRow := func(row [8]float64, rhs float64) {
		for i := 0; i < 8; i++ {
			for j := 0; j < 8; j++ {
				ata[i][j] += row[i] * row[j]
			}
			atb[i] += row[i] * rhs
		}
	}
	for i := range src {
		sx, sy, ok1 := applyH(ts, src[i].x, src[i].y)
		dx, dy, ok2 := applyH(td, dst[i].x, dst[i].y)
		if !ok1 || !ok2 {
			return cv.PerspectiveMatrix{}, false
		}
		addRow([8]float64{sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx}, dx)
		addRow([8]float64{0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy}, dy)
	}
	h, ok := solveLinear(ata, atb)
	if !ok {
		return cv.PerspectiveMatrix{}, false
	}
	hn := cv.PerspectiveMatrix{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1}
	// Denormalise: H = Tdst^-1 * Hn * Tsrc.
	full := matMul3(tdInv, matMul3(hn, ts))
	if full[8] == 0 {
		return cv.PerspectiveMatrix{}, false
	}
	// Rescale so h8 == 1 for a canonical representation.
	inv := 1 / full[8]
	for i := range full {
		full[i] *= inv
	}
	return full, true
}

// solveLinear solves the 8×8 system a*x = b with Gauss–Jordan elimination and
// partial pivoting, reporting whether the matrix was non-singular.
func solveLinear(a [8][8]float64, b [8]float64) ([8]float64, bool) {
	const n = 8
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(a[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(a[r][col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-12 {
			return [8]float64{}, false
		}
		a[col], a[piv] = a[piv], a[col]
		b[col], b[piv] = b[piv], b[col]
		p := a[col][col]
		for c := col; c < n; c++ {
			a[col][c] /= p
		}
		b[col] /= p
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a[r][col]
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				a[r][c] -= f * a[col][c]
			}
			b[r] -= f * b[col]
		}
	}
	return b, true
}

// RANSACParams configures the random-sample-consensus homography estimator used
// by [Stitcher.EstimateTransform].
type RANSACParams struct {
	// Iterations is the number of random minimal samples drawn.
	Iterations int
	// ReprojThreshold is the maximum reprojection error, in pixels, for a
	// correspondence to be counted as an inlier.
	ReprojThreshold float64
	// Seed fixes the pseudo-random sampling so estimation is deterministic.
	Seed int64
}

// estimateHomographyRANSAC robustly fits the homography mapping src onto dst from
// putative correspondences that may contain outliers. It repeatedly samples four
// pairs, fits a candidate with the normalised DLT, and keeps the model with the
// most inliers; it then refits on the full inlier set. It returns the homography,
// the inlier indices, and a bool that is false when no adequate model is found.
func estimateHomographyRANSAC(src, dst []pointF, params RANSACParams) (cv.PerspectiveMatrix, []int, bool) {
	n := len(src)
	if n < 4 {
		return cv.PerspectiveMatrix{}, nil, false
	}
	rng := rand.New(rand.NewSource(params.Seed))
	thresh2 := params.ReprojThreshold * params.ReprojThreshold

	bestInliers := []int{}
	var bestH cv.PerspectiveMatrix
	haveModel := false

	sample := make([]int, 4)
	for it := 0; it < params.Iterations; it++ {
		pickDistinct(rng, n, sample)
		s4 := []pointF{src[sample[0]], src[sample[1]], src[sample[2]], src[sample[3]]}
		d4 := []pointF{dst[sample[0]], dst[sample[1]], dst[sample[2]], dst[sample[3]]}
		h, ok := computeHomographyDLT(s4, d4)
		if !ok {
			continue
		}
		inliers := consensusSet(h, src, dst, thresh2)
		if len(inliers) > len(bestInliers) {
			bestInliers = inliers
			bestH = h
			haveModel = true
		}
	}
	if !haveModel || len(bestInliers) < 4 {
		return cv.PerspectiveMatrix{}, nil, false
	}
	// Refit on all inliers for a more accurate model.
	is := make([]pointF, len(bestInliers))
	id := make([]pointF, len(bestInliers))
	for i, idx := range bestInliers {
		is[i] = src[idx]
		id[i] = dst[idx]
	}
	if refined, ok := computeHomographyDLT(is, id); ok {
		refinedInliers := consensusSet(refined, src, dst, thresh2)
		if len(refinedInliers) >= len(bestInliers) {
			return refined, refinedInliers, true
		}
	}
	return bestH, bestInliers, true
}

// consensusSet returns the indices whose reprojection error under h is within
// the squared threshold.
func consensusSet(h cv.PerspectiveMatrix, src, dst []pointF, thresh2 float64) []int {
	var inliers []int
	for i := range src {
		px, py, ok := applyH(h, src[i].x, src[i].y)
		if !ok {
			continue
		}
		dx := px - dst[i].x
		dy := py - dst[i].y
		if dx*dx+dy*dy <= thresh2 {
			inliers = append(inliers, i)
		}
	}
	return inliers
}

// pickDistinct fills out (length 4) with four distinct indices in [0, n) drawn
// from rng. It assumes n >= 4.
func pickDistinct(rng *rand.Rand, n int, out []int) {
	for i := range out {
		for {
			c := rng.Intn(n)
			dup := false
			for j := 0; j < i; j++ {
				if out[j] == c {
					dup = true
					break
				}
			}
			if !dup {
				out[i] = c
				break
			}
		}
	}
}

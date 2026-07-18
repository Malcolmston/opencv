package matching2

import (
	"github.com/malcolmston/opencv/core"
)

// FindEssentialMat estimates the 3×3 essential matrix relating two calibrated
// views from at least eight correspondences given in pixel coordinates, using
// the shared intrinsic matrix K. Points are first back-projected to normalized
// camera coordinates with K⁻¹, the normalized eight-point algorithm is applied,
// and the two nonzero singular values are forced equal (the essential-matrix
// constraint). It reports false when there are too few points, K is singular or
// the configuration is degenerate.
func FindEssentialMat(pts1, pts2 []core.Point2d, K [3][3]float64) ([3][3]float64, bool) {
	if len(pts1) != len(pts2) || len(pts1) < 8 {
		return [3][3]float64{}, false
	}
	Kinv, ok := Mat3Inverse(K)
	if !ok {
		return [3][3]float64{}, false
	}
	n1 := normalizeByKinv(pts1, Kinv)
	n2 := normalizeByKinv(pts2, Kinv)
	F, ok := FindFundamentalMat(n1, n2)
	if !ok {
		return [3][3]float64{}, false
	}
	return conditionEssential(F), true
}

// EssentialFromFundamental converts a fundamental matrix F and shared intrinsic
// matrix K into the corresponding essential matrix E = Kᵀ·F·K, with its singular
// values conditioned to the essential form (two equal, one zero).
func EssentialFromFundamental(F, K [3][3]float64) [3][3]float64 {
	E := Mat3Mul(Mat3Transpose(K), Mat3Mul(F, K))
	return conditionEssential(E)
}

// FundamentalFromEssential converts an essential matrix E and shared intrinsic
// matrix K into the corresponding fundamental matrix F = K⁻ᵀ·E·K⁻¹, scaled to
// unit Frobenius norm. It reports false when K is singular.
func FundamentalFromEssential(E, K [3][3]float64) ([3][3]float64, bool) {
	Kinv, ok := Mat3Inverse(K)
	if !ok {
		return [3][3]float64{}, false
	}
	F := Mat3Mul(Mat3Transpose(Kinv), Mat3Mul(E, Kinv))
	return normalizeFrobenius(F), true
}

// DecomposeEssentialMat factors an essential matrix into the two possible
// rotations (R1, R2) and the translation direction t (unit length, sign
// ambiguous). Combined with ±t these give the four candidate relative poses; use
// [RecoverPose] to select the physically correct one by cheirality. Both
// returned rotations have determinant +1.
func DecomposeEssentialMat(E [3][3]float64) (R1, R2 [3][3]float64, t [3]float64) {
	u, _, v := matching2svd3(conditionEssential(E))
	// Ensure U and V are proper rotations so R1, R2 come out as rotations.
	if Mat3Det(u) < 0 {
		u = negateCol(u, 2)
	}
	if Mat3Det(v) < 0 {
		v = negateCol(v, 2)
	}
	W := [3][3]float64{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}
	vt := Mat3Transpose(v)
	R1 = Mat3Mul(u, Mat3Mul(W, vt))
	R2 = Mat3Mul(u, Mat3Mul(Mat3Transpose(W), vt))
	if Mat3Det(R1) < 0 {
		R1 = Mat3Scale(R1, -1)
	}
	if Mat3Det(R2) < 0 {
		R2 = Mat3Scale(R2, -1)
	}
	t = [3]float64{u[0][2], u[1][2], u[2][2]}
	return R1, R2, t
}

// RecoverPose recovers the relative camera rotation R and unit translation t
// between two calibrated views from the essential matrix E and the matched pixel
// points, using intrinsics K. It resolves the four-fold decomposition ambiguity
// by triangulating each candidate and choosing the pose that places the most
// points in front of both cameras (the cheirality test). good is the number of
// such points for the chosen pose. The first camera is taken as [I | 0], so t is
// recovered only up to scale.
func RecoverPose(E [3][3]float64, pts1, pts2 []core.Point2d, K [3][3]float64) (R [3][3]float64, t [3]float64, good int) {
	R1, R2, tt := DecomposeEssentialMat(E)
	Kinv, ok := Mat3Inverse(K)
	if !ok {
		return Mat3Identity(), [3]float64{}, 0
	}
	n1 := normalizeByKinv(pts1, Kinv)
	n2 := normalizeByKinv(pts2, Kinv)
	P1 := [3][4]float64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
	type cand struct {
		R [3][3]float64
		t [3]float64
	}
	cands := []cand{
		{R1, tt}, {R1, negVec3(tt)},
		{R2, tt}, {R2, negVec3(tt)},
	}
	bestGood := -1
	var bestR [3][3]float64
	var bestT [3]float64
	for _, c := range cands {
		P2 := rtProjection(c.R, c.t)
		cnt := 0
		for i := range n1 {
			X := TriangulatePoint(P1, P2, n1[i], n2[i])
			d1, _ := project34(P1, X)
			d2, _ := project34(P2, X)
			if d1 > 0 && d2 > 0 {
				cnt++
			}
		}
		if cnt > bestGood {
			bestGood = cnt
			bestR = c.R
			bestT = c.t
		}
	}
	return bestR, bestT, bestGood
}

// conditionEssential forces the singular values of a 3×3 matrix to (1, 1, 0),
// the algebraic form of an essential matrix, preserving its singular vectors.
func conditionEssential(E [3][3]float64) [3][3]float64 {
	u, _, v := matching2svd3(E)
	d := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 0}}
	return Mat3Mul(u, Mat3Mul(d, Mat3Transpose(v)))
}

// rtProjection builds the 3×4 matrix [R | t] (normalized-coordinate projection,
// i.e. K = I).
func rtProjection(R [3][3]float64, t [3]float64) [3][4]float64 {
	var P [3][4]float64
	for i := 0; i < 3; i++ {
		P[i][0] = R[i][0]
		P[i][1] = R[i][1]
		P[i][2] = R[i][2]
		P[i][3] = t[i]
	}
	return P
}

// normalizeByKinv back-projects pixel points to normalized camera coordinates
// via K⁻¹, dropping the homogeneous scale.
func normalizeByKinv(pts []core.Point2d, Kinv [3][3]float64) []core.Point2d {
	out := make([]core.Point2d, len(pts))
	for i, p := range pts {
		v := Mat3VecMul(Kinv, [3]float64{p.X, p.Y, 1})
		if v[2] == 0 {
			out[i] = core.Point2d{X: v[0], Y: v[1]}
			continue
		}
		out[i] = core.Point2d{X: v[0] / v[2], Y: v[1] / v[2]}
	}
	return out
}

// negateCol returns m with column c negated.
func negateCol(m [3][3]float64, c int) [3][3]float64 {
	for r := 0; r < 3; r++ {
		m[r][c] = -m[r][c]
	}
	return m
}

// negVec3 returns the negation of a 3-vector.
func negVec3(v [3]float64) [3]float64 { return [3]float64{-v[0], -v[1], -v[2]} }

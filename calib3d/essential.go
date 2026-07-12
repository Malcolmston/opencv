package calib3d

import "math"

// eightPointF estimates a fundamental (or, on already-calibrated coordinates,
// essential) matrix from float correspondences with the normalized eight-point
// algorithm: Hartley normalization, null space of the epipolar design matrix,
// and the rank-2 constraint enforced by zeroing the smallest singular value.
func eightPointF(p1, p2 [][2]float64) ([3][3]float64, bool) {
	if len(p1) != len(p2) || len(p1) < 8 {
		return [3][3]float64{}, false
	}
	t1, n1, ok1 := normalizePoints(p1)
	t2, n2, ok2 := normalizePoints(p2)
	if !ok1 || !ok2 {
		return [3][3]float64{}, false
	}
	rows := make([][]float64, len(n1))
	for i := range n1 {
		x, y := n1[i][0], n1[i][1]
		xp, yp := n2[i][0], n2[i][1]
		rows[i] = []float64{xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1}
	}
	f := nullspaceVec(rows, 9)
	fn := [3][3]float64{
		{f[0], f[1], f[2]},
		{f[3], f[4], f[5]},
		{f[6], f[7], f[8]},
	}
	u, s, v := svd3(fn)
	s[2] = 0
	fn = mul3(u, mul3([3][3]float64{{s[0], 0, 0}, {0, s[1], 0}, {0, 0, s[2]}}, transpose3(v)))
	res := mul3(transpose3(t2), mul3(fn, t1))
	return res, true
}

// FindEssentialMat estimates the 3×3 essential matrix relating two calibrated
// views from corresponding image points. Both point sets share the single
// intrinsic matrix K and must have equal length of at least eight.
//
// The image points are pre-multiplied by K⁻¹ to obtain normalized camera
// coordinates, on which the normalized eight-point algorithm is run. The result
// is projected onto the essential-matrix manifold by forcing its two leading
// singular values to be equal and the third to zero, so that E = U·diag(1,1,0)·Vᵀ
// up to scale. ok is false when the input is insufficient or degenerate.
func FindEssentialMat(pts1, pts2 [][2]float64, K [3][3]float64) (E [3][3]float64, ok bool) {
	if len(pts1) != len(pts2) || len(pts1) < 8 {
		return [3][3]float64{}, false
	}
	kInv, okk := inv3(K)
	if !okk {
		return [3][3]float64{}, false
	}
	n1 := normalizeByK(pts1, kInv)
	n2 := normalizeByK(pts2, kInv)
	e, okf := eightPointF(n1, n2)
	if !okf {
		return [3][3]float64{}, false
	}
	// Enforce the essential-matrix constraint: singular values (s, s, 0).
	u, s, v := svd3(e)
	avg := (s[0] + s[1]) / 2
	d := [3][3]float64{{avg, 0, 0}, {0, avg, 0}, {0, 0, 0}}
	E = mul3(u, mul3(d, transpose3(v)))
	// Scale to unit Frobenius norm for a canonical representative.
	var fro float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			fro += E[i][j] * E[i][j]
		}
	}
	fro = math.Sqrt(fro)
	if fro < 1e-18 {
		return [3][3]float64{}, false
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			E[i][j] /= fro
		}
	}
	return E, true
}

// normalizeByK maps pixel points to normalized camera coordinates via K⁻¹,
// dropping the homogeneous scale.
func normalizeByK(pts [][2]float64, kInv [3][3]float64) [][2]float64 {
	out := make([][2]float64, len(pts))
	for i, p := range pts {
		v := matVec3(kInv, [3]float64{p[0], p[1], 1})
		if math.Abs(v[2]) < 1e-18 {
			out[i] = [2]float64{v[0], v[1]}
			continue
		}
		out[i] = [2]float64{v[0] / v[2], v[1] / v[2]}
	}
	return out
}

// DecomposeEssentialMat factorises an essential matrix into the two possible
// rotations and the (sign-ambiguous, unit-norm) translation direction that are
// consistent with it. The four physical camera poses are (R1, ±t) and (R2, ±t);
// use [RecoverPose] to select the one satisfying the cheirality constraint.
//
// The decomposition follows Hartley & Zisserman: E = U·diag(1,1,0)·Vᵀ, and with
// W the 90° rotation about the third axis, R1 = U·W·Vᵀ and R2 = U·Wᵀ·Vᵀ (each
// forced to determinant +1), while t is the last column of U.
func DecomposeEssentialMat(E [3][3]float64) (R1, R2 [3][3]float64, t [3]float64) {
	u, _, v := svd3(E)
	// Force det(U) = det(V) = +1 so the rotations come out proper.
	if det3(u) < 0 {
		for r := 0; r < 3; r++ {
			u[r][2] = -u[r][2]
		}
	}
	if det3(v) < 0 {
		for r := 0; r < 3; r++ {
			v[r][2] = -v[r][2]
		}
	}
	w := [3][3]float64{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}
	vt := transpose3(v)
	R1 = mul3(u, mul3(w, vt))
	R2 = mul3(u, mul3(transpose3(w), vt))
	if det3(R1) < 0 {
		R1 = negMat(R1)
	}
	if det3(R2) < 0 {
		R2 = negMat(R2)
	}
	t = [3]float64{u[0][2], u[1][2], u[2][2]}
	return R1, R2, t
}

func negMat(m [3][3]float64) [3][3]float64 {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			m[i][j] = -m[i][j]
		}
	}
	return m
}

// RecoverPose recovers the relative camera pose (rotation R and unit-norm
// translation direction t) between two calibrated views from an essential
// matrix and the underlying point correspondences. The first camera is taken as
// the reference [I | 0]; the second is [R | t]. K is the shared intrinsic
// matrix.
//
// The four candidate poses from [DecomposeEssentialMat] are disambiguated by the
// cheirality constraint: each correspondence is triangulated and the pose under
// which the most points lie in front of both cameras (positive depth) is
// returned. good is the number of such points for the chosen pose; a small count
// indicates an unreliable estimate. The returned translation is only defined up
// to scale.
func RecoverPose(E [3][3]float64, pts1, pts2 [][2]float64, K [3][3]float64) (R [3][3]float64, t [3]float64, good int) {
	kInv, ok := inv3(K)
	if !ok {
		return [3][3]float64{}, [3]float64{}, 0
	}
	n1 := normalizeByK(pts1, kInv)
	n2 := normalizeByK(pts2, kInv)
	R1, R2, tt := DecomposeEssentialMat(E)
	cands := []struct {
		R [3][3]float64
		t [3]float64
	}{
		{R1, tt}, {R1, scale3(tt, -1)}, {R2, tt}, {R2, scale3(tt, -1)},
	}
	P1 := [3][4]float64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
	bestGood := -1
	for _, c := range cands {
		P2 := rtToP(c.R, c.t)
		cnt := 0
		for i := range n1 {
			X := triangulateOneF(P1, P2, n1[i], n2[i])
			if X[2] <= 0 {
				continue
			}
			// Depth in the second camera.
			z2 := c.R[2][0]*X[0] + c.R[2][1]*X[1] + c.R[2][2]*X[2] + c.t[2]
			if z2 > 0 {
				cnt++
			}
		}
		if cnt > bestGood {
			bestGood = cnt
			R = c.R
			t = c.t
		}
	}
	return R, t, bestGood
}

// rtToP assembles the 3×4 projection matrix [R | t].
func rtToP(R [3][3]float64, t [3]float64) [3][4]float64 {
	return [3][4]float64{
		{R[0][0], R[0][1], R[0][2], t[0]},
		{R[1][0], R[1][1], R[1][2], t[1]},
		{R[2][0], R[2][1], R[2][2], t[2]},
	}
}

// triangulateOneF reconstructs a single 3D point from its projections in two
// views by linear (DLT) triangulation, taking float image coordinates.
func triangulateOneF(P1, P2 [3][4]float64, p1, p2 [2]float64) [3]float64 {
	rows := [4][4]float64{
		subRow4(scaleRow4(P1[2], p1[0]), P1[0]),
		subRow4(scaleRow4(P1[2], p1[1]), P1[1]),
		subRow4(scaleRow4(P2[2], p2[0]), P2[0]),
		subRow4(scaleRow4(P2[2], p2[1]), P2[1]),
	}
	var ata [4][4]float64
	for _, r := range rows {
		for a := 0; a < 4; a++ {
			for b := 0; b < 4; b++ {
				ata[a][b] += r[a] * r[b]
			}
		}
	}
	dyn := make([][]float64, 4)
	for a := 0; a < 4; a++ {
		dyn[a] = make([]float64, 4)
		copy(dyn[a], ata[a][:])
	}
	x := smallestEigvec(dyn)
	if math.Abs(x[3]) < 1e-18 {
		return [3]float64{0, 0, 0}
	}
	return [3]float64{x[0] / x[3], x[1] / x[3], x[2] / x[3]}
}

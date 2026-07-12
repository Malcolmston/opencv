package aruco

import "math"

// This file holds planar-geometry helpers shared by the board, ChArUco and
// pose routines added to the package: a robust homography estimator for an
// arbitrary number of coplanar correspondences and a homography-to-pose
// factorisation. They complement the four-point solver in pose.go by accepting
// any n >= 4 correspondences (via a normalised direct-linear-transform least
// squares) and by exposing the intrinsic factorisation as a reusable step.

// homographyFromPoints fits the 3x3 homography H that best maps the planar
// source points src to the destination points dst in a least-squares sense,
// using a Hartley-normalised direct linear transform. src and dst must be the
// same length and hold at least four non-collinear points. ok is false when the
// inputs are too few or the normal equations are singular.
//
// H is scaled so that H[2][2] == 1 and maps (x, y) to (u, v) via
//
//	w = H[2][0]*x + H[2][1]*y + H[2][2]
//	u = (H[0][0]*x + H[0][1]*y + H[0][2]) / w
//	v = (H[1][0]*x + H[1][1]*y + H[1][2]) / w
func homographyFromPoints(src, dst [][2]float64) ([3][3]float64, bool) {
	if len(src) != len(dst) || len(src) < 4 {
		return [3][3]float64{}, false
	}
	ns, ts, ok := normalizePoints(src)
	if !ok {
		return [3][3]float64{}, false
	}
	nd, td, ok := normalizePoints(dst)
	if !ok {
		return [3][3]float64{}, false
	}
	// Accumulate the 8x8 normal equations A^T A x = A^T b for the eight unknowns
	// h0..h7 (with h8 pinned to 1), summed over every correspondence.
	var ata [8][8]float64
	var atb [8]float64
	addRow := func(row [8]float64, target float64) {
		for i := 0; i < 8; i++ {
			for j := 0; j < 8; j++ {
				ata[i][j] += row[i] * row[j]
			}
			atb[i] += row[i] * target
		}
	}
	for i := range ns {
		x, y := ns[i][0], ns[i][1]
		u, v := nd[i][0], nd[i][1]
		addRow([8]float64{x, y, 1, 0, 0, 0, -x * u, -y * u}, u)
		addRow([8]float64{0, 0, 0, x, y, 1, -x * v, -y * v}, v)
	}
	sol, ok := solve8(ata, atb)
	if !ok {
		return [3][3]float64{}, false
	}
	hn := [3][3]float64{
		{sol[0], sol[1], sol[2]},
		{sol[3], sol[4], sol[5]},
		{sol[6], sol[7], 1},
	}
	// Denormalise: H = Td^-1 * Hn * Ts.
	tdInv, ok := invert3(td)
	if !ok {
		return [3][3]float64{}, false
	}
	h := mul3(tdInv, mul3(hn, ts))
	if h[2][2] == 0 {
		return [3][3]float64{}, false
	}
	inv := 1 / h[2][2]
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			h[r][c] *= inv
		}
	}
	return h, true
}

// normalizePoints returns pts translated to their centroid and scaled so that
// their mean distance to the origin is sqrt(2), together with the 3x3 similarity
// transform T that performs the mapping (T applied to the homogeneous point).
// ok is false when every point coincides.
func normalizePoints(pts [][2]float64) ([][2]float64, [3][3]float64, bool) {
	n := float64(len(pts))
	var mx, my float64
	for _, p := range pts {
		mx += p[0]
		my += p[1]
	}
	mx /= n
	my /= n
	var meanDist float64
	for _, p := range pts {
		meanDist += math.Hypot(p[0]-mx, p[1]-my)
	}
	meanDist /= n
	if meanDist < 1e-12 {
		return nil, [3][3]float64{}, false
	}
	s := math.Sqrt2 / meanDist
	t := [3][3]float64{
		{s, 0, -s * mx},
		{0, s, -s * my},
		{0, 0, 1},
	}
	out := make([][2]float64, len(pts))
	for i, p := range pts {
		out[i] = [2]float64{(p[0] - mx) * s, (p[1] - my) * s}
	}
	return out, t, true
}

// mul3 returns the matrix product a*b of two 3x3 matrices.
func mul3(a, b [3][3]float64) [3][3]float64 {
	var out [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += a[i][k] * b[k][j]
			}
			out[i][j] = s
		}
	}
	return out
}

// applyH maps the point (x, y) through the homography h, returning the
// projected (u, v). ok is false when the point maps to infinity.
func applyH(h [3][3]float64, x, y float64) (u, v float64, ok bool) {
	w := h[2][0]*x + h[2][1]*y + h[2][2]
	if w == 0 {
		return 0, 0, false
	}
	return (h[0][0]*x + h[0][1]*y + h[0][2]) / w,
		(h[1][0]*x + h[1][1]*y + h[1][2]) / w, true
}

// poseFromH factors the plane-to-image homography h through the intrinsic matrix
// k into a rotation (returned as a Rodrigues vector) and translation, forcing
// the plane in front of the camera. It mirrors the closed-form step used by
// EstimatePoseSingleMarkers but takes an already-computed homography so it can
// serve multi-point board solves. ok is false when h or k is degenerate.
func poseFromH(h [3][3]float64, k [3][3]float64) (rvec, tvec [3]float64, ok bool) {
	kInv, ok := invert3(k)
	if !ok {
		return rvec, tvec, false
	}
	var b [3][3]float64
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			var s float64
			for t := 0; t < 3; t++ {
				s += kInv[r][t] * h[t][c]
			}
			b[r][c] = s
		}
	}
	b1 := [3]float64{b[0][0], b[1][0], b[2][0]}
	b2 := [3]float64{b[0][1], b[1][1], b[2][1]}
	b3 := [3]float64{b[0][2], b[1][2], b[2][2]}
	n1 := norm3(b1)
	n2 := norm3(b2)
	if n1 == 0 || n2 == 0 {
		return rvec, tvec, false
	}
	lambda := 2 / (n1 + n2)
	t := scale3(b3, lambda)
	if t[2] < 0 {
		lambda = -lambda
		t = scale3(b3, lambda)
	}
	r1 := scale3(b1, lambda)
	r2 := scale3(b2, lambda)
	r1 = normalize3(r1)
	r2 = normalize3(sub3(r2, scale3(r1, dot3(r1, r2))))
	r3 := cross3(r1, r2)
	rot := [3][3]float64{
		{r1[0], r2[0], r3[0]},
		{r1[1], r2[1], r3[1]},
		{r1[2], r2[2], r3[2]},
	}
	return rodrigues(rot), t, true
}

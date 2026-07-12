package aruco

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// EstimatePoseSingleMarkers recovers an approximate camera pose for each
// detected marker. corners holds the four image corners of each marker (as
// returned by [DetectMarkers], ordered clockwise from the top-left cell),
// markerLength is the marker's physical side length in the caller's world units,
// and k is the 3x3 pinhole camera intrinsic matrix (row-major, with fx, fy on
// the diagonal and cx, cy in the last column). dist holds radial/tangential
// distortion coefficients.
//
// It returns, for each marker, a Rodrigues rotation vector rvec and a
// translation vector tvec such that a point X in the marker frame projects to
// the camera frame as R(rvec)*X + tvec. The marker frame places the marker flat
// on its Z=0 plane, centred at the origin, with +X right and +Y up, matching
// OpenCV's convention.
//
// This is a deliberately simplified, homography-based estimator (an IPPE-style
// planar solve followed by Gram-Schmidt orthonormalisation). The following are
// deferred and noted here rather than implemented:
//
//   - dist is ignored: lens distortion is not undone, so accuracy degrades with
//     strong distortion. Pass nil when the camera is already rectified.
//   - no non-linear (Levenberg-Marquardt) refinement is performed, so the pose
//     is a closed-form estimate rather than a reprojection-optimal one.
//   - the planar sign/twofold ambiguity is resolved by forcing the marker in
//     front of the camera (positive Z); the alternate IPPE solution is not
//     returned.
//
// If a marker's homography is degenerate its rvec and tvec are both zero.
func EstimatePoseSingleMarkers(corners [][4]cv.Point, markerLength float64, k [3][3]float64, dist []float64) (rvecs, tvecs [][3]float64) {
	_ = dist // distortion handling is deferred; see the doc comment.
	half := markerLength / 2
	// Object corners in the marker plane, clockwise from the top-left, matching
	// the corner order produced by DetectMarkers.
	obj := [4][2]float64{
		{-half, half},  // top-left
		{half, half},   // top-right
		{half, -half},  // bottom-right
		{-half, -half}, // bottom-left
	}

	rvecs = make([][3]float64, len(corners))
	tvecs = make([][3]float64, len(corners))
	for i, quad := range corners {
		var img [4][2]float64
		for j := 0; j < 4; j++ {
			img[j] = [2]float64{float64(quad[j].X), float64(quad[j].Y)}
		}
		rvec, tvec, ok := poseFromHomography(obj, img, k)
		if ok {
			rvecs[i] = rvec
			tvecs[i] = tvec
		}
	}
	return rvecs, tvecs
}

// poseFromHomography estimates a single marker's pose from four planar
// correspondences and the intrinsic matrix. It fits the plane-to-image
// homography, factors out the intrinsics, and reconstructs a valid rotation and
// translation. ok is false when the homography or intrinsics are degenerate.
func poseFromHomography(obj, img [4][2]float64, k [3][3]float64) (rvec, tvec [3]float64, ok bool) {
	h, ok := solveHomography(obj, img)
	if !ok {
		return rvec, tvec, false
	}
	kInv, ok := invert3(k)
	if !ok {
		return rvec, tvec, false
	}
	// B = K^-1 H. Its first two columns are the scaled rotation columns; the
	// third is the scaled translation.
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
		// Force the marker in front of the camera.
		lambda = -lambda
		t = scale3(b3, lambda)
	}
	r1 := scale3(b1, lambda)
	r2 := scale3(b2, lambda)

	// Orthonormalise (r1, r2) with Gram-Schmidt and complete the basis.
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

// solveHomography computes the 3x3 homography mapping the four planar source
// points src to the destination points dst by solving the 8x8 linear system
// with h22 fixed to 1. ok is false when the system is singular.
func solveHomography(src, dst [4][2]float64) ([3][3]float64, bool) {
	var a [8][8]float64
	var bvec [8]float64
	for i := 0; i < 4; i++ {
		x, y := src[i][0], src[i][1]
		u, v := dst[i][0], dst[i][1]
		a[2*i] = [8]float64{x, y, 1, 0, 0, 0, -x * u, -y * u}
		bvec[2*i] = u
		a[2*i+1] = [8]float64{0, 0, 0, x, y, 1, -x * v, -y * v}
		bvec[2*i+1] = v
	}
	sol, ok := solve8(a, bvec)
	if !ok {
		return [3][3]float64{}, false
	}
	return [3][3]float64{
		{sol[0], sol[1], sol[2]},
		{sol[3], sol[4], sol[5]},
		{sol[6], sol[7], 1},
	}, true
}

// solve8 solves an 8x8 linear system by Gauss-Jordan elimination with partial
// pivoting, reporting whether the matrix was non-singular.
func solve8(a [8][8]float64, b [8]float64) ([8]float64, bool) {
	const n = 8
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(a[col][col])
		for r := col + 1; r < n; r++ {
			if math.Abs(a[r][col]) > best {
				best = math.Abs(a[r][col])
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

// invert3 returns the inverse of a 3x3 matrix and whether it is invertible.
func invert3(m [3][3]float64) ([3][3]float64, bool) {
	det := m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
	if math.Abs(det) < 1e-15 {
		return [3][3]float64{}, false
	}
	id := 1 / det
	var inv [3][3]float64
	inv[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * id
	inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * id
	inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * id
	inv[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * id
	inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * id
	inv[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * id
	inv[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * id
	inv[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * id
	inv[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * id
	return inv, true
}

// rodrigues converts a 3x3 rotation matrix to its Rodrigues rotation vector,
// whose direction is the rotation axis and whose magnitude is the angle in
// radians.
func rodrigues(r [3][3]float64) [3]float64 {
	tr := r[0][0] + r[1][1] + r[2][2]
	cosTheta := (tr - 1) / 2
	if cosTheta > 1 {
		cosTheta = 1
	} else if cosTheta < -1 {
		cosTheta = -1
	}
	theta := math.Acos(cosTheta)
	if theta < 1e-9 {
		return [3]float64{}
	}
	s := 2 * math.Sin(theta)
	axis := [3]float64{
		(r[2][1] - r[1][2]) / s,
		(r[0][2] - r[2][0]) / s,
		(r[1][0] - r[0][1]) / s,
	}
	return scale3(axis, theta)
}

// The small vector helpers below operate on 3-element vectors.

func dot3(a, b [3]float64) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

func norm3(a [3]float64) float64 { return math.Sqrt(dot3(a, a)) }

func scale3(a [3]float64, s float64) [3]float64 { return [3]float64{a[0] * s, a[1] * s, a[2] * s} }

func sub3(a, b [3]float64) [3]float64 { return [3]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }

func cross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func normalize3(a [3]float64) [3]float64 {
	n := norm3(a)
	if n == 0 {
		return a
	}
	return scale3(a, 1/n)
}

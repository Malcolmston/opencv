package cudalegacy

import (
	"math"
	"math/rand"
)

// Rodrigues converts a rotation vector (axis-angle, whose direction is the
// rotation axis and whose magnitude is the rotation angle in radians) into a
// 3×3 rotation matrix, mirroring cv::Rodrigues. It is the rotation
// representation used by [ProjectPoints] and returned by [SolvePnPRansac].
func Rodrigues(rvec [3]float64) [3][3]float64 {
	theta := math.Sqrt(rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2])
	if theta < 1e-12 {
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	kx := rvec[0] / theta
	ky := rvec[1] / theta
	kz := rvec[2] / theta
	c := math.Cos(theta)
	s := math.Sin(theta)
	v := 1 - c
	return [3][3]float64{
		{c + kx*kx*v, kx*ky*v - kz*s, kx*kz*v + ky*s},
		{ky*kx*v + kz*s, c + ky*ky*v, ky*kz*v - kx*s},
		{kz*kx*v - ky*s, kz*ky*v + kx*s, c + kz*kz*v},
	}
}

// rotationToVec is the inverse of [Rodrigues]: it extracts the axis-angle vector
// from a proper rotation matrix.
func rotationToVec(R [3][3]float64) [3]float64 {
	trace := R[0][0] + R[1][1] + R[2][2]
	cosTheta := (trace - 1) / 2
	if cosTheta > 1 {
		cosTheta = 1
	} else if cosTheta < -1 {
		cosTheta = -1
	}
	theta := math.Acos(cosTheta)
	if theta < 1e-9 {
		return [3]float64{0, 0, 0}
	}
	if math.Abs(theta-math.Pi) < 1e-6 {
		// Near 180°: recover axis from the diagonal of (R+I)/2.
		axis := [3]float64{
			math.Sqrt(math.Max((R[0][0]+1)/2, 0)),
			math.Sqrt(math.Max((R[1][1]+1)/2, 0)),
			math.Sqrt(math.Max((R[2][2]+1)/2, 0)),
		}
		if R[0][1]+R[1][0] < 0 {
			axis[1] = -axis[1]
		}
		if R[0][2]+R[2][0] < 0 {
			axis[2] = -axis[2]
		}
		return [3]float64{axis[0] * theta, axis[1] * theta, axis[2] * theta}
	}
	k := 1 / (2 * math.Sin(theta))
	return [3]float64{
		(R[2][1] - R[1][2]) * k * theta,
		(R[0][2] - R[2][0]) * k * theta,
		(R[1][0] - R[0][1]) * k * theta,
	}
}

// ProjectPoints is a CPU-backed mirror of the projection performed by OpenCV's
// cudalegacy projectPoints. It maps 3D object points, expressed in the world
// frame, into the image plane of a pinhole camera posed by rotation vector rvec
// and translation tvec, with intrinsics K and optional radial-tangential
// distortion coefficients dist (ordered k1, k2, p1, p2[, k3]). It returns one
// (u, v) image coordinate per input point.
//
// In OpenCV cudalegacy the point clouds live in float GpuMats; the 8-bit root
// [github.com/malcolmston/opencv.Mat] cannot store float coordinate triples, so
// this wrapper exchanges points as Go slices. dist may be nil or empty for an
// ideal lens.
func ProjectPoints(objectPoints [][3]float64, rvec, tvec [3]float64, K [3][3]float64, dist []float64) [][2]float64 {
	R := Rodrigues(rvec)
	out := make([][2]float64, len(objectPoints))
	fx, fy := K[0][0], K[1][1]
	cx, cy := K[0][2], K[1][2]
	var k1, k2, p1, p2, k3 float64
	if len(dist) > 0 {
		k1 = dist[0]
	}
	if len(dist) > 1 {
		k2 = dist[1]
	}
	if len(dist) > 2 {
		p1 = dist[2]
	}
	if len(dist) > 3 {
		p2 = dist[3]
	}
	if len(dist) > 4 {
		k3 = dist[4]
	}
	for i, P := range objectPoints {
		xc := R[0][0]*P[0] + R[0][1]*P[1] + R[0][2]*P[2] + tvec[0]
		yc := R[1][0]*P[0] + R[1][1]*P[1] + R[1][2]*P[2] + tvec[1]
		zc := R[2][0]*P[0] + R[2][1]*P[1] + R[2][2]*P[2] + tvec[2]
		if zc == 0 {
			zc = 1e-12
		}
		xn := xc / zc
		yn := yc / zc
		r2 := xn*xn + yn*yn
		radial := 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
		xd := xn*radial + 2*p1*xn*yn + p2*(r2+2*xn*xn)
		yd := yn*radial + p1*(r2+2*yn*yn) + 2*p2*xn*yn
		out[i] = [2]float64{fx*xd + cx, fy*yd + cy}
	}
	return out
}

// SolvePnPRansac is a CPU-backed mirror of the pose estimation performed by
// OpenCV's cudalegacy solvePnPRansac, specialised to a planar target: the object
// points are assumed to lie on the world z = 0 plane, which is the common case
// for calibration boards and marker fields. It estimates the camera rotation
// vector and translation that map objectPoints to imagePoints, using a RANSAC
// loop over minimal 4-point homographies to reject outliers, followed by a
// refit on the consensus set.
//
// dist are optional radial-tangential coefficients (see [ProjectPoints]);
// distortion is undone before pose is recovered. reprojThresh is the inlier
// reprojection error in pixels (a non-positive value defaults to 3). iterations
// bounds the RANSAC trials (a non-positive value defaults to 100). It returns
// the pose, a per-point inlier flag and ok = false when no adequate pose could
// be found (fewer than 4 points, a degenerate configuration, or no consensus).
func SolvePnPRansac(objectPoints [][3]float64, imagePoints [][2]float64, K [3][3]float64, dist []float64, reprojThresh float64, iterations int, rng *rand.Rand) (rvec, tvec [3]float64, inliers []bool, ok bool) {
	n := len(objectPoints)
	if n < 4 || len(imagePoints) != n {
		return rvec, tvec, nil, false
	}
	if reprojThresh <= 0 {
		reprojThresh = 3
	}
	if iterations <= 0 {
		iterations = 100
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}

	// Undistort image points into normalised, then pixel-like coordinates using
	// only K (distortion removed), so pose recovery sees an ideal pinhole.
	obj2D := make([][2]float64, n)
	for i, P := range objectPoints {
		obj2D[i] = [2]float64{P[0], P[1]}
	}
	norm := normalizePoints(imagePoints, K, dist)

	bestInliers := 0
	var bestRvec, bestTvec [3]float64
	var bestMask []bool

	for iter := 0; iter < iterations; iter++ {
		idx := sample4(n, rng)
		H, hok := homographyFrom(obj2D, norm, idx[:])
		if !hok {
			continue
		}
		r, t, dok := poseFromHomography(H)
		if !dok {
			continue
		}
		mask, count := countInliers(objectPoints, imagePoints, r, t, K, dist, reprojThresh)
		if count > bestInliers {
			bestInliers = count
			bestRvec, bestTvec = r, t
			bestMask = mask
		}
	}
	if bestInliers < 4 {
		return rvec, tvec, nil, false
	}

	// Refit the homography on all inliers for a stable final pose.
	var inIdx []int
	for i, in := range bestMask {
		if in {
			inIdx = append(inIdx, i)
		}
	}
	if H, hok := homographyFrom(obj2D, norm, inIdx); hok {
		if r, t, dok := poseFromHomography(H); dok {
			mask, count := countInliers(objectPoints, imagePoints, r, t, K, dist, reprojThresh)
			if count >= bestInliers {
				bestRvec, bestTvec, bestMask = r, t, mask
			}
		}
	}
	return bestRvec, bestTvec, bestMask, true
}

// normalizePoints removes radial-tangential distortion from pixel observations
// and returns their ideal normalised camera coordinates (x = (u-cx)/fx with
// distortion undone), so a homography mapping object (X, Y) to the normalised
// camera plane can be estimated and decomposed into pose without intrinsics.
func normalizePoints(pts [][2]float64, K [3][3]float64, dist []float64) [][2]float64 {
	fx, fy := K[0][0], K[1][1]
	cx, cy := K[0][2], K[1][2]
	var k1, k2, p1, p2, k3 float64
	if len(dist) > 0 {
		k1 = dist[0]
	}
	if len(dist) > 1 {
		k2 = dist[1]
	}
	if len(dist) > 2 {
		p1 = dist[2]
	}
	if len(dist) > 3 {
		p2 = dist[3]
	}
	if len(dist) > 4 {
		k3 = dist[4]
	}
	out := make([][2]float64, len(pts))
	for i, p := range pts {
		x := (p[0] - cx) / fx
		y := (p[1] - cy) / fy
		x0, y0 := x, y
		if len(dist) > 0 {
			// Iteratively invert the distortion model.
			for iter := 0; iter < 20; iter++ {
				r2 := x*x + y*y
				radial := 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
				dx := 2*p1*x*y + p2*(r2+2*x*x)
				dy := p1*(r2+2*y*y) + 2*p2*x*y
				x = (x0 - dx) / radial
				y = (y0 - dy) / radial
			}
		}
		out[i] = [2]float64{x, y}
	}
	return out
}

// sample4 returns four distinct random indices in [0, n).
func sample4(n int, rng *rand.Rand) [4]int {
	var idx [4]int
	chosen := map[int]bool{}
	for k := 0; k < 4; {
		c := rng.Intn(n)
		if chosen[c] {
			continue
		}
		chosen[c] = true
		idx[k] = c
		k++
	}
	return idx
}

// homographyFrom estimates the 3×3 homography mapping planar object points
// (X, Y) to image points (u, v) over the given index subset, by solving the DLT
// system with h33 fixed to 1. It needs at least four correspondences.
func homographyFrom(obj, img [][2]float64, idx []int) ([3][3]float64, bool) {
	var H [3][3]float64
	if len(idx) < 4 {
		return H, false
	}
	// 8 unknowns: h11..h32 (h33 = 1). Two equations per point.
	rows := len(idx) * 2
	A := make([][]float64, rows)
	b := make([]float64, rows)
	for i, id := range idx {
		X, Y := obj[id][0], obj[id][1]
		u, v := img[id][0], img[id][1]
		A[2*i] = []float64{X, Y, 1, 0, 0, 0, -u * X, -u * Y}
		b[2*i] = u
		A[2*i+1] = []float64{0, 0, 0, X, Y, 1, -v * X, -v * Y}
		b[2*i+1] = v
	}
	h, ok := solveLeastSquares(A, b, 8)
	if !ok {
		return H, false
	}
	H = [3][3]float64{
		{h[0], h[1], h[2]},
		{h[3], h[4], h[5]},
		{h[6], h[7], 1},
	}
	return H, true
}

// poseFromHomography decomposes a planar homography H that maps object-plane
// points (X, Y, 1) to normalised camera coordinates — i.e. H = [r1 r2 t] up to a
// common scale — into a rotation vector and translation. Because the intrinsics
// were already stripped by [normalizePoints], the columns of H are directly the
// scaled first two rotation columns and the translation.
func poseFromHomography(H [3][3]float64) (rvec, tvec [3]float64, ok bool) {
	h1 := [3]float64{H[0][0], H[1][0], H[2][0]}
	h2 := [3]float64{H[0][1], H[1][1], H[2][1]}
	h3 := [3]float64{H[0][2], H[1][2], H[2][2]}
	n1 := math.Sqrt(h1[0]*h1[0] + h1[1]*h1[1] + h1[2]*h1[2])
	n2 := math.Sqrt(h2[0]*h2[0] + h2[1]*h2[1] + h2[2]*h2[2])
	if n1 < 1e-12 || n2 < 1e-12 {
		return rvec, tvec, false
	}
	lambda := 2 / (n1 + n2)
	r1 := scale3(h1, 1/n1)
	r2 := scale3(h2, 1/n2)
	r3 := cross3(r1, r2)
	R := orthonormalize([3][3]float64{
		{r1[0], r2[0], r3[0]},
		{r1[1], r2[1], r3[1]},
		{r1[2], r2[2], r3[2]},
	})
	tvec = scale3(h3, lambda)
	if tvec[2] < 0 {
		// Enforce a camera in front of the plane.
		tvec = scale3(tvec, -1)
		R = [3][3]float64{
			{-R[0][0], -R[0][1], R[0][2]},
			{-R[1][0], -R[1][1], R[1][2]},
			{-R[2][0], -R[2][1], R[2][2]},
		}
	}
	return rotationToVec(R), tvec, true
}

// countInliers reprojects every object point with the candidate pose and counts
// those within thresh pixels of their observation.
func countInliers(obj [][3]float64, img [][2]float64, rvec, tvec [3]float64, K [3][3]float64, dist []float64, thresh float64) ([]bool, int) {
	proj := ProjectPoints(obj, rvec, tvec, K, dist)
	mask := make([]bool, len(obj))
	count := 0
	t2 := thresh * thresh
	for i := range obj {
		dx := proj[i][0] - img[i][0]
		dy := proj[i][1] - img[i][1]
		if dx*dx+dy*dy <= t2 {
			mask[i] = true
			count++
		}
	}
	return mask, count
}

// CompactPoints is a CPU-backed mirror of OpenCV's cv::cuda::compactPoints. Given
// two parallel point sets and a byte mask, it removes every index where the mask
// is zero and returns the surviving points, preserving order. It panics if the
// three slices do not share the same length.
func CompactPoints(points0, points1 [][2]float64, mask []uint8) (out0, out1 [][2]float64) {
	if len(points0) != len(points1) || len(points0) != len(mask) {
		panic("cudalegacy: CompactPoints requires equal-length inputs")
	}
	for i := range mask {
		if mask[i] != 0 {
			out0 = append(out0, points0[i])
			out1 = append(out1, points1[i])
		}
	}
	return out0, out1
}

// --- small linear-algebra helpers ---

func scale3(v [3]float64, s float64) [3]float64 {
	return [3]float64{v[0] * s, v[1] * s, v[2] * s}
}

func cross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// orthonormalize returns the closest proper rotation to R using Gram-Schmidt on
// its columns.
func orthonormalize(R [3][3]float64) [3][3]float64 {
	c0 := [3]float64{R[0][0], R[1][0], R[2][0]}
	c1 := [3]float64{R[0][1], R[1][1], R[2][1]}
	c0 = normalize3(c0)
	dot := c1[0]*c0[0] + c1[1]*c0[1] + c1[2]*c0[2]
	c1 = normalize3([3]float64{c1[0] - dot*c0[0], c1[1] - dot*c0[1], c1[2] - dot*c0[2]})
	c2 := cross3(c0, c1)
	return [3][3]float64{
		{c0[0], c1[0], c2[0]},
		{c0[1], c1[1], c2[1]},
		{c0[2], c1[2], c2[2]},
	}
}

func normalize3(v [3]float64) [3]float64 {
	n := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
	if n < 1e-12 {
		return v
	}
	return scale3(v, 1/n)
}

// solveLeastSquares solves the (possibly overdetermined) system A x = b in the
// least-squares sense for nUnknowns unknowns, via the normal equations
// AᵀA x = Aᵀb solved with Gaussian elimination and partial pivoting. It returns
// ok = false when the normal matrix is singular.
func solveLeastSquares(A [][]float64, b []float64, nUnknowns int) ([]float64, bool) {
	// Build AᵀA (n×n) and Aᵀb (n).
	n := nUnknowns
	ata := make([][]float64, n)
	atb := make([]float64, n)
	for i := 0; i < n; i++ {
		ata[i] = make([]float64, n)
	}
	for r := 0; r < len(A); r++ {
		row := A[r]
		for i := 0; i < n; i++ {
			atb[i] += row[i] * b[r]
			for j := 0; j < n; j++ {
				ata[i][j] += row[i] * row[j]
			}
		}
	}
	return gaussSolve(ata, atb)
}

// gaussSolve solves the square system M x = y in place with partial pivoting.
func gaussSolve(M [][]float64, y []float64) ([]float64, bool) {
	n := len(y)
	for col := 0; col < n; col++ {
		// Pivot.
		piv := col
		best := math.Abs(M[col][col])
		for r := col + 1; r < n; r++ {
			if math.Abs(M[r][col]) > best {
				best = math.Abs(M[r][col])
				piv = r
			}
		}
		if best < 1e-15 {
			return nil, false
		}
		M[col], M[piv] = M[piv], M[col]
		y[col], y[piv] = y[piv], y[col]
		// Eliminate.
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := M[r][col] / M[col][col]
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				M[r][c] -= f * M[col][c]
			}
			y[r] -= f * y[col]
		}
	}
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = y[i] / M[i][i]
	}
	return x, true
}

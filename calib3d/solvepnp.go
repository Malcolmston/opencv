package calib3d

import (
	"math"
	"math/rand"
)

// projectF projects 3D object points to floating-point pixel coordinates using
// the full pinhole model (rotation vector, translation, intrinsics with optional
// skew K[0][1], and Brown–Conrady distortion). Unlike [ProjectPoints] it does
// not round to integer pixels, so it is suitable for reprojection-error metrics.
func projectF(objPts [][3]float64, rvec, tvec [3]float64, K [3][3]float64, dist []float64) [][2]float64 {
	r := RodriguesToMatrix(rvec)
	k1, k2, p1, p2, k3 := distParams(dist)
	out := make([][2]float64, len(objPts))
	for i, X := range objPts {
		xc := r[0][0]*X[0] + r[0][1]*X[1] + r[0][2]*X[2] + tvec[0]
		yc := r[1][0]*X[0] + r[1][1]*X[1] + r[1][2]*X[2] + tvec[1]
		zc := r[2][0]*X[0] + r[2][1]*X[1] + r[2][2]*X[2] + tvec[2]
		if zc == 0 {
			zc = 1e-15
		}
		x := xc / zc
		y := yc / zc
		xd, yd := distortNormalized(x, y, k1, k2, p1, p2, k3)
		u := K[0][0]*xd + K[0][1]*yd + K[0][2]
		v := K[1][1]*yd + K[1][2]
		out[i] = [2]float64{u, v}
	}
	return out
}

// undistortPointIter inverts the Brown–Conrady model, recovering the ideal
// normalized coordinate (x, y) whose distortion equals (xd, yd), by fixed-point
// iteration.
func undistortPointIter(xd, yd, k1, k2, p1, p2, k3 float64) (x, y float64) {
	x, y = xd, yd
	for iter := 0; iter < 20; iter++ {
		r2 := x*x + y*y
		radial := 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
		dxT := 2*p1*x*y + p2*(r2+2*x*x)
		dyT := p1*(r2+2*y*y) + 2*p2*x*y
		if radial < 1e-12 {
			break
		}
		x = (xd - dxT) / radial
		y = (yd - dyT) / radial
	}
	return x, y
}

// undistortNormalize maps pixel points to ideal (distortion-free) normalized
// camera coordinates using the intrinsics K and distortion coefficients dist.
func undistortNormalize(imgPts [][2]float64, K [3][3]float64, dist []float64) [][2]float64 {
	kInv, ok := inv3(K)
	if !ok {
		return nil
	}
	k1, k2, p1, p2, k3 := distParams(dist)
	hasDist := k1 != 0 || k2 != 0 || p1 != 0 || p2 != 0 || k3 != 0
	out := make([][2]float64, len(imgPts))
	for i, p := range imgPts {
		v := matVec3(kInv, [3]float64{p[0], p[1], 1})
		xd, yd := v[0], v[1]
		if math.Abs(v[2]) > 1e-18 {
			xd, yd = v[0]/v[2], v[1]/v[2]
		}
		if hasDist {
			xd, yd = undistortPointIter(xd, yd, k1, k2, p1, p2, k3)
		}
		out[i] = [2]float64{xd, yd}
	}
	return out
}

// planeFit returns the centroid and unit normal of the best-fit plane through
// the points and the RMS orthogonal residual, so callers can detect coplanar
// configurations.
func planeFit(pts [][3]float64) (centroid, normal [3]float64, rms float64) {
	n := len(pts)
	for _, p := range pts {
		centroid = add3(centroid, p)
	}
	centroid = scale3(centroid, 1/float64(n))
	var cov [][]float64 = [][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
	for _, p := range pts {
		d := sub3(p, centroid)
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				cov[i][j] += d[i] * d[j]
			}
		}
	}
	nv := smallestEigvec(cov)
	normal = [3]float64{nv[0], nv[1], nv[2]}
	normal, _ = normalize3(normal)
	var s float64
	for _, p := range pts {
		d := dot3(sub3(p, centroid), normal)
		s += d * d
	}
	rms = math.Sqrt(s / float64(n))
	return centroid, normal, rms
}

// SolvePnP recovers the pose (rotation vector rvec and translation tvec) of a
// calibrated camera from 3D object points and their 2D image projections. K is
// the intrinsic matrix and dist the Brown–Conrady coefficients (nil for none),
// which are removed from the image points before pose estimation. objPts and
// imgPts must have equal length of at least four.
//
// The routine adapts to the geometry: for a coplanar object it factors the
// object-to-image homography (robust for planar targets), and for a general 3D
// object it uses the Direct Linear Transform to estimate the full projection
// matrix, then extracts the closest rotation with an SVD and the corresponding
// translation. ok is false when the input is insufficient or degenerate.
func SolvePnP(objPts [][3]float64, imgPts [][2]float64, K [3][3]float64, dist []float64) (rvec, tvec [3]float64, ok bool) {
	if len(objPts) != len(imgPts) || len(objPts) < 4 {
		return [3]float64{}, [3]float64{}, false
	}
	norm := undistortNormalize(imgPts, K, dist)
	if norm == nil {
		return [3]float64{}, [3]float64{}, false
	}
	// Measure planarity relative to the object's spatial extent.
	_, _, planeRes := planeFit(objPts)
	var extent float64
	c, _, _ := planeFit(objPts)
	for _, p := range objPts {
		if d := norm3(sub3(p, c)); d > extent {
			extent = d
		}
	}
	coplanar := len(objPts) < 6 || planeRes < 1e-6*math.Max(extent, 1e-9)
	if coplanar {
		return solvePnPPlanarGeneral(objPts, norm)
	}
	return solvePnPDLT(objPts, norm)
}

// solvePnPDLT estimates pose from ≥6 non-coplanar points using the DLT on
// normalized (calibrated) image coordinates: it solves for the 3×4 projection
// matrix [R | t] up to scale, then orthonormalises R and recovers t.
func solvePnPDLT(objPts [][3]float64, norm [][2]float64) (rvec, tvec [3]float64, ok bool) {
	rows := make([][]float64, 0, 2*len(objPts))
	for i, X := range objPts {
		x, y := norm[i][0], norm[i][1]
		rows = append(rows,
			[]float64{X[0], X[1], X[2], 1, 0, 0, 0, 0, -x * X[0], -x * X[1], -x * X[2], -x},
			[]float64{0, 0, 0, 0, X[0], X[1], X[2], 1, -y * X[0], -y * X[1], -y * X[2], -y},
		)
	}
	p := nullspaceVec(rows, 12)
	M := [3][3]float64{
		{p[0], p[1], p[2]},
		{p[4], p[5], p[6]},
		{p[8], p[9], p[10]},
	}
	t := [3]float64{p[3], p[7], p[11]}
	// Recover the scale from the rotational part.
	_, s, _ := svd3(M)
	scale := (s[0] + s[1] + s[2]) / 3
	if scale < 1e-15 {
		return [3]float64{}, [3]float64{}, false
	}
	// Orient so that reconstructed depths are positive for the object centroid.
	var cz float64
	for _, X := range objPts {
		cz += M[2][0]*X[0] + M[2][1]*X[1] + M[2][2]*X[2] + t[2]
	}
	sign := 1.0
	if cz < 0 {
		sign = -1
	}
	R := orthonormalize(scaleMat(M, sign))
	tvec = scale3(t, sign/scale)
	return RodriguesToVector(R), tvec, true
}

// solvePnPPlanarGeneral recovers pose for a coplanar object of arbitrary plane
// orientation. It builds an in-plane 2D parameterisation, fits the homography
// from those coordinates to the normalized image points, factors it into the
// in-plane rotation columns and translation, and composes the plane frame back
// out to yield the world-to-camera pose.
func solvePnPPlanarGeneral(objPts [][3]float64, norm [][2]float64) (rvec, tvec [3]float64, ok bool) {
	centroid, normal, _ := planeFit(objPts)
	// Build an orthonormal in-plane basis (u, v) completing normal.
	var seed [3]float64
	if math.Abs(normal[0]) < 0.9 {
		seed = [3]float64{1, 0, 0}
	} else {
		seed = [3]float64{0, 1, 0}
	}
	uAxis, _ := normalize3(sub3(seed, scale3(normal, dot3(seed, normal))))
	vAxis := cross3(normal, uAxis)
	B := colsToMat(uAxis, vAxis, normal) // columns u, v, n
	// 2D plane coordinates.
	src := make([][2]float64, len(objPts))
	for i, X := range objPts {
		d := sub3(X, centroid)
		src[i] = [2]float64{dot3(d, uAxis), dot3(d, vAxis)}
	}
	h, okh := dltHomography(src, norm)
	if !okh {
		return [3]float64{}, [3]float64{}, false
	}
	h1 := col3(h, 0)
	h2 := col3(h, 1)
	h3 := col3(h, 2)
	n1 := norm3(h1)
	n2 := norm3(h2)
	if n1 < 1e-15 || n2 < 1e-15 {
		return [3]float64{}, [3]float64{}, false
	}
	lambda := 2 / (n1 + n2)
	r1 := scale3(h1, lambda)
	r2 := scale3(h2, lambda)
	tp := scale3(h3, lambda)
	if tp[2] < 0 {
		r1 = scale3(r1, -1)
		r2 = scale3(r2, -1)
		tp = scale3(tp, -1)
	}
	r3 := cross3(r1, r2)
	Rp := orthonormalize(colsToMat(r1, r2, r3))
	// world→camera: R = Rp·Bᵀ, t = tp − R·centroid.
	R := mul3(Rp, transpose3(B))
	tvec = sub3(tp, matVec3(R, centroid))
	return RodriguesToVector(R), tvec, true
}

// scaleMat multiplies every entry of a 3×3 matrix by s.
func scaleMat(m [3][3]float64, s float64) [3][3]float64 {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			m[i][j] *= s
		}
	}
	return m
}

// pnpRansacSeed fixes the RANSAC seed so [SolvePnPRansac] is deterministic.
const pnpRansacSeed = 0x9e3779b9

// SolvePnPRansac robustly recovers camera pose from 3D–2D correspondences that
// may contain gross outliers, using Random Sample Consensus around [SolvePnP].
// K and dist describe the calibrated camera; reprojThresh is the maximum
// reprojection error, in pixels, for a correspondence to count as an inlier.
//
// It repeatedly fits the pose from a minimal random sample, scores it by the
// number of correspondences whose reprojection error is within reprojThresh, and
// finally refits on the largest consensus set. The returned inliers slice flags
// the agreeing correspondences. Like the other randomised routines in this
// package the sampler is seeded deterministically. ok is false when no adequate
// consensus is found.
func SolvePnPRansac(objPts [][3]float64, imgPts [][2]float64, K [3][3]float64, dist []float64, reprojThresh float64) (rvec, tvec [3]float64, inliers []bool, ok bool) {
	n := len(objPts)
	if n != len(imgPts) || n < 4 {
		return [3]float64{}, [3]float64{}, nil, false
	}
	if reprojThresh <= 0 {
		reprojThresh = 3.0
	}
	t2 := reprojThresh * reprojThresh
	sample := 6
	if sample > n {
		sample = 4
	}
	rng := rand.New(rand.NewSource(pnpRansacSeed))
	const iters = 500
	bestCount := -1
	var bestMask []bool
	for it := 0; it < iters; it++ {
		idx := sampleK(rng, n, sample)
		so := make([][3]float64, sample)
		si := make([][2]float64, sample)
		for j, id := range idx {
			so[j] = objPts[id]
			si[j] = imgPts[id]
		}
		rv, tv, okp := SolvePnP(so, si, K, dist)
		if !okp {
			continue
		}
		mask, count := scorePnP(objPts, imgPts, rv, tv, K, dist, t2)
		if count > bestCount {
			bestCount = count
			bestMask = mask
		}
	}
	if bestCount < 4 {
		return [3]float64{}, [3]float64{}, nil, false
	}
	// Refit on all inliers for a stable final estimate.
	var so [][3]float64
	var si [][2]float64
	for i := 0; i < n; i++ {
		if bestMask[i] {
			so = append(so, objPts[i])
			si = append(si, imgPts[i])
		}
	}
	rv, tv, okp := SolvePnP(so, si, K, dist)
	if !okp {
		return [3]float64{}, [3]float64{}, nil, false
	}
	mask, _ := scorePnP(objPts, imgPts, rv, tv, K, dist, t2)
	return rv, tv, mask, true
}

// scorePnP classifies correspondences as inliers by squared reprojection error.
func scorePnP(objPts [][3]float64, imgPts [][2]float64, rvec, tvec [3]float64, K [3][3]float64, dist []float64, t2 float64) ([]bool, int) {
	proj := projectF(objPts, rvec, tvec, K, dist)
	mask := make([]bool, len(objPts))
	count := 0
	for i := range objPts {
		e := sq(proj[i][0]-imgPts[i][0]) + sq(proj[i][1]-imgPts[i][1])
		if e <= t2 {
			mask[i] = true
			count++
		}
	}
	return mask, count
}

// sampleK draws k distinct indices in [0, n) from rng.
func sampleK(rng *rand.Rand, n, k int) []int {
	chosen := make(map[int]bool, k)
	out := make([]int, 0, k)
	for len(out) < k {
		v := rng.Intn(n)
		if chosen[v] {
			continue
		}
		chosen[v] = true
		out = append(out, v)
	}
	return out
}

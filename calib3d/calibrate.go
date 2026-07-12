package calib3d

import "math"

// CalibrateCamera estimates the intrinsic parameters and per-view extrinsic
// poses of a camera from several views of a planar calibration target, using
// Zhang's method. objectPoints[i] and imagePoints[i] are the model-plane points
// (with Z = 0) and their observed pixel coordinates in view i; all views share
// the same object geometry order. At least three views are required for a full
// five-parameter intrinsic solution. imageWidth and imageHeight describe the
// sensor and are accepted for API parity.
//
// The pipeline is the standard one: a homography is fitted per view; the
// image-of-the-absolute-conic constraints from all homographies are stacked and
// their null space gives the symmetric matrix B = K⁻ᵀK⁻¹, from which the
// intrinsics (including skew) are recovered in closed form; each view's
// extrinsics follow by back-substitution and rotation orthonormalisation; and a
// linear least-squares step estimates the two leading radial-distortion
// coefficients k1, k2. K is the 3×3 intrinsic matrix, dist holds
// [k1, k2, 0, 0, 0], rvecs/tvecs are the per-view rotation vectors and
// translations, and rms is the root-mean-square reprojection error in pixels.
// ok is false when the input is insufficient or the calibration is degenerate.
func CalibrateCamera(objectPoints [][][3]float64, imagePoints [][][2]float64, imageWidth, imageHeight int) (K [3][3]float64, dist []float64, rvecs, tvecs [][3]float64, rms float64, ok bool) {
	_ = imageWidth
	_ = imageHeight
	views := len(objectPoints)
	if views < 3 || views != len(imagePoints) {
		return [3][3]float64{}, nil, nil, nil, 0, false
	}
	homs := make([][3][3]float64, views)
	for i := 0; i < views; i++ {
		if len(objectPoints[i]) != len(imagePoints[i]) || len(objectPoints[i]) < 4 {
			return [3][3]float64{}, nil, nil, nil, 0, false
		}
		src := make([][2]float64, len(objectPoints[i]))
		for j, p := range objectPoints[i] {
			src[j] = [2]float64{p[0], p[1]}
		}
		h, okh := dltHomography(src, imagePoints[i])
		if !okh {
			return [3][3]float64{}, nil, nil, nil, 0, false
		}
		homs[i] = h
	}
	// Build the constraint rows on b = [B11,B12,B22,B13,B23,B33].
	vij := func(h [3][3]float64, p, q int) [6]float64 {
		return [6]float64{
			h[0][p] * h[0][q],
			h[0][p]*h[1][q] + h[1][p]*h[0][q],
			h[1][p] * h[1][q],
			h[2][p]*h[0][q] + h[0][p]*h[2][q],
			h[2][p]*h[1][q] + h[1][p]*h[2][q],
			h[2][p] * h[2][q],
		}
	}
	var rows [][]float64
	for _, h := range homs {
		v12 := vij(h, 0, 1)
		v11 := vij(h, 0, 0)
		v22 := vij(h, 1, 1)
		rows = append(rows, v12[:])
		diff := make([]float64, 6)
		for k := 0; k < 6; k++ {
			diff[k] = v11[k] - v22[k]
		}
		rows = append(rows, diff)
	}
	b := nullspaceVec(rows, 6)
	K, ok = intrinsicsFromB(b)
	if !ok {
		return [3][3]float64{}, nil, nil, nil, 0, false
	}
	kInv, _ := inv3(K)
	rvecs = make([][3]float64, views)
	tvecs = make([][3]float64, views)
	for i, h := range homs {
		r, t := extrinsicsFromH(h, kInv)
		rvecs[i] = r
		tvecs[i] = t
	}
	dist = estimateRadial(objectPoints, imagePoints, K, rvecs, tvecs)
	rms = reprojRMS(objectPoints, imagePoints, K, dist, rvecs, tvecs)
	return K, dist, rvecs, tvecs, rms, true
}

// intrinsicsFromB recovers the intrinsic matrix from the vectorised
// B = K⁻ᵀK⁻¹ = [B11,B12,B22,B13,B23,B33], resolving the global sign so the
// square roots are real.
func intrinsicsFromB(b []float64) ([3][3]float64, bool) {
	try := func(b []float64) ([3][3]float64, bool) {
		B11, B12, B22, B13, B23, B33 := b[0], b[1], b[2], b[3], b[4], b[5]
		den := B11*B22 - B12*B12
		if math.Abs(den) < 1e-30 || math.Abs(B11) < 1e-30 {
			return [3][3]float64{}, false
		}
		v0 := (B12*B13 - B11*B23) / den
		lambda := B33 - (B13*B13+v0*(B12*B13-B11*B23))/B11
		if lambda/B11 <= 0 || lambda*B11/den <= 0 {
			return [3][3]float64{}, false
		}
		alpha := math.Sqrt(lambda / B11)
		beta := math.Sqrt(lambda * B11 / den)
		gamma := -B12 * alpha * alpha * beta / lambda
		u0 := gamma*v0/beta - B13*alpha*alpha/lambda
		return [3][3]float64{{alpha, gamma, u0}, {0, beta, v0}, {0, 0, 1}}, true
	}
	if k, ok := try(b); ok {
		return k, true
	}
	nb := make([]float64, len(b))
	for i := range b {
		nb[i] = -b[i]
	}
	return try(nb)
}

// extrinsicsFromH recovers a view's rotation vector and translation from its
// homography and K⁻¹, orthonormalising the rotation.
func extrinsicsFromH(h [3][3]float64, kInv [3][3]float64) (rvec, tvec [3]float64) {
	h1 := matVec3(kInv, col3(h, 0))
	h2 := matVec3(kInv, col3(h, 1))
	h3 := matVec3(kInv, col3(h, 2))
	l := norm3(h1)
	if l < 1e-15 {
		return [3]float64{}, [3]float64{}
	}
	lambda := 1 / l
	r1 := scale3(h1, lambda)
	r2 := scale3(h2, lambda)
	t := scale3(h3, lambda)
	if t[2] < 0 {
		r1 = scale3(r1, -1)
		r2 = scale3(r2, -1)
		t = scale3(t, -1)
	}
	r3 := cross3(r1, r2)
	R := orthonormalize(colsToMat(r1, r2, r3))
	return RodriguesToVector(R), t
}

// estimateRadial fits the two leading radial-distortion coefficients k1, k2 by
// linear least squares from the difference between observed and ideal pinhole
// projections, and returns them in [k1, k2, 0, 0, 0] order.
func estimateRadial(objectPoints [][][3]float64, imagePoints [][][2]float64, K [3][3]float64, rvecs, tvecs [][3]float64) []float64 {
	u0, v0 := K[0][2], K[1][2]
	var A [][]float64
	var d []float64
	for i := range objectPoints {
		R := RodriguesToMatrix(rvecs[i])
		t := tvecs[i]
		for j, X := range objectPoints[i] {
			xc := R[0][0]*X[0] + R[0][1]*X[1] + R[0][2]*X[2] + t[0]
			yc := R[1][0]*X[0] + R[1][1]*X[1] + R[1][2]*X[2] + t[1]
			zc := R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + t[2]
			if math.Abs(zc) < 1e-12 {
				continue
			}
			x := xc / zc
			y := yc / zc
			r2 := x*x + y*y
			r4 := r2 * r2
			u := K[0][0]*x + K[0][1]*y + u0
			v := K[1][1]*y + v0
			obs := imagePoints[i][j]
			A = append(A, []float64{(u - u0) * r2, (u - u0) * r4})
			d = append(d, obs[0]-u)
			A = append(A, []float64{(v - v0) * r2, (v - v0) * r4})
			d = append(d, obs[1]-v)
		}
	}
	k, ok := leastSquares(A, d)
	if !ok {
		return []float64{0, 0, 0, 0, 0}
	}
	return []float64{k[0], k[1], 0, 0, 0}
}

// reprojRMS returns the root-mean-square reprojection error, in pixels, over all
// points of all views for the given intrinsics, distortion and per-view poses.
func reprojRMS(objectPoints [][][3]float64, imagePoints [][][2]float64, K [3][3]float64, dist []float64, rvecs, tvecs [][3]float64) float64 {
	var sum float64
	var n int
	for i := range objectPoints {
		proj := projectF(objectPoints[i], rvecs[i], tvecs[i], K, dist)
		for j := range proj {
			sum += sq(proj[j][0]-imagePoints[i][j][0]) + sq(proj[j][1]-imagePoints[i][j][1])
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return math.Sqrt(sum / float64(n))
}

// StereoCalibrate estimates the relative pose (rotation R and translation T)
// that maps points from the first camera's coordinate frame to the second's,
// from synchronized views of a common planar target. objectPoints[i] are the
// model points of view i; imagePoints1[i] and imagePoints2[i] are their pixel
// observations in the two cameras, whose intrinsics (K1, d1) and (K2, d2) are
// assumed known. At least one view is required.
//
// Each view's pose is recovered independently in both cameras with [SolvePnP];
// the per-view relative motion R2·R1ᵀ, t2 − R·t1 is averaged (rotations through
// their axis-angle vectors) to give a single rigid transform. rms is the
// root-mean-square reprojection error, in pixels, of the target in the second
// camera when its pose is predicted from the first camera through (R, T). ok is
// false when the input is inconsistent or pose recovery fails.
func StereoCalibrate(objectPoints [][][3]float64, imagePoints1, imagePoints2 [][][2]float64, K1 [3][3]float64, d1 []float64, K2 [3][3]float64, d2 []float64) (R [3][3]float64, T [3]float64, rms float64, ok bool) {
	views := len(objectPoints)
	if views < 1 || views != len(imagePoints1) || views != len(imagePoints2) {
		return [3][3]float64{}, [3]float64{}, 0, false
	}
	var sumRvec, sumT [3]float64
	R1s := make([][3][3]float64, views)
	t1s := make([][3]float64, views)
	valid := 0
	for i := 0; i < views; i++ {
		r1, tt1, ok1 := SolvePnP(objectPoints[i], imagePoints1[i], K1, d1)
		r2, tt2, ok2 := SolvePnP(objectPoints[i], imagePoints2[i], K2, d2)
		if !ok1 || !ok2 {
			continue
		}
		R1 := RodriguesToMatrix(r1)
		R2 := RodriguesToMatrix(r2)
		Rel := mul3(R2, transpose3(R1))
		Trel := sub3(tt2, matVec3(Rel, tt1))
		rvecRel := RodriguesToVector(Rel)
		sumRvec = add3(sumRvec, rvecRel)
		sumT = add3(sumT, Trel)
		R1s[i] = R1
		t1s[i] = tt1
		valid++
	}
	if valid == 0 {
		return [3][3]float64{}, [3]float64{}, 0, false
	}
	inv := 1 / float64(valid)
	R = RodriguesToMatrix(scale3(sumRvec, inv))
	T = scale3(sumT, inv)
	// RMS reprojection in camera 2 predicted through (R, T).
	var sum float64
	var n int
	for i := 0; i < views; i++ {
		if len(imagePoints2[i]) == 0 {
			continue
		}
		R2p := mul3(R, R1s[i])
		t2p := add3(matVec3(R, t1s[i]), T)
		proj := projectF(objectPoints[i], RodriguesToVector(R2p), t2p, K2, d2)
		for j := range proj {
			sum += sq(proj[j][0]-imagePoints2[i][j][0]) + sq(proj[j][1]-imagePoints2[i][j][1])
			n++
		}
	}
	if n > 0 {
		rms = math.Sqrt(sum / float64(n))
	}
	return R, T, rms, true
}

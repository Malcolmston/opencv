package ccalib

import "math"

// This file implements omnidirectional camera calibration: a linear
// pose-from-rays initialiser followed by Levenberg–Marquardt refinement of the
// full intrinsic + extrinsic parameter set. It is standard-library-only.

// poseFromRays recovers the rotation and translation of a planar target (object
// points with Z = 0) from the unit sphere rays their image points lift to. It
// solves the linear system enforcing that every camera-frame object point lies
// on its corresponding ray (dir × (R·X + t) = 0), then rescales and
// orthonormalises the result. ok is false when the geometry is degenerate.
//
// dirs[j] is the (unit) sphere direction of image point j and obj[j] its planar
// object coordinate.
func poseFromRays(obj [][3]float64, dirs [][3]float64) (rvec, tvec [3]float64, ok bool) {
	n := len(obj)
	if n < 4 || n != len(dirs) {
		return [3]float64{}, [3]float64{}, false
	}
	// Unknown vector u = [r1(3), r2(3), t(3)] (9). For Z = 0 the third rotation
	// column drops out. Each point contributes the two independent rows of the
	// skew-symmetric constraint [dir]_× · (r1·X + r2·Y + t) = 0.
	var rows [][]float64
	for j := 0; j < n; j++ {
		d := dirs[j]
		X, Y := obj[j][0], obj[j][1]
		// Skew-symmetric matrix of d applied to (r1*X + r2*Y + t).
		// Row 0: 0, -dz, dy ; Row 1: dz, 0, -dx ; Row 2: -dy, dx, 0.
		add := func(coef [3]float64) []float64 {
			row := make([]float64, 9)
			// r1 columns get coef*X, r2 columns coef*Y, t columns coef.
			for c := 0; c < 3; c++ {
				row[c] = coef[c] * X
				row[3+c] = coef[c] * Y
				row[6+c] = coef[c]
			}
			return row
		}
		rows = append(rows, add([3]float64{0, -d[2], d[1]}))
		rows = append(rows, add([3]float64{d[2], 0, -d[0]}))
	}
	sol := nullspaceVec(rows, 9)
	r1 := [3]float64{sol[0], sol[1], sol[2]}
	r2 := [3]float64{sol[3], sol[4], sol[5]}
	t := [3]float64{sol[6], sol[7], sol[8]}
	l1 := norm3(r1)
	l2 := norm3(r2)
	if l1 < 1e-12 || l2 < 1e-12 {
		return [3]float64{}, [3]float64{}, false
	}
	scale := 2 / (l1 + l2)
	r1 = scale3(r1, scale)
	r2 = scale3(r2, scale)
	t = scale3(t, scale)
	// Resolve the global sign so the target lies in front of the camera.
	if t[2] < 0 {
		r1 = scale3(r1, -1)
		r2 = scale3(r2, -1)
		t = scale3(t, -1)
	}
	r3 := cross3(r1, r2)
	R := orthonormalize(colsToMat(r1, r2, r3))
	return rodriguesToVector(R), t, true
}

// Calibrate estimates the intrinsic parameters and per-view poses of an
// omnidirectional camera from several views of a planar target, following the
// unified sphere model. objectPoints[i] holds the planar model points (Z = 0)
// of view i and imagePoints[i] their observed pixels; every view shares the same
// object geometry order. imageWidth and imageHeight give the sensor size (used
// to seed the principal point). guess supplies an initial intrinsic estimate:
// its focal lengths, principal point and Xi seed the optimiser (a rough guess,
// e.g. focal within ±20 %, is sufficient). At least three views are required.
//
// The pipeline seeds each view's pose with the linear [poseFromRays] solver
// using the current intrinsics, then jointly refines all intrinsics (Fx, Fy,
// Cx, Cy, Xi, K1, K2, P1, P2) and every pose with Levenberg–Marquardt to
// minimise the total reprojection error. model is the refined intrinsic model,
// rvecs/tvecs the per-view extrinsics and rms the root-mean-square reprojection
// error in pixels. ok is false when the input is insufficient or the estimate
// diverges.
//
// When fixXi is true the mirror parameter is held at guess.Xi instead of being
// optimised. Because the focal length and Xi are strongly coupled for planar
// targets — their product is far better constrained than either alone — fixing
// Xi to a known lens constant (the equivalent of OpenCV's CALIB_FIX_XI flag)
// makes the focal estimate robust to observation noise; leaving Xi free is only
// well-conditioned for wide-angle, low-noise data.
func (omnidirNS) Calibrate(objectPoints [][][3]float64, imagePoints [][][2]float64, imageWidth, imageHeight int, guess OmniModel, fixXi bool) (model OmniModel, rvecs, tvecs [][3]float64, rms float64, ok bool) {
	views := len(objectPoints)
	if views < 3 || views != len(imagePoints) {
		return OmniModel{}, nil, nil, 0, false
	}
	init := guess
	if init.Cx == 0 {
		init.Cx = float64(imageWidth) / 2
	}
	if init.Cy == 0 {
		init.Cy = float64(imageHeight) / 2
	}
	if init.Fx == 0 {
		init.Fx = float64(imageWidth) / 2
	}
	if init.Fy == 0 {
		init.Fy = init.Fx
	}
	// Seed per-view poses from the initial intrinsics.
	rvecs = make([][3]float64, views)
	tvecs = make([][3]float64, views)
	for i := 0; i < views; i++ {
		if len(objectPoints[i]) != len(imagePoints[i]) || len(objectPoints[i]) < 4 {
			return OmniModel{}, nil, nil, 0, false
		}
		dirs := make([][3]float64, len(imagePoints[i]))
		valid := true
		for j, p := range imagePoints[i] {
			d, okl := liftToSphere(p[0], p[1], init.Xi, init.Fx, init.Fy, init.Cx, init.Cy, init.Skew, init.K1, init.K2, init.P1, init.P2)
			if !okl {
				valid = false
				break
			}
			dirs[j] = d
		}
		if valid {
			r, t, okp := poseFromRays(objectPoints[i], dirs)
			if okp {
				rvecs[i], tvecs[i] = r, t
				continue
			}
		}
		// Fallback: fronto-parallel pose at unit depth.
		tvecs[i] = [3]float64{0, 0, 1}
	}
	// Build the parameter vector: 9 intrinsics + 6 per view.
	const nIntr = 9
	p := make([]float64, nIntr+6*views)
	p[0] = init.Fx
	p[1] = init.Fy
	p[2] = init.Cx
	p[3] = init.Cy
	p[4] = init.Xi
	p[5] = init.K1
	p[6] = init.K2
	p[7] = init.P1
	p[8] = init.P2
	for i := 0; i < views; i++ {
		base := nIntr + 6*i
		p[base+0] = rvecs[i][0]
		p[base+1] = rvecs[i][1]
		p[base+2] = rvecs[i][2]
		p[base+3] = tvecs[i][0]
		p[base+4] = tvecs[i][1]
		p[base+5] = tvecs[i][2]
	}
	residual := func(pp []float64) []float64 {
		fx, fy, cx, cy := pp[0], pp[1], pp[2], pp[3]
		xi, k1, k2, p1, p2 := pp[4], pp[5], pp[6], pp[7], pp[8]
		if fixXi {
			// Holding Xi constant zeroes its Jacobian column, so the optimiser
			// leaves it untouched.
			xi = init.Xi
		}
		var res []float64
		for i := 0; i < views; i++ {
			base := nIntr + 6*i
			r := rodriguesToMatrix([3]float64{pp[base], pp[base+1], pp[base+2]})
			t := [3]float64{pp[base+3], pp[base+4], pp[base+5]}
			for j, X := range objectPoints[i] {
				cam := add3(matVec3(r, X), t)
				u, v, okp := projectSphere(cam, xi, fx, fy, cx, cy, 0, k1, k2, p1, p2)
				if !okp {
					res = append(res, 1e3, 1e3)
					continue
				}
				res = append(res, u-imagePoints[i][j][0], v-imagePoints[i][j][1])
			}
		}
		return res
	}
	best, finalRMS := levenbergMarquardt(p, residual, 100)
	if math.IsNaN(finalRMS) || math.IsInf(finalRMS, 0) {
		return OmniModel{}, nil, nil, 0, false
	}
	xiOut := best[4]
	if fixXi {
		xiOut = init.Xi
	}
	model = OmniModel{
		Fx: best[0], Fy: best[1], Cx: best[2], Cy: best[3],
		Xi: xiOut, K1: best[5], K2: best[6], P1: best[7], P2: best[8],
	}
	for i := 0; i < views; i++ {
		base := nIntr + 6*i
		rvecs[i] = [3]float64{best[base], best[base+1], best[base+2]}
		tvecs[i] = [3]float64{best[base+3], best[base+4], best[base+5]}
	}
	return model, rvecs, tvecs, finalRMS, true
}

// solvePose recovers a single view's pose (rotation vector and translation) for
// a known omnidirectional model, by lifting the image points to sphere rays,
// seeding with [poseFromRays] and refining the six pose parameters with
// Levenberg–Marquardt. ok is false when the pose cannot be estimated.
func solvePose(obj [][3]float64, img [][2]float64, m OmniModel) (rvec, tvec [3]float64, ok bool) {
	if len(obj) != len(img) || len(obj) < 4 {
		return [3]float64{}, [3]float64{}, false
	}
	dirs := make([][3]float64, len(img))
	for j, p := range img {
		d, okl := liftToSphere(p[0], p[1], m.Xi, m.Fx, m.Fy, m.Cx, m.Cy, m.Skew, m.K1, m.K2, m.P1, m.P2)
		if !okl {
			return [3]float64{}, [3]float64{}, false
		}
		dirs[j] = d
	}
	r0, t0, okp := poseFromRays(obj, dirs)
	if !okp {
		r0, t0 = [3]float64{}, [3]float64{0, 0, 1}
	}
	p := []float64{r0[0], r0[1], r0[2], t0[0], t0[1], t0[2]}
	residual := func(pp []float64) []float64 {
		r := rodriguesToMatrix([3]float64{pp[0], pp[1], pp[2]})
		t := [3]float64{pp[3], pp[4], pp[5]}
		var res []float64
		for j, X := range obj {
			cam := add3(matVec3(r, X), t)
			u, v, okp := projectSphere(cam, m.Xi, m.Fx, m.Fy, m.Cx, m.Cy, m.Skew, m.K1, m.K2, m.P1, m.P2)
			if !okp {
				res = append(res, 1e3, 1e3)
				continue
			}
			res = append(res, u-img[j][0], v-img[j][1])
		}
		return res
	}
	best, _ := levenbergMarquardt(p, residual, 60)
	return [3]float64{best[0], best[1], best[2]}, [3]float64{best[3], best[4], best[5]}, true
}

// StereoCalibrate estimates the rigid transform (rotation R and translation T)
// mapping points from the first omnidirectional camera's frame into the
// second's, from synchronized views of a common planar target. objectPoints[i]
// are the model points of view i; imagePoints1[i] and imagePoints2[i] are their
// observations in the two cameras, whose omnidirectional models m1 and m2 are
// assumed known. At least one view is required.
//
// Each view's pose is recovered independently in both cameras with [solvePose];
// the per-view relative motions are averaged (rotations through their axis-angle
// vectors) into a single rigid transform. rms is the root-mean-square
// reprojection error, in pixels, of the target in the second camera when its
// pose is predicted from the first through (R, T). ok is false when pose
// recovery fails for every view.
func (omnidirNS) StereoCalibrate(objectPoints [][][3]float64, imagePoints1, imagePoints2 [][][2]float64, m1, m2 OmniModel) (R [3][3]float64, T [3]float64, rms float64, ok bool) {
	views := len(objectPoints)
	if views < 1 || views != len(imagePoints1) || views != len(imagePoints2) {
		return [3][3]float64{}, [3]float64{}, 0, false
	}
	var sumRvec, sumT [3]float64
	r1s := make([][3][3]float64, views)
	t1s := make([][3]float64, views)
	valid := 0
	for i := 0; i < views; i++ {
		rv1, t1, ok1 := solvePose(objectPoints[i], imagePoints1[i], m1)
		rv2, t2, ok2 := solvePose(objectPoints[i], imagePoints2[i], m2)
		if !ok1 || !ok2 {
			continue
		}
		R1 := rodriguesToMatrix(rv1)
		R2 := rodriguesToMatrix(rv2)
		rel := mul3(R2, transpose3(R1))
		trel := sub3(t2, matVec3(rel, t1))
		sumRvec = add3(sumRvec, rodriguesToVector(rel))
		sumT = add3(sumT, trel)
		r1s[i] = R1
		t1s[i] = t1
		valid++
	}
	if valid == 0 {
		return [3][3]float64{}, [3]float64{}, 0, false
	}
	inv := 1 / float64(valid)
	R = rodriguesToMatrix(scale3(sumRvec, inv))
	T = scale3(sumT, inv)
	var sum float64
	var n int
	for i := 0; i < views; i++ {
		if det3(r1s[i]) == 0 {
			continue
		}
		R2p := mul3(R, r1s[i])
		t2p := add3(matVec3(R, t1s[i]), T)
		proj := Omnidir.ProjectPoints(objectPoints[i], rodriguesToVector(R2p), t2p, m2.K(), m2.Xi, m2.Dist())
		for j := range proj {
			if math.IsNaN(proj[j][0]) {
				continue
			}
			sum += sq(proj[j][0]-imagePoints2[i][j][0]) + sq(proj[j][1]-imagePoints2[i][j][1])
			n++
		}
	}
	if n > 0 {
		rms = math.Sqrt(sum / float64(n))
	}
	return R, T, rms, true
}

package aruco

import "math"

// CalibrateCameraCharuco estimates a pinhole camera's intrinsic matrix from
// several views of a [CharucoBoard], the analogue of OpenCV's
// cv::aruco::calibrateCameraCharuco specialised to the common planar case.
// allCharucoCorners and allCharucoIds are parallel slices with one entry per
// view, each holding the interpolated ChArUco corners and their ids from
// [InterpolateCornersCharuco]; imageWidth and imageHeight give the picture size.
//
// It uses Zhang's homography-based method with the standard planar assumptions
// (square pixels, zero skew, principal point at the image centre), which leaves
// the focal length as the single unknown: each view's board-to-image homography
// contributes two linear constraints, solved together in least squares. It
// returns the intrinsic matrix k (with fx == fy and cx, cy at the image centre)
// and the RMS reprojection error in pixels. ok is false when too few
// non-degenerate views are supplied (at least two tilted views are needed) or
// the constraints are ill-conditioned.
//
// Tilted (non-fronto-parallel) views are essential: a set of fronto-parallel
// views leaves the focal length unobservable and yields ok=false.
func CalibrateCameraCharuco(allCharucoCorners [][][2]float64, allCharucoIds [][]int, board *CharucoBoard, imageWidth, imageHeight int) (k [3][3]float64, reprojErr float64, ok bool) {
	if board == nil || len(allCharucoCorners) != len(allCharucoIds) || imageWidth <= 0 || imageHeight <= 0 {
		return k, 0, false
	}
	cx := float64(imageWidth-1) / 2
	cy := float64(imageHeight-1) / 2
	m := [3][3]float64{
		{1, 0, -cx},
		{0, 1, -cy},
		{-cx, -cy, cx*cx + cy*cy},
	}

	type viewH struct{ h [3][3]float64 }
	var views []viewH
	var sumCC, sumCE float64
	for v := range allCharucoCorners {
		corners := allCharucoCorners[v]
		cids := allCharucoIds[v]
		if len(corners) != len(cids) || len(corners) < 4 {
			continue
		}
		var src, dst [][2]float64
		for i, id := range cids {
			obj, valid := board.chessboardCornerForID(id)
			if !valid {
				continue
			}
			src = append(src, [2]float64{obj[0], obj[1]})
			dst = append(dst, corners[i])
		}
		if len(src) < 4 {
			continue
		}
		h, hok := homographyFromPoints(src, dst)
		if !hok {
			continue
		}
		views = append(views, viewH{h})
		h1 := [3]float64{h[0][0], h[1][0], h[2][0]}
		h2 := [3]float64{h[0][1], h[1][1], h[2][1]}
		// Constraint 1: h1^T B h2 = 0, with B = w*M + diag(0,0,1).
		c1 := quadForm(m, h1, h2)
		e1 := h1[2] * h2[2]
		sumCC += c1 * c1
		sumCE += c1 * e1
		// Constraint 2: h1^T B h1 - h2^T B h2 = 0.
		c2 := quadForm(m, h1, h1) - quadForm(m, h2, h2)
		e2 := h1[2]*h1[2] - h2[2]*h2[2]
		sumCC += c2 * c2
		sumCE += c2 * e2
	}
	if len(views) < 2 || sumCC < 1e-12 {
		return k, 0, false
	}
	w := -sumCE / sumCC // w = 1 / f^2
	if w <= 0 {
		return k, 0, false
	}
	f := 1 / math.Sqrt(w)
	k = [3][3]float64{
		{f, 0, cx},
		{0, f, cy},
		{0, 0, 1},
	}

	// RMS reprojection error over every corner of every view.
	var sqSum float64
	var count int
	for v, view := range views {
		rvec, tvec, pok := poseFromH(view.h, k)
		if !pok {
			continue
		}
		rot := rvecToMatrix(rvec)
		corners := allCharucoCorners[v]
		cids := allCharucoIds[v]
		for i, id := range cids {
			obj, valid := board.chessboardCornerForID(id)
			if !valid {
				continue
			}
			u, vv, pk := projectPoint(rot, tvec, k, obj)
			if !pk {
				continue
			}
			du := u - corners[i][0]
			dv := vv - corners[i][1]
			sqSum += du*du + dv*dv
			count++
		}
	}
	if count > 0 {
		reprojErr = math.Sqrt(sqSum / float64(count))
	}
	return k, reprojErr, true
}

// quadForm returns a^T M b for a 3x3 matrix M and 3-vectors a, b.
func quadForm(m [3][3]float64, a, b [3]float64) float64 {
	var mb [3]float64
	for i := 0; i < 3; i++ {
		mb[i] = m[i][0]*b[0] + m[i][1]*b[1] + m[i][2]*b[2]
	}
	return a[0]*mb[0] + a[1]*mb[1] + a[2]*mb[2]
}

// rvecToMatrix converts a Rodrigues rotation vector to a 3x3 rotation matrix.
func rvecToMatrix(rvec [3]float64) [3][3]float64 {
	theta := norm3(rvec)
	if theta < 1e-12 {
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	kx, ky, kz := rvec[0]/theta, rvec[1]/theta, rvec[2]/theta
	c := math.Cos(theta)
	s := math.Sin(theta)
	v := 1 - c
	return [3][3]float64{
		{c + kx*kx*v, kx*ky*v - kz*s, kx*kz*v + ky*s},
		{ky*kx*v + kz*s, c + ky*ky*v, ky*kz*v - kx*s},
		{kz*kx*v - ky*s, kz*ky*v + kx*s, c + kz*kz*v},
	}
}

// projectPoint projects a board point (with the given rotation matrix, camera
// translation and intrinsics) to pixel coordinates. ok is false when the point
// falls on or behind the camera plane.
func projectPoint(rot [3][3]float64, tvec [3]float64, k [3][3]float64, obj [3]float64) (u, v float64, ok bool) {
	xc := rot[0][0]*obj[0] + rot[0][1]*obj[1] + rot[0][2]*obj[2] + tvec[0]
	yc := rot[1][0]*obj[0] + rot[1][1]*obj[1] + rot[1][2]*obj[2] + tvec[1]
	zc := rot[2][0]*obj[0] + rot[2][1]*obj[1] + rot[2][2]*obj[2] + tvec[2]
	if zc <= 1e-9 {
		return 0, 0, false
	}
	u = k[0][0]*xc/zc + k[0][2]
	v = k[1][1]*yc/zc + k[1][2]
	return u, v, true
}

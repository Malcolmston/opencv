package aruco

import (
	cv "github.com/malcolmston/opencv"
)

// UndistortImagePoints removes lens distortion from a set of pixel observations
// given the camera intrinsic matrix k and Brown-Conrady distortion coefficients
// dist, returning ideal (pinhole) pixel coordinates. dist is interpreted as
// [k1, k2, p1, p2, k3]; a shorter slice has its missing trailing coefficients
// treated as zero and nil means no distortion (the points are returned
// unchanged). Each point is undistorted by fixed-point iteration on the
// distortion model, matching OpenCV's cv::undistortPoints with P=K.
func UndistortImagePoints(points [][2]float64, k [3][3]float64, dist []float64) [][2]float64 {
	out := make([][2]float64, len(points))
	fx, fy := k[0][0], k[1][1]
	cx, cy := k[0][2], k[1][2]
	if len(dist) == 0 || fx == 0 || fy == 0 {
		copy(out, points)
		return out
	}
	var k1, k2, p1, p2, k3 float64
	get := func(i int) float64 {
		if i < len(dist) {
			return dist[i]
		}
		return 0
	}
	k1, k2, p1, p2, k3 = get(0), get(1), get(2), get(3), get(4)
	for i, pt := range points {
		x := (pt[0] - cx) / fx
		y := (pt[1] - cy) / fy
		x0, y0 := x, y
		for iter := 0; iter < 8; iter++ {
			r2 := x0*x0 + y0*y0
			radial := 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
			dxT := 2*p1*x0*y0 + p2*(r2+2*x0*x0)
			dyT := p1*(r2+2*y0*y0) + 2*p2*x0*y0
			x0 = (x - dxT) / radial
			y0 = (y - dyT) / radial
		}
		out[i] = [2]float64{x0*fx + cx, y0*fy + cy}
	}
	return out
}

// EstimatePoseBoard recovers a single camera pose for a whole [GridBoard] from
// however many of its markers were detected. corners and ids are the parallel
// slices returned by [DetectMarkers]; only markers whose id belongs to board are
// used. k is the 3x3 pinhole intrinsic matrix and dist holds Brown-Conrady
// distortion coefficients ([k1, k2, p1, p2, k3]); pass nil when the image is
// already rectified, in which case the corners are used directly.
//
// It returns a Rodrigues rotation vector rvec and translation tvec placing the
// board frame in the camera frame (a board point X projects as R(rvec)*X+tvec),
// together with markersUsed, the number of board markers that contributed. When
// distortion is supplied the corners are undistorted first, so the pose is a
// true pinhole solution. markersUsed is zero (and the vectors are zero) when no
// board marker was detected or the correspondences are degenerate.
//
// Because every detected marker contributes four correspondences, the board pose
// is far more robust to per-corner noise than any single marker's pose.
func EstimatePoseBoard(corners [][4]cv.Point, ids []int, board *GridBoard, k [3][3]float64, dist []float64) (rvec, tvec [3]float64, markersUsed int) {
	if board == nil {
		return rvec, tvec, 0
	}
	var src, dst [][2]float64
	for i, id := range ids {
		if i >= len(corners) {
			break
		}
		obj, ok := board.objectCornersForID(id)
		if !ok {
			continue
		}
		markersUsed++
		for j := 0; j < 4; j++ {
			src = append(src, [2]float64{obj[j][0], obj[j][1]})
			dst = append(dst, [2]float64{float64(corners[i][j].X), float64(corners[i][j].Y)})
		}
	}
	if markersUsed == 0 {
		return rvec, tvec, 0
	}
	dst = UndistortImagePoints(dst, k, dist)
	h, ok := homographyFromPoints(src, dst)
	if !ok {
		return rvec, tvec, 0
	}
	rvec, tvec, ok = poseFromH(h, k)
	if !ok {
		return [3]float64{}, [3]float64{}, markersUsed
	}
	return rvec, tvec, markersUsed
}

// EstimatePoseCharucoBoard recovers the camera pose from interpolated ChArUco
// chessboard corners (as returned by [InterpolateCornersCharuco]).
// charucoCorners are subpixel image points and charucoIds are their board corner
// identifiers; k and dist are the intrinsics and distortion coefficients (dist
// may be nil). It returns the board-to-camera Rodrigues rotation and translation
// and ok=false when fewer than four non-collinear corners are available or the
// solve is degenerate.
//
// ChArUco corners are localised far more precisely than raw marker corners, so
// this pose is typically the most accurate the package offers.
func EstimatePoseCharucoBoard(charucoCorners [][2]float64, charucoIds []int, board *CharucoBoard, k [3][3]float64, dist []float64) (rvec, tvec [3]float64, ok bool) {
	if board == nil || len(charucoCorners) != len(charucoIds) || len(charucoCorners) < 4 {
		return rvec, tvec, false
	}
	var src, dst [][2]float64
	for i, id := range charucoIds {
		obj, valid := board.chessboardCornerForID(id)
		if !valid {
			continue
		}
		src = append(src, [2]float64{obj[0], obj[1]})
		dst = append(dst, charucoCorners[i])
	}
	if len(src) < 4 {
		return rvec, tvec, false
	}
	dst = UndistortImagePoints(dst, k, dist)
	h, ok := homographyFromPoints(src, dst)
	if !ok {
		return rvec, tvec, false
	}
	return poseFromH(h, k)
}

// EstimatePoseSingleMarkersWithDistortion is a distortion-aware companion to
// [EstimatePoseSingleMarkers]. It undistorts each marker's corners with the
// Brown-Conrady coefficients dist ([k1, k2, p1, p2, k3]) before the planar solve,
// so the recovered pose is correct for a lens with real distortion; passing nil
// dist reproduces the undistorted estimator. Arguments and return values
// otherwise match [EstimatePoseSingleMarkers].
func EstimatePoseSingleMarkersWithDistortion(corners [][4]cv.Point, markerLength float64, k [3][3]float64, dist []float64) (rvecs, tvecs [][3]float64) {
	half := markerLength / 2
	obj := [][2]float64{
		{-half, half},
		{half, half},
		{half, -half},
		{-half, -half},
	}
	rvecs = make([][3]float64, len(corners))
	tvecs = make([][3]float64, len(corners))
	for i, quad := range corners {
		img := [][2]float64{
			{float64(quad[0].X), float64(quad[0].Y)},
			{float64(quad[1].X), float64(quad[1].Y)},
			{float64(quad[2].X), float64(quad[2].Y)},
			{float64(quad[3].X), float64(quad[3].Y)},
		}
		img = UndistortImagePoints(img, k, dist)
		h, ok := homographyFromPoints(obj, img)
		if !ok {
			continue
		}
		if rvec, tvec, ok := poseFromH(h, k); ok {
			rvecs[i] = rvec
			tvecs[i] = tvec
		}
	}
	return rvecs, tvecs
}

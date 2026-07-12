package calib3d

import "math"

// StereoRectify computes the rectification rotations and rectified projection
// matrices that make a calibrated stereo pair row-aligned, so that a physical
// point projects to the same image row in both rectified images and stereo
// matching reduces to a horizontal search. K1/d1 and K2/d2 are the intrinsics
// and distortion of the two cameras, imageSize is (width, height), and (R, T)
// is the rigid transform mapping points from the first camera frame to the
// second (Xc2 = R·Xc1 + T).
//
// It uses Bouguet's construction: the new x-axis is aligned with the baseline,
// giving a common rectified orientation Q; R1 = Q rectifies the first camera and
// R2 = Q·Rᵀ rectifies the second. The rectified pixel of a point is
// Knew·(Ri·Xci) divided by depth, which shares its row across the two cameras.
// P1 and P2 are the 3×4 rectified projection matrices (P2 carries the baseline
// term fx·Tx in its last column) and Q is the 4×4 disparity-to-depth
// reprojection matrix. The shared new intrinsics Knew are taken from the first
// camera.
func StereoRectify(K1 [3][3]float64, d1 []float64, K2 [3][3]float64, d2 []float64, imageSize [2]int, R [3][3]float64, T [3]float64) (R1, R2 [3][3]float64, P1, P2 [3][4]float64, Q [4][4]float64) {
	_ = d1
	_ = d2
	// Baseline vector in the first camera frame and its length.
	rt := transpose3(R)
	bvec := scale3(matVec3(rt, T), -1) // camera-2 center in camera-1 frame
	e1, baseline := normalize3(bvec)
	if baseline < 1e-15 {
		// Degenerate (coincident centers): fall back to identity rectification.
		e1 = [3]float64{1, 0, 0}
	}
	// Second axis perpendicular to e1 and the optical axis.
	axis := [3]float64{0, 0, 1}
	e2 := cross3(axis, e1)
	if norm3(e2) < 1e-9 {
		e2 = cross3([3]float64{0, 1, 0}, e1)
	}
	e2, _ = normalize3(e2)
	e3 := cross3(e1, e2)
	// Q has the new axes as its rows (world/cam1 -> rectified).
	Qrect := [3][3]float64{
		{e1[0], e1[1], e1[2]},
		{e2[0], e2[1], e2[2]},
		{e3[0], e3[1], e3[2]},
	}
	R1 = Qrect
	R2 = mul3(Qrect, rt)

	fx := K1[0][0]
	fy := K1[1][1]
	cx := K1[0][2]
	cy := K1[1][2]
	Tx := -baseline

	P1 = [3][4]float64{
		{fx, 0, cx, 0},
		{0, fy, cy, 0},
		{0, 0, 1, 0},
	}
	P2 = [3][4]float64{
		{fx, 0, cx, fx * Tx},
		{0, fy, cy, 0},
		{0, 0, 1, 0},
	}
	// Disparity-to-depth mapping: [X Y Z W]ᵀ = Q·[u v disparity 1]ᵀ.
	invTx := 0.0
	if math.Abs(Tx) > 1e-15 {
		invTx = 1 / Tx
	}
	Q = [4][4]float64{
		{1, 0, 0, -cx},
		{0, 1, 0, -cy},
		{0, 0, 0, fx},
		{0, 0, -invTx, 0},
	}
	return R1, R2, P1, P2, Q
}

// GetOptimalNewCameraMatrix computes a new camera matrix that controls how much
// of the distorted image is retained after undistortion, via the free-scaling
// parameter alpha. K and dist describe the camera, imageSize is (width, height).
// With alpha = 0 the result is scaled so that only valid (source-covered) pixels
// remain, cropping the black border introduced by undistortion; with alpha = 1
// all source pixels are retained, leaving black regions at the edges. Values in
// between interpolate. It also returns the pixel ROI (x, y, width, height) of the
// all-valid region.
//
// The border of the image is undistorted to trace the valid image boundary, from
// which an inner (fully-covered) and outer (all-inclusive) rectangle are formed;
// alpha blends between them and the new focal lengths and principal point map the
// selected rectangle onto the output image. When the camera has no distortion the
// input matrix is returned unchanged with a full-frame ROI.
func GetOptimalNewCameraMatrix(K [3][3]float64, dist []float64, imageSize [2]int, alpha float64) (newK [3][3]float64, roi [4]int) {
	w, h := imageSize[0], imageSize[1]
	k1, k2, p1, p2, k3 := distParams(dist)
	if k1 == 0 && k2 == 0 && p1 == 0 && p2 == 0 && k3 == 0 {
		return K, [4]int{0, 0, w, h}
	}
	fx, fy, cx, cy := K[0][0], K[1][1], K[0][2], K[1][2]
	// Undistort the image border to pixel coordinates on the ideal plane.
	const nPer = 32
	undist := func(u, v float64) (float64, float64) {
		xd := (u - cx) / fx
		yd := (v - cy) / fy
		x, y := undistortPointIter(xd, yd, k1, k2, p1, p2, k3)
		return fx*x + cx, fy*y + cy
	}
	outMinX, outMinY := math.Inf(1), math.Inf(1)
	outMaxX, outMaxY := math.Inf(-1), math.Inf(-1)
	innerLeft, innerRight := math.Inf(-1), math.Inf(1)
	innerTop, innerBottom := math.Inf(-1), math.Inf(1)
	track := func(u, v float64) (float64, float64) {
		uu, vv := undist(u, v)
		outMinX = math.Min(outMinX, uu)
		outMinY = math.Min(outMinY, vv)
		outMaxX = math.Max(outMaxX, uu)
		outMaxY = math.Max(outMaxY, vv)
		return uu, vv
	}
	fw, fh := float64(w-1), float64(h-1)
	for i := 0; i <= nPer; i++ {
		f := float64(i) / nPer
		// Top and bottom edges.
		tu, _ := track(f*fw, 0)
		bu, _ := track(f*fw, fh)
		innerTop = math.Max(innerTop, mustY(undist, f*fw, 0))
		innerBottom = math.Min(innerBottom, mustY(undist, f*fw, fh))
		_ = tu
		_ = bu
		// Left and right edges.
		_, _ = track(0, f*fh)
		_, _ = track(fw, f*fh)
		lx, _ := undist(0, f*fh)
		rx, _ := undist(fw, f*fh)
		innerLeft = math.Max(innerLeft, lx)
		innerRight = math.Min(innerRight, rx)
	}
	blend := func(inner, outer float64) float64 { return inner*(1-alpha) + outer*alpha }
	rx0 := blend(innerLeft, outMinX)
	ry0 := blend(innerTop, outMinY)
	rx1 := blend(innerRight, outMaxX)
	ry1 := blend(innerBottom, outMaxY)
	rw := rx1 - rx0
	rh := ry1 - ry0
	if rw < 1e-6 || rh < 1e-6 {
		return K, [4]int{0, 0, w, h}
	}
	sx := float64(w) / rw
	sy := float64(h) / rh
	newK = [3][3]float64{
		{fx * sx, 0, (cx - rx0) * sx},
		{0, fy * sy, (cy - ry0) * sy},
		{0, 0, 1},
	}
	// Valid ROI: the inner rectangle mapped into the output frame.
	ix0 := int(math.Ceil((innerLeft - rx0) * sx))
	iy0 := int(math.Ceil((innerTop - ry0) * sy))
	ix1 := int(math.Floor((innerRight - rx0) * sx))
	iy1 := int(math.Floor((innerBottom - ry0) * sy))
	if ix0 < 0 {
		ix0 = 0
	}
	if iy0 < 0 {
		iy0 = 0
	}
	if ix1 > w {
		ix1 = w
	}
	if iy1 > h {
		iy1 = h
	}
	roi = [4]int{ix0, iy0, maxIntV(ix1-ix0, 0), maxIntV(iy1-iy0, 0)}
	return newK, roi
}

// mustY returns only the v component of an undistortion sample.
func mustY(undist func(u, v float64) (float64, float64), u, v float64) float64 {
	_, yy := undist(u, v)
	return yy
}

func maxIntV(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// InitUndistortRectifyMap builds the per-pixel remap tables that jointly undo
// lens distortion and apply a rectification rotation. K and dist are the source
// camera parameters, R is the rectification rotation (identity for pure
// undistortion), newK is the desired output intrinsic matrix, and imageSize is
// the output size (width, height). The returned mapX and mapY are indexed
// [row][col]: output pixel (col, row) is produced by sampling the source image at
// the floating-point location (mapX[row][col], mapY[row][col]).
//
// For each output pixel the function back-projects through newK⁻¹, undoes the
// rectification with Rᵀ, applies the forward distortion model, and re-projects
// through K to find the matching source location — the inverse mapping consumed
// by a bilinear remap.
func InitUndistortRectifyMap(K [3][3]float64, dist []float64, R [3][3]float64, newK [3][3]float64, imageSize [2]int) (mapX, mapY [][]float64) {
	w, h := imageSize[0], imageSize[1]
	k1, k2, p1, p2, k3 := distParams(dist)
	fx, fy, cx, cy := K[0][0], K[1][1], K[0][2], K[1][2]
	newKInv, ok := inv3(newK)
	if !ok {
		newKInv = [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	rt := transpose3(R)
	mapX = make([][]float64, h)
	mapY = make([][]float64, h)
	for v := 0; v < h; v++ {
		mapX[v] = make([]float64, w)
		mapY[v] = make([]float64, w)
		for u := 0; u < w; u++ {
			ray := matVec3(rt, matVec3(newKInv, [3]float64{float64(u), float64(v), 1}))
			z := ray[2]
			if math.Abs(z) < 1e-15 {
				z = 1e-15
			}
			x := ray[0] / z
			y := ray[1] / z
			xd, yd := distortNormalized(x, y, k1, k2, p1, p2, k3)
			mapX[v][u] = fx*xd + cx
			mapY[v][u] = fy*yd + cy
		}
	}
	return mapX, mapY
}

package calib3d

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// matMaxDiff returns the largest absolute entrywise difference of two matrices.
func matMaxDiff(a, b [3][3]float64) float64 {
	var m float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if d := math.Abs(a[i][j] - b[i][j]); d > m {
				m = d
			}
		}
	}
	return m
}

func TestConvertPointsHomogeneousRoundTrip(t *testing.T) {
	pts := [][2]float64{{1, 2}, {-3, 4.5}, {0, 0}}
	h := ConvertPointsToHomogeneous(pts)
	for i, p := range h {
		if p[0] != pts[i][0] || p[1] != pts[i][1] || p[2] != 1 {
			t.Errorf("to-homogeneous %d: got %v", i, p)
		}
	}
	// Scale each homogeneous point; perspective division must undo it.
	for i := range h {
		s := float64(i + 2)
		h[i] = [3]float64{h[i][0] * s, h[i][1] * s, h[i][2] * s}
	}
	back := ConvertPointsFromHomogeneous(h)
	for i := range back {
		if math.Abs(back[i][0]-pts[i][0]) > 1e-12 || math.Abs(back[i][1]-pts[i][1]) > 1e-12 {
			t.Errorf("from-homogeneous %d: got %v want %v", i, back[i], pts[i])
		}
	}
}

// stereoScene builds a spread of 3D points and their float projections in two
// calibrated cameras: camera 1 is the reference, camera 2 has pose (rvec, tcam).
func stereoScene(K [3][3]float64, rvec, tcam [3]float64) (world [][3]float64, p1, p2 [][2]float64) {
	for gx := -2; gx <= 2; gx++ {
		for gy := -2; gy <= 2; gy++ {
			depth := 6.0 + 0.5*float64(gx) - 0.3*float64(gy) + 0.2*float64(gx*gy)
			world = append(world, [3]float64{float64(gx) * 0.8, float64(gy) * 0.8, depth})
		}
	}
	p1 = projectF(world, [3]float64{}, [3]float64{}, K, nil)
	p2 = projectF(world, rvec, tcam, K, nil)
	return world, p1, p2
}

func TestComputeCorrespondEpilines(t *testing.T) {
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	rvec := [3]float64{0.03, 0.08, -0.02}
	tcam := [3]float64{-1.4, 0.2, 0.15}
	_, p1, p2 := stereoScene(K, rvec, tcam)

	F, ok := eightPointF(p1, p2)
	if !ok {
		t.Fatal("eightPointF failed")
	}
	// Lines in image 2 from points in image 1: the matching point must lie on it.
	lines := ComputeCorrespondEpilines(p1, Image1, F)
	var maxd float64
	for i := range lines {
		l := lines[i]
		d := math.Abs(l[0]*p2[i][0] + l[1]*p2[i][1] + l[2])
		if d > maxd {
			maxd = d
		}
		if n := math.Hypot(l[0], l[1]); math.Abs(n-1) > 1e-9 {
			t.Errorf("line %d not normalised: |n|=%g", i, n)
		}
	}
	if maxd > 1.5 {
		t.Errorf("point-to-epiline distance too large: %g px", maxd)
	}
}

func TestFindEssentialMatRecoverPose(t *testing.T) {
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	rvec := [3]float64{0.05, 0.1, -0.04}
	trueR := RodriguesToMatrix(rvec)
	tcam := [3]float64{-1.5, 0.3, 0.2}
	_, p1, p2 := stereoScene(K, rvec, tcam)

	E, ok := FindEssentialMat(p1, p2, K)
	if !ok {
		t.Fatal("FindEssentialMat failed")
	}
	R, tdir, good := RecoverPose(E, p1, p2, K)
	if good < 20 {
		t.Errorf("cheirality inliers too few: %d", good)
	}
	if d := matMaxDiff(R, trueR); d > 0.02 {
		t.Errorf("recovered R off by %g", d)
	}
	// Translation is recovered up to scale; compare unit directions.
	tu, _ := normalize3(tcam)
	ru, _ := normalize3(tdir)
	dot := math.Abs(dot3(tu, ru))
	if dot < 0.999 {
		t.Errorf("translation direction off: |cos|=%g (t=%v recovered=%v)", dot, tu, ru)
	}
}

func TestDecomposeEssentialMatProper(t *testing.T) {
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	rvec := [3]float64{0.1, -0.05, 0.07}
	_, p1, p2 := stereoScene(K, rvec, [3]float64{-1.2, 0.1, 0.05})
	E, ok := FindEssentialMat(p1, p2, K)
	if !ok {
		t.Fatal("FindEssentialMat failed")
	}
	R1, R2, tt := DecomposeEssentialMat(E)
	if d := math.Abs(det3(R1) - 1); d > 1e-6 {
		t.Errorf("det(R1)=%g, want 1", det3(R1))
	}
	if d := math.Abs(det3(R2) - 1); d > 1e-6 {
		t.Errorf("det(R2)=%g, want 1", det3(R2))
	}
	if l := norm3(tt); math.Abs(l-1) > 1e-6 {
		t.Errorf("|t|=%g, want unit", l)
	}
}

func TestDecomposeHomographyMat(t *testing.T) {
	K := CameraMatrix{Fx: 600, Fy: 600, Cx: 320, Cy: 240}.Matrix()
	trueR := RodriguesToMatrix([3]float64{0.08, -0.06, 0.05})
	trueN := [3]float64{0, 0, 1}
	trueT := [3]float64{0.25, 0.1, 0.15} // t/d with d = 1
	// Euclidean homography He = R + t·nᵀ, then to pixel space.
	var He [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			He[i][j] = trueR[i][j] + trueT[i]*trueN[j]
		}
	}
	kInv, _ := inv3(K)
	H := mul3(K, mul3(He, kInv))

	rots, trans, norms, ok := DecomposeHomographyMat(H, K)
	if !ok {
		t.Fatal("DecomposeHomographyMat failed")
	}
	found := false
	for i := range rots {
		if matMaxDiff(rots[i], trueR) < 1e-3 &&
			vecClose(norms[i], trueN, 1e-2) &&
			math.Abs(trans[i][0]-trueT[0]) < 1e-2 &&
			math.Abs(trans[i][1]-trueT[1]) < 1e-2 &&
			math.Abs(trans[i][2]-trueT[2]) < 1e-2 {
			found = true
		}
	}
	if !found {
		t.Errorf("ground-truth decomposition not among %d solutions", len(rots))
	}
}

// pnpScene builds a non-coplanar 3D object and its float projection under a pose.
func pnpScene(K [3][3]float64, rvec, tvec [3]float64, dist []float64) (obj [][3]float64, img [][2]float64) {
	for gx := -1; gx <= 1; gx++ {
		for gy := -1; gy <= 1; gy++ {
			for gz := 0; gz <= 1; gz++ {
				obj = append(obj, [3]float64{float64(gx), float64(gy), float64(gz) * 0.7})
			}
		}
	}
	img = projectF(obj, rvec, tvec, K, dist)
	return obj, img
}

func TestSolvePnPGeneral(t *testing.T) {
	K := CameraMatrix{Fx: 520, Fy: 520, Cx: 320, Cy: 240}.Matrix()
	trueR := [3]float64{0.2, -0.15, 0.1}
	trueT := [3]float64{0.3, -0.2, 7}
	obj, img := pnpScene(K, trueR, trueT, nil)

	rvec, tvec, ok := SolvePnP(obj, img, K, nil)
	if !ok {
		t.Fatal("SolvePnP failed")
	}
	reproj := projectF(obj, rvec, tvec, K, nil)
	var maxe float64
	for i := range img {
		e := math.Hypot(reproj[i][0]-img[i][0], reproj[i][1]-img[i][1])
		if e > maxe {
			maxe = e
		}
	}
	if maxe > 1e-3 {
		t.Errorf("reprojection error too large: %g px", maxe)
	}
	if math.Abs(tvec[0]-trueT[0]) > 1e-2 || math.Abs(tvec[1]-trueT[1]) > 1e-2 || math.Abs(tvec[2]-trueT[2]) > 1e-2 {
		t.Errorf("t=%v, want %v", tvec, trueT)
	}
}

func TestSolvePnPPlanarViaSolvePnP(t *testing.T) {
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	var obj [][3]float64
	for gx := -2; gx <= 2; gx++ {
		for gy := -2; gy <= 2; gy++ {
			obj = append(obj, [3]float64{float64(gx), float64(gy), 0})
		}
	}
	trueR := [3]float64{0.15, -0.1, 0.05}
	trueT := [3]float64{0.2, -0.1, 8}
	img := projectF(obj, trueR, trueT, K, nil)
	rvec, tvec, ok := SolvePnP(obj, img, K, nil)
	if !ok {
		t.Fatal("SolvePnP (planar) failed")
	}
	reproj := projectF(obj, rvec, tvec, K, nil)
	var maxe float64
	for i := range img {
		e := math.Hypot(reproj[i][0]-img[i][0], reproj[i][1]-img[i][1])
		if e > maxe {
			maxe = e
		}
	}
	if maxe > 1e-3 {
		t.Errorf("planar reprojection error too large: %g px", maxe)
	}
}

func TestSolvePnPRansacRejectsOutliers(t *testing.T) {
	K := CameraMatrix{Fx: 520, Fy: 520, Cx: 320, Cy: 240}.Matrix()
	trueR := [3]float64{0.1, -0.2, 0.05}
	trueT := [3]float64{0.4, 0.1, 8}
	var obj [][3]float64
	for gx := -2; gx <= 2; gx++ {
		for gy := -2; gy <= 2; gy++ {
			for gz := 0; gz <= 1; gz++ {
				obj = append(obj, [3]float64{float64(gx) * 0.6, float64(gy) * 0.6, float64(gz)})
			}
		}
	}
	img := projectF(obj, trueR, trueT, K, nil)
	outliers := map[int]bool{3: true, 12: true, 25: true, 40: true}
	for i := range img {
		if outliers[i] {
			img[i][0] += 60
			img[i][1] -= 45
		}
	}
	rvec, tvec, inliers, ok := SolvePnPRansac(obj, img, K, nil, 2.0)
	if !ok {
		t.Fatal("SolvePnPRansac failed")
	}
	for i := range inliers {
		if outliers[i] && inliers[i] {
			t.Errorf("outlier %d flagged inlier", i)
		}
		if !outliers[i] && !inliers[i] {
			t.Errorf("inlier %d flagged outlier", i)
		}
	}
	if math.Abs(tvec[2]-trueT[2]) > 0.1 {
		t.Errorf("recovered t=%v, want %v", tvec, trueT)
	}
	// Determinism.
	rvec2, tvec2, _, _ := SolvePnPRansac(obj, img, K, nil, 2.0)
	if rvec != rvec2 || tvec != tvec2 {
		t.Error("SolvePnPRansac is not deterministic")
	}
}

// calibViews synthesises several views of a planar target for calibration tests.
func calibViews(K [3][3]float64, dist []float64) (obj [][][3]float64, img [][][2]float64) {
	var board [][3]float64
	for gx := -3; gx <= 3; gx++ {
		for gy := -3; gy <= 3; gy++ {
			board = append(board, [3]float64{float64(gx), float64(gy), 0})
		}
	}
	rvecs := [][3]float64{
		{0.12, 0.0, 0.0}, {0.0, 0.15, 0.0}, {0.1, 0.1, 0.02},
		{-0.12, 0.06, 0.03}, {0.05, -0.13, -0.02}, {0.18, -0.09, 0.05},
	}
	tvecs := [][3]float64{
		{0.2, 0.1, 9}, {-0.3, 0.2, 10}, {0.1, -0.2, 8.5},
		{0.0, 0.3, 9.5}, {-0.2, -0.1, 8}, {0.25, 0.05, 10.5},
	}
	for i := range rvecs {
		obj = append(obj, board)
		img = append(img, projectF(board, rvecs[i], tvecs[i], K, dist))
	}
	return obj, img
}

func TestCalibrateCameraRecoversIntrinsics(t *testing.T) {
	trueK := CameraMatrix{Fx: 800, Fy: 810, Cx: 320, Cy: 244}.Matrix()
	obj, img := calibViews(trueK, nil)

	K, dist, rvecs, tvecs, rms, ok := CalibrateCamera(obj, img, 640, 480)
	if !ok {
		t.Fatal("CalibrateCamera failed")
	}
	if len(rvecs) != len(obj) || len(tvecs) != len(obj) {
		t.Fatalf("expected %d poses, got %d/%d", len(obj), len(rvecs), len(tvecs))
	}
	if math.Abs(K[0][0]-trueK[0][0]) > 8 || math.Abs(K[1][1]-trueK[1][1]) > 8 {
		t.Errorf("focal off: got fx=%g fy=%g want %g/%g", K[0][0], K[1][1], trueK[0][0], trueK[1][1])
	}
	if math.Abs(K[0][2]-trueK[0][2]) > 5 || math.Abs(K[1][2]-trueK[1][2]) > 5 {
		t.Errorf("principal point off: got (%g,%g) want (%g,%g)", K[0][2], K[1][2], trueK[0][2], trueK[1][2])
	}
	if rms > 0.5 {
		t.Errorf("RMS reprojection error too large: %g px", rms)
	}
	if len(dist) != 5 {
		t.Errorf("expected 5 distortion coeffs, got %d", len(dist))
	}
}

func TestStereoCalibrateRecoversExtrinsics(t *testing.T) {
	K1 := CameraMatrix{Fx: 700, Fy: 700, Cx: 320, Cy: 240}.Matrix()
	K2 := CameraMatrix{Fx: 710, Fy: 710, Cx: 322, Cy: 238}.Matrix()
	trueRvec := [3]float64{0.02, -0.06, 0.01}
	trueR := RodriguesToMatrix(trueRvec)
	trueT := [3]float64{-1.6, 0.05, 0.1}

	var board [][3]float64
	for gx := -3; gx <= 3; gx++ {
		for gy := -3; gy <= 3; gy++ {
			board = append(board, [3]float64{float64(gx) * 0.5, float64(gy) * 0.5, 0})
		}
	}
	poses := [][2][3]float64{
		{{0.1, 0.05, 0.0}, {0.1, 0.0, 7}},
		{{-0.08, 0.12, 0.02}, {-0.2, 0.1, 8}},
		{{0.14, -0.1, 0.03}, {0.15, -0.15, 7.5}},
		{{0.0, 0.18, -0.02}, {0.0, 0.2, 8.5}},
	}
	var obj [][][3]float64
	var img1, img2 [][][2]float64
	for _, pose := range poses {
		r1, t1 := pose[0], pose[1]
		R1 := RodriguesToMatrix(r1)
		R2 := mul3(trueR, R1)
		t2 := add3(matVec3(trueR, t1), trueT)
		obj = append(obj, board)
		img1 = append(img1, projectF(board, r1, t1, K1, nil))
		img2 = append(img2, projectF(board, RodriguesToVector(R2), t2, K2, nil))
	}

	R, T, rms, ok := StereoCalibrate(obj, img1, img2, K1, nil, K2, nil)
	if !ok {
		t.Fatal("StereoCalibrate failed")
	}
	if d := matMaxDiff(R, trueR); d > 0.01 {
		t.Errorf("relative R off by %g", d)
	}
	for i := 0; i < 3; i++ {
		if math.Abs(T[i]-trueT[i]) > 0.05 {
			t.Errorf("relative T off: got %v want %v", T, trueT)
			break
		}
	}
	if rms > 0.5 {
		t.Errorf("stereo RMS too large: %g px", rms)
	}
}

func TestStereoRectifyRowAlignment(t *testing.T) {
	K := CameraMatrix{Fx: 600, Fy: 600, Cx: 320, Cy: 240}.Matrix()
	R := RodriguesToMatrix([3]float64{0.01, 0.03, -0.008})
	T := [3]float64{-1.5, 0.02, 0.05}

	R1, R2, P1, _, _ := StereoRectify(K, nil, K, nil, [2]int{640, 480}, R, T)
	fx := P1[0][0]
	fy := P1[1][1]
	cx := P1[0][2]
	cy := P1[1][2]
	rectPix := func(Rr [3][3]float64, Xc [3]float64) (u, v float64) {
		p := matVec3(Rr, Xc)
		return fx*p[0]/p[2] + cx, fy*p[1]/p[2] + cy
	}
	pts := [][3]float64{{0.3, 0.1, 6}, {-0.5, 0.4, 8}, {0.8, -0.3, 5}, {0, 0, 7}}
	for _, Xc1 := range pts {
		Xc2 := add3(matVec3(R, Xc1), T)
		u1, v1 := rectPix(R1, Xc1)
		u2, v2 := rectPix(R2, Xc2)
		if math.Abs(v1-v2) > 0.5 {
			t.Errorf("rectified rows not aligned: v1=%g v2=%g (X=%v)", v1, v2, Xc1)
		}
		if u1-u2 <= 0 {
			t.Errorf("expected positive disparity, got u1=%g u2=%g", u1, u2)
		}
	}
}

func TestGetOptimalNewCameraMatrixNoDistortion(t *testing.T) {
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	newK, roi := GetOptimalNewCameraMatrix(K, nil, [2]int{640, 480}, 0)
	if matMaxDiff(newK, K) != 0 {
		t.Errorf("no-distortion newK should equal K, got %v", newK)
	}
	if roi != [4]int{0, 0, 640, 480} {
		t.Errorf("roi=%v, want full frame", roi)
	}
	// With distortion the focal length should still be positive and finite.
	dist := DistCoeffs{K1: 0.15, K2: 0.02}.Slice()
	nk, _ := GetOptimalNewCameraMatrix(K, dist, [2]int{640, 480}, 0)
	if nk[0][0] <= 0 || math.IsNaN(nk[0][0]) {
		t.Errorf("distorted newK invalid: %v", nk)
	}
}

func TestInitUndistortRectifyMap(t *testing.T) {
	K := CameraMatrix{Fx: 300, Fy: 300, Cx: 32, Cy: 32}.Matrix()
	// Identity: with no distortion, no rectification, and newK == K the map is
	// the identity grid.
	mapX, mapY := InitUndistortRectifyMap(K, nil, [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, K, [2]int{64, 64})
	for _, v := range []int{0, 10, 31, 63} {
		for _, u := range []int{0, 5, 32, 63} {
			if math.Abs(mapX[v][u]-float64(u)) > 1e-9 || math.Abs(mapY[v][u]-float64(v)) > 1e-9 {
				t.Fatalf("identity map wrong at (%d,%d): (%g,%g)", u, v, mapX[v][u], mapY[v][u])
			}
		}
	}
	// With distortion the map at an undistorted pixel must point to the matching
	// distorted source pixel.
	dist := DistCoeffs{K1: 0.12, K2: 0.01}.Slice()
	obj := [][3]float64{{0.25, 0.18, 0}}
	tvec := [3]float64{0, 0, 3}
	pd := projectF(obj, [3]float64{}, tvec, K, dist)[0]
	pu := projectF(obj, [3]float64{}, tvec, K, nil)[0]
	I := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	mx, my := InitUndistortRectifyMap(K, dist, I, K, [2]int{64, 64})
	uu := int(math.Round(pu[0]))
	vv := int(math.Round(pu[1]))
	if uu >= 0 && uu < 64 && vv >= 0 && vv < 64 {
		if math.Abs(mx[vv][uu]-pd[0]) > 1.0 || math.Abs(my[vv][uu]-pd[1]) > 1.0 {
			t.Errorf("map at undistorted (%d,%d)=(%g,%g), want near distorted %v", uu, vv, mx[vv][uu], my[vv][uu], pd)
		}
	}
}

// renderChessboard draws an axis-aligned chessboard with a white quiet zone and
// returns the image together with the ground-truth inner-corner coordinates.
func renderChessboard(squaresX, squaresY, square, margin int) (*cv.Mat, [][2]float64) {
	w := squaresX*square + 2*margin
	h := squaresY*square + 2*margin
	img := cv.NewMat(h, w, 1)
	img.SetTo(255) // white background / quiet zone
	for sy := 0; sy < squaresY; sy++ {
		for sx := 0; sx < squaresX; sx++ {
			if (sx+sy)%2 == 0 {
				continue // leave white
			}
			x0 := margin + sx*square
			y0 := margin + sy*square
			for y := y0; y < y0+square; y++ {
				for x := x0; x < x0+square; x++ {
					img.Set(y, x, 0, 0)
				}
			}
		}
	}
	var corners [][2]float64
	for cy := 1; cy < squaresY; cy++ {
		for cx := 1; cx < squaresX; cx++ {
			corners = append(corners, [2]float64{float64(margin + cx*square), float64(margin + cy*square)})
		}
	}
	return img, corners
}

func TestFindChessboardCorners(t *testing.T) {
	squaresX, squaresY, square, margin := 7, 5, 40, 40
	img, truth := renderChessboard(squaresX, squaresY, square, margin)
	pattern := [2]int{squaresX - 1, squaresY - 1}

	corners, found := FindChessboardCorners(img, pattern)
	if !found {
		t.Fatalf("chessboard not found (got %d corners, want %d)", len(corners), pattern[0]*pattern[1])
	}
	if len(corners) != len(truth) {
		t.Fatalf("got %d corners, want %d", len(corners), len(truth))
	}
	// Corners are returned row-major; every one must be within a pixel of truth.
	var maxe float64
	for i := range truth {
		e := math.Hypot(corners[i][0]-truth[i][0], corners[i][1]-truth[i][1])
		if e > maxe {
			maxe = e
		}
	}
	if maxe > 1.5 {
		t.Errorf("corner localisation error too large: %g px", maxe)
	}

	// DrawChessboardCorners must run and modify a colour image without panicking.
	color := cv.NewMat(img.Rows, img.Cols, 3)
	DrawChessboardCorners(color, pattern, corners, found)
	changed := false
	for i := 0; i < len(color.Data) && !changed; i++ {
		if color.Data[i] != 0 {
			changed = true
		}
	}
	if !changed {
		t.Error("DrawChessboardCorners drew nothing")
	}
}

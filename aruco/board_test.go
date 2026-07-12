package aruco_test

import (
	"fmt"
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/aruco"
)

// --- synthetic projection helpers (package-local, distinct names) ---

func rotFromRvec(r [3]float64) [3][3]float64 {
	theta := math.Sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
	if theta < 1e-12 {
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	kx, ky, kz := r[0]/theta, r[1]/theta, r[2]/theta
	c, s := math.Cos(theta), math.Sin(theta)
	v := 1 - c
	return [3][3]float64{
		{c + kx*kx*v, kx*ky*v - kz*s, kx*kz*v + ky*s},
		{ky*kx*v + kz*s, c + ky*ky*v, ky*kz*v - kx*s},
		{kz*kx*v - ky*s, kz*ky*v + kx*s, c + kz*kz*v},
	}
}

// projectXY projects a planar board point (X, Y, 0) to a pixel, optionally with
// Brown-Conrady distortion dist = [k1,k2,p1,p2,k3] (nil for none).
func projectXY(rot [3][3]float64, t [3]float64, k [3][3]float64, dist []float64, X, Y float64) (u, v float64) {
	xc := rot[0][0]*X + rot[0][1]*Y + t[0]
	yc := rot[1][0]*X + rot[1][1]*Y + t[1]
	zc := rot[2][0]*X + rot[2][1]*Y + t[2]
	x, y := xc/zc, yc/zc
	if len(dist) > 0 {
		g := func(i int) float64 {
			if i < len(dist) {
				return dist[i]
			}
			return 0
		}
		k1, k2, p1, p2, k3 := g(0), g(1), g(2), g(3), g(4)
		r2 := x*x + y*y
		radial := 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
		xd := x*radial + 2*p1*x*y + p2*(r2+2*x*x)
		yd := y*radial + p1*(r2+2*y*y) + 2*p2*x*y
		x, y = xd, yd
	}
	return k[0][0]*x + k[0][2], k[1][1]*y + k[1][2]
}

// rotAngleBetween returns the geodesic angle between two rotation matrices.
func rotAngleBetween(a, b [3][3]float64) float64 {
	// trace(a^T b)
	var tr float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			tr += a[j][i] * b[j][i]
		}
	}
	c := (tr - 1) / 2
	if c > 1 {
		c = 1
	} else if c < -1 {
		c = -1
	}
	return math.Acos(c)
}

// --- GridBoard ---

func TestGridBoardDrawAndDetect(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict5x5)
	board := aruco.NewGridBoard(3, 2, 0.04, 0.01, d, 0)
	img := board.Draw(520, 360, 20)
	if img.Channels != 1 || img.Rows != 360 || img.Cols != 520 {
		t.Fatalf("board image shape %dx%dx%d", img.Rows, img.Cols, img.Channels)
	}
	_, ids := aruco.DetectMarkers(img, d)
	if len(ids) != 6 {
		t.Fatalf("detected %d markers, want 6 (ids=%v)", len(ids), ids)
	}
	got := map[int]bool{}
	for _, id := range ids {
		got[id] = true
	}
	for want := 0; want < 6; want++ {
		if !got[want] {
			t.Errorf("board id %d not detected (ids=%v)", want, ids)
		}
	}
}

func TestEstimatePoseBoardRecoversPose(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict5x5)
	board := aruco.NewGridBoard(4, 3, 0.04, 0.008, d, 0)
	k := [3][3]float64{{1200, 0, 640}, {0, 1200, 480}, {0, 0, 1}}
	rvecTrue := [3]float64{0.12, -0.18, 0.05}
	tvecTrue := [3]float64{0.03, -0.02, 0.6}
	rot := rotFromRvec(rvecTrue)

	obj := board.ObjectPoints()
	ids := board.Ids()
	corners := make([][4]cv.Point, len(obj))
	for i, marker := range obj {
		for j := 0; j < 4; j++ {
			u, v := projectXY(rot, tvecTrue, k, nil, marker[j][0], marker[j][1])
			corners[i][j] = cv.Point{X: int(math.Round(u)), Y: int(math.Round(v))}
		}
	}
	rvec, tvec, used := aruco.EstimatePoseBoard(corners, ids, board, k, nil)
	if used != len(ids) {
		t.Fatalf("markersUsed=%d, want %d", used, len(ids))
	}
	if a := rotAngleBetween(rot, rotFromRvec(rvec)); a > 0.02 {
		t.Errorf("rotation error %.4f rad too large", a)
	}
	for i := 0; i < 3; i++ {
		if math.Abs(tvec[i]-tvecTrue[i]) > 0.01 {
			t.Errorf("tvec[%d]=%.4f want %.4f", i, tvec[i], tvecTrue[i])
		}
	}
}

func TestEstimatePoseBoardEmpty(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict5x5)
	board := aruco.NewGridBoard(2, 2, 0.04, 0.01, d, 0)
	k := [3][3]float64{{800, 0, 320}, {0, 800, 240}, {0, 0, 1}}
	// Ids that are not on the board must contribute nothing.
	_, _, used := aruco.EstimatePoseBoard([][4]cv.Point{{{X: 1}, {X: 2}, {X: 3}, {X: 4}}}, []int{999}, board, k, nil)
	if used != 0 {
		t.Errorf("markersUsed=%d, want 0", used)
	}
}

// --- CharucoBoard ---

func TestCharucoBoardDrawDetectInterpolate(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	board := aruco.NewCharucoBoard(5, 4, 0.03, 0.02, d)
	img := board.Draw(600, 480, 20)
	corners, ids := aruco.DetectMarkers(img, d)
	if len(ids) == 0 {
		t.Fatal("no markers detected on ChArUco board")
	}
	// Every detected id must be a valid board marker id.
	boardIDs := map[int]bool{}
	for _, id := range board.Ids() {
		boardIDs[id] = true
	}
	for _, id := range ids {
		if !boardIDs[id] {
			t.Errorf("detected non-board id %d", id)
		}
	}
	cc, cids := aruco.InterpolateCornersCharuco(corners, ids, img, board)
	if len(cc) != len(cids) || len(cc) == 0 {
		t.Fatalf("interpolated %d corners / %d ids", len(cc), len(cids))
	}
	// Interior corners number (5-1)*(4-1) = 12; with all markers detected we
	// should recover all of them.
	if len(ids) == len(board.Ids()) && len(cc) != 12 {
		t.Errorf("interpolated %d corners, want 12", len(cc))
	}
	// Corners must lie inside the image.
	for _, p := range cc {
		if p[0] < 0 || p[1] < 0 || p[0] > float64(img.Cols) || p[1] > float64(img.Rows) {
			t.Errorf("charuco corner %v outside image", p)
		}
	}
}

func TestEstimatePoseCharucoBoardRecovers(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	board := aruco.NewCharucoBoard(6, 5, 0.03, 0.02, d)
	k := [3][3]float64{{1000, 0, 500}, {0, 1000, 400}, {0, 0, 1}}
	rvecTrue := [3]float64{-0.1, 0.2, -0.05}
	tvecTrue := [3]float64{0.05, 0.02, 0.7}
	rot := rotFromRvec(rvecTrue)

	chess := board.ChessboardCorners()
	cc := make([][2]float64, len(chess))
	cids := make([]int, len(chess))
	for i, p := range chess {
		u, v := projectXY(rot, tvecTrue, k, nil, p[0], p[1])
		cc[i] = [2]float64{u, v}
		cids[i] = i
	}
	rvec, tvec, ok := aruco.EstimatePoseCharucoBoard(cc, cids, board, k, nil)
	if !ok {
		t.Fatal("EstimatePoseCharucoBoard failed")
	}
	if a := rotAngleBetween(rot, rotFromRvec(rvec)); a > 1e-3 {
		t.Errorf("rotation error %.6f rad", a)
	}
	for i := 0; i < 3; i++ {
		if math.Abs(tvec[i]-tvecTrue[i]) > 1e-4 {
			t.Errorf("tvec[%d]=%.6f want %.6f", i, tvec[i], tvecTrue[i])
		}
	}
}

// --- ChArUco diamond ---

func TestCharucoDiamondDrawDetect(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	ids := [4]int{3, 7, 11, 15}
	img := aruco.DrawCharucoDiamond(d, ids, 96, 64)
	if img.Rows != 288 || img.Cols != 288 {
		t.Fatalf("diamond image %dx%d, want 288x288", img.Rows, img.Cols)
	}
	corners, mids := aruco.DetectMarkers(img, d)
	if len(mids) < 4 {
		t.Fatalf("detected %d markers, want >=4 (ids=%v)", len(mids), mids)
	}
	dc, dids := aruco.DetectCharucoDiamond(img, corners, mids, d, 96, 64)
	if len(dc) != 1 || len(dids) != 1 {
		t.Fatalf("found %d diamonds, want 1", len(dc))
	}
	got := map[int]bool{}
	for _, id := range dids[0] {
		got[id] = true
	}
	for _, want := range ids {
		if !got[want] {
			t.Errorf("diamond id %d missing (got %v)", want, dids[0])
		}
	}
	// The four central-square corners should sit near the image centre.
	for _, p := range dc[0] {
		if p.X < 72 || p.X > 216 || p.Y < 72 || p.Y > 216 {
			t.Errorf("diamond corner %v not near centre", p)
		}
	}
}

// --- RefineDetectedMarkers ---

func TestRefineDetectedMarkersRecoversDropped(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict5x5)
	board := aruco.NewGridBoard(3, 2, 0.04, 0.01, d, 0)
	img := board.Draw(520, 360, 20)
	corners, ids := aruco.DetectMarkers(img, d)
	if len(ids) != 6 {
		t.Fatalf("setup detected %d markers, want 6", len(ids))
	}
	// Drop the first detection and confirm refinement rescues it.
	dropped := ids[0]
	rc := append([][4]cv.Point{}, corners[1:]...)
	rids := append([]int{}, ids[1:]...)

	out, oids, recovered := aruco.RefineDetectedMarkers(img, board, rc, rids)
	if recovered < 1 {
		t.Fatalf("recovered %d markers, want >=1", recovered)
	}
	if len(oids) != len(out) {
		t.Fatalf("mismatched refined slices %d vs %d", len(oids), len(out))
	}
	found := false
	for _, id := range oids {
		if id == dropped {
			found = true
		}
	}
	if !found {
		t.Errorf("dropped id %d not recovered (ids=%v)", dropped, oids)
	}
}

// --- DetectorParameters, DetectMarkersWithParams, CornerSubPix ---

func TestDetectMarkersWithParams(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	canvas := cv.NewMat(160, 160, 1)
	canvas.SetTo(255)
	aruco.GenerateMarker(d, 12, 96).CopyTo(canvas, 32, 32)

	params := aruco.DefaultDetectorParameters()
	_, ids := aruco.DetectMarkersWithParams(canvas, d, params)
	if len(ids) != 1 || ids[0] != 12 {
		t.Fatalf("params detect ids=%v, want [12]", ids)
	}

	// A tight maximum-perimeter bound rejects the (large) marker.
	params.MaxMarkerPerimeterRate = 0.2
	if _, ids := aruco.DetectMarkersWithParams(canvas, d, params); len(ids) != 0 {
		t.Errorf("tight maxPerimeter: ids=%v, want none", ids)
	}
}

func TestDetectMarkersWithParamsSubpix(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	canvas := cv.NewMat(160, 160, 1)
	canvas.SetTo(255)
	aruco.GenerateMarker(d, 8, 96).CopyTo(canvas, 32, 32)

	params := aruco.DefaultDetectorParameters()
	params.CornerRefinementMethod = aruco.CornerRefineSubpix
	corners, ids := aruco.DetectMarkersWithParams(canvas, d, params)
	if len(ids) != 1 || ids[0] != 8 {
		t.Fatalf("subpix detect ids=%v, want [8]", ids)
	}
	// The marker's outer black border spans pixels [32,128); its top-left corner
	// sits at (32,32). Refined corners must still be within a pixel or two.
	tl := corners[0][0]
	if math.Abs(float64(tl.X)-32) > 3 || math.Abs(float64(tl.Y)-32) > 3 {
		t.Errorf("refined top-left %v, want near (32,32)", tl)
	}
}

func TestCornerSubPixAccuracy(t *testing.T) {
	// A black square in a white field has a sharp corner at pixel edge 49/50.
	img := cv.NewMat(100, 100, 1)
	img.SetTo(255)
	cv.Rectangle(img, cv.Point{X: 0, Y: 0}, cv.Point{X: 49, Y: 49}, cv.NewScalar(0), cv.Filled)

	guess := [][2]float64{{52, 47}}
	refined := aruco.CornerSubPix(img, guess, 5, 40, 0.001)
	// The subpixel corner should converge near (49.5, 49.5).
	if math.Abs(refined[0][0]-49.5) > 1.2 || math.Abs(refined[0][1]-49.5) > 1.2 {
		t.Errorf("refined corner %v, want near (49.5,49.5)", refined[0])
	}
}

// --- extra dictionaries ---

func TestDictionary6x6(t *testing.T) {
	d := aruco.GetPredefinedDictionary6x6(aruco.Dict6x6)
	if d.BitsPerSide() != 6 {
		t.Errorf("BitsPerSide=%d, want 6", d.BitsPerSide())
	}
	if d.Size() != 250 {
		t.Errorf("Size=%d, want 250", d.Size())
	}
	for _, id := range []int{0, 100, 249} {
		canvas := cv.NewMat(160, 160, 1)
		canvas.SetTo(255)
		aruco.GenerateMarker(d, id, 96).CopyTo(canvas, 32, 32)
		_, ids := aruco.DetectMarkers(canvas, d)
		if len(ids) != 1 || ids[0] != id {
			t.Errorf("6x6 id=%d round-trip ids=%v", id, ids)
		}
	}
}

func TestGenerateCustomDictionary(t *testing.T) {
	d := aruco.GenerateCustomDictionary(5, 30, 5, 0xabcdef123456)
	if d.BitsPerSide() != 5 {
		t.Errorf("BitsPerSide=%d, want 5", d.BitsPerSide())
	}
	if d.Size() == 0 || d.Size() > 30 {
		t.Fatalf("Size=%d, want (0,30]", d.Size())
	}
	if d.Tolerance() != 2 {
		t.Errorf("Tolerance=%d, want 2", d.Tolerance())
	}
	canvas := cv.NewMat(150, 150, 1)
	canvas.SetTo(255)
	aruco.GenerateMarker(d, 0, 90).CopyTo(canvas, 30, 30)
	_, ids := aruco.DetectMarkers(canvas, d)
	if len(ids) != 1 || ids[0] != 0 {
		t.Errorf("custom dict round-trip ids=%v, want [0]", ids)
	}
	mustPanic(t, "bad bitsPerSide", func() { aruco.GenerateCustomDictionary(3, 10, 3, 1) })
}

// --- distortion-aware pose ---

func TestUndistortImagePointsRoundTrip(t *testing.T) {
	k := [3][3]float64{{800, 0, 320}, {0, 800, 240}, {0, 0, 1}}
	dist := []float64{-0.2, 0.05, 0.001, -0.002, 0.0}
	// Start from ideal pinhole pixels, distort them, then undistort back.
	ideal := [][2]float64{{320, 240}, {400, 300}, {250, 350}, {500, 180}}
	distorted := make([][2]float64, len(ideal))
	for i, p := range ideal {
		x := (p[0] - k[0][2]) / k[0][0]
		y := (p[1] - k[1][2]) / k[1][1]
		r2 := x*x + y*y
		radial := 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
		xd := x*radial + 2*dist[2]*x*y + dist[3]*(r2+2*x*x)
		yd := y*radial + dist[2]*(r2+2*y*y) + 2*dist[3]*x*y
		distorted[i] = [2]float64{k[0][0]*xd + k[0][2], k[1][1]*yd + k[1][2]}
	}
	got := aruco.UndistortImagePoints(distorted, k, dist)
	for i := range ideal {
		if math.Abs(got[i][0]-ideal[i][0]) > 0.05 || math.Abs(got[i][1]-ideal[i][1]) > 0.05 {
			t.Errorf("undistort[%d]=%v, want %v", i, got[i], ideal[i])
		}
	}
}

func TestEstimatePoseSingleMarkerWithDistortion(t *testing.T) {
	k := [3][3]float64{{900, 0, 320}, {0, 900, 240}, {0, 0, 1}}
	dist := []float64{-0.15, 0.03, 0.0, 0.0, 0.0}
	const L = 0.05
	rvecTrue := [3]float64{0.05, 0.1, -0.03}
	tvecTrue := [3]float64{0.01, 0.0, 0.4}
	rot := rotFromRvec(rvecTrue)
	half := L / 2
	obj := [4][2]float64{{-half, half}, {half, half}, {half, -half}, {-half, -half}}
	var quad [4]cv.Point
	for j := 0; j < 4; j++ {
		u, v := projectXY(rot, tvecTrue, k, dist, obj[j][0], obj[j][1])
		quad[j] = cv.Point{X: int(math.Round(u)), Y: int(math.Round(v))}
	}
	rvecs, tvecs := aruco.EstimatePoseSingleMarkersWithDistortion([][4]cv.Point{quad}, L, k, dist)
	if math.Abs(tvecs[0][2]-tvecTrue[2]) > 0.01 {
		t.Errorf("z=%.4f want %.4f", tvecs[0][2], tvecTrue[2])
	}
	if a := rotAngleBetween(rot, rotFromRvec(rvecs[0])); a > 0.08 {
		t.Errorf("rotation error %.4f rad", a)
	}
}

// --- ChArUco camera calibration ---

func TestCalibrateCameraCharuco(t *testing.T) {
	d := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	board := aruco.NewCharucoBoard(7, 5, 0.03, 0.02, d)
	const w, h = 1024, 768
	fTrue := 1100.0
	k := [3][3]float64{{fTrue, 0, float64(w-1) / 2}, {0, fTrue, float64(h-1) / 2}, {0, 0, 1}}

	poses := []struct {
		r [3]float64
		t [3]float64
	}{
		{[3]float64{0.3, -0.1, 0.05}, [3]float64{0.02, 0.01, 0.6}},
		{[3]float64{-0.2, 0.35, -0.1}, [3]float64{-0.03, 0.02, 0.7}},
		{[3]float64{0.15, 0.25, 0.2}, [3]float64{0.0, -0.02, 0.65}},
	}
	chess := board.ChessboardCorners()
	var allCorners [][][2]float64
	var allIds [][]int
	for _, p := range poses {
		rot := rotFromRvec(p.r)
		cc := make([][2]float64, len(chess))
		cids := make([]int, len(chess))
		for i, obj := range chess {
			u, v := projectXY(rot, p.t, k, nil, obj[0], obj[1])
			cc[i] = [2]float64{u, v}
			cids[i] = i
		}
		allCorners = append(allCorners, cc)
		allIds = append(allIds, cids)
	}
	est, reproj, ok := aruco.CalibrateCameraCharuco(allCorners, allIds, board, w, h)
	if !ok {
		t.Fatal("calibration failed")
	}
	if rel := math.Abs(est[0][0]-fTrue) / fTrue; rel > 0.02 {
		t.Errorf("recovered f=%.2f want %.2f (%.3f rel)", est[0][0], fTrue, rel)
	}
	if reproj > 1.0 {
		t.Errorf("reprojection error %.4f px too large", reproj)
	}

	// Fronto-parallel-only views leave f unobservable.
	var frontoCorners [][][2]float64
	var frontoIds [][]int
	for _, tz := range []float64{0.6, 0.8} {
		rot := rotFromRvec([3]float64{0, 0, 0})
		cc := make([][2]float64, len(chess))
		cids := make([]int, len(chess))
		for i, obj := range chess {
			u, v := projectXY(rot, [3]float64{0, 0, tz}, k, nil, obj[0], obj[1])
			cc[i] = [2]float64{u, v}
			cids[i] = i
		}
		frontoCorners = append(frontoCorners, cc)
		frontoIds = append(frontoIds, cids)
	}
	if _, _, ok := aruco.CalibrateCameraCharuco(frontoCorners, frontoIds, board, w, h); ok {
		t.Error("fronto-parallel-only calibration should report ok=false")
	}
}

// --- examples ---

func ExampleNewGridBoard() {
	dict := aruco.GetPredefinedDictionary(aruco.Dict5x5)
	board := aruco.NewGridBoard(3, 2, 0.04, 0.01, dict, 0)
	img := board.Draw(520, 360, 20)
	_, ids := aruco.DetectMarkers(img, dict)
	fmt.Printf("board %dx%d markers, detected %d\n", board.MarkersX(), board.MarkersY(), len(ids))
	// Output:
	// board 3x2 markers, detected 6
}

func ExampleNewCharucoBoard() {
	dict := aruco.GetPredefinedDictionary(aruco.Dict4x4)
	board := aruco.NewCharucoBoard(5, 4, 0.03, 0.02, dict)
	fmt.Printf("%d markers, %d interior corners\n", len(board.Ids()), len(board.ChessboardCorners()))
	// Output:
	// 10 markers, 12 interior corners
}

func ExampleGetPredefinedDictionary6x6() {
	dict := aruco.GetPredefinedDictionary6x6(aruco.Dict6x6)
	fmt.Printf("%s: %d markers, %d bits/side\n", dict.Name, dict.Size(), dict.BitsPerSide())
	// Output:
	// DICT_6X6_250: 250 markers, 6 bits/side
}

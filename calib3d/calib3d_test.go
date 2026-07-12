package calib3d

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// applyHomography maps an integer point through a 3×3 homography and rounds the
// result, mirroring how the correspondences below are synthesised.
func applyHomography(h [3][3]float64, p cv.Point) cv.Point {
	x, y := float64(p.X), float64(p.Y)
	w := h[2][0]*x + h[2][1]*y + h[2][2]
	u := (h[0][0]*x + h[0][1]*y + h[0][2]) / w
	v := (h[1][0]*x + h[1][1]*y + h[1][2]) / w
	return cv.Point{X: int(math.Round(u)), Y: int(math.Round(v))}
}

func TestFindHomographyExactFourPoints(t *testing.T) {
	// A known perspective mapping a unit-ish square onto a general quad.
	src := []cv.Point{{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 100, Y: 100}, {X: 0, Y: 100}}
	dst := []cv.Point{{X: 10, Y: 12}, {X: 190, Y: 30}, {X: 210, Y: 220}, {X: 5, Y: 205}}

	h, inliers := FindHomography(src, dst, MethodDirect, 0)
	if inliers == nil {
		t.Fatal("expected a homography, got nil")
	}
	for i := range src {
		got := applyHomography(h, src[i])
		if got != dst[i] {
			t.Errorf("point %d: mapped %v, want %v", i, got, dst[i])
		}
		if !inliers[i] {
			t.Errorf("point %d should be an inlier", i)
		}
	}
}

func TestFindHomographyRANSAC(t *testing.T) {
	// Ground-truth homography used to generate correspondences.
	trueH := [3][3]float64{
		{1.1, 0.05, 12},
		{0.03, 1.08, 8},
		{0.0002, 0.0001, 1},
	}
	var src, dst []cv.Point
	outlierIdx := map[int]bool{5: true, 11: true, 17: true, 23: true}
	for i := 0; i < 30; i++ {
		p := cv.Point{X: (i * 37) % 200, Y: (i * 53) % 200}
		src = append(src, p)
		if outlierIdx[i] {
			// Deterministic gross outlier well away from the true mapping.
			dst = append(dst, cv.Point{X: (i*13)%50 + 5, Y: (i*29)%40 + 3})
		} else {
			dst = append(dst, applyHomography(trueH, p))
		}
	}

	h, inliers := FindHomography(src, dst, MethodRANSAC, 3.0)
	if inliers == nil {
		t.Fatal("RANSAC returned no homography")
	}
	for i := range src {
		if outlierIdx[i] {
			if inliers[i] {
				t.Errorf("outlier %d was flagged as an inlier", i)
			}
			continue
		}
		if !inliers[i] {
			t.Errorf("inlier %d was flagged as an outlier", i)
		}
		// The recovered model should map inliers close to their targets.
		got := applyHomography(h, src[i])
		if abs(got.X-dst[i].X) > 2 || abs(got.Y-dst[i].Y) > 2 {
			t.Errorf("inlier %d: mapped %v, want %v", i, got, dst[i])
		}
	}
}

func TestFindHomographyDeterministic(t *testing.T) {
	trueH := [3][3]float64{{1.05, 0.02, 4}, {0.01, 1.03, 6}, {0.0001, 0.0, 1}}
	var src, dst []cv.Point
	for i := 0; i < 20; i++ {
		p := cv.Point{X: (i * 41) % 180, Y: (i * 17) % 160}
		src = append(src, p)
		dst = append(dst, applyHomography(trueH, p))
	}
	h1, m1 := FindHomography(src, dst, MethodRANSAC, 3.0)
	h2, m2 := FindHomography(src, dst, MethodRANSAC, 3.0)
	if h1 != h2 {
		t.Error("RANSAC homography is not deterministic")
	}
	for i := range m1 {
		if m1[i] != m2[i] {
			t.Error("RANSAC inlier mask is not deterministic")
		}
	}
}

func TestRodriguesRoundTrip(t *testing.T) {
	cases := [][3]float64{
		{0, 0, 0},
		{0.1, -0.2, 0.3},
		{1.2, 0.4, -0.7},
		{0, 0, math.Pi / 2},
		{math.Pi / 3, 0, 0},
		{0, 0, math.Pi}, // near-π, Z axis
		{math.Pi / math.Sqrt2, math.Pi / math.Sqrt2, 0}, // π about (1,1,0)
		{0, math.Pi * 0.999, 0},                         // just under π, Y axis
	}
	for _, rvec := range cases {
		r := RodriguesToMatrix(rvec)
		// R must be orthonormal with determinant +1.
		rt := transpose3(r)
		p := mul3(rt, r)
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				want := 0.0
				if i == j {
					want = 1
				}
				if math.Abs(p[i][j]-want) > 1e-9 {
					t.Errorf("rvec %v: RᵀR not identity at [%d][%d]=%g", rvec, i, j, p[i][j])
				}
			}
		}
		if d := det3(r); math.Abs(d-1) > 1e-9 {
			t.Errorf("rvec %v: det(R)=%g, want 1", rvec, d)
		}
		back := RodriguesToVector(r)
		// Compare via the reconstructed rotation matrix to sidestep the
		// axis-angle sign ambiguity at θ = π.
		r2 := RodriguesToMatrix(back)
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				if math.Abs(r[i][j]-r2[i][j]) > 1e-6 {
					t.Errorf("rvec %v: round-trip mismatch at [%d][%d]: %g vs %g", rvec, i, j, r[i][j], r2[i][j])
				}
			}
		}
	}
}

func TestProjectPointsKnownGeometry(t *testing.T) {
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	// Identity rotation, camera 5 units back along Z.
	rvec := [3]float64{0, 0, 0}
	tvec := [3]float64{0, 0, 5}
	obj := [][3]float64{
		{0, 0, 0},   // on the optical axis -> principal point
		{1, 0, 0},   // +X
		{0, 1, 0},   // +Y
		{-1, -1, 0}, // corner
	}
	got := ProjectPoints(obj, rvec, tvec, K, nil)
	want := []cv.Point{
		{X: 320, Y: 240},
		{X: 420, Y: 240}, // 500*(1/5)+320
		{X: 320, Y: 340},
		{X: 220, Y: 140},
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("point %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestProjectPointsDistortionShiftsOutward(t *testing.T) {
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	obj := [][3]float64{{1, 1, 0}}
	tvec := [3]float64{0, 0, 5}
	undist := ProjectPoints(obj, [3]float64{}, tvec, K, nil)
	dist := DistCoeffs{K1: 0.2}.Slice()
	distd := ProjectPoints(obj, [3]float64{}, tvec, K, dist)
	// Positive K1 pushes points away from the principal point.
	du := abs(undist[0].X - 320)
	dd := abs(distd[0].X - 320)
	if dd <= du {
		t.Errorf("expected distortion to push point outward: undist dx=%d dist dx=%d", du, dd)
	}
}

func TestUndistortIdentity(t *testing.T) {
	img := cv.NewMat(20, 24, 1)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			img.Set(y, x, 0, uint8((x*7+y*11)%256))
		}
	}
	K := CameraMatrix{Fx: 400, Fy: 400, Cx: 12, Cy: 10}.Matrix()
	out := Undistort(img, K, nil) // no distortion -> identity resample
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			if out.At(y, x, 0) != img.At(y, x, 0) {
				t.Fatalf("pixel (%d,%d): got %d, want %d", x, y, out.At(y, x, 0), img.At(y, x, 0))
			}
		}
	}
}

func TestUndistortRoundTripWithProject(t *testing.T) {
	// A distorted image undistorted should place a known feature at the
	// undistorted projection. Build a synthetic image with a single bright
	// pixel at the distorted location of an object point, undistort, and check
	// the brightness lands near the undistorted projection.
	K := CameraMatrix{Fx: 300, Fy: 300, Cx: 32, Cy: 32}.Matrix()
	dist := DistCoeffs{K1: 0.1, K2: 0.01}.Slice()
	obj := [][3]float64{{0.3, 0.2, 0}}
	tvec := [3]float64{0, 0, 3}
	distortedPt := ProjectPoints(obj, [3]float64{}, tvec, K, dist)[0]
	undistPt := ProjectPoints(obj, [3]float64{}, tvec, K, nil)[0]

	img := cv.NewMat(64, 64, 1)
	if img.Cols > distortedPt.X && distortedPt.X >= 0 && img.Rows > distortedPt.Y && distortedPt.Y >= 0 {
		img.Set(distortedPt.Y, distortedPt.X, 0, 255)
	}
	out := Undistort(img, K, dist)
	// Find the brightest pixel in the output.
	var bx, by int
	var best uint8
	for y := 0; y < out.Rows; y++ {
		for x := 0; x < out.Cols; x++ {
			if v := out.At(y, x, 0); v > best {
				best = v
				bx, by = x, y
			}
		}
	}
	if abs(bx-undistPt.X) > 2 || abs(by-undistPt.Y) > 2 {
		t.Errorf("undistorted feature at (%d,%d), want near %v", bx, by, undistPt)
	}
}

func TestSolvePnPPlanarRecoversPose(t *testing.T) {
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	// A planar grid of object points (Z = 0).
	var obj [][3]float64
	for gx := -2; gx <= 2; gx++ {
		for gy := -2; gy <= 2; gy++ {
			obj = append(obj, [3]float64{float64(gx), float64(gy), 0})
		}
	}
	trueR := [3]float64{0.2, -0.15, 0.1}
	trueT := [3]float64{0.3, -0.2, 8}
	img := ProjectPoints(obj, trueR, trueT, K, nil)

	rvec, tvec, ok := SolvePnPPlanar(obj, img, K)
	if !ok {
		t.Fatal("SolvePnPPlanar failed")
	}
	// Reproject with the recovered pose and check it matches the observations.
	reproj := ProjectPoints(obj, rvec, tvec, K, nil)
	var maxErr int
	for i := range img {
		e := abs(reproj[i].X-img[i].X) + abs(reproj[i].Y-img[i].Y)
		if e > maxErr {
			maxErr = e
		}
	}
	if maxErr > 3 {
		t.Errorf("reprojection error too large: %d px", maxErr)
	}
	// Translation should be close to ground truth.
	if math.Abs(tvec[0]-trueT[0]) > 0.1 || math.Abs(tvec[1]-trueT[1]) > 0.1 || math.Abs(tvec[2]-trueT[2]) > 0.2 {
		t.Errorf("recovered t=%v, want %v", tvec, trueT)
	}
}

func TestTriangulatePoints(t *testing.T) {
	// Two cameras: identity and a lateral baseline translation.
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	// P1 = K[I|0]
	var P1 [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			P1[i][j] = K[i][j]
		}
	}
	// P2 = K[I|t] with a baseline along X.
	baseline := [3]float64{-1, 0, 0}
	var P2 [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			P2[i][j] = K[i][j]
		}
		P2[i][3] = K[i][0]*baseline[0] + K[i][1]*baseline[1] + K[i][2]*baseline[2]
	}
	world := [][3]float64{
		{0, 0, 6},
		{1, -0.5, 8},
		{-1.2, 0.7, 5},
	}
	// Project into both views.
	proj := func(P [3][4]float64, X [3]float64) cv.Point {
		x := P[0][0]*X[0] + P[0][1]*X[1] + P[0][2]*X[2] + P[0][3]
		y := P[1][0]*X[0] + P[1][1]*X[1] + P[1][2]*X[2] + P[1][3]
		w := P[2][0]*X[0] + P[2][1]*X[1] + P[2][2]*X[2] + P[2][3]
		return cv.Point{X: int(math.Round(x / w)), Y: int(math.Round(y / w))}
	}
	var pts1, pts2 []cv.Point
	for _, X := range world {
		pts1 = append(pts1, proj(P1, X))
		pts2 = append(pts2, proj(P2, X))
	}
	got := TriangulatePoints(P1, P2, pts1, pts2)
	for i := range world {
		for k := 0; k < 3; k++ {
			if math.Abs(got[i][k]-world[i][k]) > 0.1 {
				t.Errorf("point %d coord %d: got %g, want %g", i, k, got[i][k], world[i][k])
			}
		}
	}
}

func TestFindFundamentalMatEpipolarConstraint(t *testing.T) {
	// Generate correspondences that satisfy a known epipolar geometry by
	// projecting 3D points into two cameras.
	K := CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	var P1 [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			P1[i][j] = K[i][j]
		}
	}
	R := RodriguesToMatrix([3]float64{0.05, 0.1, -0.03})
	tcam := [3]float64{-1.5, 0.2, 0.1}
	var P2 [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += K[i][k] * R[k][j]
			}
			P2[i][j] = s
		}
		P2[i][3] = K[i][0]*tcam[0] + K[i][1]*tcam[1] + K[i][2]*tcam[2]
	}
	proj := func(P [3][4]float64, X [3]float64) cv.Point {
		x := P[0][0]*X[0] + P[0][1]*X[1] + P[0][2]*X[2] + P[0][3]
		y := P[1][0]*X[0] + P[1][1]*X[1] + P[1][2]*X[2] + P[1][3]
		w := P[2][0]*X[0] + P[2][1]*X[1] + P[2][2]*X[2] + P[2][3]
		return cv.Point{X: int(math.Round(x / w)), Y: int(math.Round(y / w))}
	}
	// A spread of 3D points at varied depths. More correspondences make the
	// least-squares eight-point fit robust to the ~0.5 px integer rounding of
	// the projected image coordinates.
	var world [][3]float64
	for gx := -2; gx <= 2; gx++ {
		for gy := -2; gy <= 2; gy++ {
			depth := 5.0 + 0.4*float64(gx+gy) + 0.3*float64(gx*gy)
			world = append(world, [3]float64{float64(gx) * 0.9, float64(gy) * 0.9, depth})
		}
	}
	var pts1, pts2 []cv.Point
	for _, X := range world {
		pts1 = append(pts1, proj(P1, X))
		pts2 = append(pts2, proj(P2, X))
	}
	F, ok := FindFundamentalMat(pts1, pts2)
	if !ok {
		t.Fatal("FindFundamentalMat failed")
	}
	// x2ᵀ F x1 should be near zero for all correspondences.
	var maxResid float64
	for i := range pts1 {
		x1 := [3]float64{float64(pts1[i].X), float64(pts1[i].Y), 1}
		x2 := [3]float64{float64(pts2[i].X), float64(pts2[i].Y), 1}
		fx1 := matVec3(F, x1)
		r := math.Abs(x2[0]*fx1[0] + x2[1]*fx1[1] + x2[2]*fx1[2])
		// Normalise by the epipolar line magnitude for a geometric residual.
		ln := math.Hypot(fx1[0], fx1[1])
		if ln > 1e-9 {
			r /= ln
		}
		if r > maxResid {
			maxResid = r
		}
	}
	// With integer-pixel correspondences a residual of a few pixels is expected;
	// the constraint is that F genuinely encodes the epipolar geometry.
	if maxResid > 2.0 {
		t.Errorf("epipolar residual too large: %g px", maxResid)
	}
	// F must be rank 2 (det near zero).
	if d := math.Abs(det3(F)); d > 1e-6 {
		t.Errorf("F not rank-2: det=%g", d)
	}
}

func TestCameraMatrixRoundTrip(t *testing.T) {
	c := CameraMatrix{Fx: 800, Fy: 810, Cx: 320.5, Cy: 240.5}
	k := c.Matrix()
	c2 := NewCameraMatrix(k)
	if c2 != c {
		t.Errorf("round trip: got %+v, want %+v", c2, c)
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

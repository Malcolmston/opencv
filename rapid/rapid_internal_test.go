package rapid

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// unitCube returns a unit cube centred at the origin with consistent outward
// (counter-clockwise) triangle winding.
func unitCube() *Mesh {
	v := [][3]float64{
		{-0.5, -0.5, -0.5}, // 0
		{0.5, -0.5, -0.5},  // 1
		{0.5, 0.5, -0.5},   // 2
		{-0.5, 0.5, -0.5},  // 3
		{-0.5, -0.5, 0.5},  // 4
		{0.5, -0.5, 0.5},   // 5
		{0.5, 0.5, 0.5},    // 6
		{-0.5, 0.5, 0.5},   // 7
	}
	tris := [][3]int{
		{4, 5, 6}, {4, 6, 7}, // front  (+z)
		{1, 0, 3}, {1, 3, 2}, // back   (-z)
		{1, 2, 6}, {1, 6, 5}, // right  (+x)
		{0, 4, 7}, {0, 7, 3}, // left   (-x)
		{3, 7, 6}, {3, 6, 2}, // top    (+y)
		{0, 1, 5}, {0, 5, 4}, // bottom (-y)
	}
	return &Mesh{Vertices: v, Tris: tris}
}

// renderSilhouette rasterises the mesh at the given pose as a white silhouette
// on a black background, producing a strong step edge at the object contour.
func renderSilhouette(mesh *Mesh, pose Pose, k [3][3]float64, w, h int) *cv.Mat {
	img := cv.NewMat(h, w, 1)
	r := rodrigues(pose.Rvec)
	kk := intrinsics(k)
	cam := make([][3]float64, len(mesh.Vertices))
	im := make([]cv.Point, len(mesh.Vertices))
	for i, vv := range mesh.Vertices {
		p, xc, _ := project(vv, r, pose.Tvec, kk)
		cam[i] = xc
		im[i] = cv.Point{X: int(p.X + 0.5), Y: int(p.Y + 0.5)}
	}
	for _, tri := range mesh.Tris {
		if !faceVisible(cam[tri[0]], cam[tri[1]], cam[tri[2]]) {
			continue
		}
		poly := []cv.Point{im[tri[0]], im[tri[1]], im[tri[2]]}
		cv.FillPoly(img, [][]cv.Point{poly}, cv.NewScalar(255))
	}
	return img
}

func TestRodriguesRoundTrip(t *testing.T) {
	cases := [][3]float64{
		{0, 0, 0},
		{0.1, -0.2, 0.3},
		{0, 0, math.Pi / 2},
		{1.2, -0.7, 0.4},
		{0, math.Pi - 1e-4, 0}, // near-π branch
		{math.Pi, 0, 0},        // exactly π
	}
	for _, rv := range cases {
		got := rotationToRvec(rodrigues(rv))
		// Compare rotation matrices (rvec is not unique at π).
		r1 := rodrigues(rv)
		r2 := rodrigues(got)
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				if math.Abs(r1[i][j]-r2[i][j]) > 1e-6 {
					t.Fatalf("rvec %v: rotation mismatch at [%d][%d]: %.6f vs %.6f", rv, i, j, r1[i][j], r2[i][j])
				}
			}
		}
	}
}

func TestSolveSPD(t *testing.T) {
	// Solve a small SPD system with a known solution.
	a := [][]float64{
		{4, 1, 0},
		{1, 3, 1},
		{0, 1, 2},
	}
	want := []float64{1, -2, 3}
	b := make([]float64, 3)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			b[i] += a[i][j] * want[j]
		}
	}
	// Copy since solveSPD mutates.
	ac := [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}}
	x, ok := solveSPD(ac, append([]float64(nil), b...), 3)
	if !ok {
		t.Fatal("solveSPD reported singular for SPD matrix")
	}
	for i := 0; i < 3; i++ {
		if math.Abs(x[i]-want[i]) > 1e-9 {
			t.Fatalf("x[%d]=%.9f want %.9f", i, x[i], want[i])
		}
	}
	// Singular system.
	sing := [][]float64{{1, 2}, {2, 4}}
	if _, ok := solveSPD(sing, []float64{1, 2}, 2); ok {
		t.Fatal("solveSPD should report singular")
	}
}

func TestProjectionJacobianFiniteDifference(t *testing.T) {
	k := NewCamera(500, 500, 240, 240)
	kk := intrinsics(k)
	pose := Pose{Rvec: [3]float64{0.15, -0.2, 0.1}, Tvec: [3]float64{0.2, -0.1, 6}}
	obj := [3]float64{0.3, -0.2, 0.1}
	r := rodrigues(pose.Rvec)
	proj0, xc, ok := project(obj, r, pose.Tvec, kk)
	if !ok {
		t.Fatal("point behind camera")
	}
	rx := [3]float64{xc[0] - pose.Tvec[0], xc[1] - pose.Tvec[1], xc[2] - pose.Tvec[2]}
	j := projectionJacobian(xc, rx, kk)

	const eps = 1e-6
	for c := 0; c < 6; c++ {
		var rNew [3][3]float64
		tNew := pose.Tvec
		if c < 3 {
			dO := [3]float64{}
			dO[c] = eps
			rNew = mul3(rodrigues(dO), r)
		} else {
			rNew = r
			tNew[c-3] += eps
		}
		pPert, _, _ := project(obj, rNew, tNew, kk)
		du := (pPert.X - proj0.X) / eps
		dv := (pPert.Y - proj0.Y) / eps
		if math.Abs(du-j[0][c]) > 1e-2 || math.Abs(dv-j[1][c]) > 1e-2 {
			t.Fatalf("jacobian col %d mismatch: fd=(%.4f,%.4f) analytic=(%.4f,%.4f)", c, du, dv, j[0][c], j[1][c])
		}
	}
}

func TestExtractControlPointsOnSilhouette(t *testing.T) {
	mesh := unitCube()
	k := NewCamera(500, 500, 240, 240)
	pose := Pose{Rvec: [3]float64{0.15, -0.2, 0.1}, Tvec: [3]float64{0.2, -0.1, 6}}
	cps := ExtractControlPoints(64, 20, mesh, pose, k, 480, 480)
	if len(cps) < 8 {
		t.Fatalf("expected several control points, got %d", len(cps))
	}
	for _, cp := range cps {
		n := math.Hypot(cp.Normal.X, cp.Normal.Y)
		if math.Abs(n-1) > 1e-9 {
			t.Fatalf("normal not unit: |n|=%.6f", n)
		}
		if cp.Image.X < 0 || cp.Image.X >= 480 || cp.Image.Y < 0 || cp.Image.Y >= 480 {
			t.Fatalf("control point out of image: %v", cp.Image)
		}
	}
}

// poseErr sums the reprojection distance of the mesh vertices between two poses.
func poseErr(mesh *Mesh, a, b Pose, k [3][3]float64) float64 {
	pa := ProjectVertices(mesh, a, k)
	pb := ProjectVertices(mesh, b, k)
	var s float64
	for i := range pa {
		s += pa[i].sub(pb[i]).norm()
	}
	return s / float64(len(pa))
}

func TestRapidConverges(t *testing.T) {
	mesh := unitCube()
	k := NewCamera(500, 500, 240, 240)
	truePose := Pose{Rvec: [3]float64{0.15, -0.2, 0.1}, Tvec: [3]float64{0.2, -0.1, 6.0}}
	img := renderSilhouette(mesh, truePose, k, 480, 480)

	initPose := Pose{Rvec: [3]float64{0.11, -0.15, 0.07}, Tvec: [3]float64{0.0, 0.05, 5.75}}

	tracker := NewRapid(mesh)
	e0 := poseErr(mesh, initPose, truePose, k)
	_, ratio0, rmsd0 := tracker.Proceed(img, 80, 20, k, initPose)
	if ratio0 < 0.4 {
		t.Fatalf("too few correspondences on first iteration: ratio=%.2f", ratio0)
	}

	finalPose, ratio := tracker.Compute(img, 80, 20, k, initPose, TermCriteria{MaxCount: 40, Epsilon: 1e-6})
	e1 := poseErr(mesh, finalPose, truePose, k)

	if e1 >= e0 {
		t.Fatalf("reprojection error did not decrease: e0=%.3f e1=%.3f", e0, e1)
	}
	if e1 > 0.5*e0 {
		t.Fatalf("insufficient convergence: e0=%.3f e1=%.3f", e0, e1)
	}
	_, _, rmsdF := tracker.Proceed(img, 80, 20, k, finalPose)
	if rmsdF > rmsd0 {
		t.Fatalf("rmsd did not improve: rmsd0=%.3f rmsdF=%.3f", rmsd0, rmsdF)
	}
	if ratio < 0.4 {
		t.Fatalf("final ratio too low: %.2f", ratio)
	}
	t.Logf("converged: e0=%.3f -> e1=%.3f, rmsd0=%.3f -> rmsdF=%.3f, ratio=%.2f", e0, e1, rmsd0, rmsdF, ratio)
}

func TestOLSAndGOSConverge(t *testing.T) {
	mesh := unitCube()
	k := NewCamera(500, 500, 240, 240)
	truePose := Pose{Rvec: [3]float64{0.1, -0.15, 0.05}, Tvec: [3]float64{0.1, 0.0, 6.0}}
	img := renderSilhouette(mesh, truePose, k, 480, 480)
	initPose := Pose{Rvec: [3]float64{0.07, -0.11, 0.03}, Tvec: [3]float64{0.0, 0.05, 5.8}}

	trackers := map[string]Tracker{
		"ols": NewOLSTracker(mesh, 8, 20),
		"gos": NewGOSTracker(mesh, 4, 20),
	}
	for name, tr := range trackers {
		tr.ClearState()
		e0 := poseErr(mesh, initPose, truePose, k)
		finalPose, ratio := tr.Compute(img, 80, 20, k, initPose, TermCriteria{MaxCount: 40, Epsilon: 1e-6})
		e1 := poseErr(mesh, finalPose, truePose, k)
		if e1 >= e0 {
			t.Fatalf("%s: error did not decrease e0=%.3f e1=%.3f", name, e0, e1)
		}
		if ratio < 0.4 {
			t.Fatalf("%s: ratio too low %.2f", name, ratio)
		}
		t.Logf("%s converged e0=%.3f -> e1=%.3f ratio=%.2f", name, e0, e1, ratio)
	}
}

func TestFindAndConvertCorrespondencies(t *testing.T) {
	// Build a bundle whose rows have an edge at a known column.
	rows, length := 4, 5
	width := 2*length + 1
	bundle := cv.NewMat(rows, width, 1)
	for i := 0; i < rows; i++ {
		for j := 0; j < width; j++ {
			if j >= length+1 {
				bundle.Set(i, j, 0, 255)
			}
		}
	}
	cols, resp := FindCorrespondencies(bundle)
	for i := 0; i < rows; i++ {
		if cols[i] != length && cols[i] != length+1 {
			t.Fatalf("row %d edge column = %d, want near %d", i, cols[i], length)
		}
		if resp[i] < 100 {
			t.Fatalf("row %d weak response %.1f", i, resp[i])
		}
	}
	// Corresponding source locations and control points.
	ctl := make([]ControlPoint, rows)
	locs := make([][]cv.Point, rows)
	for i := 0; i < rows; i++ {
		ctl[i] = ControlPoint{Image: Point2f{X: 10, Y: float64(i)}, Object: [3]float64{0, 0, float64(i)}, Normal: Point2f{X: 1, Y: 0}}
		locs[i] = make([]cv.Point, width)
		for j := 0; j < width; j++ {
			locs[i][j] = cv.Point{X: 10 + j - length, Y: i}
		}
	}
	matches, mask := ConvertCorrespondencies(cols, resp, locs, ctl, 50)
	if len(matches) != rows {
		t.Fatalf("expected %d matches, got %d", rows, len(matches))
	}
	for i := range mask {
		if !mask[i] {
			t.Fatalf("mask[%d] should be true", i)
		}
	}
	// High threshold rejects all.
	m2, _ := ConvertCorrespondencies(cols, resp, locs, ctl, 1e6)
	if len(m2) != 0 {
		t.Fatalf("expected zero matches with high threshold, got %d", len(m2))
	}
}

func TestDrawHelpers(t *testing.T) {
	mesh := unitCube()
	k := NewCamera(500, 500, 240, 240)
	pose := Pose{Rvec: [3]float64{0.15, -0.2, 0.1}, Tvec: [3]float64{0.2, -0.1, 6.0}}
	img := cv.NewMat(480, 480, 3)
	pts := ProjectVertices(mesh, pose, k)
	DrawWireframe(img, pts, mesh.Tris, cv.NewScalar(0, 255, 0), true)
	DrawWireframe(img, pts, mesh.Tris, cv.NewScalar(255, 0, 0), false)

	cps := ExtractControlPoints(64, 20, mesh, pose, k, 480, 480)
	gray := renderSilhouette(mesh, pose, k, 480, 480)
	bundle, locs := ExtractLineBundle(20, cps, gray)
	DrawSearchLines(img, locs, cv.NewScalar(0, 0, 255))
	cols, _ := FindCorrespondencies(bundle)
	DrawCorrespondencies(bundle, cols, nil)

	// Ensure something was drawn.
	var nonzero int
	for _, v := range img.Data {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("draw helpers produced an empty image")
	}
}

func TestExtractLineBundleEmpty(t *testing.T) {
	img := cv.NewMat(10, 10, 1)
	if b, l := ExtractLineBundle(5, nil, img); b != nil || l != nil {
		t.Fatal("empty control points should yield nil bundle")
	}
	if c, r := FindCorrespondencies(nil); c != nil || r != nil {
		t.Fatal("nil bundle should yield nil results")
	}
}

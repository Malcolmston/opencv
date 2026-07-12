package rgbd

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- synthetic-scene renderers used across the new tests ---------------------

// camDir returns the camera-frame ray direction through pixel (u,v) for
// intrinsics k (Z component 1).
func camDir(u, v int, k [3][3]float64) [3]float64 {
	fx, fy := k[0][0], k[1][1]
	cx, cy := k[0][2], k[1][2]
	return [3]float64{(float64(u) - cx) / fx, (float64(v) - cy) / fy, 1}
}

// renderPlaneFrame renders the depth and an intensity image of a world-space
// plane (n·P + off = 0) painted with texture f, seen by a camera with extrinsic
// E (world->camera).
func renderPlaneFrame(e Pose, k [3][3]float64, rows, cols int, n [3]float64, off float64, f func([3]float64) float64) (gray, depth *cv.FloatMat) {
	inv := e.Inverse()
	origin := inv.T
	gray = cv.NewFloatMat(rows, cols)
	depth = cv.NewFloatMat(rows, cols)
	nO := dot3(n, origin)
	for v := 0; v < rows; v++ {
		for u := 0; u < cols; u++ {
			dir := matVec3(inv.R, camDir(u, v, k))
			den := dot3(n, dir)
			if math.Abs(den) < 1e-9 {
				continue
			}
			z := -(nO + off) / den
			if z <= 0 {
				continue
			}
			world := add3(origin, scale3(dir, z))
			depth.Data[v*cols+u] = z
			gray.Data[v*cols+u] = f(world)
		}
	}
	return gray, depth
}

// facet is one plane of a piecewise-planar scene: n·P + off = 0.
type facet struct {
	n   [3]float64
	off float64
}

// scene returns three facets with strongly differing normals, meeting in a
// corner in front of the camera. Its varied normals fully constrain a rigid
// alignment (no rotational symmetry), unlike a single plane or a centred sphere.
func scene() []facet {
	mk := func(n, through [3]float64) facet {
		u := normalize3(n)
		return facet{n: u, off: -dot3(u, through)}
	}
	return []facet{
		mk([3]float64{0.8, 0.2, 1}, [3]float64{0, 0, 2.2}),
		mk([3]float64{-0.6, 0.5, 1}, [3]float64{0, 0, 2.6}),
		mk([3]float64{0.15, -0.7, 1}, [3]float64{0, 0, 2.4}),
	}
}

// renderFacetDepth renders the nearest-surface depth of a piecewise-planar scene
// seen by a camera with extrinsic E (world->camera).
func renderFacetDepth(e Pose, k [3][3]float64, rows, cols int, facets []facet) *cv.FloatMat {
	inv := e.Inverse()
	origin := inv.T
	d := cv.NewFloatMat(rows, cols)
	for v := 0; v < rows; v++ {
		for u := 0; u < cols; u++ {
			dir := matVec3(inv.R, camDir(u, v, k))
			best := math.Inf(1)
			for _, f := range facets {
				den := dot3(f.n, dir)
				if math.Abs(den) < 1e-9 {
					continue
				}
				z := -(dot3(f.n, origin) + f.off) / den
				if z > 0 && z < best {
					best = z
				}
			}
			if !math.IsInf(best, 1) {
				d.Data[v*cols+u] = best
			}
		}
	}
	return d
}

func rvecDiff(a, b [3]float64) float64 { return norm3(sub3(a, b)) }

// --- pose / rotation ---------------------------------------------------------

func TestRodriguesRoundTrip(t *testing.T) {
	rvecs := [][3]float64{
		{0, 0, 0}, {0.01, 0, 0}, {0.3, -0.2, 0.1}, {1.2, 0.5, -0.7},
	}
	for _, rv := range rvecs {
		r := Rodrigues(rv)
		got := InverseRodrigues(r)
		if rvecDiff(got, rv) > 1e-9 {
			t.Fatalf("round trip %v -> %v", rv, got)
		}
		// R must be orthonormal with det +1.
		if !approx(det3(r), 1, 1e-9) {
			t.Fatalf("det(R)=%g for %v", det3(r), rv)
		}
	}
}

func TestPoseComposeInverse(t *testing.T) {
	p := Pose{R: Rodrigues([3]float64{0.2, -0.1, 0.3}), T: [3]float64{0.5, -0.2, 0.1}}
	x := [3]float64{1.3, -0.4, 2.0}
	back := p.Inverse().Apply(p.Apply(x))
	for i := 0; i < 3; i++ {
		if !approx(back[i], x[i], 1e-9) {
			t.Fatalf("inverse-compose[%d]=%g want %g", i, back[i], x[i])
		}
	}
	id := p.Compose(p.Inverse())
	if !approx(id.T[0], 0, 1e-9) || !approx(id.R[0][0], 1, 1e-9) {
		t.Fatalf("p∘p⁻¹ not identity: %+v", id)
	}
}

// --- point-to-plane ICP ------------------------------------------------------

func facetCloudAndNormals(e Pose, k [3][3]float64, rows, cols int, facets []facet) (pts, normals [][3]float64) {
	depth := renderFacetDepth(e, k, rows, cols, facets)
	all := DepthTo3D(depth, k)
	nAll := Compute3DNormals(depth, k)
	for i := range all {
		if all[i][2] > 0 && norm3(nAll[i]) > 0 {
			pts = append(pts, all[i])
			normals = append(normals, nAll[i])
		}
	}
	return pts, normals
}

func TestICPPointToPlaneLowResidual(t *testing.T) {
	k := Camera{Fx: 60, Fy: 60, Cx: 20, Cy: 20}.K()
	rows, cols := 40, 40
	facets := scene()
	mTrue := Pose{R: Rodrigues([3]float64{0.02, -0.015, 0.012}), T: [3]float64{0.03, -0.02, 0.015}}

	src, _ := facetCloudAndNormals(IdentityPose(), k, rows, cols, facets)
	dst, dstN := facetCloudAndNormals(mTrue, k, rows, cols, facets)

	pose, residual := ICPPointToPlane(src, dst, dstN, 30)
	if residual > 1.5e-2 {
		t.Fatalf("point-to-plane residual too high: %g", residual)
	}
	gotRvec := InverseRodrigues(pose.R)
	trueRvec := InverseRodrigues(mTrue.R)
	if rvecDiff(gotRvec, trueRvec) > 3e-2 {
		t.Fatalf("rotation not recovered: got %v want %v", gotRvec, trueRvec)
	}
	if norm3(sub3(pose.T, mTrue.T)) > 3e-2 {
		t.Fatalf("translation not recovered: got %v want %v", pose.T, mTrue.T)
	}
}

// --- ICP odometry ------------------------------------------------------------

func TestICPOdometryRecoversMotion(t *testing.T) {
	k := Camera{Fx: 90, Fy: 90, Cx: 32, Cy: 24}.K()
	rows, cols := 48, 64
	facets := scene()
	mTrue := Pose{R: Rodrigues([3]float64{0.015, -0.012, 0.02}), T: [3]float64{0.02, -0.015, 0.01}}

	d0 := renderFacetDepth(IdentityPose(), k, rows, cols, facets)
	d1 := renderFacetDepth(mTrue, k, rows, cols, facets)

	res := ICPOdometry(d0, d1, k, DefaultOdometryOptions())
	if res.RMSError > 1.5e-2 {
		t.Fatalf("ICP odometry RMS too high: %g", res.RMSError)
	}
	gotRvec := InverseRodrigues(res.Pose.R)
	trueRvec := InverseRodrigues(mTrue.R)
	if rvecDiff(gotRvec, trueRvec) > 2e-2 {
		t.Fatalf("rotation not recovered: got %v want %v", gotRvec, trueRvec)
	}
	if norm3(sub3(res.Pose.T, mTrue.T)) > 2e-2 {
		t.Fatalf("translation not recovered: got %v want %v (rms %g)", res.Pose.T, mTrue.T, res.RMSError)
	}
}

// --- RGB-D (photometric) odometry -------------------------------------------

func texture(p [3]float64) float64 {
	return 0.5 + 0.2*math.Sin(2.3*p[0]) + 0.2*math.Cos(2.1*p[1]) + 0.05*math.Sin(1.7*(p[0]+p[1]))
}

func TestRgbdOdometryRecoversMotion(t *testing.T) {
	k := Camera{Fx: 90, Fy: 90, Cx: 32, Cy: 24}.K()
	rows, cols := 48, 64
	n := normalize3([3]float64{0.18, 0.13, 1})
	off := -dot3(n, [3]float64{0, 0, 2.5}) // plane through (0,0,2.5)
	mTrue := Pose{R: Rodrigues([3]float64{0.006, -0.005, 0.008}), T: [3]float64{0.012, -0.009, 0.006}}

	g0, d0 := renderPlaneFrame(IdentityPose(), k, rows, cols, n, off, texture)
	g1, d1 := renderPlaneFrame(mTrue, k, rows, cols, n, off, texture)

	opts := DefaultOdometryOptions()
	opts.MaxIterations = 40
	res := RgbdOdometry(g0, d0, g1, d1, k, opts)
	gotRvec := InverseRodrigues(res.Pose.R)
	trueRvec := InverseRodrigues(mTrue.R)
	if rvecDiff(gotRvec, trueRvec) > 8e-3 {
		t.Fatalf("rotation not recovered: got %v want %v (rms %g)", gotRvec, trueRvec, res.RMSError)
	}
	if norm3(sub3(res.Pose.T, mTrue.T)) > 8e-3 {
		t.Fatalf("translation not recovered: got %v want %v (rms %g)", res.Pose.T, mTrue.T, res.RMSError)
	}
}

func TestRgbdICPOdometryRecoversMotion(t *testing.T) {
	k := Camera{Fx: 90, Fy: 90, Cx: 32, Cy: 24}.K()
	rows, cols := 48, 64
	n := normalize3([3]float64{0.18, 0.13, 1})
	off := -dot3(n, [3]float64{0, 0, 2.5})
	mTrue := Pose{R: Rodrigues([3]float64{0.008, -0.006, 0.01}), T: [3]float64{0.014, -0.01, 0.007}}

	g0, d0 := renderPlaneFrame(IdentityPose(), k, rows, cols, n, off, texture)
	g1, d1 := renderPlaneFrame(mTrue, k, rows, cols, n, off, texture)

	opts := DefaultOdometryOptions()
	opts.MaxIterations = 40
	opts.ICPWeight = 30 // geometric residuals are metres, photometric are ~unit
	res := RgbdICPOdometry(g0, d0, g1, d1, k, opts)
	gotRvec := InverseRodrigues(res.Pose.R)
	trueRvec := InverseRodrigues(mTrue.R)
	if rvecDiff(gotRvec, trueRvec) > 1e-2 {
		t.Fatalf("rotation not recovered: got %v want %v (rms %g)", gotRvec, trueRvec, res.RMSError)
	}
	if norm3(sub3(res.Pose.T, mTrue.T)) > 1e-2 {
		t.Fatalf("translation not recovered: got %v want %v", res.Pose.T, mTrue.T)
	}
}

// --- TSDF volume -------------------------------------------------------------

func TestTSDFIntegrateRaycastPlane(t *testing.T) {
	k := Camera{Fx: 80, Fy: 80, Cx: 24, Cy: 24}.K()
	rows, cols := 48, 48
	const planeZ = 1.0
	depth := constDepth(rows, cols, planeZ)

	vol := NewTSDFVolume([3]int{60, 60, 60}, 0.02, [3]float64{-0.6, -0.6, 0.5}, 0.1)
	vol.Integrate(depth, k, IdentityPose())

	out := vol.Raycast(k, IdentityPose(), rows, cols)
	checked := 0
	for v := 12; v < 36; v++ {
		for u := 12; u < 36; u++ {
			z := out.At(v, u)
			if z == 0 {
				continue
			}
			if !approx(z, planeZ, 0.03) {
				t.Fatalf("raycast depth at (%d,%d)=%g, want ~%g", u, v, z, planeZ)
			}
			checked++
		}
	}
	if checked < 100 {
		t.Fatalf("too few raycast hits: %d", checked)
	}

	cloud := vol.FetchPointCloud()
	if len(cloud) == 0 {
		t.Fatal("FetchPointCloud returned no points")
	}
	near := 0
	for _, p := range cloud {
		if math.Abs(p[2]-planeZ) < 0.05 {
			near++
		}
	}
	if near < len(cloud)/2 {
		t.Fatalf("only %d/%d fetched points lie near the plane", near, len(cloud))
	}
	// Determinism.
	cloud2 := vol.FetchPointCloud()
	if len(cloud2) != len(cloud) {
		t.Fatal("FetchPointCloud is not deterministic")
	}
}

// --- depth cleaning / filtering ---------------------------------------------

func TestDepthCleanerFillsHoles(t *testing.T) {
	depth := constDepth(20, 20, 2.0)
	// Punch a few single-pixel holes.
	depth.Data[5*20+5] = 0
	depth.Data[10*20+10] = 0
	depth.Data[10*20+11] = 0
	c := NewDepthCleaner(2)
	out := c.Clean(depth)
	for _, idx := range []int{5*20 + 5, 10*20 + 10, 10*20 + 11} {
		if out.Data[idx] <= 0 {
			t.Fatalf("hole at %d not filled", idx)
		}
		if !approx(out.Data[idx], 2.0, 1e-9) {
			t.Fatalf("filled value %g, want 2.0", out.Data[idx])
		}
	}
}

func TestBilateralDepthFilterPreservesConstant(t *testing.T) {
	depth := constDepth(16, 16, 3.0)
	out := BilateralDepthFilter(depth, 2, 2.0, 0.1)
	for i, z := range out.Data {
		if !approx(z, 3.0, 1e-9) {
			t.Fatalf("filtered[%d]=%g, want 3.0", i, z)
		}
	}
	// A noisy version should be smoothed toward the mean without blowing up.
	noisy := constDepth(16, 16, 3.0)
	noisy.Data[8*16+8] = 3.4
	sm := BilateralDepthFilter(noisy, 2, 2.0, 1.0)
	if sm.At(8, 8) >= 3.4 || sm.At(8, 8) <= 3.0 {
		t.Fatalf("noisy centre not smoothed: %g", sm.At(8, 8))
	}
}

func TestRescaleDepth(t *testing.T) {
	depth := cv.NewFloatMat(2, 2)
	depth.Data = []float64{1000, 0, 2000, 500}
	out := RescaleDepth(depth, 0.001)
	want := []float64{1.0, 0, 2.0, 0.5}
	for i := range want {
		if !approx(out.Data[i], want[i], 1e-12) {
			t.Fatalf("rescaled[%d]=%g want %g", i, out.Data[i], want[i])
		}
	}
}

// --- RgbdNormals methods -----------------------------------------------------

func TestRgbdNormalsFlatPlaneAllMethods(t *testing.T) {
	depth := constDepth(40, 40, 4.0)
	for _, m := range []NormalsMethod{NormalsFALS, NormalsLINEMOD, NormalsSRI} {
		rn := NewRgbdNormals(40, 40, testK, 1, m)
		normals := rn.Compute(depth)
		checked := 0
		for v := 3; v < 37; v++ {
			for u := 3; u < 37; u++ {
				n := normals[v*40+u]
				if norm3(n) == 0 {
					continue
				}
				if !approx(math.Abs(n[2]), 1, 5e-3) {
					t.Fatalf("method %d normal at (%d,%d)=%v not along Z", m, u, v, n)
				}
				if n[2] > 0 {
					t.Fatalf("method %d normal at (%d,%d) not camera-facing: %v", m, u, v, n)
				}
				checked++
			}
		}
		if checked == 0 {
			t.Fatalf("method %d produced no interior normals", m)
		}
	}
}

// --- sparse / registration / warp -------------------------------------------

func TestDepthTo3dSparse(t *testing.T) {
	depth := constDepth(10, 10, 2.0)
	depth.Data[3*10+4] = 0 // invalid
	pixels := [][2]int{{5, 5}, {4, 3}, {20, 20}}
	pts, valid := DepthTo3dSparse(depth, testK, pixels)
	if !valid[0] || valid[1] || valid[2] {
		t.Fatalf("validity mask wrong: %v", valid)
	}
	// Pixel (5,5) at depth 2 back-projects consistently with DepthTo3D.
	full := DepthTo3D(depth, testK)
	want := full[5*10+5]
	if pts[0] != want {
		t.Fatalf("sparse point %v != dense %v", pts[0], want)
	}
}

func TestRegisterDepthDistortedZeroMatchesUndistorted(t *testing.T) {
	depth := constDepth(30, 30, 2.0)
	depth.Data[10*30+12] = 2.4
	ident := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	a := RegisterDepth(depth, testK, testK, ident, [3]float64{0, 0, 0}, 30, 30)
	b := RegisterDepthDistorted(depth, testK, testK, [5]float64{}, ident, [3]float64{0, 0, 0}, 30, 30)
	for i := range a.Data {
		if !approx(a.Data[i], b.Data[i], 1e-9) {
			t.Fatalf("distorted-zero differs from undistorted at %d: %g vs %g", i, b.Data[i], a.Data[i])
		}
	}
	// A non-zero radial coefficient must change some pixel mapping.
	c := RegisterDepthDistorted(depth, testK, testK, [5]float64{0.2, 0, 0, 0, 0}, ident, [3]float64{0, 0, 0}, 30, 30)
	diff := false
	for i := range a.Data {
		if !approx(a.Data[i], c.Data[i], 1e-6) {
			diff = true
			break
		}
	}
	if !diff {
		t.Fatal("radial distortion had no effect")
	}
}

func TestWarpFrameIdentity(t *testing.T) {
	k := Camera{Fx: 50, Fy: 50, Cx: 15, Cy: 15}.K()
	depth := constDepth(30, 30, 2.0)
	gray := cv.NewFloatMat(30, 30)
	for i := range gray.Data {
		gray.Data[i] = float64(i%7) / 7
	}
	wImg, wDepth, mask := WarpFrame(gray, depth, k, IdentityPose(), 30, 30)
	for v := 0; v < 30; v++ {
		for u := 0; u < 30; u++ {
			if !mask[v*30+u] {
				t.Fatalf("identity warp left (%d,%d) unmapped", u, v)
			}
			if !approx(wDepth.At(v, u), 2.0, 1e-9) {
				t.Fatalf("warped depth at (%d,%d)=%g want 2.0", u, v, wDepth.At(v, u))
			}
			if !approx(wImg.At(v, u), gray.At(v, u), 1e-9) {
				t.Fatalf("warped image at (%d,%d)=%g want %g", u, v, wImg.At(v, u), gray.At(v, u))
			}
		}
	}
}

// --- colored ICP -------------------------------------------------------------

func TestColoredICPRecoversMotion(t *testing.T) {
	// A cloud where colour breaks the geometric symmetry of a square ring.
	var src [][3]float64
	var col []float64
	for i := 0; i < 12; i++ {
		a := float64(i) / 12 * 2 * math.Pi
		src = append(src, [3]float64{math.Cos(a), math.Sin(a), 0.2 * math.Sin(3*a)})
		col = append(col, 0.5+0.5*math.Sin(a))
	}
	mTrue := Pose{R: Rodrigues([3]float64{0.01, 0.02, 0.03}), T: [3]float64{0.05, -0.03, 0.02}}
	dst := make([][3]float64, len(src))
	for i, p := range src {
		dst[i] = mTrue.Apply(p)
	}
	pose, err := ColoredICP(src, dst, col, col, 5.0, 30)
	if err > 1e-4 {
		t.Fatalf("colored ICP residual too high: %g", err)
	}
	if norm3(sub3(pose.T, mTrue.T)) > 1e-2 {
		t.Fatalf("translation not recovered: got %v want %v", pose.T, mTrue.T)
	}
	if rvecDiff(InverseRodrigues(pose.R), InverseRodrigues(mTrue.R)) > 1e-2 {
		t.Fatalf("rotation not recovered")
	}
}

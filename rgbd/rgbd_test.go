package rgbd

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// testK is a simple pinhole intrinsic used across the tests.
var testK = Camera{Fx: 100, Fy: 100, Cx: 32, Cy: 24}.K()

// constDepth builds a rows×cols depth map with every pixel set to z.
func constDepth(rows, cols int, z float64) *cv.FloatMat {
	d := cv.NewFloatMat(rows, cols)
	for i := range d.Data {
		d.Data[i] = z
	}
	return d
}

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestDepthTo3DConstantPlaneIsFlat(t *testing.T) {
	const z = 3.0
	depth := constDepth(48, 64, z)
	pts := DepthTo3D(depth, testK)
	if len(pts) != 48*64 {
		t.Fatalf("got %d points, want %d", len(pts), 48*64)
	}
	// Every point sits at depth z, so the cloud is the flat plane Z = z.
	for i, p := range pts {
		if !approx(p[2], z, 1e-9) {
			t.Fatalf("point %d has Z=%g, want %g", i, p[2], z)
		}
	}
	// The principal-point pixel (cx, cy) back-projects to the optical axis.
	center := pts[24*64+32]
	if !approx(center[0], 0, 1e-9) || !approx(center[1], 0, 1e-9) {
		t.Fatalf("center point = %v, want ~(0,0,%g)", center, z)
	}
	// X spans should match the pinhole geometry: pixel u=0 maps to (0-cx)*z/fx.
	corner := pts[0]
	wantX := (0 - 32.0) * z / 100.0
	wantY := (0 - 24.0) * z / 100.0
	if !approx(corner[0], wantX, 1e-9) || !approx(corner[1], wantY, 1e-9) {
		t.Fatalf("corner point = %v, want X=%g Y=%g", corner, wantX, wantY)
	}
}

func TestComputeNormalsFlatPlaneAlongAxis(t *testing.T) {
	depth := constDepth(48, 64, 5.0)
	normals := Compute3DNormals(depth, testK)
	checked := 0
	for v := 1; v < 47; v++ {
		for u := 1; u < 63; u++ {
			n := normals[v*64+u]
			// A fronto-parallel plane has a normal along the optical (Z) axis,
			// pointing back toward the camera at the origin: (0, 0, -1).
			if !approx(n[0], 0, 1e-9) || !approx(n[1], 0, 1e-9) {
				t.Fatalf("normal at (%d,%d)=%v not along Z axis", u, v, n)
			}
			if !approx(math.Abs(n[2]), 1, 1e-9) {
				t.Fatalf("normal at (%d,%d)=%v not unit length along Z", u, v, n)
			}
			if n[2] > 0 {
				t.Fatalf("normal at (%d,%d)=%v not oriented toward camera", u, v, n)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no interior normals checked")
	}
}

func TestPlaneSegmentationDominantPlane(t *testing.T) {
	var pts [][3]float64
	// A dense 20×20 grid on the plane Z = 0.
	const grid = 20
	planeCount := 0
	for i := 0; i < grid; i++ {
		for j := 0; j < grid; j++ {
			pts = append(pts, [3]float64{float64(i) * 0.1, float64(j) * 0.1, 0})
			planeCount++
		}
	}
	// A scattering of far-off points that must not join the plane.
	off := [][3]float64{
		{0.5, 0.5, 5}, {0.2, 0.9, 6}, {1.5, 0.3, 7}, {0.8, 1.2, 8},
		{1.1, 1.1, 9}, {0.4, 0.7, 10},
	}
	pts = append(pts, off...)

	opts := DefaultPlaneOptions()
	opts.DistanceThreshold = 0.01
	opts.MaxPlanes = 1
	planes, labels := PlaneSegmentation(pts, opts)
	if len(planes) != 1 {
		t.Fatalf("got %d planes, want 1", len(planes))
	}
	if planes[0].Inliers != planeCount {
		t.Fatalf("dominant plane has %d inliers, want %d", planes[0].Inliers, planeCount)
	}
	// The recovered normal must be along Z (the plane is Z = 0).
	if !approx(math.Abs(planes[0].Normal[2]), 1, 1e-6) {
		t.Fatalf("plane normal %v not along Z", planes[0].Normal)
	}
	// Off-plane points stay unlabeled.
	for k := 0; k < len(off); k++ {
		if labels[planeCount+k] != -1 {
			t.Fatalf("off-plane point %d was labeled %d", k, labels[planeCount+k])
		}
	}
	// Determinism: identical inputs give identical inlier counts.
	planes2, _ := PlaneSegmentation(pts, opts)
	if planes2[0].Inliers != planes[0].Inliers {
		t.Fatal("PlaneSegmentation is not deterministic")
	}
}

func TestICPRecoversKnownTransform(t *testing.T) {
	// A source cloud: a small asymmetric constellation of points.
	src := [][3]float64{
		{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
		{1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
		{0.5, 0.2, 0.7}, {0.3, 0.8, 0.4}, {0.9, 0.5, 0.1}, {0.2, 0.4, 0.9},
	}
	// A known rotation of 30° about the Z axis and a translation.
	ang := math.Pi / 6
	ca, sa := math.Cos(ang), math.Sin(ang)
	rTrue := [3][3]float64{{ca, -sa, 0}, {sa, ca, 0}, {0, 0, 1}}
	tTrue := [3]float64{0.5, -0.3, 0.2}
	dst := make([][3]float64, len(src))
	for i, p := range src {
		dst[i] = add3(matVec3(rTrue, p), tTrue)
	}

	r, tr, err := ICP(src, dst, 50)
	if err > 1e-6 {
		t.Fatalf("ICP residual error %g too large", err)
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if !approx(r[i][j], rTrue[i][j], 1e-4) {
				t.Fatalf("R[%d][%d]=%g, want %g", i, j, r[i][j], rTrue[i][j])
			}
		}
		if !approx(tr[i], tTrue[i], 1e-4) {
			t.Fatalf("t[%d]=%g, want %g", i, tr[i], tTrue[i])
		}
	}
	// The recovered transform should map src onto dst.
	for i, p := range src {
		got := add3(matVec3(r, p), tr)
		for k := 0; k < 3; k++ {
			if !approx(got[k], dst[i][k], 1e-4) {
				t.Fatalf("mapped src[%d]=%v, want %v", i, got, dst[i])
			}
		}
	}
}

func TestVoxelDownsampleReducesAndPreservesExtent(t *testing.T) {
	// A dense 30×30×1 grid of points spaced 0.05 apart, spanning [0, 1.45].
	var pts [][3]float64
	for i := 0; i < 30; i++ {
		for j := 0; j < 30; j++ {
			pts = append(pts, [3]float64{float64(i) * 0.05, float64(j) * 0.05, 0})
		}
	}
	minX, minY, maxX, maxY := extent(pts)

	out := VoxelDownsample(pts, 0.2)
	if len(out) >= len(pts) {
		t.Fatalf("downsample did not reduce count: %d -> %d", len(pts), len(out))
	}
	if len(out) == 0 {
		t.Fatal("downsample produced no points")
	}
	oMinX, oMinY, oMaxX, oMaxY := extent(out)
	// Averaging keeps the extent within roughly one leaf of the original.
	const tol = 0.2
	if oMinX < minX-tol || oMinX > minX+tol || oMaxX > maxX+tol || oMaxX < maxX-tol {
		t.Fatalf("X extent not preserved: [%g,%g] vs [%g,%g]", oMinX, oMaxX, minX, maxX)
	}
	if oMinY < minY-tol || oMinY > minY+tol || oMaxY > maxY+tol || oMaxY < maxY-tol {
		t.Fatalf("Y extent not preserved: [%g,%g] vs [%g,%g]", oMinY, oMaxY, minY, maxY)
	}
	// Determinism.
	out2 := VoxelDownsample(pts, 0.2)
	if len(out2) != len(out) {
		t.Fatal("VoxelDownsample is not deterministic")
	}
	for i := range out {
		if out[i] != out2[i] {
			t.Fatalf("VoxelDownsample point %d differs across runs", i)
		}
	}
}

// extent returns the axis-aligned XY bounds of a point cloud.
func extent(pts [][3]float64) (minX, minY, maxX, maxY float64) {
	minX, minY = math.Inf(1), math.Inf(1)
	maxX, maxY = math.Inf(-1), math.Inf(-1)
	for _, p := range pts {
		minX = math.Min(minX, p[0])
		minY = math.Min(minY, p[1])
		maxX = math.Max(maxX, p[0])
		maxY = math.Max(maxY, p[1])
	}
	return
}

func TestRegisterDepthIdentityPose(t *testing.T) {
	// With identical intrinsics and an identity pose, registration reproduces
	// the input depth pixel-for-pixel.
	depth := constDepth(48, 64, 2.0)
	// Vary a couple of pixels to make the mapping non-trivial.
	depth.Data[10*64+20] = 2.5
	depth.Data[30*64+40] = 1.5
	ident := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	out := RegisterDepth(depth, testK, testK, ident, [3]float64{0, 0, 0}, 48, 64)
	for v := 0; v < 48; v++ {
		for u := 0; u < 64; u++ {
			if !approx(out.At(v, u), depth.At(v, u), 1e-9) {
				t.Fatalf("registered depth at (%d,%d)=%g, want %g", u, v, out.At(v, u), depth.At(v, u))
			}
		}
	}
}

func TestRegisterDepthTranslationShift(t *testing.T) {
	// A pure translation of +Z (moving the colour camera back) leaves the
	// projected pixel put but increases its depth by the shift, verifying the
	// transform is applied.
	depth := constDepth(20, 20, 4.0)
	ident := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	out := RegisterDepth(depth, testK, testK, ident, [3]float64{0, 0, 1.0}, 20, 20)
	// The principal-point pixel maps to itself with depth 4+1=5.
	if !approx(out.At(10, 10), 5.0, 1e-6) {
		t.Fatalf("shifted center depth = %g, want 5.0", out.At(10, 10))
	}
}

func TestSVD3Reconstructs(t *testing.T) {
	a := [3][3]float64{{4, 1, 2}, {1, 3, 0}, {2, 0, 5}}
	u, s, v := svd3(a)
	// Reconstruct U·diag(s)·Vᵀ and compare to A.
	var recon [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				recon[i][j] += u[i][k] * s[k] * v[j][k]
			}
		}
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if !approx(recon[i][j], a[i][j], 1e-9) {
				t.Fatalf("SVD reconstruction[%d][%d]=%g, want %g", i, j, recon[i][j], a[i][j])
			}
		}
	}
	// Singular values are non-increasing.
	if s[0] < s[1] || s[1] < s[2] {
		t.Fatalf("singular values not sorted: %v", s)
	}
}

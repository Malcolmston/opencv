package stereo2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// makeStereoPair builds a synthetic rectified stereo pair with a known constant
// disparity. Each column of the left image holds a deterministic pseudo-random
// intensity that is constant down the column; the right image is the left image
// shifted left by d0 so that left(y,x) == right(y,x-d0). Block matching should
// therefore recover disparity d0 for every interior pixel.
func makeStereoPair(rows, cols, d0 int) (*cv.Mat, *cv.Mat) {
	left := cv.NewMat(rows, cols, 1)
	right := cv.NewMat(rows, cols, 1)
	col := make([]uint8, cols+d0+8)
	// Deterministic, high-frequency, non-repeating pattern.
	seed := uint32(0x1234567)
	for i := range col {
		seed = seed*1664525 + 1013904223
		col[i] = uint8((seed >> 16) & 0xff)
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			left.Set(y, x, 0, col[x])
			// right(y,x) matches left(y,x+d0)
			right.Set(y, x, 0, col[x+d0])
		}
	}
	return left, right
}

func TestBlockMatchSAD_KnownDisparity(t *testing.T) {
	const rows, cols, d0 = 24, 40, 5
	left, right := makeStereoPair(rows, cols, d0)
	disp := BlockMatchSAD(left, right, 0, 12, 5)

	r := 2
	checked := 0
	for y := r; y < rows-r; y++ {
		for x := d0 + r; x < cols-r; x++ {
			if !disp.Valid(y, x) {
				continue
			}
			got := disp.At(y, x)
			if math.Abs(float64(got-float32(d0))) > 0.001 {
				t.Fatalf("pixel (%d,%d): got disparity %v, want %d", y, x, got, d0)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no interior pixels were validated")
	}
}

func TestBlockMatchSSDandNCC_KnownDisparity(t *testing.T) {
	const rows, cols, d0 = 20, 36, 4
	left, right := makeStereoPair(rows, cols, d0)
	for _, tc := range []struct {
		name string
		disp *DisparityMap
	}{
		{"SSD", BlockMatchSSD(left, right, 0, 10, 5)},
		{"NCC", BlockMatchNCC(left, right, 0, 10, 5)},
	} {
		checked := 0
		for y := 3; y < rows-3; y++ {
			for x := d0 + 3; x < cols-3; x++ {
				if !tc.disp.Valid(y, x) {
					continue
				}
				if math.Abs(float64(tc.disp.At(y, x)-float32(d0))) > 0.001 {
					t.Fatalf("%s pixel (%d,%d): got %v want %d", tc.name, y, x, tc.disp.At(y, x), d0)
				}
				checked++
			}
		}
		if checked == 0 {
			t.Fatalf("%s: no interior pixels validated", tc.name)
		}
	}
}

func TestCostVolumePipelineMatchesBlockMatcher(t *testing.T) {
	const rows, cols, d0 = 18, 34, 6
	left, right := makeStereoPair(rows, cols, d0)

	vol := BuildCostVolume(left, right, 0, 12, false)
	agg := AggregateBoxFilter(vol, 2)
	disp := agg.ToDisparity(0)

	direct := BlockMatchSAD(left, right, 0, 12, 5)

	for y := 2; y < rows-2; y++ {
		for x := d0 + 2; x < cols-2; x++ {
			if !direct.Valid(y, x) {
				continue
			}
			if !disp.Valid(y, x) {
				t.Fatalf("cost-volume pipeline invalid at (%d,%d) where direct is valid", y, x)
			}
			if disp.At(y, x) != direct.At(y, x) {
				t.Fatalf("pipeline mismatch at (%d,%d): volume %v vs direct %v", y, x, disp.At(y, x), direct.At(y, x))
			}
		}
	}
}

func TestCensusMatcher_KnownDisparity(t *testing.T) {
	const rows, cols, d0 = 22, 40, 5
	left, right := makeStereoPair(rows, cols, d0)
	cm := NewCensusMatcher(0, 12, 5, 5, 2)
	disp := cm.Compute(left, right)

	checked := 0
	for y := 5; y < rows-5; y++ {
		for x := d0 + 5; x < cols-5; x++ {
			if !disp.Valid(y, x) {
				continue
			}
			if disp.At(y, x) != float32(d0) {
				t.Fatalf("census (%d,%d): got %v want %d", y, x, disp.At(y, x), d0)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("census: no interior pixels validated")
	}
}

func TestHammingDistance(t *testing.T) {
	cases := []struct {
		a, b uint64
		want int
	}{
		{0, 0, 0},
		{0xFF, 0x00, 8},
		{0b1010, 0b0101, 4},
		{0xFFFFFFFFFFFFFFFF, 0, 64},
	}
	for _, c := range cases {
		if got := HammingDistance(c.a, c.b); got != c.want {
			t.Fatalf("HammingDistance(%x,%x)=%d want %d", c.a, c.b, got, c.want)
		}
	}
}

func TestSGMMatcher_KnownDisparity(t *testing.T) {
	const rows, cols, d0 = 24, 44, 6
	left, right := makeStereoPair(rows, cols, d0)
	sgm := NewSGMMatcher(0, 14)
	sgm.Subpixel = false
	disp := sgm.Compute(left, right)

	checked := 0
	for y := 4; y < rows-4; y++ {
		for x := d0 + 4; x < cols-4; x++ {
			if !disp.Valid(y, x) {
				continue
			}
			if disp.At(y, x) != float32(d0) {
				t.Fatalf("sgm (%d,%d): got %v want %d", y, x, disp.At(y, x), d0)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("sgm: no interior pixels validated")
	}
}

func TestSemiGlobalAggregatePrefersSmoothField(t *testing.T) {
	// Single-pixel cost volume where the raw per-pixel minimum disagrees with
	// its neighbours; SGM smoothing along the row should pull it to the
	// consensus disparity.
	rows, cols, D := 1, 5, 4
	vol := NewCostVolume(rows, cols, D, 0)
	for x := 0; x < cols; x++ {
		for d := 0; d < D; d++ {
			vol.Set(0, x, d, 10)
		}
		vol.Set(0, x, 1, 0) // everyone prefers d=1 ...
	}
	vol.Set(0, 2, 1, 9) // ... except the middle pixel, which barely prefers d=3
	vol.Set(0, 2, 3, 1)

	raw := vol.ToDisparity(0)
	if raw.At(0, 2) != 3 {
		t.Fatalf("raw winner at center = %v, want 3", raw.At(0, 2))
	}
	agg := SemiGlobalAggregate(vol, 5, 40, FourPaths())
	sm := agg.ToDisparity(0)
	if sm.At(0, 2) != 1 {
		t.Fatalf("SGM-smoothed winner at center = %v, want 1", sm.At(0, 2))
	}
}

func TestLeftRightCheckRejectsInconsistent(t *testing.T) {
	left := NewDisparityMap(1, 6)
	right := NewDisparityMap(1, 6)
	// Consistent pixel: left disparity 2 at x=4 -> right x=2 must be 2.
	left.Set(0, 4, 2)
	right.Set(0, 2, 2)
	// Inconsistent pixel: left disparity 3 at x=5 -> right x=2 holds 2 (!=3).
	left.Set(0, 5, 3)

	out := LeftRightCheck(left, right, 0)
	if !out.Valid(0, 4) || out.At(0, 4) != 2 {
		t.Fatalf("consistent pixel wrongly rejected: valid=%v val=%v", out.Valid(0, 4), out.At(0, 4))
	}
	if out.Valid(0, 5) {
		t.Fatalf("inconsistent pixel not rejected")
	}
}

func TestSubpixelParabola(t *testing.T) {
	// Symmetric costs around the true minimum give zero offset; a skew gives a
	// predictable fractional shift.
	vol := NewCostVolume(1, 2, 3, 0)
	// Pixel 0: symmetric V -> winner index 1, offset 0 -> disparity 1.
	vol.Set(0, 0, 0, 4)
	vol.Set(0, 0, 1, 0)
	vol.Set(0, 0, 2, 4)
	// Pixel 1: costs 1,0,4 -> offset = (1-4)/(2*(1-0+4)) = -0.3 -> 0.7.
	vol.Set(0, 1, 0, 1)
	vol.Set(0, 1, 1, 0)
	vol.Set(0, 1, 2, 4)

	disp := RefineSubpixelParabola(vol, 0)
	if math.Abs(float64(disp.At(0, 0)-1.0)) > 1e-5 {
		t.Fatalf("pixel0 subpixel = %v, want 1.0", disp.At(0, 0))
	}
	if math.Abs(float64(disp.At(0, 1)-0.7)) > 1e-5 {
		t.Fatalf("pixel1 subpixel = %v, want 0.7", disp.At(0, 1))
	}
}

func TestSpeckleFilter(t *testing.T) {
	d := NewDisparityMap(5, 5)
	// A big region of disparity 10.
	for y := 0; y < 5; y++ {
		for x := 0; x < 4; x++ {
			d.Set(y, x, 10)
		}
	}
	// A lone speckle of disparity 2 in the corner.
	d.Set(0, 4, 2)

	out := SpeckleFilter(d, 3, 1)
	if out.Valid(0, 4) {
		t.Fatalf("speckle not removed")
	}
	if !out.Valid(2, 2) || out.At(2, 2) != 10 {
		t.Fatalf("large region wrongly removed at (2,2)")
	}
}

func TestMedianFilterDisparity(t *testing.T) {
	d := NewDisparityMap(3, 3)
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			d.Set(y, x, 5)
		}
	}
	d.Set(1, 1, 100) // outlier
	out := MedianFilterDisparity(d, 1)
	if out.At(1, 1) != 5 {
		t.Fatalf("median filter did not suppress outlier: got %v", out.At(1, 1))
	}
}

func TestFillInvalidHorizontal(t *testing.T) {
	d := NewDisparityMap(1, 5)
	d.Set(0, 0, 3)
	d.Set(0, 4, 7)
	// interior 1,2,3 invalid -> filled with min(3,7)=3
	out := FillInvalidHorizontal(d)
	for x := 1; x <= 3; x++ {
		if out.At(0, x) != 3 {
			t.Fatalf("fill at x=%d = %v, want 3", x, out.At(0, x))
		}
	}
}

func TestDepthDisparityRoundTrip(t *testing.T) {
	disp := NewDisparityMap(2, 2)
	disp.Set(0, 0, 10)
	disp.Set(0, 1, 20)
	disp.Set(1, 0, 5)
	// (1,1) stays invalid
	depth := DepthFromDisparity(disp, 100, 0.1) // Z = 10/d
	// d=10 -> Z=1.0 ; d=20 -> Z=0.5 ; d=5 -> Z=2.0
	if math.Abs(float64(depth.At(0, 0)-1.0)) > 1e-5 {
		t.Fatalf("depth(0,0)=%v want 1.0", depth.At(0, 0))
	}
	if math.Abs(float64(depth.At(0, 1)-0.5)) > 1e-5 {
		t.Fatalf("depth(0,1)=%v want 0.5", depth.At(0, 1))
	}
	if depth.Valid(1, 1) {
		t.Fatalf("invalid disparity produced a depth")
	}
	back := DisparityFromDepth(depth, 100, 0.1)
	if math.Abs(float64(back.At(0, 0)-10)) > 1e-4 {
		t.Fatalf("round-trip disparity(0,0)=%v want 10", back.At(0, 0))
	}
}

func TestPointCloudFromDepth(t *testing.T) {
	depth := NewDepthMap(3, 3)
	cam := NewCamera(50, 50, 1, 1) // principal point at (1,1)
	depth.Set(1, 1, 2.0)           // center pixel at Z=2
	depth.Set(0, 2, 4.0)
	pc := PointCloudFromDepth(depth, cam, nil)
	if pc.Len() != 2 {
		t.Fatalf("point count = %d, want 2", pc.Len())
	}
	// Center pixel is at principal point -> X=Y=0, Z=2.
	found := false
	for _, p := range pc.Points {
		if math.Abs(p.Z-2.0) < 1e-9 {
			if math.Abs(p.X) > 1e-9 || math.Abs(p.Y) > 1e-9 {
				t.Fatalf("center point off-axis: %+v", p)
			}
			found = true
		}
	}
	if !found {
		t.Fatal("center point missing")
	}
}

func TestReprojectImageTo3D(t *testing.T) {
	disp := NewDisparityMap(3, 3)
	disp.Set(1, 1, 4)
	Q := QMatrix(1, 1, 50, 0.1) // f=50, baseline=0.1
	pc := ReprojectImageTo3D(disp, Q, nil)
	if pc.Len() != 1 {
		t.Fatalf("point count = %d, want 1", pc.Len())
	}
	// W = d/baseline = 4/0.1 = 40 ; Z = f/W = 50/40 = 1.25.
	p := pc.Points[0]
	if math.Abs(p.Z-1.25) > 1e-6 {
		t.Fatalf("Z=%v want 1.25", p.Z)
	}
	// x=y=cx=cy -> X=Y=0.
	if math.Abs(p.X) > 1e-9 || math.Abs(p.Y) > 1e-9 {
		t.Fatalf("expected on-axis point, got %+v", p)
	}
}

func TestFitPlaneAxisAligned(t *testing.T) {
	// Points on the plane z = 3 (normal (0,0,1), D=-3).
	pts := []Point3D{
		{0, 0, 3}, {1, 0, 3}, {0, 1, 3}, {1, 1, 3}, {2, 3, 3}, {-1, 2, 3},
	}
	pl, err := FitPlane(pts)
	if err != nil {
		t.Fatal(err)
	}
	// Normal should be ±(0,0,1).
	if math.Abs(math.Abs(pl.C)-1) > 1e-6 || math.Abs(pl.A) > 1e-6 || math.Abs(pl.B) > 1e-6 {
		t.Fatalf("normal not aligned with z: %+v", pl)
	}
	// Every point lies on the plane.
	for _, p := range pts {
		if math.Abs(pl.DistanceTo(p)) > 1e-6 {
			t.Fatalf("point %+v not on plane, dist %v", p, pl.DistanceTo(p))
		}
	}
}

func TestFitPlaneTilted(t *testing.T) {
	// Plane z = x (normal proportional to (1,0,-1)).
	var pts []Point3D
	for x := 0; x < 5; x++ {
		for y := 0; y < 5; y++ {
			pts = append(pts, Point3D{float64(x), float64(y), float64(x)})
		}
	}
	pl, err := FitPlane(pts)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range pts {
		if math.Abs(pl.DistanceTo(p)) > 1e-6 {
			t.Fatalf("tilted plane residual too large for %+v: %v", p, pl.DistanceTo(p))
		}
	}
	// Normal should have no y component.
	if math.Abs(pl.B) > 1e-6 {
		t.Fatalf("normal has y component: %+v", pl)
	}
}

func TestFitPlaneRANSAC(t *testing.T) {
	// Inliers on z=0, plus gross outliers.
	var pts []Point3D
	for x := 0; x < 6; x++ {
		for y := 0; y < 6; y++ {
			pts = append(pts, Point3D{float64(x), float64(y), 0})
		}
	}
	pts = append(pts, Point3D{2, 2, 50}, Point3D{3, 1, -40}, Point3D{0, 5, 33})

	pl, inliers, err := FitPlaneRANSAC(pts, 200, 0.5, 12345)
	if err != nil {
		t.Fatal(err)
	}
	if len(inliers) < 36 {
		t.Fatalf("RANSAC found only %d inliers, want >= 36", len(inliers))
	}
	// Fitted plane should be ~ z=0.
	if math.Abs(math.Abs(pl.C)-1) > 1e-3 {
		t.Fatalf("RANSAC plane normal not aligned with z: %+v", pl)
	}
	for _, idx := range inliers {
		if pts[idx].Z != 0 {
			t.Fatalf("outlier index %d classified as inlier", idx)
		}
	}
}

func TestPoint3DVectorOps(t *testing.T) {
	a := Point3D{1, 0, 0}
	b := Point3D{0, 1, 0}
	if c := a.Cross(b); c.X != 0 || c.Y != 0 || c.Z != 1 {
		t.Fatalf("cross = %+v want (0,0,1)", c)
	}
	if d := a.Dot(b); d != 0 {
		t.Fatalf("dot = %v want 0", d)
	}
	if n := (Point3D{3, 4, 0}).Norm(); math.Abs(n-5) > 1e-12 {
		t.Fatalf("norm = %v want 5", n)
	}
	if s := a.Add(b).Sub(b); s != a {
		t.Fatalf("add/sub round trip failed: %+v", s)
	}
	if s := (Point3D{1, 2, 3}).Scale(2); s != (Point3D{2, 4, 6}) {
		t.Fatalf("scale = %+v", s)
	}
}

func TestDisparityMapToMat(t *testing.T) {
	d := NewDisparityMap(1, 3)
	d.Set(0, 0, 0)
	d.Set(0, 1, 5)
	d.Set(0, 2, 10)
	m := d.ToMat()
	if m.At(0, 0, 0) != 0 || m.At(0, 2, 0) != 255 {
		t.Fatalf("ToMat scaling wrong: %d %d", m.At(0, 0, 0), m.At(0, 2, 0))
	}
	if m.At(0, 1, 0) != 128 { // 5/10*255 = 127.5 -> 128
		t.Fatalf("ToMat mid value = %d want 128", m.At(0, 1, 0))
	}
}

// BenchmarkSGMCompute benchmarks the heaviest routine: a full semi-global match
// (census cost + eight-path aggregation + sub-pixel extraction).
func BenchmarkSGMCompute(b *testing.B) {
	const rows, cols, d0 = 64, 96, 8
	left, right := makeStereoPair(rows, cols, d0)
	sgm := NewSGMMatcher(0, 16)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sgm.Compute(left, right)
	}
}

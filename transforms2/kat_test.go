package transforms2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// grad builds a single-channel gradient image whose sample at (x, y) is a
// distinct, deterministic value.
func grad(w, h int) *cv.Mat {
	m := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := (x*7 + y*3) % 256
			m.Data[y*w+x] = uint8(v)
		}
	}
	return m
}

func gaussBlob(w, h int, cx, cy, sigma float64) *cv.Mat {
	m := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx := float64(x) - cx
			dy := float64(y) - cy
			v := 255 * math.Exp(-(dx*dx+dy*dy)/(2*sigma*sigma))
			m.Data[y*w+x] = uint8(v + 0.5)
		}
	}
	return m
}

func TestSampleChannelExactInteger(t *testing.T) {
	m := grad(5, 4)
	for y := 0; y < 4; y++ {
		for x := 0; x < 5; x++ {
			got := SampleChannel(m, float64(x), float64(y), 0, InterpBilinear, BorderReplicate, 0)
			want := float64(m.At(y, x, 0))
			if math.Abs(got-want) > 1e-9 {
				t.Fatalf("bilinear at (%d,%d)=%v want %v", x, y, got, want)
			}
		}
	}
}

func TestBicubicPreservesConstant(t *testing.T) {
	m := cv.NewMat(6, 6, 1)
	for i := range m.Data {
		m.Data[i] = 100
	}
	for _, xy := range [][2]float64{{2.3, 3.7}, {0.5, 0.5}, {4.9, 1.1}} {
		got := SampleChannel(m, xy[0], xy[1], 0, InterpBicubic, BorderReplicate, 0)
		if math.Abs(got-100) > 1e-9 {
			t.Fatalf("bicubic constant at %v = %v, want 100", xy, got)
		}
	}
}

func TestBorderModesSampling(t *testing.T) {
	m := grad(4, 4)
	// Nearest, replicate: out-of-range left maps to column 0.
	got := SampleChannel(m, -3, 1, 0, InterpNearest, BorderReplicate, 0)
	want := float64(m.At(1, 0, 0))
	if got != want {
		t.Fatalf("replicate got %v want %v", got, want)
	}
	// Constant border returns fill.
	got = SampleChannel(m, -3, 1, 0, InterpNearest, BorderConstant, 42)
	if got != 42 {
		t.Fatalf("constant got %v want 42", got)
	}
}

func TestWarpAffineIdentity(t *testing.T) {
	m := grad(7, 5)
	out := WarpAffine(m, AffineIdentity(), 7, 5, InterpBilinear, BorderReplicate, 0)
	for i := range m.Data {
		if out.Data[i] != m.Data[i] {
			t.Fatalf("identity warp changed data at %d: %d vs %d", i, out.Data[i], m.Data[i])
		}
	}
}

func TestTranslateIntegerShift(t *testing.T) {
	m := grad(6, 4)
	out := Translate(m, 2, 1, InterpNearest, BorderReplicate, 0)
	// out(x,y) = src(x-2, y-1) with replication.
	for y := 0; y < 4; y++ {
		for x := 0; x < 6; x++ {
			sx := x - 2
			sy := y - 1
			if sx < 0 {
				sx = 0
			}
			if sy < 0 {
				sy = 0
			}
			if out.At(y, x, 0) != m.At(sy, sx, 0) {
				t.Fatalf("translate mismatch at (%d,%d)", x, y)
			}
		}
	}
}

func TestGetAffineTransformRecovers(t *testing.T) {
	src := [3]cv.Point2f{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 0, Y: 3}}
	want := AffineRotationAround(1, 1, 30, 1.2)
	var dst [3]cv.Point2f
	for i, p := range src {
		x, y := ApplyAffine(want, p.X, p.Y)
		dst[i] = cv.Point2f{X: x, Y: y}
	}
	got := GetAffineTransform(src, dst)
	for i := 0; i < 6; i++ {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("coeff %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestComposeInvertAffineIdentity(t *testing.T) {
	m := AffineRotationAround(3, 2, 40, 0.8)
	inv, ok := InvertAffine(m)
	if !ok {
		t.Fatal("not invertible")
	}
	id := ComposeAffine(m, inv)
	exp := AffineIdentity()
	for i := 0; i < 6; i++ {
		if math.Abs(id[i]-exp[i]) > 1e-9 {
			t.Fatalf("compose*inverse coeff %d = %v", i, id[i])
		}
	}
}

func TestScaleNearestDouble(t *testing.T) {
	m := grad(2, 2)
	out := Scale(m, 2, 2, InterpNearest)
	if out.Cols != 4 || out.Rows != 4 {
		t.Fatalf("size %dx%d", out.Cols, out.Rows)
	}
	// Columns/rows should map [0,0,1,1].
	idx := []int{0, 0, 1, 1}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if out.At(y, x, 0) != m.At(idx[y], idx[x], 0) {
				t.Fatalf("double mismatch at (%d,%d)", x, y)
			}
		}
	}
}

func TestGetPerspectiveTransformSquare(t *testing.T) {
	src := [4]cv.Point2f{{X: 0, Y: 0}, {X: 2, Y: 0}, {X: 2, Y: 2}, {X: 0, Y: 2}}
	dst := [4]cv.Point2f{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	m := GetPerspectiveTransform(src, dst)
	x, y := ApplyPerspective(m, 1, 1)
	if math.Abs(x-2) > 1e-9 || math.Abs(y-2) > 1e-9 {
		t.Fatalf("center mapped to (%v,%v), want (2,2)", x, y)
	}
}

func TestRemapMatchesWarpAffine(t *testing.T) {
	m := grad(9, 7)
	aff := AffineRotationAround(4, 3, 15, 1)
	mapX, mapY := MakeAffineMaps(aff, 9, 7)
	viaRemap := Remap(m, mapX, mapY, InterpBilinear, BorderConstant, 0)
	viaWarp := WarpAffine(m, aff, 9, 7, InterpBilinear, BorderConstant, 0)
	for i := range viaWarp.Data {
		if viaRemap.Data[i] != viaWarp.Data[i] {
			t.Fatalf("remap vs warp differ at %d: %d vs %d", i, viaRemap.Data[i], viaWarp.Data[i])
		}
	}
}

func TestRemapIdentity(t *testing.T) {
	m := grad(8, 6)
	mapX, mapY := MakeIdentityMaps(8, 6)
	out := Remap(m, mapX, mapY, InterpBilinear, BorderReplicate, 0)
	for i := range m.Data {
		if out.Data[i] != m.Data[i] {
			t.Fatalf("identity remap differ at %d", i)
		}
	}
}

func TestWarpPolarConstantInterior(t *testing.T) {
	m := cv.NewMat(20, 20, 1)
	for i := range m.Data {
		m.Data[i] = 77
	}
	polar := WarpPolar(m, 10, 10, 8, 16, 24, PolarLinear, InterpBilinear, BorderReplicate, 0)
	// Every polar sample lies within radius 8 of the centre, inside the image,
	// so all should read the constant value.
	for i := range polar.Data {
		if polar.Data[i] != 77 {
			t.Fatalf("polar sample %d = %d, want 77", i, polar.Data[i])
		}
	}
}

func TestDelaunaySquare(t *testing.T) {
	pts := []cv.Point2f{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	tris := DelaunayTriangulation(pts)
	if len(tris) != 2 {
		t.Fatalf("expected 2 triangles, got %d", len(tris))
	}
	for _, tr := range tris {
		for _, idx := range tr {
			if idx < 0 || idx >= 4 {
				t.Fatalf("triangle index out of range: %d", idx)
			}
		}
	}
}

func TestPiecewiseAffineIdentity(t *testing.T) {
	m := grad(12, 10)
	pts := []cv.Point2f{
		{X: 0, Y: 0}, {X: 11, Y: 0}, {X: 11, Y: 9}, {X: 0, Y: 9}, {X: 5, Y: 4},
	}
	out := PiecewiseAffineWarp(m, pts, pts, nil, 12, 10, InterpBilinear, BorderReplicate, 0)
	// Interior pixels should reproduce the source exactly (identity affine at
	// integer sample points).
	for y := 2; y < 8; y++ {
		for x := 2; x < 10; x++ {
			if out.At(y, x, 0) != m.At(y, x, 0) {
				t.Fatalf("piecewise identity mismatch at (%d,%d): %d vs %d", x, y, out.At(y, x, 0), m.At(y, x, 0))
			}
		}
	}
}

func TestTPSInterpolatesControlPoints(t *testing.T) {
	from := []cv.Point2f{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}, {X: 5, Y: 5}}
	to := []cv.Point2f{{X: 1, Y: 0}, {X: 9, Y: 1}, {X: 11, Y: 9}, {X: -1, Y: 11}, {X: 6, Y: 4}}
	tps := NewThinPlateSpline(from, to, 0)
	for i := range from {
		x, y := tps.Transform(from[i].X, from[i].Y)
		if math.Abs(x-to[i].X) > 1e-6 || math.Abs(y-to[i].Y) > 1e-6 {
			t.Fatalf("control %d mapped to (%v,%v), want (%v,%v)", i, x, y, to[i].X, to[i].Y)
		}
	}
}

func TestTPSIdentityWarp(t *testing.T) {
	m := grad(16, 16)
	pts := []cv.Point2f{{X: 0, Y: 0}, {X: 15, Y: 0}, {X: 15, Y: 15}, {X: 0, Y: 15}, {X: 8, Y: 8}}
	tps := NewThinPlateSpline(pts, pts, 0)
	out := tps.Warp(m, 16, 16, InterpBilinear, BorderReplicate, 0)
	for y := 2; y < 14; y++ {
		for x := 2; x < 14; x++ {
			d := int(out.At(y, x, 0)) - int(m.At(y, x, 0))
			if d < -1 || d > 1 {
				t.Fatalf("TPS identity differs at (%d,%d): %d vs %d", x, y, out.At(y, x, 0), m.At(y, x, 0))
			}
		}
	}
}

func TestUndistortZeroIsIdentity(t *testing.T) {
	m := grad(20, 16)
	cam := CameraMatrix{Fx: 30, Fy: 30, Cx: 10, Cy: 8}
	out := Undistort(m, cam, DistortionCoeffs{}, InterpBilinear, BorderReplicate, 0)
	for i := range m.Data {
		if out.Data[i] != m.Data[i] {
			t.Fatalf("zero-distortion undistort changed data at %d", i)
		}
	}
}

func TestDistortUndistortRoundtrip(t *testing.T) {
	cam := CameraMatrix{Fx: 200, Fy: 210, Cx: 160, Cy: 120}
	dist := DistortionCoeffs{K1: -0.2, K2: 0.05, P1: 0.001, P2: -0.001, K3: 0.01}
	for _, p := range []cv.Point2f{{X: 160, Y: 120}, {X: 60, Y: 40}, {X: 250, Y: 200}, {X: 100, Y: 180}} {
		dx, dy := DistortPoint(cam, dist, p.X, p.Y)
		ux, uy := UndistortPoint(cam, dist, dx, dy)
		if math.Abs(ux-p.X) > 1e-3 || math.Abs(uy-p.Y) > 1e-3 {
			t.Fatalf("roundtrip (%v,%v) -> (%v,%v)", p.X, p.Y, ux, uy)
		}
	}
}

func TestEnhancedCorrelationIdentical(t *testing.T) {
	m := grad(10, 10)
	if rho := EnhancedCorrelationCoefficient(m, m); math.Abs(rho-1) > 1e-9 {
		t.Fatalf("ECC identical = %v, want 1", rho)
	}
}

func TestFindTransformECCIdentity(t *testing.T) {
	m := gaussBlob(40, 40, 20, 20, 8)
	w, rho, err := FindTransformECC(m, m, MotionTranslation, DefaultECCCriteria())
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(w[2]) > 1e-4 || math.Abs(w[5]) > 1e-4 {
		t.Fatalf("identity translation = (%v,%v)", w[2], w[5])
	}
	if math.Abs(rho-1) > 1e-6 {
		t.Fatalf("identity rho = %v", rho)
	}
}

func TestFindTransformECCTranslation(t *testing.T) {
	tmpl := gaussBlob(48, 48, 24, 24, 9)
	image := Translate(tmpl, 2, 3, InterpBilinear, BorderReplicate, 0)
	w, rho, err := FindTransformECC(tmpl, image, MotionTranslation, DefaultECCCriteria())
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(w[2]-2) > 0.3 || math.Abs(w[5]-3) > 0.3 {
		t.Fatalf("recovered translation (%v,%v), want (2,3)", w[2], w[5])
	}
	if rho < 0.99 {
		t.Fatalf("final rho too low: %v", rho)
	}
}

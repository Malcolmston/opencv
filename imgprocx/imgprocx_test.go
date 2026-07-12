package imgprocx

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// almost reports whether a and b are within tol.
func almost(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestGetAffineTransform(t *testing.T) {
	src := [3]cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 0, Y: 10}}
	dst := [3]cv.Point{{X: 3, Y: 4}, {X: 23, Y: 6}, {X: 1, Y: 24}}
	m := GetAffineTransform(src, dst)
	rows := FromAffineMatrix(m)
	// The transform must map every source point onto its destination exactly.
	for i := 0; i < 3; i++ {
		got := ApplyAffine(rows, src[i])
		if !almost(got.X, float64(dst[i].X), 1e-9) || !almost(got.Y, float64(dst[i].Y), 1e-9) {
			t.Fatalf("point %d: got (%.6f,%.6f) want (%d,%d)", i, got.X, got.Y, dst[i].X, dst[i].Y)
		}
	}
}

func TestGetAffineTransformCollinearPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on collinear source points")
		}
	}()
	src := [3]cv.Point{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: 2}}
	dst := [3]cv.Point{{X: 0, Y: 0}, {X: 5, Y: 1}, {X: 9, Y: 4}}
	GetAffineTransform(src, dst)
}

func TestEstimateAffine2D(t *testing.T) {
	// A known affine transform.
	true6 := [2][3]float64{{1.2, 0.1, 5}, {-0.05, 0.9, -3}}
	var src, dst []cv.Point
	// Inliers on a grid, destinations rounded to the nearest pixel.
	for gy := 0; gy < 6; gy++ {
		for gx := 0; gx < 6; gx++ {
			p := cv.Point{X: gx * 7, Y: gy * 7}
			q := ApplyAffine(true6, p)
			src = append(src, p)
			dst = append(dst, cv.Point{X: int(math.Round(q.X)), Y: int(math.Round(q.Y))})
		}
	}
	// Inject gross outliers at known positions.
	outlierIdx := map[int]bool{}
	for _, i := range []int{4, 11, 20, 33} {
		dst[i].X += 60
		dst[i].Y -= 55
		outlierIdx[i] = true
	}
	m, inliers := EstimateAffine2D(src, dst)
	if inliers == nil {
		t.Fatal("expected an inlier mask")
	}
	// Recovered parameters must be close to the truth.
	for r := 0; r < 2; r++ {
		for c := 0; c < 3; c++ {
			if !almost(m[r][c], true6[r][c], 0.1) {
				t.Errorf("param[%d][%d]=%.4f want ~%.4f", r, c, m[r][c], true6[r][c])
			}
		}
	}
	// Outliers flagged out, inliers flagged in.
	for i := range src {
		if outlierIdx[i] && inliers[i] {
			t.Errorf("index %d is an outlier but was flagged inlier", i)
		}
		if !outlierIdx[i] && !inliers[i] {
			t.Errorf("index %d is an inlier but was flagged outlier", i)
		}
	}
	// Determinism: identical inputs yield identical output.
	m2, inliers2 := EstimateAffine2D(src, dst)
	if m2 != m {
		t.Error("EstimateAffine2D is not deterministic (model differs)")
	}
	for i := range inliers {
		if inliers[i] != inliers2[i] {
			t.Errorf("EstimateAffine2D non-deterministic mask at %d", i)
		}
	}
}

func TestEstimateAffinePartial2D(t *testing.T) {
	angle := 20.0 * math.Pi / 180
	scale := 1.3
	a := scale * math.Cos(angle)
	b := scale * math.Sin(angle)
	true4 := [2][3]float64{{a, -b, 4}, {b, a, -6}}
	var src, dst []cv.Point
	for gy := 0; gy < 5; gy++ {
		for gx := 0; gx < 5; gx++ {
			p := cv.Point{X: gx * 9, Y: gy * 9}
			q := ApplyAffine(true4, p)
			src = append(src, p)
			dst = append(dst, cv.Point{X: int(math.Round(q.X)), Y: int(math.Round(q.Y))})
		}
	}
	dst[7].X += 40 // one outlier
	dst[7].Y += 40
	m, inliers := EstimateAffinePartial2D(src, dst)
	if inliers == nil {
		t.Fatal("expected an inlier mask")
	}
	// The similarity constraint must hold: m[0][0]==m[1][1] and m[0][1]==-m[1][0].
	if !almost(m[0][0], m[1][1], 1e-9) || !almost(m[0][1], -m[1][0], 1e-9) {
		t.Errorf("recovered matrix is not a similarity: %+v", m)
	}
	for r := 0; r < 2; r++ {
		for c := 0; c < 3; c++ {
			if !almost(m[r][c], true4[r][c], 0.1) {
				t.Errorf("param[%d][%d]=%.4f want ~%.4f", r, c, m[r][c], true4[r][c])
			}
		}
	}
	if inliers[7] {
		t.Error("outlier at index 7 was flagged inlier")
	}
}

func TestIntegralImage(t *testing.T) {
	rows, cols := 12, 15
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8((x*3+y)%256))
			img.Set(y, x, 1, uint8((x+y*2)%256))
			img.Set(y, x, 2, uint8((x*x+y)%256))
		}
	}
	sum, sqSum := IntegralImage(img)
	if len(sum) != rows+1 || len(sum[0]) != cols+1 {
		t.Fatalf("sum table size %dx%d, want %dx%d", len(sum), len(sum[0]), rows+1, cols+1)
	}
	// pixelIntensity mirrors the package's definition (sum over channels).
	pixelIntensity := func(y, x int) float64 {
		return float64(img.At(y, x, 0)) + float64(img.At(y, x, 1)) + float64(img.At(y, x, 2))
	}
	rects := [][4]int{{0, 0, cols, rows}, {2, 3, 10, 9}, {5, 5, 6, 6}, {0, 0, 1, 1}, {7, 1, 15, 12}}
	for _, r := range rects {
		x0, y0, x1, y1 := r[0], r[1], r[2], r[3]
		var direct float64
		var directSq float64
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				v := pixelIntensity(y, x)
				direct += v
				directSq += v * v
			}
		}
		if got := RectSum(sum, x0, y0, x1, y1); !almost(got, direct, 1e-6) {
			t.Errorf("rect %v sum=%.1f want %.1f", r, got, direct)
		}
		if got := RectSum(sqSum, x0, y0, x1, y1); !almost(got, directSq, 1e-6) {
			t.Errorf("rect %v sqSum=%.1f want %.1f", r, got, directSq)
		}
	}
}

func TestPhaseCorrelate(t *testing.T) {
	rows, cols := 32, 48
	a := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			a.Set(y, x, 0, uint8((x*11+y*7+x*y)%256))
		}
	}
	for _, want := range []struct{ dx, dy int }{{5, 3}, {-4, 6}, {0, 0}, {12, -9}} {
		b := cv.NewMat(rows, cols, 1)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				sy := ((y-want.dy)%rows + rows) % rows
				sx := ((x-want.dx)%cols + cols) % cols
				b.Set(y, x, 0, a.At(sy, sx, 0))
			}
		}
		shift, resp := PhaseCorrelate(a, b)
		if int(math.Round(shift.X)) != want.dx || int(math.Round(shift.Y)) != want.dy {
			t.Errorf("shift got (%.2f,%.2f) want (%d,%d)", shift.X, shift.Y, want.dx, want.dy)
		}
		if resp < 0.99 {
			t.Errorf("response %.4f too low for exact shift", resp)
		}
	}
}

func TestGetGaborKernel(t *testing.T) {
	const ksize = 31
	k := GetGaborKernel(ksize, 4, 0, 8, 0.5, 0)
	if k.Rows != ksize || k.Cols != ksize {
		t.Fatalf("kernel size %dx%d, want %dx%d", k.Rows, k.Cols, ksize, ksize)
	}
	if len(k.Data) != ksize*ksize {
		t.Fatalf("kernel data length %d, want %d", len(k.Data), ksize*ksize)
	}
	var sum float64
	for _, v := range k.Data {
		sum += v
	}
	mean := sum / float64(len(k.Data))
	if math.Abs(mean) > 0.01 {
		t.Errorf("Gabor kernel mean %.5f is not near zero", mean)
	}
}

func TestGetGaborKernelEvenPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for even ksize")
		}
	}()
	GetGaborKernel(30, 4, 0, 8, 0.5, 0)
}

func TestCornerSubPix(t *testing.T) {
	m := cv.NewMat(40, 40, 1)
	for y := 12; y < 30; y++ {
		for x := 12; x < 30; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	corners := []cv.Point{{X: 12, Y: 12}, {X: 29, Y: 29}}
	win := 4
	out := CornerSubPix(m, corners, win)
	if len(out) != len(corners) {
		t.Fatalf("got %d refined corners, want %d", len(out), len(corners))
	}
	for i, p := range out {
		if math.IsNaN(p.X) || math.IsNaN(p.Y) {
			t.Errorf("corner %d refined to NaN", i)
		}
		// Refined corner must stay within the search window of the input.
		if math.Abs(p.X-float64(corners[i].X)) > float64(win) ||
			math.Abs(p.Y-float64(corners[i].Y)) > float64(win) {
			t.Errorf("corner %d moved outside window: %+v from %v", i, p, corners[i])
		}
	}
	// Determinism.
	out2 := CornerSubPix(m, corners, win)
	for i := range out {
		if out[i] != out2[i] {
			t.Errorf("CornerSubPix non-deterministic at %d", i)
		}
	}
}

func TestLinearPolarCenterSample(t *testing.T) {
	src := cv.NewMat(20, 20, 1)
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			src.Set(y, x, 0, uint8((x*5+y*3)%256))
		}
	}
	center := Point2f{X: 10, Y: 10}
	dst := LinearPolar(src, center, 8, 16, 24)
	if dst.Rows != 24 || dst.Cols != 16 {
		t.Fatalf("polar size %dx%d, want 24x16", dst.Rows, dst.Cols)
	}
	// Column 0 is radius 0, so every row samples the centre pixel.
	want := src.At(10, 10, 0)
	for row := 0; row < dst.Rows; row++ {
		if got := dst.At(row, 0, 0); got != want {
			t.Errorf("polar[%d][0]=%d, want centre value %d", row, got, want)
		}
	}
}

func TestLogPolarDims(t *testing.T) {
	src := cv.NewMat(20, 20, 1)
	src.SetTo(100)
	dst := LogPolar(src, Point2f{X: 10, Y: 10}, 10, 18, 30)
	if dst.Rows != 30 || dst.Cols != 18 {
		t.Fatalf("log-polar size %dx%d, want 30x18", dst.Rows, dst.Cols)
	}
	// On a constant image every sampled value equals the constant.
	if got := dst.At(0, 0, 0); got != 100 {
		t.Errorf("log-polar sample %d, want 100", got)
	}
}

func TestAffineMatrixRoundTrip(t *testing.T) {
	m := [2][3]float64{{1.5, 0.2, 3}, {-0.4, 2.1, -7}}
	a := ToAffineMatrix(m)
	back := FromAffineMatrix(a)
	if back != m {
		t.Errorf("round trip mismatch: %+v vs %+v", back, m)
	}
	// ToAffineMatrix must agree with cv's evaluation convention.
	p := cv.Point{X: 4, Y: 9}
	got := ApplyAffine(m, p)
	wantX := a[0]*4 + a[1]*9 + a[2]
	wantY := a[3]*4 + a[4]*9 + a[5]
	if !almost(got.X, wantX, 1e-9) || !almost(got.Y, wantY, 1e-9) {
		t.Errorf("ApplyAffine %+v, want (%.3f,%.3f)", got, wantX, wantY)
	}
}

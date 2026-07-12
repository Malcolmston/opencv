package imgprocx

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func eq(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestGetGaussianKernel(t *testing.T) {
	// Small aperture with sigma<=0 returns the exact fixed table.
	k := GetGaussianKernel(3, 0)
	want := []float64{0.25, 0.5, 0.25}
	for i := range want {
		if !eq(k[i], want[i], 1e-12) {
			t.Fatalf("k[%d]=%.6f want %.6f", i, k[i], want[i])
		}
	}
	// Explicit sigma: coefficients must be symmetric, peaked centre, sum to 1.
	k = GetGaussianKernel(5, 1.0)
	var sum float64
	for _, v := range k {
		sum += v
	}
	if !eq(sum, 1.0, 1e-12) {
		t.Errorf("kernel does not sum to 1: %.6f", sum)
	}
	if !eq(k[0], k[4], 1e-12) || !eq(k[1], k[3], 1e-12) {
		t.Errorf("kernel not symmetric: %v", k)
	}
	if !(k[2] > k[1] && k[1] > k[0]) {
		t.Errorf("kernel not peaked at centre: %v", k)
	}
	// Hand value: unnormalised exp(-x^2/2) at x=-2..2 then normalise.
	raw := []float64{math.Exp(-2), math.Exp(-0.5), 1, math.Exp(-0.5), math.Exp(-2)}
	var rs float64
	for _, v := range raw {
		rs += v
	}
	if !eq(k[0], raw[0]/rs, 1e-12) {
		t.Errorf("k[0]=%.8f want %.8f", k[0], raw[0]/rs)
	}
}

func TestGetGaussianKernelEvenPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for even ksize")
		}
	}()
	GetGaussianKernel(4, 1)
}

func TestGetDerivKernels(t *testing.T) {
	// dx=1, dy=0, ksize=3, unnormalised: kx is the difference [-1,0,1],
	// ky the smoothing [1,2,1].
	kx, ky := GetDerivKernels(1, 0, 3, false)
	wantKx := []float64{-1, 0, 1}
	wantKy := []float64{1, 2, 1}
	for i := range wantKx {
		if kx[i] != wantKx[i] || ky[i] != wantKy[i] {
			t.Fatalf("ksize3: kx=%v ky=%v", kx, ky)
		}
	}
	// ksize=5 derivative row is [-1,-2,0,2,1]; smoothing is [1,4,6,4,1].
	kx, ky = GetDerivKernels(1, 0, 5, false)
	wantKx = []float64{-1, -2, 0, 2, 1}
	wantKy = []float64{1, 4, 6, 4, 1}
	for i := range wantKx {
		if kx[i] != wantKx[i] || ky[i] != wantKy[i] {
			t.Fatalf("ksize5: kx=%v ky=%v", kx, ky)
		}
	}
	// Second derivative, ksize=3: [1,-2,1].
	kx, _ = GetDerivKernels(2, 0, 3, false)
	for i, w := range []float64{1, -2, 1} {
		if kx[i] != w {
			t.Fatalf("d2 ksize3 kx=%v", kx)
		}
	}
	// Normalised smoothing kernel sums to 1.
	_, ky = GetDerivKernels(1, 0, 3, true)
	if !eq(ky[0]+ky[1]+ky[2], 1.0, 1e-12) {
		t.Errorf("normalised smoothing kernel sums to %.6f", ky[0]+ky[1]+ky[2])
	}
}

func TestSpatialGradient(t *testing.T) {
	// Horizontal ramp I(y,x)=x. Interior Sobel dx = 8, dy = 0.
	rows, cols := 6, 6
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Set(y, x, 0, uint8(x))
		}
	}
	dx, dy := SpatialGradient(m)
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			if !eq(dx.At(y, x), 8, 1e-9) {
				t.Errorf("dx[%d][%d]=%.3f want 8", y, x, dx.At(y, x))
			}
			if !eq(dy.At(y, x), 0, 1e-9) {
				t.Errorf("dy[%d][%d]=%.3f want 0", y, x, dy.At(y, x))
			}
		}
	}
	// Vertical ramp: dy = 8, dx = 0 in the interior.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Set(y, x, 0, uint8(y))
		}
	}
	dx, dy = SpatialGradient(m)
	if !eq(dy.At(2, 3), 8, 1e-9) || !eq(dx.At(2, 3), 0, 1e-9) {
		t.Errorf("vertical ramp: dx=%.3f dy=%.3f want dx=0 dy=8", dx.At(2, 3), dy.At(2, 3))
	}
}

func TestCreateHanningWindow(t *testing.T) {
	w := CreateHanningWindow(5, 5)
	// 1-D Hann of length 5 is [0, 0.5, 1, 0.5, 0]; 2-D is the outer product.
	col := []float64{0, 0.5, 1, 0.5, 0}
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			want := col[y] * col[x]
			if !eq(w.At(y, x), want, 1e-12) {
				t.Errorf("w[%d][%d]=%.4f want %.4f", y, x, w.At(y, x), want)
			}
		}
	}
	// Edges are exactly zero; centre is exactly one.
	if w.At(0, 0) != 0 || !eq(w.At(2, 2), 1, 1e-12) {
		t.Errorf("corner=%.4f centre=%.4f", w.At(0, 0), w.At(2, 2))
	}
}

func TestAccumulateFamily(t *testing.T) {
	rows, cols := 3, 4
	a := cv.NewMat(rows, cols, 1)
	b := cv.NewMat(rows, cols, 1)
	for i := range a.Data {
		a.Data[i] = uint8(i + 1)
		b.Data[i] = uint8(2 * (i + 1))
	}
	sum := cv.NewFloatMat(rows, cols)
	Accumulate(a, sum)
	Accumulate(b, sum)
	for i := range sum.Data {
		want := float64(a.Data[i]) + float64(b.Data[i])
		if !eq(sum.Data[i], want, 1e-9) {
			t.Fatalf("Accumulate[%d]=%.1f want %.1f", i, sum.Data[i], want)
		}
	}
	sq := cv.NewFloatMat(rows, cols)
	AccumulateSquare(a, sq)
	if !eq(sq.Data[0], 1, 1e-9) || !eq(sq.Data[1], 4, 1e-9) {
		t.Errorf("AccumulateSquare front: %.1f %.1f", sq.Data[0], sq.Data[1])
	}
	prod := cv.NewFloatMat(rows, cols)
	AccumulateProduct(a, b, prod)
	for i := range prod.Data {
		want := float64(a.Data[i]) * float64(b.Data[i])
		if !eq(prod.Data[i], want, 1e-9) {
			t.Fatalf("AccumulateProduct[%d]=%.1f want %.1f", i, prod.Data[i], want)
		}
	}
	// Weighted: start dst=10 everywhere, alpha=0.25, src=100 -> 0.75*10+0.25*100=32.5.
	w := cv.NewFloatMat(rows, cols)
	for i := range w.Data {
		w.Data[i] = 10
	}
	src := cv.NewMat(rows, cols, 1)
	src.SetTo(100)
	AccumulateWeighted(src, w, 0.25)
	for i := range w.Data {
		if !eq(w.Data[i], 32.5, 1e-9) {
			t.Fatalf("AccumulateWeighted[%d]=%.4f want 32.5", i, w.Data[i])
		}
	}
}

func TestBlendLinear(t *testing.T) {
	rows, cols := 2, 2
	a := cv.NewMat(rows, cols, 1)
	b := cv.NewMat(rows, cols, 1)
	a.SetTo(40)
	b.SetTo(80)
	w1 := cv.NewFloatMat(rows, cols)
	w2 := cv.NewFloatMat(rows, cols)
	// Equal weights -> midpoint 60.
	for i := range w1.Data {
		w1.Data[i] = 1
		w2.Data[i] = 1
	}
	out := BlendLinear(a, b, w1, w2)
	if out.At(0, 0, 0) != 60 {
		t.Errorf("equal-weight blend = %d want 60", out.At(0, 0, 0))
	}
	// 3:1 weighting -> (3*40 + 1*80)/4 = 50.
	for i := range w1.Data {
		w1.Data[i] = 3
		w2.Data[i] = 1
	}
	out = BlendLinear(a, b, w1, w2)
	if out.At(0, 0, 0) != 50 {
		t.Errorf("3:1 blend = %d want 50", out.At(0, 0, 0))
	}
	// Zero weights -> zero output.
	for i := range w1.Data {
		w1.Data[i] = 0
		w2.Data[i] = 0
	}
	out = BlendLinear(a, b, w1, w2)
	if out.At(1, 1, 0) != 0 {
		t.Errorf("zero-weight blend = %d want 0", out.At(1, 1, 0))
	}
}

func TestDistanceTransformWithLabels(t *testing.T) {
	// All foreground except one background pixel at (row=1,col=2).
	rows, cols := 4, 5
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(255)
	bgY, bgX := 1, 2
	m.Set(bgY, bgX, 0, 0)
	dist, labels := DistanceTransformWithLabels(m)
	bgIdx := bgY*cols + bgX
	if dist.At(bgY, bgX) != 0 {
		t.Errorf("background distance = %.3f want 0", dist.At(bgY, bgX))
	}
	// With a single background pixel every label is that pixel's index and the
	// distance is the chamfer distance to it.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if labels[y][x] != bgIdx {
				t.Errorf("label[%d][%d]=%d want %d", y, x, labels[y][x], bgIdx)
			}
			dxAbs := math.Abs(float64(x - bgX))
			dyAbs := math.Abs(float64(y - bgY))
			mn := math.Min(dxAbs, dyAbs)
			mx := math.Max(dxAbs, dyAbs)
			want := mn*math.Sqrt2 + (mx - mn)
			if !eq(dist.At(y, x), want, 1e-9) {
				t.Errorf("dist[%d][%d]=%.4f want %.4f", y, x, dist.At(y, x), want)
			}
		}
	}
}

func TestFloodFillFixedRange(t *testing.T) {
	// Left half value 10, right half value 200; zero tolerance fills the left
	// half only (4-connected).
	rows, cols := 4, 6
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x < 3 {
				m.Set(y, x, 0, 10)
			} else {
				m.Set(y, x, 0, 200)
			}
		}
	}
	area, rect := FloodFill(m, cv.Point{X: 0, Y: 0}, cv.Scalar{99}, &FloodFillOptions{FixedRange: true})
	if area != rows*3 {
		t.Errorf("area = %d want %d", area, rows*3)
	}
	if rect.X != 0 || rect.Y != 0 || rect.Width != 3 || rect.Height != 4 {
		t.Errorf("rect = %+v want {0 0 3 4}", rect)
	}
	if m.At(0, 0, 0) != 99 || m.At(3, 2, 0) != 99 {
		t.Errorf("left half not repainted")
	}
	if m.At(0, 3, 0) != 200 {
		t.Errorf("right half changed: %d", m.At(0, 3, 0))
	}
}

func TestFloodFillFloatingRangeAndMask(t *testing.T) {
	// A gentle horizontal ramp 0,1,2,3,4,5. Floating range with UpDiff=1 walks
	// the whole ramp from the left; fixed range with the same tolerance stops.
	rows, cols := 1, 6
	ramp := func() *cv.Mat {
		m := cv.NewMat(rows, cols, 1)
		for x := 0; x < cols; x++ {
			m.Set(0, x, 0, uint8(x))
		}
		return m
	}
	m := ramp()
	area, _ := FloodFill(m, cv.Point{X: 0, Y: 0}, cv.Scalar{255},
		&FloodFillOptions{LoDiff: 1, UpDiff: 1}) // floating range
	if area != 6 {
		t.Errorf("floating-range area = %d want 6", area)
	}
	m = ramp()
	area, _ = FloodFill(m, cv.Point{X: 0, Y: 0}, cv.Scalar{255},
		&FloodFillOptions{FixedRange: true, LoDiff: 1, UpDiff: 1})
	if area != 2 { // only values 0 and 1 are within +/-1 of seed value 0
		t.Errorf("fixed-range area = %d want 2", area)
	}
	// Mask barrier: block column 2 so the fill cannot cross it.
	m = cv.NewMat(1, 6, 1)
	m.SetTo(50)
	mask := cv.NewMat(1, 6, 1)
	mask.Set(0, 2, 0, 1) // barrier
	area, _ = FloodFill(m, cv.Point{X: 0, Y: 0}, cv.Scalar{255},
		&FloodFillOptions{FixedRange: true, Mask: mask, MaskOnly: true})
	if area != 2 { // columns 0 and 1 before the barrier
		t.Errorf("masked area = %d want 2", area)
	}
	if m.At(0, 0, 0) != 50 {
		t.Errorf("MaskOnly should not change the image, got %d", m.At(0, 0, 0))
	}
	if mask.At(0, 0, 0) != 1 || mask.At(0, 1, 0) != 1 {
		t.Errorf("mask not marked for filled pixels")
	}
}

func TestHoughLinesPointSet(t *testing.T) {
	// Points on the vertical line x=3. For theta=0, rho = x = 3 for all of them.
	pts := []cv.Point{{X: 3, Y: 0}, {X: 3, Y: 1}, {X: 3, Y: 2}, {X: 3, Y: 3}}
	lines := HoughLinesPointSet(pts, 5, 3, 0, 10, 1, 0, math.Pi/2, math.Pi/180)
	if len(lines) == 0 {
		t.Fatal("no lines found")
	}
	best := lines[0]
	if best.Votes != 4 {
		t.Errorf("top votes = %d want 4", best.Votes)
	}
	if !eq(best.Rho, 3, 1e-9) || !eq(best.Theta, 0, 1e-9) {
		t.Errorf("top line rho=%.3f theta=%.3f want rho=3 theta=0", best.Rho, best.Theta)
	}
}

func TestCornerMinEigenValAndVecs(t *testing.T) {
	// A bright square on a dark field has a strong corner where its two edges
	// meet: the minimum eigenvalue there exceeds that on a straight edge.
	rows, cols := 30, 30
	m := cv.NewMat(rows, cols, 1)
	for y := 8; y < 22; y++ {
		for x := 8; x < 22; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	minEig := CornerMinEigenVal(m, 3, 3)
	corner := minEig.At(8, 8) // top-left corner of the square
	edge := minEig.At(8, 15)  // middle of the top edge
	if !(corner > edge) {
		t.Errorf("corner minEig %.1f should exceed edge minEig %.1f", corner, edge)
	}
	// Eigen decomposition must be internally consistent: lambda1 >= lambda2 and
	// unit eigenvectors.
	ev := CornerEigenValsAndVecs(m, 3, 3)
	e := ev[8][8]
	if e.Lambda1 < e.Lambda2 {
		t.Errorf("lambda1 %.3f < lambda2 %.3f", e.Lambda1, e.Lambda2)
	}
	if !eq(math.Hypot(e.X1, e.Y1), 1, 1e-9) || !eq(math.Hypot(e.X2, e.Y2), 1, 1e-9) {
		t.Errorf("eigenvectors not unit length: %+v", e)
	}
}

func TestPreCornerDetect(t *testing.T) {
	rows, cols := 30, 30
	m := cv.NewMat(rows, cols, 1)
	for y := 8; y < 22; y++ {
		for x := 8; x < 22; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	pc := PreCornerDetect(m, 3)
	if pc.Rows != rows || pc.Cols != cols {
		t.Fatalf("size %dx%d want %dx%d", pc.Rows, pc.Cols, rows, cols)
	}
	// The response must be non-trivial (a flat region gives exactly zero).
	var maxAbs float64
	for _, v := range pc.Data {
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	if maxAbs == 0 {
		t.Error("PreCornerDetect produced an all-zero response")
	}
	if pc.At(0, 0) != 0 {
		t.Errorf("flat corner of the field should be 0, got %.3f", pc.At(0, 0))
	}
}

func TestEMD(t *testing.T) {
	// Perfect diagonal match: zero cost.
	d := EMD([]float64{1, 1}, []float64{1, 1}, [][]float64{{0, 2}, {2, 0}})
	if !eq(d, 0, 1e-9) {
		t.Errorf("diagonal EMD = %.6f want 0", d)
	}
	// One supply of mass 2 split across two demands of cost 3 and 5:
	// total cost 8 over flow 2 -> 4.
	d = EMD([]float64{2}, []float64{1, 1}, [][]float64{{3, 5}})
	if !eq(d, 4, 1e-9) {
		t.Errorf("split EMD = %.6f want 4", d)
	}
	// 1-D distributions with ground distance |i-j|: optimal cost 0.1, flow 1.
	d = EMD([]float64{0.4, 0.6}, []float64{0.5, 0.5}, [][]float64{{0, 1}, {1, 0}})
	if !eq(d, 0.1, 1e-9) {
		t.Errorf("1-D EMD = %.6f want 0.1", d)
	}
}

func TestIntegralTilted(t *testing.T) {
	rows, cols := 5, 6
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8((x*2+y*3)%17+1))
		}
	}
	tilted := IntegralTilted(img)
	if len(tilted) != rows+1 || len(tilted[0]) != cols+1 {
		t.Fatalf("table size %dx%d want %dx%d", len(tilted), len(tilted[0]), rows+1, cols+1)
	}
	// Brute-force the definition: tilted[Y][X] = sum over y<Y with
	// |x-X+1| <= Y-y-1 of the pixel value.
	pix := func(y, x int) float64 {
		if y < 0 || y >= rows || x < 0 || x >= cols {
			return 0
		}
		return float64(img.At(y, x, 0))
	}
	for Y := 0; Y <= rows; Y++ {
		for X := 0; X <= cols; X++ {
			var want float64
			for y := 0; y < Y; y++ {
				w := Y - y - 1
				for x := X - 1 - w; x <= X-1+w; x++ {
					want += pix(y, x)
				}
			}
			if got := TiltedSum(tilted, X, Y); !eq(got, want, 1e-9) {
				t.Errorf("tilted[%d][%d]=%.1f want %.1f", Y, X, got, want)
			}
		}
	}
}

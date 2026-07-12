package cv

import (
	"math"
	"testing"
)

// fmFrom builds a rows×cols FloatMat from a flat value list (row-major).
func fmFrom(rows, cols int, vals ...float64) *FloatMat {
	m := NewFloatMat(rows, cols)
	if len(vals) != rows*cols {
		panic("fmFrom: wrong number of values")
	}
	copy(m.Data, vals)
	return m
}

// floatClose reports whether a and b differ by no more than tol.
func floatClose(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestTraceAndDeterminant(t *testing.T) {
	m := fmFrom(2, 2, 1, 2, 3, 4)
	if tr := Trace(m); tr != 5 {
		t.Fatalf("Trace = %v, want 5", tr)
	}
	if d := Determinant(m); !floatClose(d, -2, 1e-9) {
		t.Fatalf("Determinant = %v, want -2", d)
	}
	m3 := fmFrom(3, 3, 6, 1, 1, 4, -2, 5, 2, 8, 7)
	if d := Determinant(m3); !floatClose(d, -306, 1e-6) {
		t.Fatalf("Determinant 3x3 = %v, want -306", d)
	}
}

func TestInvertRoundTrip(t *testing.T) {
	m := fmFrom(2, 2, 4, 7, 2, 6)
	inv, ok := Invert(m)
	if !ok {
		t.Fatal("Invert reported singular")
	}
	prod := Gemm(m, inv, 1, nil, 0, false, false)
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			want := 0.0
			if x == y {
				want = 1
			}
			if !floatClose(prod.Data[y*2+x], want, 1e-9) {
				t.Fatalf("M*inv[%d,%d] = %v, want %v", y, x, prod.Data[y*2+x], want)
			}
		}
	}
	// Singular matrix.
	if _, ok := Invert(fmFrom(2, 2, 1, 2, 2, 4)); ok {
		t.Fatal("expected singular matrix to fail Invert")
	}
}

func TestSolveLinear(t *testing.T) {
	a := fmFrom(2, 2, 2, 1, 1, 3)
	b := fmFrom(2, 1, 5, 10)
	x, ok := Solve(a, b)
	if !ok {
		t.Fatal("Solve failed")
	}
	// 2x+y=5, x+3y=10 -> x=1, y=3.
	if !floatClose(x.Data[0], 1, 1e-9) || !floatClose(x.Data[1], 3, 1e-9) {
		t.Fatalf("Solve = %v, want [1 3]", x.Data)
	}
}

func TestSetIdentityAndCompleteSymm(t *testing.T) {
	m := NewFloatMat(3, 3)
	SetIdentity(m, 2)
	if m.Data[0] != 2 || m.Data[4] != 2 || m.Data[8] != 2 || m.Data[1] != 0 {
		t.Fatalf("SetIdentity wrong: %v", m.Data)
	}
	s := fmFrom(2, 2, 1, 5, 0, 2)
	CompleteSymm(s, false) // upper -> lower
	if s.Data[2] != 5 {
		t.Fatalf("CompleteSymm = %v, want lower[1,0]=5", s.Data)
	}
}

func TestGemmWithBias(t *testing.T) {
	a := fmFrom(2, 3, 1, 2, 3, 4, 5, 6)
	b := fmFrom(3, 2, 7, 8, 9, 10, 11, 12)
	c := fmFrom(2, 2, 1, 1, 1, 1)
	out := Gemm(a, b, 1, c, 1, false, false)
	// a*b = [[58,64],[139,154]]; +c -> [[59,65],[140,155]]
	want := []float64{59, 65, 140, 155}
	for i, w := range want {
		if !floatClose(out.Data[i], w, 1e-9) {
			t.Fatalf("Gemm = %v, want %v", out.Data, want)
		}
	}
}

func TestGemmTranspose(t *testing.T) {
	a := fmFrom(2, 2, 1, 2, 3, 4)
	// a^T * a
	out := Gemm(a, a, 1, nil, 0, true, false)
	want := []float64{10, 14, 14, 20}
	for i, w := range want {
		if !floatClose(out.Data[i], w, 1e-9) {
			t.Fatalf("Gemm transpose = %v, want %v", out.Data, want)
		}
	}
}

func TestEigenSymmetric(t *testing.T) {
	m := fmFrom(2, 2, 2, 1, 1, 2)
	vals, vecs := Eigen(m)
	if !floatClose(vals[0], 3, 1e-6) || !floatClose(vals[1], 1, 1e-6) {
		t.Fatalf("Eigen values = %v, want [3 1]", vals)
	}
	// Each row is a unit eigenvector: M v = lambda v.
	for r := 0; r < 2; r++ {
		vx := vecs.Data[r*2+0]
		vy := vecs.Data[r*2+1]
		if !floatClose(vx*vx+vy*vy, 1, 1e-6) {
			t.Fatalf("eigenvector %d not unit: (%v,%v)", r, vx, vy)
		}
		mvx := m.Data[0]*vx + m.Data[1]*vy
		mvy := m.Data[2]*vx + m.Data[3]*vy
		if !floatClose(mvx, vals[r]*vx, 1e-5) || !floatClose(mvy, vals[r]*vy, 1e-5) {
			t.Fatalf("M v != lambda v for eigenvector %d", r)
		}
	}
}

func TestSVDecompReconstruct(t *testing.T) {
	a := fmFrom(3, 2, 1, 2, 3, 4, 5, 6)
	w, u, vt := SVDecomp(a)
	// Reconstruct: A = U diag(w) Vt.
	rows, n := a.Rows, a.Cols
	for y := 0; y < rows; y++ {
		for x := 0; x < n; x++ {
			var v float64
			for k := 0; k < n; k++ {
				v += u.Data[y*n+k] * w[k] * vt.Data[k*n+x]
			}
			if !floatClose(v, a.Data[y*n+x], 1e-6) {
				t.Fatalf("SVD reconstruct[%d,%d] = %v, want %v", y, x, v, a.Data[y*n+x])
			}
		}
	}
	if w[0] < w[1] {
		t.Fatalf("singular values not descending: %v", w)
	}
}

func TestMahalanobisEuclidean(t *testing.T) {
	// With identity inverse covariance, Mahalanobis == Euclidean distance.
	v1 := fmFrom(1, 3, 1, 2, 3)
	v2 := fmFrom(1, 3, 4, 6, 3)
	icov := NewFloatMat(3, 3)
	SetIdentity(icov, 1)
	d := Mahalanobis(v1, v2, icov)
	if !floatClose(d, 5, 1e-9) { // sqrt(9+16+0)=5
		t.Fatalf("Mahalanobis = %v, want 5", d)
	}
}

func TestCalcCovarMatrix(t *testing.T) {
	// Two variables perfectly correlated.
	data := fmFrom(3, 2, 1, 2, 2, 4, 3, 6)
	covar, mean := CalcCovarMatrix(data, true)
	if !floatClose(mean.Data[0], 2, 1e-9) || !floatClose(mean.Data[1], 4, 1e-9) {
		t.Fatalf("mean = %v, want [2 4]", mean.Data)
	}
	// var(x)=2/3, cov(x,y)=4/3, var(y)=8/3.
	if !floatClose(covar.Data[0], 2.0/3, 1e-9) || !floatClose(covar.Data[3], 8.0/3, 1e-9) {
		t.Fatalf("covar diagonal = %v", covar.Data)
	}
}

func TestPCARoundTrip(t *testing.T) {
	// Points along the line y = 2x; one principal component suffices.
	data := fmFrom(4, 2, 0, 0, 1, 2, 2, 4, 3, 6)
	mean, ev, vals := PCACompute(data, 1)
	if len(vals) != 1 || ev.Rows != 1 {
		t.Fatalf("PCACompute components wrong")
	}
	proj := PCAProject(data, mean, ev)
	recon := PCABackProject(proj, mean, ev)
	for i := range data.Data {
		if !floatClose(recon.Data[i], data.Data[i], 1e-6) {
			t.Fatalf("PCA reconstruct[%d] = %v, want %v", i, recon.Data[i], data.Data[i])
		}
	}
}

func TestReduce(t *testing.T) {
	m := fmFrom(2, 3, 1, 2, 3, 4, 5, 6)
	rowSum := Reduce(m, true, ReduceSum) // 1x3
	if rowSum.Data[0] != 5 || rowSum.Data[1] != 7 || rowSum.Data[2] != 9 {
		t.Fatalf("Reduce toRow sum = %v", rowSum.Data)
	}
	colMax := Reduce(m, false, ReduceMax) // 2x1
	if colMax.Data[0] != 3 || colMax.Data[1] != 6 {
		t.Fatalf("Reduce toCol max = %v", colMax.Data)
	}
	rowAvg := Reduce(m, true, ReduceAvg)
	if !floatClose(rowAvg.Data[0], 2.5, 1e-9) {
		t.Fatalf("Reduce avg = %v", rowAvg.Data)
	}
}

func TestRepeat(t *testing.T) {
	m := fmFrom(1, 2, 1, 2)
	r := Repeat(m, 2, 3)
	if r.Rows != 2 || r.Cols != 6 {
		t.Fatalf("Repeat dims = %dx%d", r.Rows, r.Cols)
	}
	if r.Data[0] != 1 || r.Data[1] != 2 || r.Data[2] != 1 || r.Data[6] != 1 {
		t.Fatalf("Repeat data = %v", r.Data)
	}
}

func TestExpLogPowSqrt(t *testing.T) {
	m := fmFrom(1, 3, 1, 2, 3)
	back := Log(Exp(m))
	for i := range m.Data {
		if !floatClose(back.Data[i], m.Data[i], 1e-9) {
			t.Fatalf("Log(Exp) mismatch at %d", i)
		}
	}
	p := Pow(fmFrom(1, 2, 2, 3), 2)
	if p.Data[0] != 4 || p.Data[1] != 9 {
		t.Fatalf("Pow = %v", p.Data)
	}
	s := Sqrt(fmFrom(1, 2, 9, 16))
	if s.Data[0] != 3 || s.Data[1] != 4 {
		t.Fatalf("Sqrt = %v", s.Data)
	}
}

func TestCartPolarRoundTrip(t *testing.T) {
	x := fmFrom(1, 3, 1, 0, -1)
	y := fmFrom(1, 3, 0, 2, -1)
	mag, ang := CartToPolar(x, y, false)
	if !floatClose(mag.Data[0], 1, 1e-9) || !floatClose(mag.Data[1], 2, 1e-9) {
		t.Fatalf("magnitude = %v", mag.Data)
	}
	bx, by := PolarToCart(mag, ang, false)
	for i := range x.Data {
		if !floatClose(bx.Data[i], x.Data[i], 1e-9) || !floatClose(by.Data[i], y.Data[i], 1e-9) {
			t.Fatalf("PolarToCart round trip mismatch at %d: (%v,%v) want (%v,%v)",
				i, bx.Data[i], by.Data[i], x.Data[i], y.Data[i])
		}
	}
	// Phase in degrees.
	deg := Phase(fmFrom(1, 1, 0), fmFrom(1, 1, 1), true)
	if !floatClose(deg.Data[0], 90, 1e-6) {
		t.Fatalf("Phase deg = %v, want 90", deg.Data[0])
	}
}

func TestScaleAdd(t *testing.T) {
	a := fmFrom(1, 3, 1, 2, 3)
	b := fmFrom(1, 3, 10, 20, 30)
	r := ScaleAdd(a, 2, b)
	if r.Data[0] != 12 || r.Data[1] != 24 || r.Data[2] != 36 {
		t.Fatalf("ScaleAdd = %v", r.Data)
	}
}

func TestSortAndSortIdx(t *testing.T) {
	m := fmFrom(2, 3, 3, 1, 2, 9, 7, 8)
	s := Sort(m, true, false)
	if s.Data[0] != 1 || s.Data[1] != 2 || s.Data[2] != 3 {
		t.Fatalf("Sort row0 = %v", s.Data[:3])
	}
	idx := SortIdx(m, true, false)
	// Row 0 = [3,1,2] -> ascending indices [1,2,0].
	if idx[0][0] != 1 || idx[0][1] != 2 || idx[0][2] != 0 {
		t.Fatalf("SortIdx row0 = %v", idx[0])
	}
	// Descending column sort.
	sc := Sort(m, false, true)
	if sc.Data[0] != 9 { // max of column 0 (3 vs 9)
		t.Fatalf("Sort col desc = %v", sc.Data)
	}
}

func TestMinMaxIdx(t *testing.T) {
	m := fmFrom(2, 2, 4, 1, 9, 3)
	mn, mx, mnIdx, mxIdx := MinMaxIdx(m)
	if mn != 1 || mnIdx != 1 || mx != 9 || mxIdx != 2 {
		t.Fatalf("MinMaxIdx = %v %v %v %v", mn, mx, mnIdx, mxIdx)
	}
}

func TestFindNonZeroAndCount(t *testing.T) {
	m := NewMat(2, 2, 1)
	m.Data = []uint8{0, 5, 0, 7}
	pts := FindNonZero(m)
	if len(pts) != 2 || pts[0] != (Point{X: 1, Y: 0}) || pts[1] != (Point{X: 1, Y: 1}) {
		t.Fatalf("FindNonZero = %v", pts)
	}
	if CountNonZero(m) != 2 {
		t.Fatalf("CountNonZero wrong")
	}
}

func TestTransformChannels(t *testing.T) {
	// Swap R and B channels via a transform matrix, add bias to G.
	src := NewMat(1, 1, 3)
	src.Data = []uint8{10, 20, 30}
	m := fmFrom(3, 4,
		0, 0, 1, 0,
		0, 1, 0, 5,
		1, 0, 0, 0)
	out := Transform(src, m)
	if out.Data[0] != 30 || out.Data[1] != 25 || out.Data[2] != 10 {
		t.Fatalf("Transform = %v, want [30 25 10]", out.Data)
	}
}

func TestExtractInsertChannel(t *testing.T) {
	src := NewMat(1, 2, 3)
	src.Data = []uint8{1, 2, 3, 4, 5, 6}
	ch := ExtractChannel(src, 1)
	if ch.Channels != 1 || ch.Data[0] != 2 || ch.Data[1] != 5 {
		t.Fatalf("ExtractChannel = %v", ch.Data)
	}
	dst := NewMat(1, 2, 3)
	InsertChannel(ch, dst, 2)
	if dst.Data[2] != 2 || dst.Data[5] != 5 {
		t.Fatalf("InsertChannel = %v", dst.Data)
	}
}

func TestMixChannels(t *testing.T) {
	src := NewMat(1, 1, 3)
	src.Data = []uint8{10, 20, 30}
	dst := NewMat(1, 1, 3)
	// Reverse channel order.
	MixChannels([]*Mat{src}, []*Mat{dst}, [][2]int{{0, 2}, {1, 1}, {2, 0}})
	if dst.Data[0] != 30 || dst.Data[1] != 20 || dst.Data[2] != 10 {
		t.Fatalf("MixChannels = %v", dst.Data)
	}
}

func TestDFTRoundTrip(t *testing.T) {
	re := fmFrom(2, 2, 1, 2, 3, 4)
	im := NewFloatMat(2, 2)
	fr, fi := DFT(re, im)
	// DC term equals sum of all samples.
	if !floatClose(fr.Data[0], 10, 1e-9) || !floatClose(fi.Data[0], 0, 1e-9) {
		t.Fatalf("DFT DC = (%v,%v), want (10,0)", fr.Data[0], fi.Data[0])
	}
	br, _ := IDFT(fr, fi, true)
	for i := range re.Data {
		if !floatClose(br.Data[i], re.Data[i], 1e-9) {
			t.Fatalf("IDFT(DFT) mismatch at %d: %v vs %v", i, br.Data[i], re.Data[i])
		}
	}
}

func TestDCTRoundTrip(t *testing.T) {
	src := fmFrom(4, 4,
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16)
	back := IDCT(DCT(src))
	for i := range src.Data {
		if !floatClose(back.Data[i], src.Data[i], 1e-9) {
			t.Fatalf("IDCT(DCT) mismatch at %d: %v vs %v", i, back.Data[i], src.Data[i])
		}
	}
}

func TestMulSpectrums(t *testing.T) {
	// (1+2i)*(3+4i) = -5+10i.
	aRe := fmFrom(1, 1, 1)
	aIm := fmFrom(1, 1, 2)
	bRe := fmFrom(1, 1, 3)
	bIm := fmFrom(1, 1, 4)
	re, im := MulSpectrums(aRe, aIm, bRe, bIm, false)
	if !floatClose(re.Data[0], -5, 1e-9) || !floatClose(im.Data[0], 10, 1e-9) {
		t.Fatalf("MulSpectrums = (%v,%v), want (-5,10)", re.Data[0], im.Data[0])
	}
	// With conjugate: (1+2i)*(3-4i)=11+2i.
	rc, ic := MulSpectrums(aRe, aIm, bRe, bIm, true)
	if !floatClose(rc.Data[0], 11, 1e-9) || !floatClose(ic.Data[0], 2, 1e-9) {
		t.Fatalf("MulSpectrums conj = (%v,%v), want (11,2)", rc.Data[0], ic.Data[0])
	}
}

func TestCreateHanningWindow(t *testing.T) {
	w := CreateHanningWindow(5, 5)
	if w.Data[0] != 0 { // corner endpoint is zero
		t.Fatalf("Hanning corner = %v, want 0", w.Data[0])
	}
	// Center is the maximum, equal to 1 for odd size.
	if !floatClose(w.At(2, 2), 1, 1e-9) {
		t.Fatalf("Hanning center = %v, want 1", w.At(2, 2))
	}
}

func TestPhaseCorrelateShift(t *testing.T) {
	// Build an image and a copy shifted right by 2, wrapping around.
	a := NewFloatMat(8, 8)
	for y := 2; y < 5; y++ {
		for x := 1; x < 4; x++ {
			a.Data[y*8+x] = 1
		}
	}
	b := NewFloatMat(8, 8)
	for i := range a.Data {
		y := i / 8
		x := i % 8
		b.Data[y*8+((x+2)%8)] = a.Data[i]
	}
	sx, sy, _ := PhaseCorrelate(a, b)
	if !floatClose(sx, 2, 1e-6) || !floatClose(sy, 0, 1e-6) {
		t.Fatalf("PhaseCorrelate shift = (%v,%v), want (2,0)", sx, sy)
	}
}

func TestPointPolygonTest(t *testing.T) {
	sq := []Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	if PointPolygonTest(sq, Point{5, 5}, false) != 1 {
		t.Fatal("inside point not +1")
	}
	if PointPolygonTest(sq, Point{20, 5}, false) != -1 {
		t.Fatal("outside point not -1")
	}
	if PointPolygonTest(sq, Point{0, 5}, false) != 0 {
		t.Fatal("edge point not 0")
	}
	d := PointPolygonTest(sq, Point{5, 5}, true)
	if !floatClose(d, 5, 1e-9) {
		t.Fatalf("distance = %v, want 5", d)
	}
	if PointPolygonTest(sq, Point{15, 5}, true) >= 0 {
		t.Fatal("outside distance should be negative")
	}
}

func TestIsContourConvex(t *testing.T) {
	sq := []Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	if !IsContourConvex(sq) {
		t.Fatal("square should be convex")
	}
	concave := []Point{{0, 0}, {10, 0}, {5, 5}, {10, 10}, {0, 10}}
	if IsContourConvex(concave) {
		t.Fatal("arrowhead should be non-convex")
	}
}

func TestMinEnclosingCircle(t *testing.T) {
	pts := []Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	c, r := MinEnclosingCircle(pts)
	if !floatClose(c.X, 5, 1e-6) || !floatClose(c.Y, 5, 1e-6) {
		t.Fatalf("center = %v", c)
	}
	if !floatClose(r, math.Hypot(5, 5), 1e-6) {
		t.Fatalf("radius = %v, want %v", r, math.Hypot(5, 5))
	}
}

func TestFitLineHorizontal(t *testing.T) {
	pts := []Point{{0, 5}, {1, 5}, {2, 5}, {10, 5}}
	vx, vy, _, y0 := FitLine(pts)
	if math.Abs(vy) > 1e-6 || math.Abs(math.Abs(vx)-1) > 1e-6 {
		t.Fatalf("direction = (%v,%v), want horizontal", vx, vy)
	}
	if !floatClose(y0, 5, 1e-9) {
		t.Fatalf("y0 = %v, want 5", y0)
	}
}

func TestHuMomentsAndMatchShapes(t *testing.T) {
	src := NewMat(20, 20, 1)
	for y := 5; y < 15; y++ {
		for x := 5; x < 15; x++ {
			src.Set(y, x, 0, 255)
		}
	}
	m := ImageMoments(src)
	hu := HuMoments(m)
	// A symmetric square has near-zero higher-order Hu moments.
	if math.Abs(hu[2]) > 1e-6 {
		t.Fatalf("hu[2] = %v, expected ~0 for symmetric shape", hu[2])
	}
	if d := MatchShapes(m, m); !floatClose(d, 0, 1e-9) {
		t.Fatalf("MatchShapes with self = %v, want 0", d)
	}
}

func TestPerspectiveTransform(t *testing.T) {
	// Translation by (3, -2).
	pm := PerspectiveMatrix{1, 0, 3, 0, 1, -2, 0, 0, 1}
	out := PerspectiveTransform([]Point2f{{1, 1}, {4, 5}}, pm)
	if !floatClose(out[0].X, 4, 1e-9) || !floatClose(out[0].Y, -1, 1e-9) {
		t.Fatalf("out[0] = %v", out[0])
	}
	if !floatClose(out[1].X, 7, 1e-9) || !floatClose(out[1].Y, 3, 1e-9) {
		t.Fatalf("out[1] = %v", out[1])
	}
}

func TestDrawMarkerAndArrowed(t *testing.T) {
	m := NewMat(40, 40, 1)
	DrawMarker(m, Point{20, 20}, NewScalar(255), MarkerCross, 10, 1)
	if m.At(20, 20, 0) != 255 {
		t.Fatal("DrawMarker did not draw center")
	}
	m2 := NewMat(40, 40, 1)
	ArrowedLine(m2, Point{5, 20}, Point{35, 20}, NewScalar(255), 1, 0.2)
	// Some pixel near the tip should be set (the barbs).
	if CountNonZero(m2) < 30 {
		t.Fatalf("ArrowedLine drew too few pixels: %d", CountNonZero(m2))
	}
}

func TestGetTextSize(t *testing.T) {
	size, baseline := GetTextSize("AB", 2)
	// 2 glyphs: 2*(5+1)*2 - 2 = 22 wide; height 7*2=14.
	if size.X != 22 || size.Y != 14 || baseline != 2 {
		t.Fatalf("GetTextSize = %v baseline %d", size, baseline)
	}
}

func TestClipLine(t *testing.T) {
	rect := Rect{X: 0, Y: 0, Width: 10, Height: 10}
	p1, p2, ok := ClipLine(rect, Point{-5, 5}, Point{15, 5})
	if !ok {
		t.Fatal("line should be visible")
	}
	if p1.X != 0 || p2.X != 9 {
		t.Fatalf("ClipLine = %v %v", p1, p2)
	}
	if _, _, ok := ClipLine(rect, Point{-5, -5}, Point{-1, -1}); ok {
		t.Fatal("fully outside line should be clipped away")
	}
}

func TestEllipse2Poly(t *testing.T) {
	pts := Ellipse2Poly(Point{50, 50}, 20, 10, 0, 0, 360, 90)
	// Angles 0,90,180,270,360 -> at least 4 distinct sample points.
	if len(pts) < 4 {
		t.Fatalf("Ellipse2Poly returned %d points", len(pts))
	}
	// First point at angle 0: (center.X + axesX, center.Y).
	if pts[0].X != 70 || pts[0].Y != 50 {
		t.Fatalf("Ellipse2Poly[0] = %v, want (70,50)", pts[0])
	}
}

func TestFillConvexPoly(t *testing.T) {
	m := NewMat(20, 20, 1)
	tri := []Point{{2, 2}, {17, 2}, {2, 17}}
	FillConvexPoly(m, tri, NewScalar(255))
	if m.At(3, 3, 0) != 255 {
		t.Fatal("interior not filled")
	}
	if m.At(17, 17, 0) != 0 {
		t.Fatal("exterior filled")
	}
}

func TestBoxPoints(t *testing.T) {
	r := RotatedRect{CenterX: 5, CenterY: 5, Width: 4, Height: 2, Angle: 0}
	p := BoxPoints(r)
	// Axis-aligned: corners at x in {3,7}, y in {4,6}.
	for _, pt := range p {
		if pt.X != 3 && pt.X != 7 {
			t.Fatalf("BoxPoints x = %v", pt.X)
		}
		if pt.Y != 4 && pt.Y != 6 {
			t.Fatalf("BoxPoints y = %v", pt.Y)
		}
	}
}

func TestIntegral(t *testing.T) {
	src := NewMat(3, 3, 1)
	for i := range src.Data {
		src.Data[i] = 1
	}
	ii := Integral(src)
	// Full sum = 9 at (3,3).
	if ii.At(3, 3) != 9 {
		t.Fatalf("Integral full sum = %v, want 9", ii.At(3, 3))
	}
	// Sum of the bottom-right 2x2 block via inclusion-exclusion = 4.
	sum := ii.At(3, 3) - ii.At(1, 3) - ii.At(3, 1) + ii.At(1, 1)
	if sum != 4 {
		t.Fatalf("Integral block sum = %v, want 4", sum)
	}
	sq := IntegralSquared(src)
	if sq.At(3, 3) != 9 {
		t.Fatalf("IntegralSquared full = %v, want 9", sq.At(3, 3))
	}
}

func TestAccumulate(t *testing.T) {
	src := NewMat(2, 2, 1)
	src.Data = []uint8{1, 2, 3, 4}
	acc := NewFloatMat(2, 2)
	Accumulate(src, acc)
	Accumulate(src, acc)
	if acc.Data[3] != 8 {
		t.Fatalf("Accumulate = %v, want 8", acc.Data[3])
	}
	AccumulateWeighted(src, acc, 0.5)
	// (1-0.5)*8 + 0.5*4 = 6.
	if !floatClose(acc.Data[3], 6, 1e-9) {
		t.Fatalf("AccumulateWeighted = %v, want 6", acc.Data[3])
	}
	sq := NewFloatMat(2, 2)
	AccumulateSquare(src, sq)
	if sq.Data[3] != 16 {
		t.Fatalf("AccumulateSquare = %v, want 16", sq.Data[3])
	}
}

func TestGetGaborKernel(t *testing.T) {
	k := GetGaborKernel(9, 2, 0, 4, 1, 0)
	if k.Rows != 9 || k.Cols != 9 {
		t.Fatalf("Gabor size = %dx%d", k.Rows, k.Cols)
	}
	// The central sample (xr=yr=0) has envelope 1 and cos(psi)=1.
	if !floatClose(k.At(4, 4), 1, 1e-9) {
		t.Fatalf("Gabor center = %v, want 1", k.At(4, 4))
	}
}

func TestSpatialGradientAndPreCorner(t *testing.T) {
	// Vertical edge: left half dark, right half bright.
	src := NewMat(5, 5, 1)
	for y := 0; y < 5; y++ {
		for x := 3; x < 5; x++ {
			src.Set(y, x, 0, 255)
		}
	}
	dx, dy := SpatialGradient(src)
	if dx.At(2, 2) <= 0 {
		t.Fatalf("SpatialGradient dx at edge = %v, expected positive", dx.At(2, 2))
	}
	if math.Abs(dy.At(2, 2)) > 1e-6 {
		t.Fatalf("SpatialGradient dy on vertical edge = %v, expected ~0", dy.At(2, 2))
	}
	pc := PreCornerDetect(src)
	if pc.Rows != 5 || pc.Cols != 5 {
		t.Fatalf("PreCornerDetect size = %dx%d", pc.Rows, pc.Cols)
	}
}

func TestApplyColorMap(t *testing.T) {
	src := NewMat(1, 3, 1)
	src.Data = []uint8{0, 128, 255}
	out := ApplyColorMap(src, ColormapJet)
	if out.Channels != 3 {
		t.Fatalf("ApplyColorMap channels = %d, want 3", out.Channels)
	}
	// Grayscale colormap is the identity across channels.
	g := ApplyColorMap(src, ColormapGray)
	if g.Data[3] != 128 || g.Data[4] != 128 || g.Data[5] != 128 {
		t.Fatalf("Grayscale colormap = %v", g.Data[3:6])
	}
}

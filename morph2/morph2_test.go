package morph2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// mat builds a single-channel Mat of the given size from a flat sample slice.
func mat(rows, cols int, data []uint8) *cv.Mat {
	if len(data) != rows*cols {
		panic("test: data length mismatch")
	}
	m := cv.NewMat(rows, cols, 1)
	copy(m.Data, data)
	return m
}

// filled builds a rows*cols image with a fill value.
func filled(rows, cols int, v uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = v
	}
	return m
}

// has2x2 reports whether the binary image contains a fully-foreground 2x2 block.
func has2x2(m *cv.Mat) bool {
	for y := 0; y < m.Rows-1; y++ {
		for x := 0; x < m.Cols-1; x++ {
			if m.Data[idx(y, x, m.Cols)] != 0 &&
				m.Data[idx(y, x+1, m.Cols)] != 0 &&
				m.Data[idx(y+1, x, m.Cols)] != 0 &&
				m.Data[idx(y+1, x+1, m.Cols)] != 0 {
				return true
			}
		}
	}
	return false
}

func TestElement(t *testing.T) {
	if got := RectElement(3, 3).Count(); got != 9 {
		t.Fatalf("RectElement count = %d, want 9", got)
	}
	if got := CrossElement(3, 3).Count(); got != 5 {
		t.Fatalf("CrossElement count = %d, want 5", got)
	}
	if got := DiamondElement(1).Count(); got != 5 {
		t.Fatalf("DiamondElement(1) count = %d, want 5", got)
	}
	// Round-trip through a Mat kernel.
	e := CrossElement(3, 3)
	back := ElementFromMat(e.ToMat(), -1, -1)
	if back.Count() != e.Count() {
		t.Fatalf("ElementFromMat round-trip count mismatch")
	}
	// Reflect of a single off-centre cell.
	a := &Element{Rows: 3, Cols: 3, AnchorY: 1, AnchorX: 1, cells: make([]bool, 9)}
	a.Set(0, 0, true)
	r := a.Reflect()
	if !r.At(2, 2) || r.At(0, 0) {
		t.Fatalf("Reflect did not map (0,0) to (2,2)")
	}
}

func TestErodeDilate(t *testing.T) {
	src := cv.NewMat(5, 5, 1)
	src.Data[idx(2, 2, 5)] = 255
	e := RectElement(3, 3)

	d := Dilate(src, e)
	if CountForeground(d) != 9 {
		t.Fatalf("dilate count = %d, want 9", CountForeground(d))
	}
	// Erosion of an isolated pixel removes it.
	if CountForeground(Erode(src, e)) != 0 {
		t.Fatalf("erode of isolated pixel should be empty")
	}
	// Erode(Dilate) recovers the single pixel.
	er := Erode(d, e)
	if CountForeground(er) != 1 || er.Data[idx(2, 2, 5)] != 255 {
		t.Fatalf("erode of dilated block should be the centre pixel")
	}
}

func TestOpenClose(t *testing.T) {
	e := RectElement(3, 3)
	// Opening removes an isolated speck.
	speck := cv.NewMat(5, 5, 1)
	speck.Data[idx(2, 2, 5)] = 255
	if CountForeground(Open(speck, e)) != 0 {
		t.Fatalf("open should remove isolated speck")
	}
	// Closing fills a single-pixel hole.
	hole := filled(5, 5, 255)
	hole.Data[idx(2, 2, 5)] = 0
	c := Close(hole, e)
	if c.Data[idx(2, 2, 5)] != 255 {
		t.Fatalf("close should fill single-pixel hole")
	}
}

func TestGradient(t *testing.T) {
	src := cv.NewMat(5, 5, 1)
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			src.Data[idx(y, x, 5)] = 255
		}
	}
	g := Gradient(src, RectElement(3, 3))
	if g.Data[idx(2, 2, 5)] != 0 {
		t.Fatalf("gradient centre = %d, want 0", g.Data[idx(2, 2, 5)])
	}
	if g.Data[idx(0, 0, 5)] != 255 {
		t.Fatalf("gradient corner = %d, want 255", g.Data[idx(0, 0, 5)])
	}
}

func TestHitOrMiss(t *testing.T) {
	src := cv.NewMat(5, 5, 1)
	src.Data[idx(3, 3, 5)] = 255 // isolated interior point
	for _, p := range [][2]int{{0, 0}, {0, 1}, {1, 0}, {1, 1}} {
		src.Data[idx(p[0], p[1], 5)] = 255 // 2x2 block far from the point
	}
	hit := &Element{Rows: 3, Cols: 3, AnchorY: 1, AnchorX: 1, cells: make([]bool, 9)}
	hit.Set(1, 1, true)
	miss := RectElement(3, 3)
	miss.Set(1, 1, false) // the eight neighbours must be background
	out := HitOrMiss(src, hit, miss)
	if CountForeground(out) != 1 || out.Data[idx(3, 3, 5)] != 255 {
		t.Fatalf("hit-or-miss should detect exactly the isolated point, got %d", CountForeground(out))
	}
}

func TestZhangSuen(t *testing.T) {
	// Single pixel is invariant.
	one := cv.NewMat(5, 5, 1)
	one.Data[idx(2, 2, 5)] = 255
	if !Equal(ZhangSuenThinning(one), one) {
		t.Fatalf("Zhang-Suen should preserve a single pixel")
	}
	// A solid rectangle thins to a 1px medial line: fewer pixels, no 2x2 block.
	rect := cv.NewMat(7, 9, 1)
	for y := 2; y <= 4; y++ {
		for x := 1; x <= 7; x++ {
			rect.Data[idx(y, x, 9)] = 255
		}
	}
	th := ZhangSuenThinning(rect)
	if CountForeground(th) == 0 || CountForeground(th) >= CountForeground(rect) {
		t.Fatalf("thinning should reduce foreground")
	}
	if has2x2(th) {
		t.Fatalf("thinned result should contain no 2x2 block")
	}
}

func TestGuoHall(t *testing.T) {
	rect := cv.NewMat(7, 9, 1)
	for y := 2; y <= 4; y++ {
		for x := 1; x <= 7; x++ {
			rect.Data[idx(y, x, 9)] = 255
		}
	}
	th := GuoHallThinning(rect)
	if CountForeground(th) == 0 || has2x2(th) {
		t.Fatalf("Guo-Hall result should be a non-empty thin skeleton")
	}
}

func TestSkeletonSubset(t *testing.T) {
	src := cv.NewMat(9, 9, 1)
	for y := 2; y <= 6; y++ {
		for x := 2; x <= 6; x++ {
			src.Data[idx(y, x, 9)] = 255
		}
	}
	sk := Skeleton(src, RectElement(3, 3))
	if CountForeground(sk) == 0 {
		t.Fatalf("skeleton should be non-empty")
	}
	for i := range sk.Data {
		if sk.Data[i] != 0 && src.Data[i] == 0 {
			t.Fatalf("skeleton must be a subset of the source")
		}
	}
}

func TestReconstructByDilation(t *testing.T) {
	mask := mat(1, 7, []uint8{255, 255, 0, 0, 0, 255, 255})
	marker := mat(1, 7, []uint8{255, 0, 0, 0, 0, 0, 0})
	got := ReconstructByDilation(marker, mask, Conn4)
	want := []uint8{255, 255, 0, 0, 0, 0, 0}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("reconstruction = %v, want %v", got.Data, want)
		}
	}
}

func TestFillHoles(t *testing.T) {
	src := filled(5, 5, 255)
	src.Data[idx(2, 2, 5)] = 0 // isolated interior hole
	out := FillHoles(src, Conn4)
	if CountForeground(out) != 25 {
		t.Fatalf("fill holes should fill the interior hole, got %d", CountForeground(out))
	}
}

func TestClearBorder(t *testing.T) {
	src := cv.NewMat(5, 5, 1)
	src.Data[idx(0, 0, 5)] = 255 // touches border
	src.Data[idx(2, 2, 5)] = 255 // interior
	out := ClearBorder(src, Conn4)
	if out.Data[idx(0, 0, 5)] != 0 {
		t.Fatalf("border component should be cleared")
	}
	if out.Data[idx(2, 2, 5)] != 255 {
		t.Fatalf("interior component should survive")
	}
	if CountForeground(out) != 1 {
		t.Fatalf("expected one surviving component, got %d", CountForeground(out))
	}
}

func TestRegionalMaxima(t *testing.T) {
	src := filled(5, 5, 10)
	for _, p := range [][2]int{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		src.Data[idx(p[0], p[1], 5)] = 200
	}
	out := RegionalMaxima(src, Conn8)
	if CountForeground(out) != 4 {
		t.Fatalf("regional maxima count = %d, want 4", CountForeground(out))
	}
	if out.Data[idx(1, 1, 5)] != 255 || out.Data[idx(0, 0, 5)] != 0 {
		t.Fatalf("regional maxima mislabelled")
	}
}

func TestHMaxima(t *testing.T) {
	src := filled(5, 5, 50)
	src.Data[idx(2, 2, 5)] = 100
	out := HMaxima(src, 20, Conn4)
	if out.Data[idx(2, 2, 5)] != 80 {
		t.Fatalf("h-maxima centre = %d, want 80", out.Data[idx(2, 2, 5)])
	}
	if out.Data[idx(0, 0, 5)] != 50 {
		t.Fatalf("h-maxima background = %d, want 50", out.Data[idx(0, 0, 5)])
	}
}

func TestExtendedMaxima(t *testing.T) {
	src := filled(5, 5, 50)
	src.Data[idx(2, 2, 5)] = 100
	out := ExtendedMaxima(src, 20, Conn4)
	if CountForeground(out) != 1 || out.Data[idx(2, 2, 5)] != 255 {
		t.Fatalf("extended maxima should mark the single dome")
	}
}

func TestRegionalMinima(t *testing.T) {
	src := filled(5, 5, 50)
	src.Data[idx(2, 2, 5)] = 0
	out := RegionalMinima(src, Conn4)
	if out.Data[idx(2, 2, 5)] != 255 || CountForeground(out) != 1 {
		t.Fatalf("regional minima should mark the single basin")
	}
}

func TestImposeMinima(t *testing.T) {
	src := filled(5, 5, 100)
	mk := cv.NewMat(5, 5, 1)
	mk.Data[idx(2, 2, 5)] = 255
	out := ImposeMinima(src, mk, Conn4)
	if out.Data[idx(2, 2, 5)] != 0 {
		t.Fatalf("imposed minimum should be 0, got %d", out.Data[idx(2, 2, 5)])
	}
	for i := range out.Data {
		if out.Data[i] < out.Data[idx(2, 2, 5)] {
			t.Fatalf("imposed minimum must be the global minimum")
		}
	}
}

func TestDistanceL1(t *testing.T) {
	src := filled(5, 5, 255)
	src.Data[idx(0, 0, 5)] = 0
	g := DistanceTransformL1(src)
	cases := map[[2]int]float64{{0, 0}: 0, {2, 2}: 4, {4, 4}: 8, {0, 4}: 4}
	for p, want := range cases {
		if got := g.At(p[0], p[1]); got != want {
			t.Fatalf("L1 dist at %v = %v, want %v", p, got, want)
		}
	}
}

func TestDistanceExact(t *testing.T) {
	src := filled(5, 5, 255)
	src.Data[idx(0, 0, 5)] = 0
	g := DistanceTransformExact(src)
	if math.Abs(g.At(3, 4)-5) > 1e-9 || math.Abs(g.At(4, 3)-5) > 1e-9 {
		t.Fatalf("exact 3-4-5 distance wrong: %v %v", g.At(3, 4), g.At(4, 3))
	}
	if math.Abs(g.At(2, 2)-math.Sqrt(8)) > 1e-9 {
		t.Fatalf("exact diagonal distance = %v, want %v", g.At(2, 2), math.Sqrt(8))
	}
}

func TestDistanceChebyshev(t *testing.T) {
	src := filled(5, 5, 255)
	src.Data[idx(0, 0, 5)] = 0
	g := DistanceTransformChebyshev(src)
	if g.At(2, 3) != 3 || g.At(4, 1) != 4 {
		t.Fatalf("chebyshev distances wrong: %v %v", g.At(2, 3), g.At(4, 1))
	}
}

func TestGranulometry(t *testing.T) {
	src := cv.NewMat(9, 9, 1)
	for y := 3; y <= 5; y++ {
		for x := 3; x <= 5; x++ {
			src.Data[idx(y, x, 9)] = 255
		}
	}
	ps := Granulometry(src, ShapeRect, 2, 1)
	if len(ps.Sizes) != 3 {
		t.Fatalf("expected sizes [0 1 2], got %v", ps.Sizes)
	}
	if math.Abs(ps.Spectrum[0]) > 1e-9 || math.Abs(ps.Spectrum[1]-1) > 1e-9 {
		t.Fatalf("pattern spectrum = %v, want [0 1]", ps.Spectrum)
	}
	if math.Abs(ps.Mean()-2) > 1e-9 {
		t.Fatalf("granulometric mean = %v, want 2", ps.Mean())
	}
}

func TestWatershed(t *testing.T) {
	gray := mat(1, 9, []uint8{0, 1, 2, 3, 4, 3, 2, 1, 0})
	markers := NewLabelMap(1, 9)
	markers.Set(0, 0, 1)
	markers.Set(0, 8, 2)
	out := Watershed(gray, markers, Conn4)
	want := []int32{1, 1, 1, 1, WatershedRidge, 2, 2, 2, 2}
	for i := range want {
		if out.Labels[i] != want[i] {
			t.Fatalf("watershed labels = %v, want %v", out.Labels, want)
		}
	}
	if CountForeground(out.Ridges()) != 1 {
		t.Fatalf("expected exactly one ridge pixel")
	}
}

func TestConnectedComponentMarkers(t *testing.T) {
	src := cv.NewMat(5, 5, 1)
	src.Data[idx(0, 0, 5)] = 255
	src.Data[idx(0, 1, 5)] = 255
	src.Data[idx(4, 4, 5)] = 255
	lm := ConnectedComponentMarkers(src, Conn4)
	if lm.NumLabels() != 2 {
		t.Fatalf("expected 2 components, got %d", lm.NumLabels())
	}
	if lm.At(0, 0) != lm.At(0, 1) || lm.At(0, 0) == lm.At(4, 4) {
		t.Fatalf("component labelling incorrect")
	}
}

func TestSetOps(t *testing.T) {
	a := mat(1, 4, []uint8{255, 255, 0, 0})
	b := mat(1, 4, []uint8{0, 255, 255, 0})
	if !Equal(Union(a, b), mat(1, 4, []uint8{255, 255, 255, 0})) {
		t.Fatalf("union wrong")
	}
	if !Equal(Intersection(a, b), mat(1, 4, []uint8{0, 255, 0, 0})) {
		t.Fatalf("intersection wrong")
	}
	if !Equal(Difference(a, b), mat(1, 4, []uint8{255, 0, 0, 0})) {
		t.Fatalf("difference wrong")
	}
	if !Equal(Complement(a), mat(1, 4, []uint8{0, 0, 255, 255})) {
		t.Fatalf("complement wrong")
	}
}

func TestMorphologyEx(t *testing.T) {
	src := cv.NewMat(5, 5, 1)
	src.Data[idx(2, 2, 5)] = 255
	e := RectElement(3, 3)
	if !Equal(MorphologyEx(src, e, OpDilate, 1), Dilate(src, e)) {
		t.Fatalf("MorphologyEx OpDilate mismatch")
	}
	if !Equal(MorphologyEx(src, e, OpOpen, 1), Open(src, e)) {
		t.Fatalf("MorphologyEx OpOpen mismatch")
	}
}

func makeReliefWithMarkers(n int) (*cv.Mat, *LabelMap) {
	img := cv.NewMat(n, n, 1)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			// A ridge down the middle column separates two basins.
			d := x - n/2
			if d < 0 {
				d = -d
			}
			v := 255 - d*255/(n/2+1)
			img.Data[idx(y, x, n)] = uint8(v)
		}
	}
	m := NewLabelMap(n, n)
	for y := 0; y < n; y++ {
		m.Set(y, 0, 1)
		m.Set(y, n-1, 2)
	}
	return img, m
}

func BenchmarkWatershed(b *testing.B) {
	img, markers := makeReliefWithMarkers(128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Watershed(img, markers, Conn8)
	}
}

func BenchmarkReconstructByDilation(b *testing.B) {
	mask := filled(128, 128, 200)
	marker := cv.NewMat(128, 128, 1)
	marker.Data[idx(64, 64, 128)] = 200
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ReconstructByDilation(marker, mask, Conn8)
	}
}

func BenchmarkExactEuclidean(b *testing.B) {
	src := filled(256, 256, 255)
	src.Data[idx(0, 0, 256)] = 0
	src.Data[idx(255, 255, 256)] = 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DistanceTransformExact(src)
	}
}

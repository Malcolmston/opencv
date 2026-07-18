package edges2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// verticalStep builds a rows×cols image whose columns < boundary are 0 and
// whose columns >= boundary are 255: a single vertical step edge.
func verticalStep(rows, cols, boundary int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := boundary; x < cols; x++ {
			m.Data[y*cols+x] = 255
		}
	}
	return m
}

// blank builds a black rows×cols image.
func blank(rows, cols int) *cv.Mat { return cv.NewMat(rows, cols, 1) }

func TestSobelVerticalEdge(t *testing.T) {
	src := verticalStep(9, 9, 5)
	f := Sobel(src)
	// At the boundary column Gx must be strongly positive (dark→bright) and
	// Gy must be near zero. In flat regions both must vanish.
	gxEdge, gyEdge := f.At(4, 5)
	if gxEdge <= 0 {
		t.Fatalf("expected positive Gx at edge, got %v", gxEdge)
	}
	if math.Abs(gyEdge) > 1e-9 {
		t.Fatalf("expected ~0 Gy at edge, got %v", gyEdge)
	}
	gxFlat, gyFlat := f.At(4, 1)
	if gxFlat != 0 || gyFlat != 0 {
		t.Fatalf("expected zero gradient in flat region, got (%v,%v)", gxFlat, gyFlat)
	}
	// Sobel Gx of a full-amplitude step is 4*255 = 1020.
	if gxEdge != 1020 {
		t.Fatalf("expected Gx == 1020, got %v", gxEdge)
	}
}

func TestScharrPrewittRobertsRespondToEdge(t *testing.T) {
	src := verticalStep(9, 9, 5)
	for name, gf := range map[string]*GradientField{
		"Scharr":  Scharr(src),
		"Prewitt": Prewitt(src),
		"Roberts": Roberts(src),
	} {
		mag := gf.Magnitude()
		if mag.At(4, 5) <= 0 && mag.At(4, 4) <= 0 {
			t.Fatalf("%s: expected a response near the edge", name)
		}
		if mag.At(4, 1) != 0 {
			t.Fatalf("%s: expected zero response in flat region, got %v", name, mag.At(4, 1))
		}
	}
}

func TestFloatGridToMat(t *testing.T) {
	g := NewFloatGrid(1, 3)
	g.Data[0] = -5
	g.Data[1] = 100
	g.Data[2] = 400
	m := g.ToMat()
	if m.Data[0] != 0 || m.Data[1] != 100 || m.Data[2] != 255 {
		t.Fatalf("ToMat clamp wrong: %v", m.Data)
	}
	n := g.ToMatNormalized()
	if n.Data[0] != 0 || n.Data[2] != 255 {
		t.Fatalf("ToMatNormalized endpoints wrong: %v", n.Data)
	}
	if n.Data[1] <= n.Data[0] || n.Data[1] >= n.Data[2] {
		t.Fatalf("ToMatNormalized not monotonic: %v", n.Data)
	}
}

func TestDoubleThreshold(t *testing.T) {
	g := NewFloatGrid(1, 3)
	g.Data[0] = 5
	g.Data[1] = 15
	g.Data[2] = 25
	m := DoubleThreshold(g, 10, 20)
	if m.Data[0] != 0 || m.Data[1] != 128 || m.Data[2] != 255 {
		t.Fatalf("DoubleThreshold levels wrong: %v", m.Data)
	}
}

func TestCannyStepEdge(t *testing.T) {
	src := verticalStep(15, 15, 8)
	edges := Canny(src, 50, 150, 1.0)
	// There must be edge pixels, and they must concentrate near column 8.
	count := 0
	edgeInBand := 0
	for y := 0; y < edges.Rows; y++ {
		for x := 0; x < edges.Cols; x++ {
			if edges.Data[y*edges.Cols+x] != 0 {
				count++
				if x >= 6 && x <= 9 {
					edgeInBand++
				}
			}
		}
	}
	if count == 0 {
		t.Fatal("Canny found no edges on a step image")
	}
	if edgeInBand*2 < count {
		t.Fatalf("Canny edges not concentrated at the boundary: %d/%d in band", edgeInBand, count)
	}
}

func TestHysteresisConnectsWeak(t *testing.T) {
	// A row of magnitudes: one strong seed with weak neighbours must all be
	// kept; an isolated weak pixel must be dropped.
	g := NewFloatGrid(1, 6)
	g.Data = []float64{0, 12, 12, 30, 0, 12}
	out := Hysteresis(g, 10, 20)
	want := []uint8{0, 255, 255, 255, 0, 0}
	for i := range want {
		if out.Data[i] != want[i] {
			t.Fatalf("Hysteresis got %v want %v", out.Data, want)
		}
	}
}

func TestMarrHildreth(t *testing.T) {
	src := verticalStep(15, 15, 8)
	edges := MarrHildreth(src, 1.2, 1.0)
	found := false
	for y := 0; y < edges.Rows; y++ {
		for x := 6; x <= 9; x++ {
			if edges.Data[y*edges.Cols+x] != 0 {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("MarrHildreth found no zero crossings near the edge")
	}
	// A flat image must produce no zero crossings.
	flat := cv.NewMat(10, 10, 1)
	for i := range flat.Data {
		flat.Data[i] = 128
	}
	fe := MarrHildreth(flat, 1.2, 1.0)
	for _, v := range fe.Data {
		if v != 0 {
			t.Fatal("MarrHildreth found edges in a flat image")
		}
	}
}

func TestDifferenceOfGaussians(t *testing.T) {
	src := verticalStep(15, 15, 8)
	dog := DifferenceOfGaussians(src, 1.0, 2.0)
	_, max := dog.Abs().MinMax()
	if max <= 0 {
		t.Fatal("DoG has no response on a step edge")
	}
}

func TestHoughLinesHorizontal(t *testing.T) {
	edges := blank(20, 20)
	const y0 = 5
	for x := 0; x < 20; x++ {
		edges.Data[y0*20+x] = 255
	}
	lines := HoughLines(edges, 1, math.Pi/180, 10)
	if len(lines) == 0 {
		t.Fatal("no lines detected for a horizontal line")
	}
	l := lines[0]
	if math.Abs(l.Theta-math.Pi/2) > math.Pi/180+1e-9 {
		t.Fatalf("expected theta≈pi/2, got %v", l.Theta)
	}
	if math.Abs(l.Rho-y0) > 1 {
		t.Fatalf("expected rho≈%d, got %v", y0, l.Rho)
	}
	if l.Votes != 20 {
		t.Fatalf("expected 20 votes, got %d", l.Votes)
	}
}

func TestHoughLinesVertical(t *testing.T) {
	edges := blank(20, 20)
	const x0 = 7
	for y := 0; y < 20; y++ {
		edges.Data[y*20+x0] = 255
	}
	lines := HoughLines(edges, 1, math.Pi/180, 10)
	if len(lines) == 0 {
		t.Fatal("no lines detected for a vertical line")
	}
	l := lines[0]
	// A vertical line has a horizontal normal, so sin(theta)≈0 (theta near 0
	// or near pi are equivalent representations) and |rho|≈x0.
	if math.Abs(math.Sin(l.Theta)) > 0.05 {
		t.Fatalf("expected a vertical line (sin theta≈0), got theta=%v", l.Theta)
	}
	if math.Abs(math.Abs(l.Rho)-x0) > 1 {
		t.Fatalf("expected |rho|≈%d, got %v", x0, l.Rho)
	}
}

func TestHoughLinesP(t *testing.T) {
	edges := blank(20, 20)
	const y0 = 5
	for x := 2; x <= 15; x++ {
		edges.Data[y0*20+x] = 255
	}
	segs := HoughLinesP(edges, 1, math.Pi/180, 10, 5, 2)
	if len(segs) == 0 {
		t.Fatal("no probabilistic segments detected")
	}
	s := segs[0]
	if s.Length() < 10 {
		t.Fatalf("expected a long segment, got length %v", s.Length())
	}
	if math.Abs(s.Y1-y0) > 1.5 || math.Abs(s.Y2-y0) > 1.5 {
		t.Fatalf("segment not on the drawn row: %+v", s)
	}
}

func TestHoughCircles(t *testing.T) {
	img := DrawCircles(blank(41, 41), []Circle{{X: 20, Y: 20, Radius: 10}}, 255)
	circles := HoughCircles(img, 6, 14, 15, 50, 150, 5)
	if len(circles) == 0 {
		t.Fatal("no circles detected")
	}
	c := circles[0]
	if math.Abs(c.X-20) > 2 || math.Abs(c.Y-20) > 2 {
		t.Fatalf("circle centre off: %+v", c)
	}
	if math.Abs(c.Radius-10) > 2 {
		t.Fatalf("circle radius off: %+v", c)
	}
}

func TestLineToSegment(t *testing.T) {
	// Horizontal line rho=5, theta=pi/2 must clip to y=5 across the width.
	l := Line{Rho: 5, Theta: math.Pi / 2}
	seg, ok := l.ToSegment(20, 20)
	if !ok {
		t.Fatal("expected the line to intersect the image")
	}
	if math.Abs(seg.Y1-5) > 1e-6 || math.Abs(seg.Y2-5) > 1e-6 {
		t.Fatalf("clipped segment not on y=5: %+v", seg)
	}
}

func TestLinkEdges(t *testing.T) {
	edges := blank(10, 10)
	// A single connected diagonal of five pixels.
	for i := 0; i < 5; i++ {
		edges.Data[i*10+i] = 255
	}
	// A separate isolated pixel that must be dropped by minLength.
	edges.Data[9*10+0] = 255
	chains := LinkEdges(edges, 3)
	if len(chains) != 1 {
		t.Fatalf("expected 1 chain, got %d", len(chains))
	}
	if chains[0].Length() != 5 {
		t.Fatalf("expected chain length 5, got %d", chains[0].Length())
	}
	min, max := chains[0].BoundingBox()
	if min != (Point{0, 0}) || max != (Point{4, 4}) {
		t.Fatalf("bounding box wrong: %+v %+v", min, max)
	}
}

func TestOrientationHistogram(t *testing.T) {
	src := verticalStep(12, 12, 6)
	hist := OrientationHistogram(Sobel(src), 8, false)
	// Gradients point in +x (angle 0), so bin 0 must dominate.
	argmax := 0
	for i, v := range hist {
		if v > hist[argmax] {
			argmax = i
		}
		_ = v
	}
	if argmax != 0 {
		t.Fatalf("expected orientation bin 0 to dominate, got %d (%v)", argmax, hist)
	}
}

func TestHOG(t *testing.T) {
	src := verticalStep(16, 16, 8)
	opts := DefaultHOGOptions()
	d1 := HOG(src, opts)
	d2 := HOG(src, opts)
	// cellsY=cellsX=2, one 2x2 block, 9 bins -> 36 values.
	if len(d1) != 36 {
		t.Fatalf("expected descriptor length 36, got %d", len(d1))
	}
	for i := range d1 {
		if d1[i] != d2[i] {
			t.Fatal("HOG is not deterministic")
		}
	}
	var ss float64
	for _, v := range d1 {
		ss += v * v
	}
	norm := math.Sqrt(ss)
	if norm < 0.9 || norm > 1.01 {
		t.Fatalf("expected an L2-normalised block (~1), got %v", norm)
	}
}

func TestLSD(t *testing.T) {
	src := verticalStep(20, 20, 10)
	segs := LSD(src, DefaultLSDOptions())
	if len(segs) == 0 {
		t.Fatal("LSD found no segments on a step edge")
	}
	s := segs[0]
	if s.Length() < 10 {
		t.Fatalf("expected a long vertical segment, got length %v", s.Length())
	}
	// A vertical segment has a near-zero horizontal component.
	if math.Abs(math.Cos(s.Angle())) > 0.3 {
		t.Fatalf("expected a near-vertical segment, angle %v", s.Angle())
	}
}

func TestSegmentMethods(t *testing.T) {
	s := Segment{X1: 0, Y1: 0, X2: 3, Y2: 4}
	if s.Length() != 5 {
		t.Fatalf("length: got %v want 5", s.Length())
	}
	mx, my := s.Midpoint()
	if mx != 1.5 || my != 2 {
		t.Fatalf("midpoint: got (%v,%v)", mx, my)
	}
	if math.Abs(s.Angle()-math.Atan2(4, 3)) > 1e-9 {
		t.Fatalf("angle wrong: %v", s.Angle())
	}
}

func TestStructuredEdges(t *testing.T) {
	src := verticalStep(20, 20, 10)
	m := StructuredEdges(src, 1.0)
	// Edge energy must concentrate at the boundary column and fall off with
	// distance from it.
	edgeVal := m.Data[10*20+10]
	farVal := m.Data[10*20+2]
	if edgeVal == 0 {
		t.Fatal("StructuredEdges gave no response at the boundary")
	}
	if edgeVal <= farVal {
		t.Fatalf("StructuredEdges energy not peaked at the edge: edge=%d far=%d", edgeVal, farVal)
	}
}

func TestGradientMagnitudeFlat(t *testing.T) {
	flat := cv.NewMat(8, 8, 1)
	for i := range flat.Data {
		flat.Data[i] = 200
	}
	m := GradientMagnitude(flat)
	for _, v := range m.Data {
		if v != 0 {
			t.Fatal("gradient magnitude of a flat image is nonzero")
		}
	}
}

func BenchmarkCanny(b *testing.B) {
	// A checkerboard exercises the full pipeline on many edges.
	const n = 128
	src := cv.NewMat(n, n, 1)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if (x/8+y/8)%2 == 0 {
				src.Data[y*n+x] = 255
			}
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Canny(src, 50, 150, 1.0)
	}
}

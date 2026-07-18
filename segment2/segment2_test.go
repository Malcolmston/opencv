package segment2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// twoColorImage builds a rows×cols 3-channel image whose left half (x < cols/2)
// is colour a and right half is colour b.
func twoColorImage(rows, cols int, a, b [3]uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			c := a
			if x >= cols/2 {
				c = b
			}
			base := (y*cols + x) * 3
			m.Data[base] = c[0]
			m.Data[base+1] = c[1]
			m.Data[base+2] = c[2]
		}
	}
	return m
}

func TestKMeansSegmentTwoColors(t *testing.T) {
	img := twoColorImage(4, 8, [3]uint8{10, 10, 10}, [3]uint8{240, 240, 240})
	res := KMeansSegment(img, 2, 20)
	if res.Labels.CountRegions() != 2 {
		t.Fatalf("want 2 regions, got %d", res.Labels.CountRegions())
	}
	// All left pixels share one label, all right pixels another.
	left := res.Labels.At(0, 0)
	right := res.Labels.At(0, 7)
	if left == right {
		t.Fatalf("left and right must differ")
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			want := left
			if x >= 4 {
				want = right
			}
			if got := res.Labels.At(y, x); got != want {
				t.Fatalf("label mismatch at (%d,%d): got %d want %d", x, y, got, want)
			}
		}
	}
	if res.Compactness != 0 {
		t.Fatalf("two flat colours must cluster with zero compactness, got %v", res.Compactness)
	}
}

func TestQuantizeColorsIdentity(t *testing.T) {
	img := twoColorImage(4, 8, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255})
	out := QuantizeColors(img, 2, 20)
	for i := range img.Data {
		if out.Data[i] != img.Data[i] {
			t.Fatalf("quantize of 2-colour image with k=2 must be identity at %d: %d != %d", i, out.Data[i], img.Data[i])
		}
	}
}

func TestMeanShiftFilterDeterministicAndFlat(t *testing.T) {
	img := cv.NewMat(6, 6, 3)
	for i := range img.Data {
		img.Data[i] = 100
	}
	a := MeanShiftFilter(img, 3, 20, 5)
	b := MeanShiftFilter(img, 3, 20, 5)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("mean shift must be deterministic")
		}
		if a.Data[i] != 100 {
			t.Fatalf("flat image must be preserved, got %d", a.Data[i])
		}
	}
}

func TestMeanShiftSegmentTwoColors(t *testing.T) {
	img := twoColorImage(6, 8, [3]uint8{20, 20, 20}, [3]uint8{220, 220, 220})
	lm := MeanShiftSegment(img, MeanShiftParams{SpatialRadius: 3, ColorRadius: 30, MaxIter: 5, Epsilon: 1}, 1)
	if lm.CountRegions() != 2 {
		t.Fatalf("want 2 regions, got %d", lm.CountRegions())
	}
	if lm.At(0, 0) == lm.At(0, 7) {
		t.Fatalf("distinct colours must land in distinct regions")
	}
}

func TestDistanceTransformKnown(t *testing.T) {
	// 4x4 all foreground except the single background pixel at (0,0).
	m := cv.NewMat(4, 4, 1)
	for i := range m.Data {
		m.Data[i] = 255
	}
	m.Data[0] = 0
	d := DistanceTransform(m)
	check := func(x, y int, want float64) {
		got := d[y*4+x]
		if math.Abs(got-want) > 1e-9 {
			t.Fatalf("dist at (%d,%d) = %v, want %v", x, y, got, want)
		}
	}
	check(0, 0, 0)
	check(1, 0, 1)
	check(0, 1, 1)
	check(1, 1, math.Sqrt2)
	check(2, 0, 2)
	check(3, 3, math.Sqrt(18))
}

func TestGradientMagnitudeUniform(t *testing.T) {
	m := cv.NewMat(5, 5, 1)
	for i := range m.Data {
		m.Data[i] = 77
	}
	g := GradientMagnitude(m)
	for i, v := range g {
		if v != 0 {
			t.Fatalf("uniform image gradient must be 0 at %d, got %v", i, v)
		}
	}
}

func TestWatershedTwoBasins(t *testing.T) {
	rows, cols := 5, 7
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := uint8(10)
			if x >= 4 {
				v = 240
			} else if x == 3 {
				v = 125
			}
			img.Data[y*cols+x] = v
		}
	}
	markers := make([]int, rows*cols)
	markers[2*cols+0] = 1
	markers[2*cols+6] = 2
	labels := Watershed(img, markers)
	if labels[2*cols+1] != 1 {
		t.Fatalf("left interior should be basin 1, got %d", labels[2*cols+1])
	}
	if labels[2*cols+5] != 2 {
		t.Fatalf("right interior should be basin 2, got %d", labels[2*cols+5])
	}
}

func TestRegionGrowTwoColors(t *testing.T) {
	img := twoColorImage(4, 8, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255})
	mask := RegionGrow(img, cv.Point{X: 0, Y: 0}, 10, Conn4)
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			want := uint8(0)
			if x < 4 {
				want = 255
			}
			if mask.Data[y*8+x] != want {
				t.Fatalf("region grow mask at (%d,%d) = %d want %d", x, y, mask.Data[y*8+x], want)
			}
		}
	}
}

func TestSeededRegionGrowPartition(t *testing.T) {
	img := twoColorImage(4, 8, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255})
	seeds := NewLabelMap(4, 8)
	seeds.Set(0, 0, 1)
	seeds.Set(0, 7, 2)
	lm := SeededRegionGrow(img, seeds, Conn4)
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			want := 1
			if x >= 4 {
				want = 2
			}
			if got := lm.At(y, x); got != want {
				t.Fatalf("seeded grow at (%d,%d) = %d want %d", x, y, got, want)
			}
		}
	}
}

func TestSplitAndMergeTwoColors(t *testing.T) {
	img := twoColorImage(8, 8, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255})
	lm := SplitAndMerge(img, 20, 1)
	if lm.NumLabels != 2 {
		t.Fatalf("want 2 regions, got %d", lm.NumLabels)
	}
	if lm.At(0, 0) == lm.At(0, 7) {
		t.Fatalf("split-and-merge must separate the two halves")
	}
	if lm.At(0, 0) != lm.At(7, 3) {
		t.Fatalf("left half must share a label")
	}
}

func TestFelzenszwalbTwoColors(t *testing.T) {
	img := twoColorImage(10, 12, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255})
	lm := Felzenszwalb(img, 0, 300, 5)
	if lm.NumLabels != 2 {
		t.Fatalf("want 2 regions, got %d", lm.NumLabels)
	}
	if lm.At(5, 0) == lm.At(5, 11) {
		t.Fatalf("halves must be distinct regions")
	}
}

func TestSLICCoverage(t *testing.T) {
	rows, cols := 20, 20
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			b := (y*cols + x) * 3
			img.Data[b] = uint8(x * 12 % 256)
			img.Data[b+1] = uint8(y * 12 % 256)
			img.Data[b+2] = 128
		}
	}
	lm := SLIC(img, 10, 10, 5)
	if lm.NumLabels < 2 || lm.NumLabels > 40 {
		t.Fatalf("unexpected superpixel count %d", lm.NumLabels)
	}
	for i, l := range lm.Labels {
		if l < 0 || l >= lm.NumLabels {
			t.Fatalf("pixel %d has invalid label %d", i, l)
		}
	}
}

func TestSLICOSeparatesColors(t *testing.T) {
	img := twoColorImage(20, 20, [3]uint8{0, 0, 0}, [3]uint8{255, 255, 255})
	lm := SLICO(img, 8, 8)
	// No superpixel should straddle the strong colour edge: the label at the
	// far left must not equal the label at the far right.
	if lm.At(10, 0) == lm.At(10, 19) {
		t.Fatalf("SLICO must not merge across a strong colour boundary")
	}
}

func TestMaxFlowKnown(t *testing.T) {
	// s=0, t=3. Capacities give a known max flow of 5.
	g := NewFlowGraph(4, 0, 3)
	g.AddEdge(0, 1, 3, 0)
	g.AddEdge(0, 2, 2, 0)
	g.AddEdge(1, 2, 1, 0)
	g.AddEdge(1, 3, 2, 0)
	g.AddEdge(2, 3, 3, 0)
	flow, reach := g.MaxFlow()
	if math.Abs(flow-5) > 1e-9 {
		t.Fatalf("max flow = %v, want 5", flow)
	}
	if !reach[0] {
		t.Fatalf("source must be on source side")
	}
	if reach[3] {
		t.Fatalf("sink must not be on source side")
	}
}

func TestGrabCutSquare(t *testing.T) {
	rows, cols := 16, 16
	img := cv.NewMat(rows, cols, 3)
	for y := 5; y < 11; y++ {
		for x := 5; x < 11; x++ {
			b := (y*cols + x) * 3
			img.Data[b] = 255
			img.Data[b+1] = 255
			img.Data[b+2] = 255
		}
	}
	mask := GrabCut(img, cv.Rect{X: 3, Y: 3, Width: 10, Height: 10}, 3)
	if mask.Data[8*cols+8] != 255 {
		t.Fatalf("square centre must be foreground")
	}
	if mask.Data[0] != 0 {
		t.Fatalf("image corner must be background")
	}
	var fg int
	for _, v := range mask.Data {
		if v == 255 {
			fg++
		}
	}
	if fg < 30 || fg > 70 {
		t.Fatalf("foreground area %d out of expected range for a 36-pixel square", fg)
	}
}

func TestOtsuBimodal(t *testing.T) {
	img := twoColorImage(6, 6, [3]uint8{40, 40, 40}, [3]uint8{210, 210, 210})
	th := OtsuThreshold(img)
	// Otsu selects the lower edge of the empty gap between the two modes, so the
	// bright class (values > th) is exactly the 210-valued pixels.
	if th < 40 || th >= 210 {
		t.Fatalf("Otsu threshold %v must separate the two modes", th)
	}
}

func TestConnectedComponentsTwoBlobs(t *testing.T) {
	rows, cols := 5, 5
	m := cv.NewMat(rows, cols, 1)
	// Blob A: top-left 2x2. Blob B: single pixel bottom-right.
	m.Data[0*cols+0] = 255
	m.Data[0*cols+1] = 255
	m.Data[1*cols+0] = 255
	m.Data[1*cols+1] = 255
	m.Data[4*cols+4] = 255
	lm := ConnectedComponents(m, Conn4)
	if lm.NumLabels != 3 {
		t.Fatalf("want background + 2 components = 3, got %d", lm.NumLabels)
	}
	if lm.At(0, 0) == 0 || lm.At(4, 4) == 0 {
		t.Fatalf("foreground pixels must not be background")
	}
	if lm.At(0, 0) == lm.At(4, 4) {
		t.Fatalf("disconnected blobs must have distinct labels")
	}
	if lm.At(0, 0) != lm.At(1, 1) {
		t.Fatalf("connected pixels must share a label")
	}
}

func TestThresholdComponentsAndStats(t *testing.T) {
	rows, cols := 5, 5
	img := cv.NewMat(rows, cols, 1)
	for x := 0; x < 2; x++ {
		for y := 0; y < 2; y++ {
			img.Data[y*cols+x] = 255
		}
	}
	img.Data[4*cols+4] = 255
	lm := ThresholdComponents(img, 128, Conn4)
	stats := ComponentStats(lm)
	if len(stats) != 2 {
		t.Fatalf("want 2 components, got %d", len(stats))
	}
	big := LargestComponent(lm)
	sizes := lm.RegionSizes()
	if sizes[big] != 4 {
		t.Fatalf("largest component area = %d, want 4", sizes[big])
	}
	filtered := FilterComponentsBySize(lm, 2)
	if n := len(ComponentStats(filtered)); n != 1 {
		t.Fatalf("filtering minArea=2 should leave 1 foreground component, got %d", n)
	}
}

func TestActiveContourContracts(t *testing.T) {
	img := cv.NewMat(40, 40, 1) // uniform -> no external force, pure contraction
	var ring []cv.Point
	cx, cy, r := 20.0, 20.0, 15.0
	n := 16
	for i := 0; i < n; i++ {
		th := 2 * math.Pi * float64(i) / float64(n)
		ring = append(ring, cv.Point{
			X: int(cx + r*math.Cos(th) + 0.5),
			Y: int(cy + r*math.Sin(th) + 0.5),
		})
	}
	before := ContourArea(ring)
	p := ActiveContourParams{Alpha: 0.2, Beta: 0.1, Gamma: 1, EdgeWeight: 0, Sigma: 0, Iterations: 60}
	out := ActiveContour(img, ring, p)
	after := ContourArea(out)
	if after >= before {
		t.Fatalf("tension should shrink the snake: before=%v after=%v", before, after)
	}
	if after <= 0 {
		t.Fatalf("contour collapsed to zero area")
	}
}

func TestContourAreaAndPerimeter(t *testing.T) {
	sq := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
	if a := ContourArea(sq); math.Abs(a-100) > 1e-9 {
		t.Fatalf("square area = %v, want 100", a)
	}
	if p := ContourPerimeter(sq); math.Abs(p-40) > 1e-9 {
		t.Fatalf("square perimeter = %v, want 40", p)
	}
}

func TestLabelMapHelpers(t *testing.T) {
	lm := NewLabelMap(2, 4)
	// Left two columns label 0, right two label 5 (sparse).
	for y := 0; y < 2; y++ {
		lm.Set(y, 2, 5)
		lm.Set(y, 3, 5)
	}
	if lm.CountRegions() != 2 {
		t.Fatalf("want 2 regions, got %d", lm.CountRegions())
	}
	lm.Compact()
	if lm.NumLabels != 2 {
		t.Fatalf("compact should give NumLabels 2, got %d", lm.NumLabels)
	}
	sizes := lm.RegionSizes()
	if sizes[0] != 4 || sizes[1] != 4 {
		t.Fatalf("region sizes = %v, want [4 4]", sizes)
	}
	rects := lm.BoundingRects()
	if rects[1].X != 2 || rects[1].Width != 2 || rects[1].Height != 2 {
		t.Fatalf("region 1 rect = %+v", rects[1])
	}
	bm := lm.BoundaryMask()
	if bm.Data[0*4+1] == 0 {
		t.Fatalf("boundary between columns 1 and 2 should be marked")
	}
	col := lm.Colorize()
	if col.Channels != 3 {
		t.Fatalf("colorize must be 3-channel")
	}
	m := lm.RegionMask(1)
	if m.Data[0*4+2] != 255 || m.Data[0*4+0] != 0 {
		t.Fatalf("region mask wrong")
	}
}

func TestLabelMapMeanColors(t *testing.T) {
	img := twoColorImage(2, 4, [3]uint8{10, 10, 10}, [3]uint8{200, 200, 200})
	lm := NewLabelMap(2, 4)
	for y := 0; y < 2; y++ {
		for x := 2; x < 4; x++ {
			lm.Set(y, x, 1)
		}
	}
	means := lm.MeanColors(img)
	if math.Abs(means[0][0]-10) > 1e-9 || math.Abs(means[1][0]-200) > 1e-9 {
		t.Fatalf("mean colours wrong: %v", means)
	}
}

func BenchmarkGrabCut(b *testing.B) {
	rows, cols := 40, 40
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := (y*cols + x) * 3
			if x >= 12 && x < 28 && y >= 12 && y < 28 {
				img.Data[base] = 230
				img.Data[base+1] = 200
				img.Data[base+2] = 180
			} else {
				img.Data[base] = 30
				img.Data[base+1] = 40
				img.Data[base+2] = 50
			}
		}
	}
	rect := cv.Rect{X: 8, Y: 8, Width: 24, Height: 24}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GrabCut(img, rect, 2)
	}
}

package segmentation

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// drawDisc paints a filled circle of value v (single channel) into m.
func drawDisc(m *cv.Mat, cx, cy, r int, v uint8) {
	for y := cy - r; y <= cy+r; y++ {
		for x := cx - r; x <= cx+r; x++ {
			if x < 0 || x >= m.Cols || y < 0 || y >= m.Rows {
				continue
			}
			dx, dy := x-cx, y-cy
			if dx*dx+dy*dy <= r*r {
				m.Set(y, x, 0, v)
			}
		}
	}
}

// quadImage builds a rows x cols three-channel image split into four solid-colour
// quadrants, a canonical multi-region test picture.
func quadImage(rows, cols int) *cv.Mat {
	img := cv.NewMat(rows, cols, 3)
	colors := [4][3]uint8{{220, 20, 20}, {20, 220, 20}, {20, 20, 220}, {220, 220, 20}}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			q := 0
			if x >= cols/2 {
				q |= 1
			}
			if y >= rows/2 {
				q |= 2
			}
			img.SetPixel(y, x, colors[q][:])
		}
	}
	return img
}

func TestEfficientGraphSegmentationQuadrants(t *testing.T) {
	img := quadImage(40, 40)
	lm := EfficientGraphSegmentation(img, 0, 300, 1)
	if lm.Count != 4 {
		t.Fatalf("Count = %d, want 4 quadrant regions", lm.Count)
	}
	// Each quadrant centre must carry a distinct label.
	labels := map[int]bool{
		lm.At(10, 10): true,
		lm.At(30, 10): true,
		lm.At(10, 30): true,
		lm.At(30, 30): true,
	}
	if len(labels) != 4 {
		t.Errorf("quadrant centres share labels: %v", labels)
	}
	// Region sizes sum to the pixel total.
	sum := 0
	for _, s := range lm.RegionSizes() {
		sum += s
	}
	if sum != 40*40 {
		t.Errorf("region sizes sum to %d, want %d", sum, 40*40)
	}
}

func TestEfficientGraphSegmentationDeterministic(t *testing.T) {
	img := quadImage(32, 32)
	a := EfficientGraphSegmentation(img, 0.8, 200, 5)
	b := EfficientGraphSegmentation(img, 0.8, 200, 5)
	if a.Count != b.Count {
		t.Fatalf("Count differs: %d vs %d", a.Count, b.Count)
	}
	for i := range a.Labels {
		if a.Labels[i] != b.Labels[i] {
			t.Fatalf("labels differ at %d", i)
		}
	}
}

func TestSLICSuperpixels(t *testing.T) {
	img := quadImage(40, 40)
	lm := SLIC(img, 10, 20, 5)
	// A 40x40 image at spacing 10 seeds a 4x4 grid, so expect roughly 16 regions.
	if lm.Count < 9 || lm.Count > 30 {
		t.Fatalf("SLIC produced %d superpixels, want ~16", lm.Count)
	}
	// Every pixel is labelled and sizes sum to the total.
	sum := 0
	for _, s := range lm.RegionSizes() {
		sum += s
	}
	if sum != 40*40 {
		t.Errorf("labelling incomplete: sizes sum to %d", sum)
	}
	// Determinism.
	lm2 := SLIC(img, 10, 20, 5)
	for i := range lm.Labels {
		if lm.Labels[i] != lm2.Labels[i] {
			t.Fatalf("SLIC not deterministic at %d", i)
		}
	}
}

func TestMultiOtsuThreeLevels(t *testing.T) {
	// Three horizontal bands at 30/130/220.
	rows, cols := 30, 30
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		v := uint8(30)
		switch {
		case y >= 2*rows/3:
			v = 220
		case y >= rows/3:
			v = 130
		}
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, v)
		}
	}
	th := MultiOtsu(img, 3)
	if len(th) != 2 {
		t.Fatalf("got %d thresholds, want 2", len(th))
	}
	if !(th[0] > 30 && th[0] <= 130) || !(th[1] > 130 && th[1] <= 220) {
		t.Errorf("thresholds %v do not separate the 30/130/220 bands", th)
	}
	lab, _ := MultiOtsuThreshold(img, 3)
	if lab.At(2, 5, 0) != 0 || lab.At(15, 5, 0) != 1 || lab.At(28, 5, 0) != 2 {
		t.Errorf("class labels wrong: %d %d %d, want 0 1 2",
			lab.At(2, 5, 0), lab.At(15, 5, 0), lab.At(28, 5, 0))
	}
}

func TestDistanceTransformValues(t *testing.T) {
	// 7x7 with a solid 5x5 foreground block inset by one pixel.
	m := cv.NewMat(7, 7, 1)
	for y := 1; y <= 5; y++ {
		for x := 1; x <= 5; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	dt := DistanceTransform(m)
	at := func(x, y int) float64 { return dt[y*7+x] }
	if at(0, 0) != 0 {
		t.Errorf("background distance = %v, want 0", at(0, 0))
	}
	if at(1, 1) != 1 {
		t.Errorf("corner foreground distance = %v, want 1", at(1, 1))
	}
	if at(3, 3) != 3 {
		t.Errorf("centre distance = %v, want 3", at(3, 3))
	}
}

func TestDistanceTransformWatershedSplitsTouchingDiscs(t *testing.T) {
	rows, cols := 24, 30
	m := cv.NewMat(rows, cols, 1)
	// Two overlapping discs forming a peanut with a thin waist at x=13.
	drawDisc(m, 8, 12, 6, 255)
	drawDisc(m, 18, 12, 6, 255)
	lm := DistanceTransformWatershed(m, 0.7)
	// Background plus the two blobs.
	if lm.Count != 3 {
		t.Fatalf("Count = %d, want 3 (background + 2 blobs)", lm.Count)
	}
	l1 := lm.At(8, 12)
	l2 := lm.At(18, 12)
	if l1 == 0 || l2 == 0 {
		t.Errorf("disc centres landed on background: %d %d", l1, l2)
	}
	if l1 == l2 {
		t.Errorf("touching discs were not separated (both label %d)", l1)
	}
	if lm.At(0, 0) != 0 {
		t.Errorf("corner background label = %d, want 0", lm.At(0, 0))
	}
}

// stripeLabelMap builds a three-vertical-stripe LabelMap and matching colour
// image, used for the RAG merge tests. Stripe colours are supplied by caller.
func stripeLabelMap(rows, cols int, ca, cb, cc [3]uint8) (*LabelMap, *cv.Mat) {
	lm := newLabelMap(rows, cols)
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var lbl int
			var col [3]uint8
			switch {
			case x < cols/3:
				lbl, col = 0, ca
			case x < 2*cols/3:
				lbl, col = 1, cb
			default:
				lbl, col = 2, cc
			}
			lm.Labels[y*cols+x] = lbl
			img.SetPixel(y, x, col[:])
		}
	}
	lm.Count = 3
	return lm, img
}

func TestRAGMergeByColor(t *testing.T) {
	// Stripes A and B are near in colour; C is far. A colour threshold between the
	// two distances must merge A+B and keep C separate.
	lm, img := stripeLabelMap(30, 30, [3]uint8{200, 10, 10}, [3]uint8{188, 20, 18}, [3]uint8{10, 10, 210})
	rag := BuildRAG(lm, img)
	if rag.Regions() != 3 {
		t.Fatalf("RAG has %d regions, want 3", rag.Regions())
	}
	merged := rag.MergeByColor(40)
	if merged.Count != 2 {
		t.Fatalf("after MergeByColor Count = %d, want 2", merged.Count)
	}
	// A (x=3) and B (x=15) now share a label; C (x=27) differs.
	if merged.At(3, 5) != merged.At(15, 5) {
		t.Errorf("similar stripes A and B were not merged")
	}
	if merged.At(3, 5) == merged.At(27, 5) {
		t.Errorf("distinct stripe C was merged into A/B")
	}
}

func TestRAGMergeBySize(t *testing.T) {
	// A large region with a tiny 2x2 speck of the same colour: MergeBySize folds
	// the speck away.
	rows, cols := 20, 20
	lm := newLabelMap(rows, cols)
	img := cv.NewMat(rows, cols, 3)
	for i := range img.Data {
		img.Data[i] = 100
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			lm.Labels[y*cols+x] = 0
		}
	}
	lm.Labels[5*cols+5] = 1
	lm.Labels[5*cols+6] = 1
	lm.Labels[6*cols+5] = 1
	lm.Labels[6*cols+6] = 1
	lm.Count = 2
	rag := BuildRAG(lm, img)
	merged := rag.MergeBySize(10)
	if merged.Count != 1 {
		t.Fatalf("after MergeBySize Count = %d, want 1", merged.Count)
	}
}

func TestIntelligentScissorsFollowsEdge(t *testing.T) {
	rows, cols := 20, 20
	const edge = 10
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := uint8(40)
			if x >= edge {
				v = 210
			}
			img.SetPixel(y, x, []uint8{v, v, v})
		}
	}
	sc := NewIntelligentScissors(img)
	sc.BuildMap(cv.Point{X: edge, Y: 0})
	path := sc.Trace(cv.Point{X: edge, Y: rows - 1})
	if len(path) == 0 {
		t.Fatal("empty contour")
	}
	if path[0] != (cv.Point{X: edge, Y: 0}) {
		t.Errorf("path starts at %+v, want seed", path[0])
	}
	if path[len(path)-1] != (cv.Point{X: edge, Y: rows - 1}) {
		t.Errorf("path ends at %+v, want target", path[len(path)-1])
	}
	// The live wire snaps to the edge column.
	for _, p := range path {
		if p.X < edge-1 || p.X > edge+1 {
			t.Errorf("path strayed from the edge at %+v", p)
		}
	}
}

func TestKMeansSegmentationThreeColors(t *testing.T) {
	lm, centers := KMeansSegmentation(stripeImage(30, 30), 3, 10)
	if lm.Count != 3 {
		t.Fatalf("Count = %d, want 3 colour clusters", lm.Count)
	}
	if len(centers) != 3 {
		t.Fatalf("got %d centres, want 3", len(centers))
	}
	// The left stripe (x=3) is internally consistent across rows.
	if lm.At(3, 5) != lm.At(3, 20) {
		t.Errorf("left stripe split across clusters")
	}
	// The left stripe (x=3) and the right stripe (x=27) are different colours.
	if lm.At(3, 5) == lm.At(27, 5) {
		t.Errorf("far-apart colours share a cluster")
	}
}

// stripeImage builds a three-colour vertical-stripe image.
func stripeImage(rows, cols int) *cv.Mat {
	_, img := stripeLabelMap(rows, cols, [3]uint8{200, 10, 10}, [3]uint8{10, 200, 10}, [3]uint8{10, 10, 200})
	return img
}

func TestRegionGrowingTwoSeeds(t *testing.T) {
	rows, cols := 20, 20
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := uint8(50)
			if x >= cols/2 {
				v = 200
			}
			img.Set(y, x, 0, v)
		}
	}
	lm := RegionGrowing(img, []cv.Point{{X: 5, Y: 10}, {X: 15, Y: 10}}, 30)
	if lm.Count != 2 {
		t.Fatalf("Count = %d, want 2 grown regions", lm.Count)
	}
	// Two points in the left half (x=2, x=5) share a region.
	if lm.At(2, 10) != lm.At(5, 10) {
		t.Errorf("left region not homogeneous")
	}
	// Left half (x=5) and right half (x=15) are different regions.
	if lm.At(5, 5) == lm.At(15, 5) {
		t.Errorf("the two halves share a region")
	}
}

func TestMeanShiftSegmentationTwoRegions(t *testing.T) {
	rows, cols := 24, 24
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := 60
			if x >= cols/2 {
				base = 190
			}
			if (x+y)%2 == 0 {
				base += 6
			} else {
				base -= 6
			}
			img.SetPixel(y, x, []uint8{uint8(base), uint8(base), uint8(base)})
		}
	}
	lm := MeanShiftSegmentation(img, 4, 30, 8)
	if lm.Count < 2 || lm.Count > 4 {
		t.Fatalf("Count = %d, want ~2 regions", lm.Count)
	}
	// Left half (x=6) and right half (x=18) stay in different regions.
	if lm.At(6, 5) == lm.At(18, 5) {
		t.Errorf("left and right regions were merged")
	}
	// Determinism.
	lm2 := MeanShiftSegmentation(img, 4, 30, 8)
	for i := range lm.Labels {
		if lm.Labels[i] != lm2.Labels[i] {
			t.Fatalf("MeanShiftSegmentation not deterministic at %d", i)
		}
	}
}

func TestSelectiveSearchProposals(t *testing.T) {
	img := quadImage(40, 40)
	boxes := SelectiveSearchSegmentation(img, 0, 300, 1)
	if len(boxes) < 4 {
		t.Fatalf("got %d proposals, want at least the 4 initial regions", len(boxes))
	}
	// The largest proposal (first) should cover a large fraction of the image,
	// reflecting a fully merged region.
	big := boxes[0].Width * boxes[0].Height
	if big < 40*40/2 {
		t.Errorf("largest proposal area = %d, want a near-full-image box", big)
	}
	// Determinism.
	boxes2 := SelectiveSearchSegmentation(img, 0, 300, 1)
	if len(boxes) != len(boxes2) {
		t.Fatalf("proposal count not deterministic: %d vs %d", len(boxes), len(boxes2))
	}
	for i := range boxes {
		if boxes[i] != boxes2[i] {
			t.Fatalf("proposals differ at %d", i)
		}
	}
}

func TestLabelMapBoundaryAndColorize(t *testing.T) {
	img := quadImage(20, 20)
	lm := EfficientGraphSegmentation(img, 0, 300, 1)
	bm := lm.BoundaryMask()
	boundary := 0
	for _, v := range bm.Data {
		if v == 255 {
			boundary++
		}
	}
	if boundary == 0 {
		t.Error("boundary mask has no boundary pixels for a 4-region image")
	}
	col := lm.Colorize()
	if col.Channels != 3 || col.Rows != 20 || col.Cols != 20 {
		t.Errorf("colorize produced wrong shape %dx%dx%d", col.Rows, col.Cols, col.Channels)
	}
	rects := lm.BoundingRects()
	if len(rects) != lm.Count {
		t.Errorf("BoundingRects returned %d rects for %d regions", len(rects), lm.Count)
	}
}

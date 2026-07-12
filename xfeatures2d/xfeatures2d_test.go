package xfeatures2d

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- Test image builders -------------------------------------------------

// whiteBackground returns a rows×cols single-channel image filled with 255.
func whiteBackground(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(255)
	return m
}

// checkerboard returns an (n·sq)×(n·sq) checkerboard of sq-pixel squares.
func checkerboard(n, sq int) *cv.Mat {
	size := n * sq
	m := cv.NewMat(size, size, 1)
	for by := 0; by < n; by++ {
		for bx := 0; bx < n; bx++ {
			var val uint8
			if (bx+by)%2 == 0 {
				val = 255
			}
			for y := 0; y < sq; y++ {
				for x := 0; x < sq; x++ {
					m.Data[(by*sq+y)*size+(bx*sq+x)] = val
				}
			}
		}
	}
	return m
}

// squareGrid returns a black image with a grid of separated white squares, whose
// convex corners are detectable by the FAST/AGAST segment test.
func squareGrid(n, sq, gap int) *cv.Mat {
	step := sq + gap
	size := n*step + gap
	m := cv.NewMat(size, size, 1)
	for by := 0; by < n; by++ {
		for bx := 0; bx < n; bx++ {
			ox := gap + bx*step
			oy := gap + by*step
			for y := 0; y < sq; y++ {
				for x := 0; x < sq; x++ {
					m.Data[(oy+y)*size+(ox+x)] = 255
				}
			}
		}
	}
	return m
}

// --- SimpleBlobDetector --------------------------------------------------

func TestSimpleBlobDetectorFindsCircles(t *testing.T) {
	img := whiteBackground(100, 100)
	truth := []cv.Point{{X: 25, Y: 25}, {X: 75, Y: 25}, {X: 25, Y: 75}, {X: 75, Y: 75}}
	const radius = 12
	for _, c := range truth {
		cv.Circle(img, c, radius, cv.NewScalar(0), cv.Filled)
	}
	// A too-small blob (filtered by area) and a thin non-circular bar (filtered
	// by inertia) that must not be reported.
	cv.Circle(img, cv.Point{X: 50, Y: 50}, 2, cv.NewScalar(0), cv.Filled)
	cv.Rectangle(img, cv.Point{X: 49, Y: 3}, cv.Point{X: 51, Y: 40}, cv.NewScalar(0), cv.Filled)

	d := NewSimpleBlobDetector()
	kps := d.Detect(img)

	if len(kps) != len(truth) {
		t.Fatalf("expected %d blobs, got %d: %+v", len(truth), len(kps), kps)
	}

	// Every detected center must be near a true circle centre and every true
	// centre must be matched.
	for _, want := range truth {
		matched := false
		for _, kp := range kps {
			if math.Hypot(float64(kp.Pt.X-want.X), float64(kp.Pt.Y-want.Y)) <= 2 {
				matched = true
				break
			}
		}
		if !matched {
			t.Errorf("no blob detected near true centre %v", want)
		}
	}

	// The blob diameter should be close to the drawn circle diameter (24).
	for _, kp := range kps {
		if kp.Size < 18 || kp.Size > 30 {
			t.Errorf("blob size %.1f far from expected diameter ~24", kp.Size)
		}
	}
}

func TestSimpleBlobDetectorFiltersEverythingWhenAreaTooLarge(t *testing.T) {
	img := whiteBackground(100, 100)
	cv.Circle(img, cv.Point{X: 50, Y: 50}, 12, cv.NewScalar(0), cv.Filled)

	d := NewSimpleBlobDetector()
	d.MinArea = 100000 // no blob can be this large
	if kps := d.Detect(img); len(kps) != 0 {
		t.Fatalf("expected 0 blobs when MinArea excludes all, got %d", len(kps))
	}
}

func TestSimpleBlobDetectorDeterministic(t *testing.T) {
	img := whiteBackground(80, 80)
	cv.Circle(img, cv.Point{X: 40, Y: 40}, 10, cv.NewScalar(0), cv.Filled)
	d := NewSimpleBlobDetector()
	a := d.Detect(img)
	b := d.Detect(img)
	if len(a) != len(b) {
		t.Fatalf("non-deterministic count: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic keypoint %d: %+v vs %+v", i, a[i], b[i])
		}
	}
}

// --- AGAST ---------------------------------------------------------------

func TestAGASTDetectsCorners(t *testing.T) {
	img := squareGrid(3, 20, 12)
	kps := NewAGAST(20).Detect(img)
	if len(kps) == 0 {
		t.Fatal("AGAST found no corners on a grid of squares")
	}
	// Every square contributes convex corners; expect at least one per square.
	if len(kps) < 9 {
		t.Errorf("expected several corners (>=9) on a 3×3 square grid, got %d", len(kps))
	}
	for _, kp := range kps {
		if kp.Response <= 0 {
			t.Errorf("AGAST keypoint has non-positive response %.1f", kp.Response)
		}
	}
}

func TestAGASTThresholdMonotone(t *testing.T) {
	img := squareGrid(3, 20, 12)
	low := NewAGAST(15).Detect(img)
	high := NewAGAST(80).Detect(img)
	if len(high) > len(low) {
		t.Errorf("a higher threshold should not yield more corners: low=%d high=%d", len(low), len(high))
	}
}

// --- GFTTDetector --------------------------------------------------------

func TestGFTTDetectsCheckerboardCorners(t *testing.T) {
	img := checkerboard(4, 20)
	g := NewGFTTDetector(50)
	kps := g.Detect(img)
	if len(kps) == 0 {
		t.Fatal("GFTT found no corners on a checkerboard")
	}
	// Interior grid junctions sit at multiples of 20. At least one detected
	// corner should be close to such a junction.
	near := false
	for _, kp := range kps {
		for _, j := range []int{20, 40, 60} {
			if abs(kp.Pt.X-j) <= 2 {
				for _, k := range []int{20, 40, 60} {
					if abs(kp.Pt.Y-k) <= 2 {
						near = true
					}
				}
			}
		}
	}
	if !near {
		t.Error("no GFTT corner near an interior checkerboard junction")
	}
}

func TestGFTTHarrisAnnotation(t *testing.T) {
	img := checkerboard(4, 20)
	g := NewGFTTDetector(20)
	g.AnnotateHarris = true
	kps := g.Detect(img)
	if len(kps) == 0 {
		t.Fatal("no corners")
	}
	anyNonZero := false
	for _, kp := range kps {
		if kp.Response != 0 {
			anyNonZero = true
		}
	}
	if !anyNonZero {
		t.Error("AnnotateHarris left every response zero")
	}
}

// --- StarDetector --------------------------------------------------------

func TestStarDetectorFindsBlob(t *testing.T) {
	img := cv.NewMat(60, 60, 1) // black background
	cv.Circle(img, cv.Point{X: 30, Y: 30}, 8, cv.NewScalar(255), cv.Filled)
	kps := NewStarDetector().Detect(img)
	if len(kps) == 0 {
		t.Fatal("StarDetector found no keypoints on a bright blob")
	}
	best := kps[0]
	for _, kp := range kps {
		if kp.Response > best.Response {
			best = kp
		}
	}
	if math.Hypot(float64(best.Pt.X-30), float64(best.Pt.Y-30)) > 6 {
		t.Errorf("strongest Star keypoint %v far from blob centre (30,30)", best.Pt)
	}
}

// --- HarrisLaplace -------------------------------------------------------

func TestHarrisLaplaceDetects(t *testing.T) {
	img := squareGrid(3, 20, 12)
	kps := NewHarrisLaplace().Detect(img)
	if len(kps) == 0 {
		t.Fatal("HarrisLaplace found no keypoints on a corner-rich image")
	}
	for _, kp := range kps {
		if kp.Size <= 0 {
			t.Errorf("HarrisLaplace keypoint has non-positive size %.2f", kp.Size)
		}
	}
}

// --- BRISK ---------------------------------------------------------------

// texturedImage builds a deterministic ramp image and stamps an identical
// textured patch at each of the given centres. The patch is patchSize×patchSize.
func texturedImage(rows, cols, patchSize int, centres []cv.Point) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	// A gentle non-constant background so orientation is well defined everywhere.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = uint8((x*3 + y) % 200)
		}
	}
	half := patchSize / 2
	for _, c := range centres {
		for i := 0; i < patchSize; i++ {
			for j := 0; j < patchSize; j++ {
				val := uint8((i*9 + j*5 + (i*j%17)*3) % 256)
				x := c.X - half + j
				y := c.Y - half + i
				if x >= 0 && x < cols && y >= 0 && y < rows {
					m.Data[y*cols+x] = val
				}
			}
		}
	}
	return m
}

func TestBRISKFixedLengthDescriptor(t *testing.T) {
	b := NewBRISK(20)
	img := texturedImage(120, 120, 40, []cv.Point{{X: 30, Y: 30}, {X: 90, Y: 30}})
	kps := []KeyPoint{
		{Pt: cv.Point{X: 30, Y: 30}, Size: 0},
		{Pt: cv.Point{X: 90, Y: 30}, Size: 0},
	}
	kept, descs := b.Compute(img, kps)
	if len(kept) != len(descs) {
		t.Fatalf("keypoint/descriptor length mismatch: %d vs %d", len(kept), len(descs))
	}
	if len(descs) != 2 {
		t.Fatalf("expected 2 descriptors, got %d", len(descs))
	}
	want := b.DescriptorSizeBytes()
	if want <= 0 {
		t.Fatalf("descriptor byte size must be positive, got %d", want)
	}
	for i, d := range descs {
		if len(d) != want {
			t.Errorf("descriptor %d has length %d, want fixed %d", i, len(d), want)
		}
	}
}

func TestBRISKIdenticalPatchesHammingZero(t *testing.T) {
	b := NewBRISK(20)
	img := texturedImage(120, 120, 40, []cv.Point{{X: 30, Y: 30}, {X: 90, Y: 30}})
	// The two keypoints sit at the centres of two identical patches, so their
	// descriptors must be bit-for-bit equal (Hamming distance 0).
	kps := []KeyPoint{
		{Pt: cv.Point{X: 30, Y: 30}, Size: 0},
		{Pt: cv.Point{X: 90, Y: 30}, Size: 0},
	}
	_, descs := b.Compute(img, kps)
	if len(descs) != 2 {
		t.Fatalf("expected 2 descriptors, got %d", len(descs))
	}
	if d := HammingDistance(descs[0], descs[1]); d != 0 {
		t.Errorf("identical patches gave Hamming distance %d, want 0", d)
	}
}

func TestBRISKDifferentPatchesDiffer(t *testing.T) {
	b := NewBRISK(20)
	img := texturedImage(120, 120, 40, []cv.Point{{X: 30, Y: 30}})
	// One keypoint on the textured patch, one on the plain background ramp.
	kps := []KeyPoint{
		{Pt: cv.Point{X: 30, Y: 30}, Size: 0},
		{Pt: cv.Point{X: 60, Y: 90}, Size: 0},
	}
	_, descs := b.Compute(img, kps)
	if len(descs) != 2 {
		t.Fatalf("expected 2 descriptors, got %d", len(descs))
	}
	if d := HammingDistance(descs[0], descs[1]); d == 0 {
		t.Error("different neighbourhoods produced identical descriptors (Hamming 0)")
	}
}

func TestBRISKDeterministic(t *testing.T) {
	b := NewBRISK(20)
	img := texturedImage(120, 120, 40, []cv.Point{{X: 60, Y: 60}})
	kps := []KeyPoint{{Pt: cv.Point{X: 60, Y: 60}, Size: 0}}
	_, d1 := b.Compute(img, kps)
	_, d2 := b.Compute(img, kps)
	if len(d1) != 1 || len(d2) != 1 {
		t.Fatalf("expected 1 descriptor each, got %d and %d", len(d1), len(d2))
	}
	if HammingDistance(d1[0], d2[0]) != 0 {
		t.Error("BRISK is not deterministic")
	}
}

func TestBRISKDropsBorderKeypoints(t *testing.T) {
	b := NewBRISK(20)
	img := texturedImage(120, 120, 40, []cv.Point{{X: 60, Y: 60}})
	kps := []KeyPoint{
		{Pt: cv.Point{X: 1, Y: 1}, Size: 0},   // too close to the border
		{Pt: cv.Point{X: 60, Y: 60}, Size: 0}, // fine
	}
	kept, descs := b.Compute(img, kps)
	if len(kept) != 1 || len(descs) != 1 {
		t.Fatalf("expected border keypoint dropped, got %d kept", len(kept))
	}
	if kept[0].Pt.X != 60 || kept[0].Pt.Y != 60 {
		t.Errorf("wrong keypoint retained: %v", kept[0].Pt)
	}
}

func TestBRISKDetectAndCompute(t *testing.T) {
	b := NewBRISK(20)
	img := squareGrid(3, 20, 12)
	kps, descs := b.DetectAndCompute(img)
	if len(kps) != len(descs) {
		t.Fatalf("keypoint/descriptor mismatch: %d vs %d", len(kps), len(descs))
	}
	if len(kps) == 0 {
		t.Fatal("DetectAndCompute produced no features")
	}
	for _, d := range descs {
		if len(d) != b.DescriptorSizeBytes() {
			t.Fatalf("descriptor length %d != %d", len(d), b.DescriptorSizeBytes())
		}
	}
}

// --- Helpers -------------------------------------------------------------

func TestHammingDistance(t *testing.T) {
	if got := HammingDistance([]byte{0x00, 0xFF}, []byte{0x0F, 0xF0}); got != 8 {
		t.Errorf("HammingDistance = %d, want 8", got)
	}
	if got := HammingDistance([]byte{0xAB}, []byte{0xAB}); got != 0 {
		t.Errorf("HammingDistance of equal bytes = %d, want 0", got)
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

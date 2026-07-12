package xfeatures2d

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// tiledImage returns a rows×cols single-channel image whose intensity is exactly
// periodic with the given period in both axes, using a deterministic
// non-symmetric pattern. Because it is perfectly periodic, two points a whole
// period apart have identical neighbourhoods for any radius that stays inside
// the image, which makes it ideal for "identical patch -> distance 0" and
// "shifted copy matches" descriptor tests.
func tiledImage(rows, cols, period int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			u := x % period
			v := y % period
			m.Data[y*cols+x] = uint8((u*7 + v*13 + u*v*3) % 256)
		}
	}
	return m
}

// filledSquare returns a black image with a single bright filled square.
func filledSquare(size, x0, y0, side int) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	for y := y0; y < y0+side; y++ {
		for x := x0; x < x0+side; x++ {
			m.Data[y*size+x] = 255
		}
	}
	return m
}

// The three canonical points of a period-40 tiled image: A and B are one period
// apart (identical neighbourhoods); D is half a period away in y (distinct).
const (
	tilePeriod = 40
	tileAx     = 70
	tileAy     = 110
	tileBx     = 110 // A + period
	tileBy     = 110
	tileDx     = 70
	tileDy     = 130 // A + period/2 in y
)

func tileKeypoints(size float64) (a, b, d KeyPoint) {
	return KeyPoint{Pt: cv.Point{X: tileAx, Y: tileAy}, Size: size, Angle: -1},
		KeyPoint{Pt: cv.Point{X: tileBx, Y: tileBy}, Size: size, Angle: -1},
		KeyPoint{Pt: cv.Point{X: tileDx, Y: tileDy}, Size: size, Angle: -1}
}

// --- BRIEF ---------------------------------------------------------------

func TestBRIEF(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	b := NewBRIEF(32)
	a, bb, d := tileKeypoints(0)
	_, descs := b.Compute(img, []KeyPoint{a, bb, d})
	if len(descs[0]) != 32 {
		t.Fatalf("descriptor length %d, want 32", len(descs[0]))
	}
	if h := HammingDistance(descs[0], descs[1]); h != 0 {
		t.Errorf("identical neighbourhoods gave Hamming %d, want 0", h)
	}
	if h := HammingDistance(descs[0], descs[2]); h < 32 {
		t.Errorf("distinct neighbourhoods gave Hamming %d, want large", h)
	}
}

// --- FREAK ---------------------------------------------------------------

func TestFREAK(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	f := NewFREAK(20)
	if f.DescriptorSizeBits() <= 0 {
		t.Fatal("FREAK has no descriptor bits")
	}
	a, bb, d := tileKeypoints(0)
	_, descs := f.Compute(img, []KeyPoint{a, bb, d})
	if h := HammingDistance(descs[0], descs[1]); h != 0 {
		t.Errorf("identical neighbourhoods gave Hamming %d, want 0", h)
	}
	if h := HammingDistance(descs[0], descs[2]); h < f.DescriptorSizeBits()/8 {
		t.Errorf("distinct neighbourhoods gave Hamming %d, want large", h)
	}
}

func TestFREAKDetectAndCompute(t *testing.T) {
	img := squareGrid(3, 20, 12)
	f := NewFREAK(20)
	kps, descs := f.DetectAndCompute(img)
	if len(kps) != len(descs) || len(kps) == 0 {
		t.Fatalf("DetectAndCompute: %d kps, %d descs", len(kps), len(descs))
	}
	for _, dsc := range descs {
		if len(dsc) != f.DescriptorSizeBytes() {
			t.Fatalf("descriptor length %d, want %d", len(dsc), f.DescriptorSizeBytes())
		}
	}
}

// --- LATCH ---------------------------------------------------------------

func TestLATCH(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	l := NewLATCH(32)
	a, bb, d := tileKeypoints(0)
	_, descs := l.Compute(img, []KeyPoint{a, bb, d})
	if h := HammingDistance(descs[0], descs[1]); h != 0 {
		t.Errorf("identical neighbourhoods gave Hamming %d, want 0", h)
	}
	if h := HammingDistance(descs[0], descs[2]); h < 16 {
		t.Errorf("distinct neighbourhoods gave Hamming %d, want large", h)
	}
}

// --- LUCID ---------------------------------------------------------------

func TestLUCID(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	l := NewLUCID(5, 2)
	a, bb, d := tileKeypoints(0)
	_, descs := l.Compute(img, []KeyPoint{a, bb, d})
	if len(descs[0]) != l.DescriptorSize() {
		t.Fatalf("descriptor length %d, want %d", len(descs[0]), l.DescriptorSize())
	}
	if dd := LUCIDDistance(descs[0], descs[1]); dd != 0 {
		t.Errorf("identical neighbourhoods gave LUCID distance %d, want 0", dd)
	}
	if dd := LUCIDDistance(descs[0], descs[2]); dd == 0 {
		t.Error("distinct neighbourhoods gave LUCID distance 0")
	}
}

// --- BEBLID / TEBLID / BoostDesc -----------------------------------------

func TestBEBLID(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	b := NewBEBLID(32)
	a, bb, d := tileKeypoints(0)
	_, descs := b.Compute(img, []KeyPoint{a, bb, d})
	if h := HammingDistance(descs[0], descs[1]); h != 0 {
		t.Errorf("BEBLID identical gave Hamming %d, want 0", h)
	}
	if h := HammingDistance(descs[0], descs[2]); h < 20 {
		t.Errorf("BEBLID distinct gave Hamming %d, want large", h)
	}
}

func TestTEBLID(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	b := NewTEBLID(32)
	a, bb, d := tileKeypoints(0)
	_, descs := b.Compute(img, []KeyPoint{a, bb, d})
	if h := HammingDistance(descs[0], descs[1]); h != 0 {
		t.Errorf("TEBLID identical gave Hamming %d, want 0", h)
	}
	if h := HammingDistance(descs[0], descs[2]); h < 20 {
		t.Errorf("TEBLID distinct gave Hamming %d, want large", h)
	}
}

func TestBoostDesc(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	b := NewBoostDesc(32)
	a, bb, d := tileKeypoints(0)
	_, descs := b.Compute(img, []KeyPoint{a, bb, d})
	if h := HammingDistance(descs[0], descs[1]); h != 0 {
		t.Errorf("BoostDesc identical gave Hamming %d, want 0", h)
	}
	if h := HammingDistance(descs[0], descs[2]); h < 20 {
		t.Errorf("BoostDesc distinct gave Hamming %d, want large", h)
	}
}

// --- DAISY / VGG ---------------------------------------------------------

func TestDAISY(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	d := NewDAISY()
	a, bb, dd := tileKeypoints(0)
	_, descs := d.Compute(img, []KeyPoint{a, bb, dd})
	if len(descs[0]) != d.DescriptorSize() {
		t.Fatalf("descriptor length %d, want %d", len(descs[0]), d.DescriptorSize())
	}
	if l2 := L2Distance(descs[0], descs[1]); l2 > 1e-9 {
		t.Errorf("DAISY identical gave L2 %g, want 0", l2)
	}
	near := L2Distance(descs[0], descs[1])
	far := L2Distance(descs[0], descs[2])
	if far <= near {
		t.Errorf("DAISY distinct L2 %g not greater than identical L2 %g", far, near)
	}
}

func TestVGG(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	v := NewVGG()
	a, bb, dd := tileKeypoints(0)
	_, descs := v.Compute(img, []KeyPoint{a, bb, dd})
	if len(descs[0]) != v.DescriptorSize() {
		t.Fatalf("descriptor length %d, want %d", len(descs[0]), v.DescriptorSize())
	}
	if l2 := L2Distance(descs[0], descs[1]); l2 > 1e-9 {
		t.Errorf("VGG identical gave L2 %g, want 0", l2)
	}
	if far := L2Distance(descs[0], descs[2]); far <= 1e-6 {
		t.Errorf("VGG distinct gave L2 %g, want > 0", far)
	}
}

// --- SURF ----------------------------------------------------------------

func TestSURFDescriptor(t *testing.T) {
	img := tiledImage(220, 220, tilePeriod)
	s := NewSURF(100)
	a, bb, dd := tileKeypoints(0)
	_, descs := s.Compute(img, []KeyPoint{a, bb, dd})
	if len(descs[0]) != 64 {
		t.Fatalf("descriptor length %d, want 64", len(descs[0]))
	}
	// A and B are a shifted copy of each other (one period apart): their
	// descriptors must coincide and be closer than to the distinct point.
	near := L2Distance(descs[0], descs[1])
	far := L2Distance(descs[0], descs[2])
	if near > 1e-9 {
		t.Errorf("SURF shifted-copy L2 %g, want 0", near)
	}
	if far <= near {
		t.Errorf("SURF distinct L2 %g not greater than shifted-copy L2 %g", far, near)
	}
}

func TestSURFDetect(t *testing.T) {
	img := cv.NewMat(140, 140, 1)
	cv.Circle(img, cv.Point{X: 70, Y: 70}, 9, cv.NewScalar(255), cv.Filled)
	s := NewSURF(50)
	kps := s.Detect(img)
	if len(kps) == 0 {
		t.Fatal("SURF found no keypoints on a bright blob")
	}
	best := kps[0]
	for _, kp := range kps {
		if kp.Response > best.Response {
			best = kp
		}
	}
	if math.Hypot(float64(best.Pt.X-70), float64(best.Pt.Y-70)) > 8 {
		t.Errorf("strongest SURF keypoint %v far from blob centre (60,60)", best.Pt)
	}
}

func TestSURFDetectAndCompute(t *testing.T) {
	img := squareGrid(3, 20, 12)
	s := NewSURF(50)
	kps, descs := s.DetectAndCompute(img)
	if len(kps) != len(descs) {
		t.Fatalf("kp/desc mismatch %d vs %d", len(kps), len(descs))
	}
}

// --- MSDDetector ---------------------------------------------------------

func TestMSDDetector(t *testing.T) {
	img := squareGrid(3, 20, 12)
	m := NewMSDDetector()
	m.Threshold = 100
	kps := m.Detect(img)
	if len(kps) == 0 {
		t.Fatal("MSD found no keypoints on a corner-rich image")
	}
	for _, kp := range kps {
		if kp.Response < m.Threshold {
			t.Errorf("MSD keypoint response %g below threshold %g", kp.Response, m.Threshold)
		}
	}
	// Determinism.
	kps2 := m.Detect(img)
	if len(kps) != len(kps2) {
		t.Fatalf("MSD not deterministic: %d vs %d", len(kps), len(kps2))
	}
}

// --- TBMR ----------------------------------------------------------------

func TestTBMR(t *testing.T) {
	img := filledSquare(120, 40, 40, 40)
	d := NewTBMR()
	kps := d.Detect(img)
	if len(kps) == 0 {
		t.Fatal("TBMR found no regions on a square blob")
	}
	found := false
	for _, kp := range kps {
		if kp.Size <= 0 {
			t.Errorf("TBMR keypoint has non-positive size %g", kp.Size)
		}
		if math.Hypot(float64(kp.Pt.X-60), float64(kp.Pt.Y-60)) < 12 {
			found = true
		}
	}
	if !found {
		t.Error("no TBMR region near the square centre (60,60)")
	}
}

// --- PCTSignatures / SQFD ------------------------------------------------

func TestPCTSignatures(t *testing.T) {
	img := checkerboard(6, 20)
	p := NewPCTSignatures()
	sig := p.ComputeSignature(img)
	if len(sig) == 0 {
		t.Fatal("PCTSignatures produced an empty signature")
	}
	var w float64
	for _, s := range sig {
		w += s.Weight
	}
	if math.Abs(w-1) > 1e-6 {
		t.Errorf("signature weights sum to %g, want 1", w)
	}
	// SQFD of a signature with itself is 0.
	if d := SQFD(sig, sig, 1.0); d > 1e-9 {
		t.Errorf("SQFD of identical signatures = %g, want 0", d)
	}
	// A visibly different image gives a positive distance.
	other := p.ComputeSignature(filledSquare(120, 30, 30, 60))
	if d := SQFD(sig, other, 1.0); d <= 1e-6 {
		t.Errorf("SQFD of different signatures = %g, want > 0", d)
	}
}

// --- Matchers ------------------------------------------------------------

func TestMatchBruteForceHamming(t *testing.T) {
	q := [][]byte{{0x00}, {0xFF}}
	tr := [][]byte{{0xF0}, {0x01}}
	m := MatchBruteForceHamming(q, tr)
	if len(m) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(m))
	}
	if m[0].TrainIdx != 1 { // 0x00 nearest to 0x01 (distance 1)
		t.Errorf("query 0 matched train %d, want 1", m[0].TrainIdx)
	}
	if m[1].TrainIdx != 0 { // 0xFF nearest to 0xF0 (distance 4)
		t.Errorf("query 1 matched train %d, want 0", m[1].TrainIdx)
	}
}

// buildGMSScene creates two keypoint sets and matches: a dense cluster of
// consistent (inlier) matches translated by a fixed offset, plus scattered
// random (outlier) matches whose targets are geometrically inconsistent.
func buildGMSScene() (kp1, kp2 []KeyPoint, matches []DMatch) {
	const off = 20
	// Inliers: a 10x10 grid block of matches translated by (off, off).
	for gy := 0; gy < 10; gy++ {
		for gx := 0; gx < 10; gx++ {
			x := 40 + gx*4
			y := 40 + gy*4
			i := len(kp1)
			kp1 = append(kp1, KeyPoint{Pt: cv.Point{X: x, Y: y}, Size: 7, Angle: 0})
			kp2 = append(kp2, KeyPoint{Pt: cv.Point{X: x + off, Y: y + off}, Size: 7, Angle: 0})
			matches = append(matches, DMatch{QueryIdx: i, TrainIdx: i, Distance: 1})
		}
	}
	inliers := len(matches)
	// Outliers: query points spread out, targets placed pseudo-randomly.
	for k := 0; k < 40; k++ {
		x := 5 + (k*37)%180
		y := 5 + (k*53)%180
		tx := 5 + (k*91)%180
		ty := 5 + (k*29)%180
		i := len(kp1)
		kp1 = append(kp1, KeyPoint{Pt: cv.Point{X: x, Y: y}, Size: 7, Angle: 0})
		kp2 = append(kp2, KeyPoint{Pt: cv.Point{X: tx, Y: ty}, Size: 7, Angle: 0})
		matches = append(matches, DMatch{QueryIdx: i, TrainIdx: i, Distance: 1})
	}
	_ = inliers
	return kp1, kp2, matches
}

func TestMatchGMS(t *testing.T) {
	kp1, kp2, matches := buildGMSScene()
	kept := MatchGMS(200, 200, 200, 200, kp1, kp2, matches, false, false, 6)
	if len(kept) == 0 {
		t.Fatal("GMS rejected every match")
	}
	if len(kept) >= len(matches) {
		t.Fatalf("GMS kept %d of %d matches, expected outliers to be filtered", len(kept), len(matches))
	}
	// The retained matches should be dominated by the consistent cluster: check
	// that most kept matches obey the (off, off) translation.
	consistent := 0
	for _, m := range kept {
		q := kp1[m.QueryIdx].Pt
		tt := kp2[m.TrainIdx].Pt
		if tt.X-q.X == 20 && tt.Y-q.Y == 20 {
			consistent++
		}
	}
	if float64(consistent) < 0.8*float64(len(kept)) {
		t.Errorf("only %d/%d kept matches are geometrically consistent", consistent, len(kept))
	}
}

func TestMatchLOGOS(t *testing.T) {
	kp1, kp2, matches := buildGMSScene()
	kept := NewLOGOSMatcher().Filter(kp1, kp2, matches)
	if len(kept) == 0 {
		t.Fatal("LOGOS rejected every match")
	}
	if len(kept) >= len(matches) {
		t.Fatalf("LOGOS kept %d of %d, expected outliers filtered", len(kept), len(matches))
	}
	consistent := 0
	for _, m := range kept {
		q := kp1[m.QueryIdx].Pt
		tt := kp2[m.TrainIdx].Pt
		if tt.X-q.X == 20 && tt.Y-q.Y == 20 {
			consistent++
		}
	}
	if float64(consistent) < 0.8*float64(len(kept)) {
		t.Errorf("only %d/%d LOGOS matches are consistent", consistent, len(kept))
	}
}

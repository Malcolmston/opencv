package linedescriptor

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// orientationDiff returns the difference between two line orientations modulo π
// (so that opposite directions count as the same orientation), in [0, π/2].
func orientationDiff(a, b float64) float64 {
	d := angleDiff(a, b)
	if d > math.Pi/2 {
		d = math.Pi - d
	}
	return d
}

func dist(a, b cv.Point) float64 {
	return math.Hypot(float64(a.X-b.X), float64(a.Y-b.Y))
}

// endpointsMatch reports whether the segment kl matches the expected endpoints
// (either orientation) with each endpoint within tol pixels.
func endpointsMatch(kl KeyLine, ea, eb cv.Point, tol float64) bool {
	direct := dist(kl.StartPoint, ea) <= tol && dist(kl.EndPoint, eb) <= tol
	swapped := dist(kl.StartPoint, eb) <= tol && dist(kl.EndPoint, ea) <= tol
	return direct || swapped
}

// TestLSDDetectorRecoversSegments draws a single straight segment of known
// endpoints and angle in its own image and checks the detector recovers a
// segment with the right orientation and endpoints. Each case uses an isolated
// image so the two parallel gradient edges of the drawn bar are not disturbed
// by crossings.
func TestLSDDetectorRecoversSegments(t *testing.T) {
	cases := []struct {
		name     string
		a, b     cv.Point
		expAngle float64
	}{
		{"horizontal", cv.Point{X: 15, Y: 45}, cv.Point{X: 75, Y: 45}, 0},
		{"vertical", cv.Point{X: 45, Y: 15}, cv.Point{X: 45, Y: 75}, math.Pi / 2},
		{"diagonal", cv.Point{X: 15, Y: 15}, cv.Point{X: 75, Y: 75}, math.Pi / 4},
		{"antidiagonal", cv.Point{X: 15, Y: 75}, cv.Point{X: 75, Y: 15}, -math.Pi / 4},
	}
	det := NewLSDDetector()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			img := cv.NewMat(90, 90, 1)
			cv.Line(img, c.a, c.b, cv.NewScalar(255), 3)
			lines := det.Detect(img)
			if len(lines) == 0 {
				t.Fatalf("%s: no segments detected", c.name)
			}
			found := false
			for _, kl := range lines {
				if orientationDiff(kl.Angle, c.expAngle) < 0.12 && endpointsMatch(kl, c.a, c.b, 5) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("%s: no detected segment matched angle %.3f and endpoints %v-%v; got %d segments, first=%v",
					c.name, c.expAngle, c.a, c.b, len(lines), lines[0])
			}
		})
	}
}

// TestLSDDetectorSortedByResponse verifies the detector returns segments in
// descending Response order.
func TestLSDDetectorSortedByResponse(t *testing.T) {
	img := cv.NewMat(100, 100, 1)
	cv.Line(img, cv.Point{X: 10, Y: 20}, cv.Point{X: 90, Y: 20}, cv.NewScalar(255), 3) // long
	cv.Line(img, cv.Point{X: 10, Y: 70}, cv.Point{X: 40, Y: 70}, cv.NewScalar(255), 3) // short
	lines := NewLSDDetector().Detect(img)
	if len(lines) < 2 {
		t.Fatalf("expected at least 2 segments, got %d", len(lines))
	}
	for i := 1; i < len(lines); i++ {
		if lines[i-1].Response < lines[i].Response {
			t.Fatalf("segments not sorted by descending response at %d: %.1f < %.1f",
				i, lines[i-1].Response, lines[i].Response)
		}
	}
}

// TestLSDDetectorDeterministic checks two runs on the same input are identical.
func TestLSDDetectorDeterministic(t *testing.T) {
	img := cv.NewMat(80, 80, 1)
	cv.Line(img, cv.Point{X: 10, Y: 30}, cv.Point{X: 70, Y: 30}, cv.NewScalar(255), 3)
	cv.Line(img, cv.Point{X: 10, Y: 10}, cv.Point{X: 70, Y: 70}, cv.NewScalar(255), 3)
	det := NewLSDDetector()
	a := det.Detect(img)
	b := det.Detect(img)
	if len(a) != len(b) {
		t.Fatalf("nondeterministic count: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("nondeterministic segment %d: %v vs %v", i, a[i], b[i])
		}
	}
}

// TestDescriptorFixedLength checks every code has the advertised byte length.
func TestDescriptorFixedLength(t *testing.T) {
	bd := NewBinaryDescriptor()
	img := cv.NewMat(120, 120, 1)
	cv.Line(img, cv.Point{X: 30, Y: 60}, cv.Point{X: 90, Y: 60}, cv.NewScalar(255), 3)
	cv.Line(img, cv.Point{X: 40, Y: 30}, cv.Point{X: 40, Y: 100}, cv.NewScalar(255), 3)
	lines := []KeyLine{newKeyLine(30, 60, 90, 60), newKeyLine(40, 30, 40, 100)}
	got, codes := bd.Compute(img, lines)
	if len(got) != len(lines) {
		t.Fatalf("Compute returned %d keylines, want %d", len(got), len(lines))
	}
	if len(codes) != len(lines) {
		t.Fatalf("Compute returned %d codes, want %d", len(codes), len(lines))
	}
	want := bd.DescriptorSize()
	if want != 4 {
		t.Fatalf("default DescriptorSize = %d, want 4", want)
	}
	for i, c := range codes {
		if len(c) != want {
			t.Fatalf("code %d has length %d, want %d", i, len(c), want)
		}
	}
}

// TestIdenticalPatchesHammingZero checks that describing the same line twice,
// and two identical lines, gives Hamming distance 0.
func TestIdenticalPatchesHammingZero(t *testing.T) {
	bd := NewBinaryDescriptor()
	img := cv.NewMat(120, 120, 1)
	cv.Line(img, cv.Point{X: 30, Y: 60}, cv.Point{X: 90, Y: 60}, cv.NewScalar(255), 3)
	kl := newKeyLine(30, 60, 90, 60)
	_, c1 := bd.Compute(img, []KeyLine{kl})
	_, c2 := bd.Compute(img, []KeyLine{kl, kl})
	if HammingDistance(c1[0], c2[0]) != 0 {
		t.Fatalf("same line described twice differ by %d bits", HammingDistance(c1[0], c2[0]))
	}
	if HammingDistance(c2[0], c2[1]) != 0 {
		t.Fatalf("two identical lines differ by %d bits", HammingDistance(c2[0], c2[1]))
	}
}

// TestShiftedCopyHammingZero checks that a segment and its rigidly shifted copy
// (with the underlying image content shifted by the same integer offset) yield
// identical codes, demonstrating the descriptor's translation invariance.
func TestShiftedCopyHammingZero(t *testing.T) {
	bd := NewBinaryDescriptor()
	const dx, dy = 6, 8

	a := cv.NewMat(140, 140, 1)
	cv.Line(a, cv.Point{X: 40, Y: 70}, cv.Point{X: 100, Y: 70}, cv.NewScalar(255), 3)
	b := cv.NewMat(140, 140, 1)
	cv.Line(b, cv.Point{X: 40 + dx, Y: 70 + dy}, cv.Point{X: 100 + dx, Y: 70 + dy}, cv.NewScalar(255), 3)

	_, ca := bd.Compute(a, []KeyLine{newKeyLine(40, 70, 100, 70)})
	_, cb := bd.Compute(b, []KeyLine{newKeyLine(40+dx, 70+dy, 100+dx, 70+dy)})
	if HammingDistance(ca[0], cb[0]) != 0 {
		t.Fatalf("shifted copy differs by %d bits (a=%08b b=%08b)",
			HammingDistance(ca[0], cb[0]), ca[0], cb[0])
	}
}

// makeThin builds an image with a thin bright line at the given endpoints.
func makeThin(a, b cv.Point) *cv.Mat {
	img := cv.NewMat(140, 140, 1)
	cv.Line(img, a, b, cv.NewScalar(255), 3)
	return img
}

// makeStep builds an image whose bottom half (below the segment) is bright,
// giving a single step edge along the segment.
func makeStep(edgeY int) *cv.Mat {
	img := cv.NewMat(140, 140, 1)
	for y := edgeY; y < 140; y++ {
		for x := 0; x < 140; x++ {
			img.Data[y*140+x] = 255
		}
	}
	return img
}

// TestMatcherPairsShiftedCopies builds two structurally different segments (a
// thin ridge and a step edge), shifts both to make a train set, and checks the
// matcher pairs each query to its own shifted copy rather than to the other
// segment.
func TestMatcherPairsShiftedCopies(t *testing.T) {
	bd := NewBinaryDescriptor()
	const dx, dy = 5, 7

	// Query descriptors.
	thinA := makeThin(cv.Point{X: 40, Y: 70}, cv.Point{X: 100, Y: 70})
	_, qThin := bd.Compute(thinA, []KeyLine{newKeyLine(40, 70, 100, 70)})
	stepA := makeStep(70)
	_, qStep := bd.Compute(stepA, []KeyLine{newKeyLine(40, 70, 100, 70)})

	// The two query codes must be distinguishable for the pairing to be meaningful.
	if HammingDistance(qThin[0], qStep[0]) == 0 {
		t.Fatalf("thin and step descriptors are identical; test cannot discriminate")
	}

	// Train descriptors: the same two structures shifted by (dx,dy). Deliberately
	// place the ridge first and the step second so a correct match must reorder.
	thinB := makeThin(cv.Point{X: 40 + dx, Y: 70 + dy}, cv.Point{X: 100 + dx, Y: 70 + dy})
	_, tThin := bd.Compute(thinB, []KeyLine{newKeyLine(40+dx, 70+dy, 100+dx, 70+dy)})
	stepB := makeStep(70 + dy)
	_, tStep := bd.Compute(stepB, []KeyLine{newKeyLine(40+dx, 70+dy, 100+dx, 70+dy)})

	query := [][]byte{qThin[0], qStep[0]} // 0: thin, 1: step
	train := [][]byte{tStep[0], tThin[0]} // 0: step, 1: thin (swapped order)
	wantTrain := []int{1, 0}              // thin->1, step->0

	m := NewBinaryDescriptorMatcher()
	matches := m.Match(query, train)
	if len(matches) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(matches))
	}
	for _, mt := range matches {
		if mt.TrainIdx != wantTrain[mt.QueryIdx] {
			t.Fatalf("query %d matched train %d (dist %d), want %d",
				mt.QueryIdx, mt.TrainIdx, mt.Distance, wantTrain[mt.QueryIdx])
		}
		if mt.Distance != 0 {
			t.Fatalf("query %d matched its shifted copy with distance %d, want 0",
				mt.QueryIdx, mt.Distance)
		}
	}
}

// TestKnnMatchOrdering checks KnnMatch returns k candidates sorted by ascending
// distance with deterministic tie-breaking.
func TestKnnMatchOrdering(t *testing.T) {
	m := NewBinaryDescriptorMatcher()
	query := [][]byte{{0b0000_0000}}
	train := [][]byte{
		{0b0000_1111}, // dist 4
		{0b0000_0001}, // dist 1
		{0b0000_0011}, // dist 2
		{0b0000_0001}, // dist 1 (tie with index 1; higher index)
	}
	knn := m.KnnMatch(query, train, 3)
	if len(knn) != 1 || len(knn[0]) != 3 {
		t.Fatalf("KnnMatch shape = %d x %v, want 1 x 3", len(knn), knn)
	}
	got := knn[0]
	// Expect train 1 (dist 1) first (lower index tie-break), then 3 (dist 1),
	// then 2 (dist 2).
	wantTrain := []int{1, 3, 2}
	wantDist := []int{1, 1, 2}
	for i := range got {
		if got[i].TrainIdx != wantTrain[i] || got[i].Distance != wantDist[i] {
			t.Fatalf("knn[%d] = {train %d dist %d}, want {train %d dist %d}",
				i, got[i].TrainIdx, got[i].Distance, wantTrain[i], wantDist[i])
		}
	}
}

// TestHammingDistanceMismatchPanics checks the length guard.
func TestHammingDistanceMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on length mismatch")
		}
	}()
	HammingDistance([]byte{0}, []byte{0, 0})
}

// TestDrawKeylines checks the output is a 3-channel image with the segment
// pixels coloured.
func TestDrawKeylines(t *testing.T) {
	img := cv.NewMat(40, 40, 1)
	lines := []KeyLine{newKeyLine(5, 20, 35, 20)}
	out := DrawKeylines(img, lines, cv.NewScalar(255, 0, 0), 1)
	if out.Channels != 3 {
		t.Fatalf("output channels = %d, want 3", out.Channels)
	}
	if out.Rows != 40 || out.Cols != 40 {
		t.Fatalf("output size = %dx%d, want 40x40", out.Rows, out.Cols)
	}
	// The midpoint of the drawn line should carry the red channel.
	i := (20*40 + 20) * 3
	if out.Data[i] != 255 || out.Data[i+1] != 0 || out.Data[i+2] != 0 {
		t.Fatalf("midpoint colour = (%d,%d,%d), want (255,0,0)",
			out.Data[i], out.Data[i+1], out.Data[i+2])
	}
	// The source image must be unmodified.
	if img.Channels != 1 {
		t.Fatal("DrawKeylines mutated the source image")
	}
}

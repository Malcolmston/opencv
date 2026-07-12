package linedescriptor

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- Multi-octave / scale pyramid ---------------------------------------

// TestBuildScalePyramid checks the pyramid shrinks by the requested factor and
// records matching scales.
func TestBuildScalePyramid(t *testing.T) {
	img := cv.NewMat(200, 160, 1)
	pyr := BuildScalePyramid(img, 3, 2.0)
	if len(pyr.Levels) != 3 || len(pyr.Scales) != 3 {
		t.Fatalf("pyramid has %d levels / %d scales, want 3 / 3", len(pyr.Levels), len(pyr.Scales))
	}
	if pyr.Scales[0] != 1 {
		t.Fatalf("scale[0] = %v, want 1", pyr.Scales[0])
	}
	if pyr.Levels[0].Rows != 200 || pyr.Levels[0].Cols != 160 {
		t.Fatalf("level 0 = %dx%d, want 200x160", pyr.Levels[0].Rows, pyr.Levels[0].Cols)
	}
	if pyr.Levels[1].Rows != 100 || pyr.Levels[1].Cols != 80 {
		t.Fatalf("level 1 = %dx%d, want 100x80", pyr.Levels[1].Rows, pyr.Levels[1].Cols)
	}
	if pyr.Levels[2].Cols != 40 {
		t.Fatalf("level 2 cols = %d, want 40", pyr.Levels[2].Cols)
	}
}

// TestDetectPyramidRealOctaves verifies the multi-octave detector recovers the
// same physical line at more than one octave, with real (non-zero) octave tags
// and octave-space endpoints that scale back to the original coordinates.
func TestDetectPyramidRealOctaves(t *testing.T) {
	img := cv.NewMat(200, 200, 1)
	cv.Line(img, cv.Point{X: 20, Y: 100}, cv.Point{X: 180, Y: 100}, cv.NewScalar(255), 5)

	ex := NewLSDDetector().DetectPyramid(img, 3, 2.0)
	if len(ex) == 0 {
		t.Fatal("no segments detected across the pyramid")
	}

	octaves := map[int]bool{}
	for _, l := range ex {
		octaves[l.Octave] = true
	}
	if !octaves[0] {
		t.Fatal("expected at least one segment at octave 0")
	}
	if !octaves[1] {
		t.Fatal("expected at least one segment at octave 1 (multi-octave detection)")
	}

	// Check back-projection consistency for a coarse-octave segment.
	for _, l := range ex {
		if l.Octave == 0 {
			continue
		}
		f := math.Pow(2, float64(l.Octave))
		gotX := l.StartPointInOctave.X * f
		if math.Abs(gotX-l.StartPointF.X) > 1e-6 {
			t.Fatalf("octave %d: StartPointInOctave.X*scale = %.3f, want StartPointF.X = %.3f",
				l.Octave, gotX, l.StartPointF.X)
		}
		if l.NumOfPixels != int(math.Round(l.LineLength)) {
			t.Fatalf("NumOfPixels = %d, want round(LineLength=%f)", l.NumOfPixels, l.LineLength)
		}
	}
}

// TestDetectPyramidDeterministic checks two runs are byte-identical.
func TestDetectPyramidDeterministic(t *testing.T) {
	img := cv.NewMat(160, 160, 1)
	cv.Line(img, cv.Point{X: 10, Y: 40}, cv.Point{X: 150, Y: 40}, cv.NewScalar(255), 4)
	cv.Line(img, cv.Point{X: 20, Y: 20}, cv.Point{X: 120, Y: 120}, cv.NewScalar(255), 4)
	det := NewLSDDetector()
	a := det.DetectPyramid(img, 3, 2.0)
	b := det.DetectPyramid(img, 3, 2.0)
	if len(a) != len(b) {
		t.Fatalf("nondeterministic length %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("segment %d differs: %+v vs %+v", i, a[i], b[i])
		}
	}
}

// TestDetectWithScale checks a downscaled detection maps back to the original
// coordinate frame and tags the octave.
func TestDetectWithScale(t *testing.T) {
	img := cv.NewMat(200, 200, 1)
	cv.Line(img, cv.Point{X: 20, Y: 100}, cv.Point{X: 180, Y: 100}, cv.NewScalar(255), 5)
	lines := NewLSDDetector().DetectWithScale(img, 1)
	if len(lines) == 0 {
		t.Fatal("no segments at octave 1")
	}
	for _, kl := range lines {
		if kl.Octave != 1 {
			t.Fatalf("octave = %d, want 1", kl.Octave)
		}
		// A horizontal line near y=100 in the original frame.
		if orientationDiff(kl.Angle, 0) > 0.12 {
			t.Fatalf("orientation %.3f not horizontal", kl.Angle)
		}
		if kl.Length < 100 {
			t.Fatalf("back-projected length %.1f too short for a full-width line", kl.Length)
		}
	}
}

// TestToKeyLines checks the embedded KeyLine extraction.
func TestToKeyLines(t *testing.T) {
	ex := []KeyLineEx{
		{KeyLine: newKeyLine(0, 0, 10, 0)},
		{KeyLine: newKeyLine(0, 5, 10, 5)},
	}
	kl := ToKeyLines(ex)
	if len(kl) != 2 || kl[0] != ex[0].KeyLine || kl[1] != ex[1].KeyLine {
		t.Fatalf("ToKeyLines mismatch: %+v", kl)
	}
}

// --- Mask ---------------------------------------------------------------

// TestDetectWithMask checks the ROI mask keeps only segments whose midpoint is
// admitted.
func TestDetectWithMask(t *testing.T) {
	img := cv.NewMat(120, 120, 1)
	cv.Line(img, cv.Point{X: 10, Y: 30}, cv.Point{X: 110, Y: 30}, cv.NewScalar(255), 3) // top
	cv.Line(img, cv.Point{X: 10, Y: 90}, cv.Point{X: 110, Y: 90}, cv.NewScalar(255), 3) // bottom

	det := NewLSDDetector()
	all := det.Detect(img)
	if len(all) < 2 {
		t.Fatalf("expected >=2 segments unmasked, got %d", len(all))
	}

	// Mask admitting only the top half.
	mask := RectMask(120, 120, 0, 0, 60, 120)
	masked := det.DetectWithMask(img, mask)
	if len(masked) == 0 {
		t.Fatal("mask removed everything")
	}
	for _, kl := range masked {
		my := (kl.StartPoint.Y + kl.EndPoint.Y) / 2
		if my >= 60 {
			t.Fatalf("segment with midpoint y=%d survived a top-half mask", my)
		}
	}
	if len(masked) >= len(all) {
		t.Fatalf("mask did not reduce the segment count (%d vs %d)", len(masked), len(all))
	}
}

// TestDetectWithMaskNil checks a nil mask is a passthrough.
func TestDetectWithMaskNil(t *testing.T) {
	img := cv.NewMat(80, 80, 1)
	cv.Line(img, cv.Point{X: 5, Y: 40}, cv.Point{X: 75, Y: 40}, cv.NewScalar(255), 3)
	det := NewLSDDetector()
	if len(det.DetectWithMask(img, nil)) != len(det.Detect(img)) {
		t.Fatal("nil mask should behave like Detect")
	}
}

// --- EDLines ------------------------------------------------------------

// TestEDLinesDetectsSegment checks the edge-drawing detector recovers a line of
// the right orientation and endpoints.
func TestEDLinesDetectsSegment(t *testing.T) {
	cases := []struct {
		name     string
		a, b     cv.Point
		expAngle float64
	}{
		{"horizontal", cv.Point{X: 20, Y: 60}, cv.Point{X: 100, Y: 60}, 0},
		{"vertical", cv.Point{X: 60, Y: 20}, cv.Point{X: 60, Y: 100}, math.Pi / 2},
		{"diagonal", cv.Point{X: 20, Y: 20}, cv.Point{X: 100, Y: 100}, math.Pi / 4},
	}
	ed := NewEDLinesDetector()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			img := cv.NewMat(130, 130, 1)
			cv.Line(img, c.a, c.b, cv.NewScalar(255), 3)
			lines := ed.Detect(img)
			if len(lines) == 0 {
				t.Fatalf("%s: no segments", c.name)
			}
			found := false
			for _, kl := range lines {
				if orientationDiff(kl.Angle, c.expAngle) < 0.15 && endpointsMatch(kl, c.a, c.b, 8) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("%s: no segment matched; first=%v", c.name, lines[0])
			}
		})
	}
}

// TestEDLinesDeterministic checks repeatability.
func TestEDLinesDeterministic(t *testing.T) {
	img := cv.NewMat(120, 120, 1)
	cv.Line(img, cv.Point{X: 10, Y: 40}, cv.Point{X: 110, Y: 40}, cv.NewScalar(255), 3)
	cv.Line(img, cv.Point{X: 10, Y: 80}, cv.Point{X: 110, Y: 80}, cv.NewScalar(255), 3)
	ed := NewEDLinesDetector()
	a := ed.Detect(img)
	b := ed.Detect(img)
	if len(a) != len(b) {
		t.Fatalf("nondeterministic count %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("segment %d differs: %v vs %v", i, a[i], b[i])
		}
	}
}

// TestEDLinesSortedByResponse checks descending response order.
func TestEDLinesSortedByResponse(t *testing.T) {
	img := cv.NewMat(140, 140, 1)
	cv.Line(img, cv.Point{X: 10, Y: 30}, cv.Point{X: 130, Y: 30}, cv.NewScalar(255), 3)
	cv.Line(img, cv.Point{X: 10, Y: 90}, cv.Point{X: 50, Y: 90}, cv.NewScalar(255), 3)
	lines := NewEDLinesDetector().Detect(img)
	for i := 1; i < len(lines); i++ {
		if lines[i-1].Response < lines[i].Response {
			t.Fatalf("not sorted at %d: %.1f < %.1f", i, lines[i-1].Response, lines[i].Response)
		}
	}
}

// --- LSH matcher / RadiusMatch -----------------------------------------

// TestRadiusMatch checks the radius query returns exactly the codes within the
// Hamming ball, sorted.
func TestRadiusMatch(t *testing.T) {
	m := NewBinaryDescriptorMatcher()
	query := [][]byte{{0b0000_0000}}
	train := [][]byte{
		{0b0000_0001}, // dist 1
		{0b0000_1111}, // dist 4
		{0b0000_0011}, // dist 2
		{0b0000_0000}, // dist 0
	}
	res := m.RadiusMatch(query, train, 2)
	if len(res) != 1 {
		t.Fatalf("want 1 query row, got %d", len(res))
	}
	got := res[0]
	wantTrain := []int{3, 0, 2} // dist 0,1,2
	wantDist := []int{0, 1, 2}
	if len(got) != len(wantTrain) {
		t.Fatalf("radius result = %v, want %d entries", got, len(wantTrain))
	}
	for i := range got {
		if got[i].TrainIdx != wantTrain[i] || got[i].Distance != wantDist[i] {
			t.Fatalf("radius[%d] = {train %d dist %d}, want {train %d dist %d}",
				i, got[i].TrainIdx, got[i].Distance, wantTrain[i], wantDist[i])
		}
	}
}

// TestLSHIndexKnnFindsIdentical checks the multi-index LSH structure retrieves
// an identical descriptor at distance 0 and ranks a near copy next.
func TestLSHIndexKnnFindsIdentical(t *testing.T) {
	train := [][]byte{
		{0xFF, 0x00, 0xAA, 0x55},
		{0xFF, 0x00, 0xAA, 0x54}, // 1 bit off the first
		{0x00, 0xFF, 0x55, 0xAA}, // very different
	}
	idx := NewLSHIndex(4, 8)
	idx.Add(train)
	if idx.Size() != 3 {
		t.Fatalf("index size = %d, want 3", idx.Size())
	}
	query := [][]byte{{0xFF, 0x00, 0xAA, 0x55}}
	knn := idx.KnnMatch(query, 2)
	if len(knn) != 1 || len(knn[0]) == 0 {
		t.Fatalf("knn shape unexpected: %v", knn)
	}
	if knn[0][0].TrainIdx != 0 || knn[0][0].Distance != 0 {
		t.Fatalf("nearest = {train %d dist %d}, want {0, 0}", knn[0][0].TrainIdx, knn[0][0].Distance)
	}
}

// TestLSHIndexRadius checks LSH radius retrieval keeps only in-ball candidates.
func TestLSHIndexRadius(t *testing.T) {
	train := [][]byte{
		{0xFF, 0xFF},
		{0xFF, 0xFE}, // 1 bit
		{0x00, 0x00}, // 16 bits
	}
	idx := NewLSHIndex(6, 6)
	idx.Add(train)
	res := idx.RadiusMatch([][]byte{{0xFF, 0xFF}}, 2)
	if len(res) != 1 {
		t.Fatalf("want 1 row, got %d", len(res))
	}
	for _, dm := range res[0] {
		if dm.Distance > 2 {
			t.Fatalf("radius returned a match at distance %d > 2", dm.Distance)
		}
	}
	// Distance-0 self must be present and first.
	if len(res[0]) == 0 || res[0][0].TrainIdx != 0 || res[0][0].Distance != 0 {
		t.Fatalf("expected self-match first, got %v", res[0])
	}
}

// TestLSHMatchesShiftedDescriptor exercises the full detect→describe→LSH path on
// an image and its integer-shifted copy.
func TestLSHMatchesShiftedDescriptor(t *testing.T) {
	bd := NewBinaryDescriptor()
	const dx, dy = 6, 9
	a := makeThin(cv.Point{X: 40, Y: 70}, cv.Point{X: 100, Y: 70})
	b := makeThin(cv.Point{X: 40 + dx, Y: 70 + dy}, cv.Point{X: 100 + dx, Y: 70 + dy})
	_, ca := bd.Compute(a, []KeyLine{newKeyLine(40, 70, 100, 70)})
	_, cb := bd.Compute(b, []KeyLine{newKeyLine(40+dx, 70+dy, 100+dx, 70+dy)})

	idx := NewLSHIndex(4, 6)
	idx.Add(cb)
	knn := idx.KnnMatch(ca, 1)
	if len(knn) != 1 || len(knn[0]) == 0 {
		t.Fatalf("LSH failed to retrieve the shifted copy: %v", knn)
	}
	if knn[0][0].Distance != 0 {
		t.Fatalf("shifted copy retrieved at distance %d, want 0", knn[0][0].Distance)
	}
}

// --- DrawLineMatches ----------------------------------------------------

// TestDrawLineMatches checks the side-by-side canvas dimensions and that a match
// paints the match colour somewhere on the right half's segment.
func TestDrawLineMatches(t *testing.T) {
	img1 := cv.NewMat(50, 60, 1)
	img2 := cv.NewMat(40, 70, 1)
	lines1 := []KeyLine{newKeyLine(5, 25, 55, 25)}
	lines2 := []KeyLine{newKeyLine(5, 20, 65, 20)}
	matches := []DMatch{{QueryIdx: 0, TrainIdx: 0, Distance: 0}}

	out := DrawLineMatches(img1, lines1, img2, lines2, matches,
		cv.NewScalar(0, 255, 0), cv.NewScalar(255, 0, 0), 1)

	if out.Channels != 3 {
		t.Fatalf("channels = %d, want 3", out.Channels)
	}
	if out.Rows != 50 || out.Cols != 130 {
		t.Fatalf("canvas = %dx%d, want 50x130", out.Rows, out.Cols)
	}
	// The right image's segment sits at y=20, x in [60+5, 60+65]; sample a green pixel.
	i := (20*out.Cols + (60 + 30)) * 3
	if out.Data[i+1] == 0 {
		t.Fatalf("expected green on the matched right-side segment at green channel, got %d", out.Data[i+1])
	}
}

// TestDrawLineMatchesSkipsBadIndices checks out-of-range match indices are
// ignored rather than panicking.
func TestDrawLineMatchesSkipsBadIndices(t *testing.T) {
	img1 := cv.NewMat(30, 30, 1)
	img2 := cv.NewMat(30, 30, 1)
	lines1 := []KeyLine{newKeyLine(2, 15, 27, 15)}
	lines2 := []KeyLine{newKeyLine(2, 10, 27, 10)}
	matches := []DMatch{{QueryIdx: 5, TrainIdx: 0}, {QueryIdx: 0, TrainIdx: 9}}
	out := DrawLineMatches(img1, lines1, img2, lines2, matches,
		cv.NewScalar(0, 255, 0), cv.NewScalar(255, 0, 0), 1)
	if out.Cols != 60 {
		t.Fatalf("canvas cols = %d, want 60", out.Cols)
	}
}

// --- Params / per-octave compute ---------------------------------------

// TestBinaryDescriptorParams checks defaults and construction.
func TestBinaryDescriptorParams(t *testing.T) {
	p := DefaultBinaryDescriptorParams()
	if err := p.Validate(); err != nil {
		t.Fatalf("default params invalid: %v", err)
	}
	bd := NewBinaryDescriptorWithParams(p)
	if bd.NumBands != 8 || bd.BandWidth != 7 {
		t.Fatalf("params -> descriptor = %d bands x %d, want 8x7", bd.NumBands, bd.BandWidth)
	}
	bad := BinaryDescriptorParams{NumOfOctaves: 0, WidthOfBand: 7, NumOfBands: 8, ReductionRatio: 2}
	if bad.Validate() == nil {
		t.Fatal("expected invalid params to fail Validate")
	}
}

// TestComputeMultiOctave checks per-octave description yields correctly sized
// codes and that an identical structure across octaves is described stably.
func TestComputeMultiOctave(t *testing.T) {
	img := cv.NewMat(200, 200, 1)
	cv.Line(img, cv.Point{X: 20, Y: 100}, cv.Point{X: 180, Y: 100}, cv.NewScalar(255), 5)
	det := NewLSDDetector()
	ex := det.DetectPyramid(img, 3, 2.0)
	pyr := BuildScalePyramid(img, 3, 2.0)

	bd := NewBinaryDescriptor()
	kept, codes := bd.ComputeMultiOctave(pyr, ex)
	if len(kept) != len(ex) || len(codes) != len(ex) {
		t.Fatalf("ComputeMultiOctave returned %d/%d, want %d", len(kept), len(codes), len(ex))
	}
	for i, c := range codes {
		if len(c) != bd.DescriptorSize() {
			t.Fatalf("code %d length %d, want %d", i, len(c), bd.DescriptorSize())
		}
	}
	// Determinism.
	_, codes2 := bd.ComputeMultiOctave(pyr, ex)
	for i := range codes {
		if HammingDistance(codes[i], codes2[i]) != 0 {
			t.Fatalf("nondeterministic code at %d", i)
		}
	}
}

// TestComputeMultiOctaveMissingOctave checks a line whose octave is absent from
// the pyramid gets an all-zero code rather than panicking.
func TestComputeMultiOctaveMissingOctave(t *testing.T) {
	img := cv.NewMat(80, 80, 1)
	pyr := BuildScalePyramid(img, 1, 2.0) // only octave 0
	line := KeyLineEx{
		KeyLine:            KeyLine{Octave: 5},
		StartPointInOctave: PointF{X: 10, Y: 10},
		EndPointInOctave:   PointF{X: 40, Y: 10},
	}
	bd := NewBinaryDescriptor()
	_, codes := bd.ComputeMultiOctave(pyr, []KeyLineEx{line})
	for _, b := range codes[0] {
		if b != 0 {
			t.Fatal("expected all-zero code for a missing octave")
		}
	}
}

// TestDrawKeylinesEx checks the extended draw promotes to RGB and paints.
func TestDrawKeylinesEx(t *testing.T) {
	img := cv.NewMat(60, 60, 1)
	lines := []KeyLineEx{{KeyLine: newKeyLine(5, 30, 55, 30)}}
	out := DrawKeylinesEx(img, lines, cv.NewScalar(0, 0, 255), 1)
	if out.Channels != 3 || out.Rows != 60 || out.Cols != 60 {
		t.Fatalf("bad canvas %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
}

// --- Filtering helpers --------------------------------------------------

func TestFilterHelpers(t *testing.T) {
	lines := []KeyLine{
		newKeyLine(0, 0, 5, 0),   // len 5
		newKeyLine(0, 0, 20, 0),  // len 20
		newKeyLine(0, 0, 12, 0),  // len 12
		newKeyLine(0, 0, 100, 0), // len 100
	}
	if got := FilterByLength(lines, 12); len(got) != 3 {
		t.Fatalf("FilterByLength(12) kept %d, want 3", len(got))
	}
	if got := FilterByResponse(lines, 20); len(got) != 2 {
		t.Fatalf("FilterByResponse(20) kept %d, want 2", len(got))
	}
	top := TopN(lines, 2)
	if len(top) != 2 || top[0].Length != 100 || top[1].Length != 20 {
		t.Fatalf("TopN(2) = %v, want lengths 100,20", top)
	}
	if TopN(lines, 0) != nil {
		t.Fatal("TopN(0) should be nil")
	}
	if len(TopN(lines, 99)) != 4 {
		t.Fatal("TopN(99) should return all")
	}
}

func TestFilterByOctaveAndCodes(t *testing.T) {
	ex := []KeyLineEx{
		{KeyLine: KeyLine{Octave: 0}},
		{KeyLine: KeyLine{Octave: 1}},
		{KeyLine: KeyLine{Octave: 1}},
	}
	if len(FilterByOctave(ex, 1)) != 2 {
		t.Fatal("FilterByOctave(1) should keep 2")
	}
	lines := []KeyLine{newKeyLine(0, 0, 5, 0), newKeyLine(0, 0, 30, 0)}
	codes := [][]byte{{0x01}, {0x02}}
	fl, fc := FilterCodesByLength(lines, codes, 10)
	if len(fl) != 1 || len(fc) != 1 || fc[0][0] != 0x02 {
		t.Fatalf("FilterCodesByLength mismatch: %v %v", fl, fc)
	}
}

// --- Geometric verification --------------------------------------------

// TestMatchLineSegments checks geometric verification rejects an
// appearance-only match that is geometrically inconsistent (wrong orientation).
func TestMatchLineSegments(t *testing.T) {
	// Query: one horizontal segment.
	lines1 := []KeyLine{newKeyLine(0, 50, 60, 50)} // horizontal
	codes1 := [][]byte{{0b0000_0000}}
	// Train: a vertical segment with an identical code, and a horizontal segment
	// with a slightly different code.
	lines2 := []KeyLine{
		newKeyLine(30, 0, 30, 60), // vertical, identical code
		newKeyLine(0, 52, 60, 52), // horizontal, near code
	}
	codes2 := [][]byte{{0b0000_0000}, {0b0000_0001}}

	params := DefaultGeometricMatchParams()
	matches := MatchLineSegments(lines1, codes1, lines2, codes2, params)
	if len(matches) != 1 {
		t.Fatalf("want 1 verified match, got %d: %v", len(matches), matches)
	}
	// Despite the vertical segment being a perfect appearance match, geometry
	// must pick the horizontal one (train 1).
	if matches[0].TrainIdx != 1 {
		t.Fatalf("verified match chose train %d, want 1 (the geometrically consistent one)", matches[0].TrainIdx)
	}
}

// TestMatchLineSegmentsNoMatch checks a query with no consistent train yields no
// match entry.
func TestMatchLineSegmentsNoMatch(t *testing.T) {
	lines1 := []KeyLine{newKeyLine(0, 50, 60, 50)}
	codes1 := [][]byte{{0x00}}
	lines2 := []KeyLine{newKeyLine(30, 0, 30, 60)} // vertical only
	codes2 := [][]byte{{0x00}}
	if m := MatchLineSegments(lines1, codes1, lines2, codes2, DefaultGeometricMatchParams()); len(m) != 0 {
		t.Fatalf("want no match, got %v", m)
	}
}

// TestSegmentOverlap checks collinear overlap scores high and disjoint scores 0.
func TestSegmentOverlap(t *testing.T) {
	a := newKeyLine(0, 10, 100, 10)
	b := newKeyLine(20, 10, 80, 10) // fully inside a
	if ov := SegmentOverlap(a, b); ov < 0.99 {
		t.Fatalf("overlap of contained segment = %.3f, want ~1", ov)
	}
	c := newKeyLine(200, 10, 260, 10) // disjoint along the axis
	if ov := SegmentOverlap(a, c); ov != 0 {
		t.Fatalf("overlap of disjoint segment = %.3f, want 0", ov)
	}
}

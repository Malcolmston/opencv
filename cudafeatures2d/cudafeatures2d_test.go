package cudafeatures2d

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// checkerboard renders a size×size single-channel checkerboard with the given
// cell side. Checkerboards have strong, well-localised corners at every cell
// junction, which every corner detector here should find.
func checkerboard(size, cell int) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if ((x/cell)+(y/cell))%2 == 0 {
				m.Data[y*size+x] = 255
			}
		}
	}
	return m
}

// squares renders a light background with several dark filled squares whose
// corners are strong, well-isolated interest points (a pattern FAST/ORB detect
// reliably, unlike a checkerboard). The squares sit well inside the border so
// they survive the ORB patch-radius margin. The (20,20) top-left corner is
// asserted on by the corner tests.
func squares(size int) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	m.SetTo(200)
	rects := [][4]int{
		{20, 20, 16, 16}, {70, 30, 16, 16}, {40, 75, 16, 16}, {90, 90, 16, 16},
	}
	for _, r := range rects {
		for y := r[1]; y < r[1]+r[3] && y < size; y++ {
			for x := r[0]; x < r[0]+r[2] && x < size; x++ {
				m.Data[y*size+x] = 20
			}
		}
	}
	return m
}

func TestGpuMatUploadDownloadClone(t *testing.T) {
	src := squares(64)
	var g GpuMat
	if !g.Empty() {
		t.Fatal("zero GpuMat should be empty")
	}
	g.Upload(src)
	if g.Empty() {
		t.Fatal("GpuMat empty after Upload")
	}
	r, c := g.Size()
	if r != 64 || c != 64 {
		t.Fatalf("Size = %d,%d want 64,64", r, c)
	}
	if g.Rows() != 64 || g.Cols() != 64 || g.Channels() != 1 {
		t.Fatalf("dims = %d,%d,%d", g.Rows(), g.Cols(), g.Channels())
	}
	got := g.Download()
	if got == nil || got.Rows != 64 {
		t.Fatal("Download returned wrong matrix")
	}
	// Download is a copy: mutating it must not touch the device data.
	got.Data[0] ^= 0xFF
	if g.Download().Data[0] == got.Data[0] {
		t.Fatal("Download did not return an independent copy")
	}
	// Clone independence.
	cl := g.Clone()
	cl.Upload(nil)
	if g.Empty() {
		t.Fatal("clearing a clone affected the original")
	}
}

func TestNewGpuMatEmptyInputs(t *testing.T) {
	if !NewGpuMat(nil).Empty() {
		t.Fatal("NewGpuMat(nil) should be empty")
	}
	if !NewGpuMat(&cv.Mat{}).Empty() {
		t.Fatal("NewGpuMat(empty) should be empty")
	}
	empty := NewGpuMat(nil)
	if empty.Download() != nil {
		t.Fatal("empty Download should be nil")
	}
	if empty.Rows() != 0 || empty.Cols() != 0 || empty.Channels() != 0 {
		t.Fatal("empty dims should be zero")
	}
	r, c := empty.Size()
	if r != 0 || c != 0 {
		t.Fatal("empty Size should be 0,0")
	}
}

func TestStreamNoOp(t *testing.T) {
	s := NewStream()
	if s == nil {
		t.Fatal("NewStream returned nil")
	}
	s.WaitForCompletion() // must not panic or block
}

func TestORBDetectAndComputeSelfMatch(t *testing.T) {
	img := NewGpuMat(squares(128))
	orb := CreateORB(200)
	kps, desc := orb.DetectAndCompute(img, nil)
	if len(kps) < 4 {
		t.Fatalf("expected several ORB keypoints, got %d", len(kps))
	}
	if desc.Empty() {
		t.Fatal("descriptors GpuMat is empty")
	}
	if desc.Rows() != len(kps) {
		t.Fatalf("descriptor rows %d != keypoints %d", desc.Rows(), len(kps))
	}
	if desc.Cols() != orb.DescriptorSize() {
		t.Fatalf("descriptor width %d != DescriptorSize %d", desc.Cols(), orb.DescriptorSize())
	}
	if orb.DefaultNorm() != NormHamming {
		t.Fatal("ORB default norm should be Hamming")
	}

	// Matching the descriptors against themselves must yield an exact
	// (distance-zero) match for every keypoint. Because some squares are
	// identical their descriptors can coincide, so the matched train index is
	// not necessarily the query index — but the distance is always zero.
	matcher := CreateBFMatcher(orb.DefaultNorm())
	matches := matcher.Match(desc, desc)
	if len(matches) != len(kps) {
		t.Fatalf("expected %d self matches, got %d", len(kps), len(matches))
	}
	for _, m := range matches {
		if m.Distance != 0 {
			t.Errorf("self match %d -> %d distance = %v, want 0", m.QueryIdx, m.TrainIdx, m.Distance)
		}
	}
}

func TestORBConvertRoundTrip(t *testing.T) {
	img := NewGpuMat(squares(128))
	orb := CreateORB(50)
	_, desc := orb.DetectAndCompute(img, nil)
	rows := orb.Convert(desc)
	if len(rows) != desc.Rows() {
		t.Fatalf("Convert rows %d != %d", len(rows), desc.Rows())
	}
	// Round-trip back to a GpuMat and confirm the bytes are preserved.
	back := DescriptorsToGpuMat(rows)
	if back.Rows() != desc.Rows() || back.Cols() != desc.Cols() {
		t.Fatal("round-trip changed descriptor dimensions")
	}
	bd := back.Download()
	dd := desc.Download()
	for i := range dd.Data {
		if bd.Data[i] != dd.Data[i] {
			t.Fatalf("descriptor byte %d changed on round trip", i)
		}
	}
}

func TestORBComputeMatchesDetectAndCompute(t *testing.T) {
	img := NewGpuMat(squares(128))
	orb := CreateORB(80)
	kps := orb.Detect(img, nil)
	if len(kps) == 0 {
		t.Fatal("no keypoints detected")
	}
	outKps, desc := orb.Compute(img, kps)
	if len(outKps) != len(kps) {
		t.Fatalf("Compute changed keypoint count: %d vs %d", len(outKps), len(kps))
	}
	// DetectAndCompute should agree with Detect+Compute.
	_, desc2 := orb.DetectAndCompute(img, nil)
	if desc.Rows() != desc2.Rows() || desc.Cols() != desc2.Cols() {
		t.Fatal("Compute and DetectAndCompute produced different descriptor shapes")
	}
}

func TestORBAsyncEqualsSync(t *testing.T) {
	img := NewGpuMat(squares(128))
	orb := CreateORB(60)
	k1, d1 := orb.DetectAndCompute(img, nil)
	k2, d2 := orb.DetectAndComputeAsync(img, nil, NewStream())
	if len(k1) != len(k2) || d1.Rows() != d2.Rows() {
		t.Fatal("async result differs from sync result")
	}
}

func TestORBMask(t *testing.T) {
	img := NewGpuMat(squares(128))
	orb := CreateORB(200)
	all := orb.Detect(img, nil)

	// Mask keeps only the top-left quadrant.
	mask := cv.NewMat(128, 128, 1)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			mask.Data[y*128+x] = 255
		}
	}
	masked := orb.Detect(img, NewGpuMat(mask))
	if len(masked) == 0 {
		t.Fatal("mask removed all keypoints")
	}
	if len(masked) >= len(all) {
		t.Fatalf("mask did not reduce keypoints: %d vs %d", len(masked), len(all))
	}
	for _, kp := range masked {
		if kp.Pt.X >= 64 || kp.Pt.Y >= 64 {
			t.Fatalf("masked keypoint outside allowed region: %+v", kp.Pt)
		}
	}
}

func TestFastFeatureDetector(t *testing.T) {
	img := NewGpuMat(squares(128))
	fast := CreateFastFeatureDetector(20, true)
	kps := fast.Detect(img, nil)
	if len(kps) < 4 {
		t.Fatalf("FAST found too few corners: %d", len(kps))
	}
	// Async must agree.
	kps2 := fast.DetectAsync(img, nil, NewStream())
	if len(kps2) != len(kps) {
		t.Fatalf("FAST async count %d != sync %d", len(kps2), len(kps))
	}
	// Every corner should be a genuine FAST response (non-zero cornerness).
	for _, kp := range kps {
		if kp.Response <= 0 {
			t.Errorf("FAST keypoint has non-positive response: %+v", kp)
		}
	}
}

func TestCornersDetectorFindsSquareCorners(t *testing.T) {
	img := NewGpuMat(squares(128))
	det := CreateGoodFeaturesToTrackDetector(50, 0.01, 5, 3)
	corners := det.Detect(img, nil)
	if len(corners) < 4 {
		t.Fatalf("expected at least 4 corners, got %d", len(corners))
	}
	// The strongest corners should sit near an actual square corner. Check that
	// at least one detected point is within 3px of the (20,20) square's corner.
	near := false
	for _, p := range corners {
		if abs(p.X-20) <= 3 && abs(p.Y-20) <= 3 {
			near = true
			break
		}
	}
	if !near {
		t.Errorf("no corner detected near the (20,20) square corner; got %v", corners)
	}

	// Async agreement.
	if len(det.DetectAsync(img, nil, NewStream())) != len(corners) {
		t.Fatal("corners async differs from sync")
	}
}

func TestCornersDetectorMask(t *testing.T) {
	img := NewGpuMat(squares(128))
	det := CreateGoodFeaturesToTrackDetector(100, 0.01, 3, 3)
	all := det.Detect(img, nil)

	mask := cv.NewMat(128, 128, 1)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			mask.Data[y*128+x] = 255
		}
	}
	masked := det.Detect(img, NewGpuMat(mask))
	if len(masked) >= len(all) {
		t.Fatalf("mask did not reduce corners: %d vs %d", len(masked), len(all))
	}
	for _, p := range masked {
		if p.X >= 64 || p.Y >= 64 {
			t.Fatalf("masked corner outside region: %+v", p)
		}
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

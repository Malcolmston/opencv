package stereo

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// regionMode returns the most common disparity over the interior of the shifted
// region (avoiding edges and the flat inset) plus the valid fraction, reading a
// uint8 map. It mirrors modeDisparity in stereo_test.go but is local so the two
// test files stay independent.
func regionMode(d *cv.Mat, rx, ry, rw, rh int) (mode int, validFrac float64) {
	counts := map[int]int{}
	total, valid := 0, 0
	for y := ry + 7; y < ry+rh-2; y++ {
		for x := rx + 7; x < rx+rw-2; x++ {
			total++
			v := d.Data[y*d.Cols+x]
			if v == InvalidDisparity {
				continue
			}
			valid++
			counts[int(v)]++
		}
	}
	best, bestN := 0, -1
	for k, n := range counts {
		if n > bestN {
			best, bestN = k, n
		}
	}
	if total == 0 {
		return 0, 0
	}
	return best, float64(valid) / float64(total)
}

func TestStereoSGMEightPathRecoversDisparity(t *testing.T) {
	const w, h, disp = 72, 48, 8
	const rx, ry, rw, rh = 28, 8, 36, 30
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)

	for _, mode := range []SGMMode{ModeSGBM, ModeHH} {
		sg := StereoSGM{NumDisparities: 16, BlockSize: 5, Mode: mode, Disp12MaxDiff: 2}
		d := sg.Compute(left, right)
		if d.Rows != h || d.Cols != w {
			t.Fatalf("mode %d: unexpected shape %dx%d", mode, d.Rows, d.Cols)
		}
		m, frac := regionMode(d, rx, ry, rw, rh)
		if m < disp-1 || m > disp+1 {
			t.Errorf("mode %d: recovered disparity %d, want %d ±1", mode, m, disp)
		}
		if frac < 0.4 {
			t.Errorf("mode %d: valid fraction %.2f, want >= 0.4", mode, frac)
		}
	}
}

func TestStereoSGMSubpixel(t *testing.T) {
	const w, h, disp = 72, 40, 8
	const rx, ry, rw, rh = 28, 6, 36, 28
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)

	sg := StereoSGM{NumDisparities: 16, BlockSize: 5, Mode: ModeHH, Disp12MaxDiff: 2}
	df := sg.ComputeFloat(left, right)
	// Count how many interior samples land within ±1 of the true disparity and
	// confirm at least one carries a genuine fractional part.
	sawFraction := false
	within, total := 0, 0
	for y := ry + 8; y < ry+rh-2; y++ {
		for x := rx + 8; x < rx+rw-2; x++ {
			v := df.At(y, x)
			if v == InvalidDisparityF {
				continue
			}
			total++
			if v >= disp-1 && v <= disp+1 {
				within++
			}
			if f := v - float32(int(v)); f > 0.01 && f < 0.99 {
				sawFraction = true
			}
		}
	}
	if total == 0 || float64(within)/float64(total) < 0.5 {
		t.Fatalf("subpixel disparities off: %d/%d within tolerance", within, total)
	}
	if !sawFraction {
		t.Error("expected at least one sub-pixel (fractional) disparity")
	}
}

func TestStereoSGMLRCheckRejectsOcclusion(t *testing.T) {
	// A half-plane shift creates an occluded band just left of the step where the
	// left-right check must invalidate pixels.
	const w, h, disp = 64, 24, 10
	left, right := buildHalfShift(w, h, disp)

	strict := StereoSGM{NumDisparities: 16, BlockSize: 5, Mode: ModeHH, Disp12MaxDiff: 1}
	loose := StereoSGM{NumDisparities: 16, BlockSize: 5, Mode: ModeHH, Disp12MaxDiff: -1}
	ds := strict.Compute(left, right)
	dl := loose.Compute(left, right)

	countInvalid := func(d *cv.Mat) int {
		n := 0
		for _, v := range d.Data {
			if v == InvalidDisparity {
				n++
			}
		}
		return n
	}
	if countInvalid(ds) <= countInvalid(dl) {
		t.Errorf("LR check did not reject more pixels: strict=%d loose=%d",
			countInvalid(ds), countInvalid(dl))
	}
}

// buildHalfShift makes a pair whose right half is shifted, producing a genuine
// occlusion boundary for the LR-consistency test.
func buildHalfShift(w, h, disp int) (left, right *cv.Mat) {
	right = cv.NewMat(h, w, 1)
	left = cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			right.Data[y*w+x] = texture(x, y)
		}
	}
	copy(left.Data, right.Data)
	for y := 0; y < h; y++ {
		for x := w / 2; x < w; x++ {
			sx := x - disp
			if sx < 0 {
				sx = 0
			}
			left.Data[y*w+x] = texture(sx, y)
		}
	}
	return left, right
}

func TestMatchingCostVolumeAndWTA(t *testing.T) {
	const w, h, disp = 64, 32, 8
	const rx, ry, rw, rh = 24, 6, 36, 22
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)

	for _, ct := range []CostType{CostSAD, CostSSD, CostAD, CostCensus} {
		vol := MatchingCostVolume(left, right, 0, 16, 5, ct)
		if vol.Rows != h || vol.Cols != w || vol.NumDisparities != 16 {
			t.Fatalf("cost %d: bad volume shape", ct)
		}
		d := vol.WinnerTakeAll()
		m, _ := regionMode(d, rx, ry, rw, rh)
		if m < disp-1 || m > disp+1 {
			t.Errorf("cost %d: WTA disparity %d, want %d ±1", ct, m, disp)
		}
	}
}

func TestCensusTransformAndHamming(t *testing.T) {
	m := cv.NewMat(5, 5, 1)
	// Centre brighter than all neighbours -> all comparison bits set (neighbour<centre).
	for i := range m.Data {
		m.Data[i] = 10
	}
	m.Data[2*5+2] = 200
	codes, rows, cols := CensusTransform(m, 3, 3)
	if rows != 5 || cols != 5 {
		t.Fatalf("bad census dims %dx%d", rows, cols)
	}
	center := codes[2*5+2]
	if got := HammingDistance64(center, 0); got != 8 {
		t.Errorf("expected 8 set bits for a peak centre, got %d", got)
	}
	if HammingDistance64(center, center) != 0 {
		t.Error("Hamming distance to self must be 0")
	}
}

func TestValidateDisparityRejectsInconsistent(t *testing.T) {
	const w, h = 10, 4
	dl := cv.NewMat(h, w, 1)
	dr := cv.NewMat(h, w, 1)
	// Consistent: left x=5 has d=3 -> right x=2 must also read 3.
	for y := 0; y < h; y++ {
		dl.Data[y*w+5] = 3
		dr.Data[y*w+2] = 3
		// Inconsistent: left x=8 has d=4 but right x=4 reads 9.
		dl.Data[y*w+8] = 4
		dr.Data[y*w+4] = 9
	}
	out := ValidateDisparity(dl, dr, 1, InvalidDisparity)
	for y := 0; y < h; y++ {
		if out.Data[y*w+5] != 3 {
			t.Errorf("consistent pixel wrongly rejected: %d", out.Data[y*w+5])
		}
		if out.Data[y*w+8] != InvalidDisparity {
			t.Errorf("inconsistent pixel not rejected: %d", out.Data[y*w+8])
		}
	}
	// Disabled check clones input untouched.
	cl := ValidateDisparity(dl, dr, -1, InvalidDisparity)
	if cl.Data[8] != dl.Data[8] {
		t.Error("disabled check should clone unchanged")
	}
}

func TestGetValidDisparityROI(t *testing.T) {
	roi1 := Rect{X: 0, Y: 0, Width: 100, Height: 100}
	roi2 := Rect{X: 0, Y: 0, Width: 100, Height: 100}
	r := GetValidDisparityROI(roi1, roi2, 0, 16, 5)
	// xmin = max(0, 0+15)+2 = 17; xmax = min(100,100-0)-2 = 98; width = 81.
	if r.X != 17 || r.Width != 81 {
		t.Errorf("ROI x/width = %d/%d, want 17/81", r.X, r.Width)
	}
	if r.Y != 2 || r.Height != 96 {
		t.Errorf("ROI y/height = %d/%d, want 2/96", r.Y, r.Height)
	}
	// Degenerate overlap -> empty.
	empty := GetValidDisparityROI(Rect{0, 0, 5, 5}, Rect{90, 90, 5, 5}, 0, 16, 5)
	if !empty.Empty() {
		t.Errorf("expected empty ROI, got %+v", empty)
	}
}

func TestComputeConfidence(t *testing.T) {
	const w, h, disp = 64, 24, 8
	const rx, ry, rw, rh = 24, 4, 36, 16
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)
	vol := MatchingCostVolume(left, right, 0, 16, 5, CostSAD)
	conf := ComputeConfidence(vol)
	if conf.Rows != h || conf.Cols != w {
		t.Fatalf("bad confidence shape")
	}
	// Confident textured region should exceed the flat inset's confidence.
	texConf := int(conf.Data[(ry+9)*w+(rx+14)])
	flatConf := int(conf.Data[(ry+3)*w+(rx+3)])
	if texConf <= flatConf {
		t.Errorf("expected higher confidence in textured region: tex=%d flat=%d", texConf, flatConf)
	}
}

func TestRefineSubpixel(t *testing.T) {
	const w, h, disp = 64, 24, 8
	const rx, ry, rw, rh = 24, 4, 36, 16
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)
	vol := MatchingCostVolume(left, right, 0, 16, 5, CostSAD)
	d := vol.WinnerTakeAll()
	df := RefineSubpixel(d, vol)
	if df.Rows != h || df.Cols != w {
		t.Fatalf("bad refined shape")
	}
	// Interior region should be near the true disparity.
	within, total := 0, 0
	for y := ry + 8; y < ry+rh-2; y++ {
		for x := rx + 8; x < rx+rw-2; x++ {
			v := df.At(y, x)
			if v == InvalidDisparityF {
				continue
			}
			total++
			if v >= disp-1 && v <= disp+1 {
				within++
			}
		}
	}
	if total == 0 || float64(within)/float64(total) < 0.5 {
		t.Fatalf("refined disparities off: %d/%d", within, total)
	}
}

func TestDisparityWLSFilterFillsHole(t *testing.T) {
	const w, h = 20, 20
	disp := cv.NewMat(h, w, 1)
	guide := cv.NewMat(h, w, 1)
	for i := range disp.Data {
		disp.Data[i] = 12
		guide.Data[i] = 100
	}
	// Punch an invalid hole.
	for y := 8; y < 12; y++ {
		for x := 8; x < 12; x++ {
			disp.Data[y*w+x] = InvalidDisparity
		}
	}
	out := DisparityWLSFilter{Lambda: 2, SigmaColor: 10, Iterations: 80}.Filter(disp, guide)
	// The hole must be filled close to the surrounding disparity.
	got := int(out.Data[10*w+10])
	if got < 10 || got > 13 {
		t.Errorf("WLS did not fill hole to ~12, got %d", got)
	}
}

func TestDisparityWLSFilterWithConfidence(t *testing.T) {
	const w, h = 16, 16
	disp := cv.NewMat(h, w, 1)
	guide := cv.NewMat(h, w, 1)
	for i := range disp.Data {
		disp.Data[i] = 9
		guide.Data[i] = 80
	}
	vol := MatchingCostVolume(disp, disp, 0, 8, 3, CostSAD) // self-match, high confidence
	conf := ComputeConfidence(vol)
	out := DisparityWLSFilter{Lambda: 1, SigmaColor: 12}.FilterWithConfidence(disp, conf, guide)
	if out.Rows != h || out.Cols != w {
		t.Fatal("bad WLS output shape")
	}
}

func TestPrefilters(t *testing.T) {
	const w, h = 12, 8
	m := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.Data[y*w+x] = uint8((x * 20) % 256)
		}
	}
	xs := ApplyXSobelPrefilter(m, 31)
	if xs.Rows != h || xs.Cols != w {
		t.Fatal("xsobel bad shape")
	}
	// A flat column-constant region has no vertical edges; a linear horizontal
	// ramp yields a constant positive derivative -> above the cap centre (31).
	if int(xs.Data[3*w+5]) <= 31 {
		t.Errorf("expected positive x-gradient response, got %d", xs.Data[3*w+5])
	}
	nr := ApplyNormalizedPrefilter(m, 5, 31)
	if nr.Rows != h || nr.Cols != w {
		t.Fatal("normalized bad shape")
	}
}

func TestBlockMatcherPipeline(t *testing.T) {
	const w, h, disp = 72, 40, 8
	const rx, ry, rw, rh = 28, 8, 36, 26
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)
	bm := BlockMatcher{
		NumDisparities:    16,
		BlockSize:         7,
		PreFilterType:     PrefilterXSobel,
		PreFilterCap:      31,
		SpeckleWindowSize: 20,
		SpeckleRange:      2,
		Disp12MaxDiff:     2,
	}
	d := bm.Compute(left, right)
	m, frac := regionMode(d, rx, ry, rw, rh)
	if m < disp-1 || m > disp+1 {
		t.Errorf("BlockMatcher disparity %d, want %d ±1", m, disp)
	}
	if frac < 0.3 {
		t.Errorf("BlockMatcher valid fraction %.2f too low", frac)
	}
	// Left border band must be invalid.
	for y := 0; y < h; y++ {
		for x := 0; x < 14; x++ {
			if d.Data[y*w+x] != InvalidDisparity {
				t.Fatalf("border not invalid at (%d,%d): %d", x, y, d.Data[y*w+x])
			}
		}
	}
}

func TestQuasiDenseStereo(t *testing.T) {
	const w, h, disp = 72, 40, 6
	const rx, ry, rw, rh = 28, 8, 36, 26
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)
	q := QuasiDenseStereo{NumDisparities: 16, CorrWinSize: 5, CorrThreshold: 0.6, DisparityGradient: 2}
	d := q.Process(left, right)
	if d.Rows != h || d.Cols != w {
		t.Fatalf("bad quasidense shape")
	}
	m, frac := regionMode(d, rx, ry, rw, rh)
	if m < disp-1 || m > disp+1 {
		t.Errorf("QuasiDenseStereo disparity %d, want %d ±1", m, disp)
	}
	if frac < 0.3 {
		t.Errorf("QuasiDenseStereo coverage %.2f too low", frac)
	}
}

func TestDeterminism(t *testing.T) {
	const w, h, disp = 64, 32, 8
	const rx, ry, rw, rh = 24, 6, 36, 22
	left, right := syntheticPair(w, h, disp, rx, ry, rw, rh)
	a := StereoSGM{NumDisparities: 16, BlockSize: 5, Mode: ModeHH}.Compute(left, right)
	b := StereoSGM{NumDisparities: 16, BlockSize: 5, Mode: ModeHH}.Compute(left, right)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("StereoSGM not deterministic at %d: %d vs %d", i, a.Data[i], b.Data[i])
		}
	}
	qa := QuasiDenseStereo{NumDisparities: 16}.Process(left, right)
	qb := QuasiDenseStereo{NumDisparities: 16}.Process(left, right)
	for i := range qa.Data {
		if qa.Data[i] != qb.Data[i] {
			t.Fatalf("QuasiDenseStereo not deterministic at %d", i)
		}
	}
}

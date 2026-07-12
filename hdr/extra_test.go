package hdr

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// texturedImage builds a deterministic, non-periodic multi-scale value-noise
// texture from a seeded RNG, suitable for exercising the MTB aligner (which
// needs unambiguous structure at every pyramid level; periodic patterns would
// alias the shift search).
func texturedImage(rows, cols int, seed int64) *cv.Mat {
	rng := rand.New(rand.NewSource(seed))
	acc := make([]float64, rows*cols)
	// Sum several octaves of bilinearly-upsampled random grids, from coarse to
	// near-per-pixel, so structure is unambiguous at every pyramid level.
	for _, grid := range []int{2, 4, 8, 16, 32} {
		const amp = 51.0
		gc := grid + 1
		noise := make([]float64, gc*gc)
		for i := range noise {
			noise[i] = rng.Float64()
		}
		for y := 0; y < rows; y++ {
			gy := float64(y) / float64(rows) * float64(grid)
			y0 := int(gy)
			fy := gy - float64(y0)
			for x := 0; x < cols; x++ {
				gx := float64(x) / float64(cols) * float64(grid)
				x0 := int(gx)
				fx := gx - float64(x0)
				v00 := noise[y0*gc+x0]
				v10 := noise[y0*gc+x0+1]
				v01 := noise[(y0+1)*gc+x0]
				v11 := noise[(y0+1)*gc+x0+1]
				top := v00*(1-fx) + v10*fx
				bot := v01*(1-fx) + v11*fx
				acc[y*cols+x] += amp * (top*(1-fy) + bot*fy)
			}
		}
	}
	m := cv.NewMat(rows, cols, 3)
	for i, v := range acc {
		z := clamp8(v)
		m.Data[i*3+0], m.Data[i*3+1], m.Data[i*3+2] = z, z, z
	}
	return m
}

// --- AlignMTB ----------------------------------------------------------------

// cropScene extracts a rows x cols window from a larger 3-channel scene with its
// top-left corner at (ox, oy). Both frames in the alignment test are fully
// populated crops (no shifted-in black borders), so the recovered shift is the
// crop offset itself.
func cropScene(scene *cv.Mat, oy, ox, rows, cols int) *cv.Mat {
	out := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			si := ((oy+y)*scene.Cols + (ox + x)) * 3
			di := (y*cols + x) * 3
			copy(out.Data[di:di+3], scene.Data[si:si+3])
		}
	}
	return out
}

func TestAlignMTBShiftRecovery(t *testing.T) {
	const margin = 8
	scene := texturedImage(64+2*margin, 64+2*margin, 1)
	a := NewAlignMTB(4)
	// ref is the centred window; src is offset by (dx, dy) within the scene.
	ref := cropScene(scene, margin, margin, 64, 64)
	cases := []struct{ dx, dy int }{{3, 2}, {-4, 1}, {2, -3}, {0, 0}}
	for _, tc := range cases {
		src := cropScene(scene, margin+tc.dy, margin+tc.dx, 64, 64)
		dx, dy := a.CalculateShift(ref, src)
		if abs(dx-tc.dx) > 1 || abs(dy-tc.dy) > 1 {
			t.Fatalf("offset (%d,%d): recovered (%d,%d)", tc.dx, tc.dy, dx, dy)
		}
	}
}

func TestAlignMTBComputeBitmaps(t *testing.T) {
	m := texturedImage(32, 40, 7)
	a := NewAlignMTB(0)
	tb, eb := a.ComputeBitmaps(m)
	if tb.Rows != 32 || tb.Cols != 40 || eb.Rows != 32 || eb.Cols != 40 {
		t.Fatalf("bitmap dims wrong: %dx%d / %dx%d", tb.Rows, tb.Cols, eb.Rows, eb.Cols)
	}
	// Roughly half the pixels are above the median.
	frac := float64(tb.Count()) / float64(32*40)
	if frac < 0.3 || frac > 0.7 {
		t.Fatalf("threshold fraction %.2f not near 0.5", frac)
	}
	// Every set threshold bit implies the pixel differs from the median, so it
	// must also be set in the exclusion bitmap (default ExclusionRange 4).
	if eb.Count() == 0 {
		t.Fatal("exclusion bitmap empty")
	}
}

func TestAlignMTBProcess(t *testing.T) {
	base := texturedImage(48, 48, 3)
	a := NewAlignMTB(4)
	stack := []*cv.Mat{a.Shift(base, 2, -1), base.Clone(), a.Shift(base, -3, 2)}
	aligned := a.Process(stack)
	if len(aligned) != 3 {
		t.Fatalf("expected 3 aligned frames, got %d", len(aligned))
	}
	// The reference (middle) frame is returned unchanged.
	if !bytes.Equal(aligned[1].Data, base.Data) {
		t.Fatal("reference frame altered")
	}
	// Aligned frames should agree with the reference better than the raw frames
	// over the central region (avoiding shifted-in black borders).
	center := func(m *cv.Mat) float64 { return meanAbsDiffCenter(m, base, 6) }
	if center(aligned[0]) >= center(stack[0]) {
		t.Fatalf("frame 0 not better aligned: before=%.2f after=%.2f", center(stack[0]), center(aligned[0]))
	}
	if center(aligned[2]) >= center(stack[2]) {
		t.Fatalf("frame 2 not better aligned: before=%.2f after=%.2f", center(stack[2]), center(aligned[2]))
	}
}

// meanAbsDiffCenter compares two same-size Mats over an inset central window.
func meanAbsDiffCenter(a, b *cv.Mat, inset int) float64 {
	var s float64
	var n int
	for y := inset; y < a.Rows-inset; y++ {
		for x := inset; x < a.Cols-inset; x++ {
			i := (y*a.Cols + x) * a.Channels
			s += math.Abs(float64(a.Data[i]) - float64(b.Data[i]))
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return s / float64(n)
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// meanDarkQuarter returns the mean sample over the darkest (leftmost) quarter of
// the columns, where the high-range test scene has its deepest shadows.
func meanDarkQuarter(m *cv.Mat) float64 {
	w := m.Cols / 4
	if w < 1 {
		w = 1
	}
	var s float64
	var n int
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < w; x++ {
			for c := 0; c < m.Channels; c++ {
				s += float64(m.Data[(y*m.Cols+x)*m.Channels+c])
				n++
			}
		}
	}
	return s / float64(n)
}

// --- Robust merge ------------------------------------------------------------

func TestMergeRobertsonAndFuncRecoverRatios(t *testing.T) {
	rows, cols := 24, 24
	imgs, times := syntheticBracket(rows, cols)
	resp, err := CalibrateRobertson(imgs, times, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	rad, err := MergeRobertson(imgs, times, resp)
	if err != nil {
		t.Fatal(err)
	}
	meanCol := func(r *Radiance, x int) float64 {
		var s float64
		for y := 0; y < rows; y++ {
			s += r.At(y, x, 0)
		}
		return s / float64(rows)
	}
	c1, c2 := 6, 16
	recRatio := meanCol(rad, c2) / meanCol(rad, c1)
	trueRatio := sceneRadiance(c2, cols) / sceneRadiance(c1, cols)
	if relErr := math.Abs(recRatio-trueRatio) / trueRatio; relErr > 0.2 {
		t.Fatalf("robertson ratio %.3f vs true %.3f (relErr %.3f)", recRatio, trueRatio, relErr)
	}

	// MergeDebevecFunc with a nil weight equals MergeDebevec.
	respD, _ := CalibrateDebevec(imgs, times, 0, 0)
	base, _ := MergeDebevec(imgs, times, respD)
	same, err := MergeDebevecFunc(imgs, times, respD, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range base.Data {
		if math.Abs(base.Data[i]-same.Data[i]) > 1e-9 {
			t.Fatalf("nil-weight MergeDebevecFunc differs at %d: %v vs %v", i, base.Data[i], same.Data[i])
		}
	}
	// Gaussian weighting still produces finite positive radiance.
	gw, err := MergeDebevecFunc(imgs, times, respD, GaussianWeight)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range gw.Data {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("gaussian-weighted radiance invalid: %v", v)
		}
	}

	if _, err := MergeRobertson(imgs, times, nil); err == nil {
		t.Fatal("expected nil-response error")
	}
	if _, err := MergeRobertson(imgs, times, LinearResponse(1)); err == nil {
		t.Fatal("expected channel-mismatch error")
	}
}

func TestWeightFuncs(t *testing.T) {
	// Clipping points are strongly down-weighted relative to mid-grey.
	for name, w := range map[string]WeightFunc{
		"hat": HatWeight, "tent": TentWeight, "gaussian": GaussianWeight,
	} {
		if w(0) >= w(127) || w(255) >= w(127) {
			t.Fatalf("%s: extremes not down-weighted (w0=%.3f w127=%.3f w255=%.3f)",
				name, w(0), w(127), w(255))
		}
	}
	if TentWeight(0) != 0 || TentWeight(255) != 0 {
		t.Fatalf("tent should reject clipping points")
	}
	if UniformWeight(0) != 0 || UniformWeight(255) != 0 || UniformWeight(100) != 1 {
		t.Fatalf("uniform weight wrong")
	}
}

func TestMergeMertensProcessor(t *testing.T) {
	imgs, times := syntheticBracket(24, 24)
	mp := NewMergeMertensProcessor(NewMergeMertensParams())
	a, err := mp.Process(imgs)
	if err != nil {
		t.Fatal(err)
	}
	b, err := mp.ProcessWithExposures(imgs, times)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Data, b.Data) {
		t.Fatal("exposure times should not change Mertens fusion")
	}
	if _, err := mp.ProcessWithExposures(imgs, times[:2]); err == nil {
		t.Fatal("expected length-mismatch error")
	}
}

// --- Response accessors ------------------------------------------------------

func TestResponseAccessors(t *testing.T) {
	imgs, times := syntheticBracket(24, 24)
	resp, err := CalibrateDebevec(imgs, times, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Response(0, 200) <= resp.Response(0, 40) {
		t.Fatal("response not increasing")
	}
	cc := resp.ChannelCurve(0)
	cc[0] = -999 // mutating the copy must not affect the response
	if resp.Curve[0][0] == -999 {
		t.Fatal("ChannelCurve returned an alias")
	}
	lg := resp.LogResponse(0)
	if math.Abs(lg[128]-math.Log(resp.Curve[0][128])) > 1e-9 {
		t.Fatal("LogResponse mismatch")
	}

	// A deliberately broken curve is detected and repaired.
	bad := LinearResponse(1)
	bad.Curve[0][100] = bad.Curve[0][110] + 5 // introduce a dip
	if bad.IsMonotonic(0) {
		t.Fatal("expected non-monotonic detection")
	}
	bad.EnforceMonotonic()
	if !bad.IsMonotonic(1e-9) {
		t.Fatal("EnforceMonotonic did not repair the curve")
	}

	norm := LinearResponse(3)
	norm.Normalize()
	for c := 0; c < 3; c++ {
		if math.Abs(norm.Curve[c][128]-1) > 1e-9 {
			t.Fatalf("Normalize channel %d: entry128=%v", c, norm.Curve[c][128])
		}
	}
}

// --- Radiance stats ----------------------------------------------------------

func TestRadianceStats(t *testing.T) {
	r := highRangeRadiance(16, 32)
	mn, mx := r.MinMax()
	if mn <= 0 || mx <= mn {
		t.Fatalf("MinMax wrong: %v..%v", mn, mx)
	}
	if r.Mean() <= 0 {
		t.Fatalf("Mean should be positive, got %v", r.Mean())
	}
	if dr := r.DynamicRange(); dr < 10 {
		t.Fatalf("dynamic range %.2f stops too small for e^-4..e^4 scene", dr)
	}
	if la := r.LogAverageLuminance(); la <= 0 {
		t.Fatalf("log-average luminance non-positive: %v", la)
	}
	lm := r.LuminanceFloatMat()
	if lm.Rows != 16 || lm.Cols != 32 {
		t.Fatal("luminance float mat wrong size")
	}
	scaled := r.Scale(2)
	if math.Abs(scaled.Data[10]-2*r.Data[10]) > 1e-9 {
		t.Fatal("Scale wrong")
	}
	norm := r.Normalized()
	_, nmax := norm.MinMax()
	if math.Abs(nmax-1) > 1e-9 {
		t.Fatalf("Normalized max should be 1, got %v", nmax)
	}
}

// --- Radiance I/O ------------------------------------------------------------

func TestPFMRoundTrip(t *testing.T) {
	r := highRangeRadiance(5, 7)
	var buf bytes.Buffer
	if err := WritePFM(&buf, r); err != nil {
		t.Fatal(err)
	}
	got, err := ReadPFM(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got.Rows != r.Rows || got.Cols != r.Cols || got.Channels != r.Channels {
		t.Fatalf("PFM dims changed: %dx%dx%d", got.Rows, got.Cols, got.Channels)
	}
	for i := range r.Data {
		if math.Abs(got.Data[i]-r.Data[i]) > 1e-5*math.Abs(r.Data[i])+1e-6 {
			t.Fatalf("PFM sample %d: got %v want %v", i, got.Data[i], r.Data[i])
		}
	}
	// Single-channel path.
	g := NewRadiance(3, 3, 1)
	for i := range g.Data {
		g.Data[i] = float64(i) * 0.5
	}
	var gb bytes.Buffer
	if err := WritePFM(&gb, g); err != nil {
		t.Fatal(err)
	}
	g2, err := ReadPFM(&gb)
	if err != nil {
		t.Fatal(err)
	}
	if g2.Channels != 1 {
		t.Fatalf("expected 1 channel, got %d", g2.Channels)
	}
}

func TestRGBERoundTrip(t *testing.T) {
	r := highRangeRadiance(6, 9)
	var buf bytes.Buffer
	if err := WriteHDR(&buf, r); err != nil {
		t.Fatal(err)
	}
	got, err := ReadHDR(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got.Rows != r.Rows || got.Cols != r.Cols || got.Channels != 3 {
		t.Fatalf("RGBE dims changed: %dx%dx%d", got.Rows, got.Cols, got.Channels)
	}
	// RGBE has an 8-bit mantissa; allow a few percent relative error.
	for i := range r.Data {
		want := r.Data[i]
		if math.Abs(got.Data[i]-want) > 0.03*math.Abs(want)+1e-4 {
			t.Fatalf("RGBE sample %d: got %v want %v", i, got.Data[i], want)
		}
	}
	if err := WriteHDR(&buf, NewRadiance(2, 2, 1)); err == nil {
		t.Fatal("expected error for single-channel RGBE")
	}
}

// --- Colormap ----------------------------------------------------------------

func TestApplyColorMap(t *testing.T) {
	f := cv.NewFloatMat(4, 8)
	for i := range f.Data {
		f.Data[i] = float64(i)
	}
	for _, cm := range []ColorMap{ColorMapGray, ColorMapJet, ColorMapHot, ColorMapInferno} {
		out := ApplyColorMap(f, cm)
		if out.Channels != 3 || out.Rows != 4 || out.Cols != 8 {
			t.Fatalf("colormap %d bad output dims", cm)
		}
	}
	// Grayscale endpoints map to black and white.
	gray := ApplyColorMap(f, ColorMapGray)
	if gray.Data[0] != 0 {
		t.Fatalf("gray low end not black: %d", gray.Data[0])
	}
	last := (len(gray.Data) - 3)
	if gray.Data[last] != 255 {
		t.Fatalf("gray high end not white: %d", gray.Data[last])
	}
	vis := highRangeRadiance(8, 8).Visualize(ColorMapJet)
	if vis.Channels != 3 {
		t.Fatal("visualize should be 3-channel")
	}
}

// --- New tonemappers ---------------------------------------------------------

func TestTonemapDurandAndGradient(t *testing.T) {
	r := highRangeRadiance(32, 40)
	// naive linear crushes the shadows (the dark left columns) to near zero.
	naiveShadow := meanDarkQuarter(naiveLinear(r))
	for name, op := range map[string]Tonemap{
		"durand":   NewTonemapDurand(),
		"gradient": NewTonemapMantiukGradient(),
	} {
		out := op.Process(r)
		if out.Rows != r.Rows || out.Cols != r.Cols || out.Channels != 3 {
			t.Fatalf("%s: bad dims", name)
		}
		s := stdMat(out)
		if s < 8 {
			t.Fatalf("%s: std %.2f too low (no detail)", name, s)
		}
		// The whole point of tone mapping: shadow detail that naive linear
		// scaling crushes is lifted into the visible range.
		if sh := meanDarkQuarter(out); sh <= naiveShadow {
			t.Fatalf("%s: shadows not lifted (%.2f <= naive %.2f)", name, sh, naiveShadow)
		}
	}
	// Grayscale path.
	g := NewRadiance(16, 16, 1)
	for i := range g.Data {
		g.Data[i] = math.Exp(float64(i%16)/15*6 - 3)
	}
	if NewTonemapDurand().Process(g).Channels != 1 {
		t.Fatal("durand grayscale channel mismatch")
	}
	if NewTonemapMantiukGradient().Process(g).Channels != 1 {
		t.Fatal("gradient grayscale channel mismatch")
	}
}

// --- Detail / edge-preserving ------------------------------------------------

func TestDetailEnhanceAndEdgePreserving(t *testing.T) {
	imgs, _ := syntheticBracket(40, 40)
	src := imgs[3]
	enh := DetailEnhance(src, 3, 0.15, 2.5)
	if enh.Rows != src.Rows || enh.Cols != src.Cols || enh.Channels != src.Channels {
		t.Fatal("DetailEnhance changed dims")
	}
	// Detail enhancement should not reduce overall local contrast.
	if stdMat(enh) < stdMat(src)-1 {
		t.Fatalf("DetailEnhance reduced contrast: %.2f -> %.2f", stdMat(src), stdMat(enh))
	}
	smoothed := EdgePreservingFilter(src, 3, 0.1)
	if smoothed.Rows != src.Rows {
		t.Fatal("EdgePreservingFilter changed dims")
	}
	// Edge-preserving smoothing should not increase noise/contrast.
	if stdMat(smoothed) > stdMat(src)+1 {
		t.Fatalf("smoothing increased contrast: %.2f -> %.2f", stdMat(src), stdMat(smoothed))
	}
}

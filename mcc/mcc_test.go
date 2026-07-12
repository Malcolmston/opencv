package mcc_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/mcc"
)

// applyDistortion simulates a camera's color error by mapping each reference
// color through a fixed 3x3 gain (in [0,1] space) plus a small black-level
// offset, returning the "measured" colors in the 0..255 range.
func applyDistortion(ref [][3]float64) [][3]float64 {
	// A deliberately non-identity, mildly cross-channel matrix.
	mat := [3][3]float64{
		{0.90, 0.10, 0.05},
		{0.08, 0.85, 0.07},
		{0.04, 0.12, 0.95},
	}
	off := [3]float64{6, 4, 8}
	out := make([][3]float64, len(ref))
	for i, c := range ref {
		r := c[0] / 255
		g := c[1] / 255
		b := c[2] / 255
		nr := mat[0][0]*r + mat[0][1]*g + mat[0][2]*b
		ng := mat[1][0]*r + mat[1][1]*g + mat[1][2]*b
		nb := mat[2][0]*r + mat[2][1]*g + mat[2][2]*b
		out[i] = [3]float64{
			clamp255(nr*255 + off[0]),
			clamp255(ng*255 + off[1]),
			clamp255(nb*255 + off[2]),
		}
	}
	return out
}

func clamp255(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}

func TestReferenceCharts(t *testing.T) {
	for _, ct := range []mcc.CheckerType{mcc.Macbeth24, mcc.Vinyl} {
		ref := mcc.ReferenceChart(ct)
		if len(ref) != 24 {
			t.Fatalf("%s: got %d patches, want 24", ct, len(ref))
		}
		if ct.Rows() != 4 || ct.Cols() != 6 || ct.NumPatches() != 24 {
			t.Errorf("%s: geometry rows=%d cols=%d n=%d", ct, ct.Rows(), ct.Cols(), ct.NumPatches())
		}
		// White patch (index 18) should be near L*=100 and neutral.
		white := ref[18].Lab
		if white[0] < 95 {
			t.Errorf("%s: white L*=%.1f, want >=95", ct, white[0])
		}
		if math.Abs(white[1]) > 5 || math.Abs(white[2]) > 5 {
			t.Errorf("%s: white a*b* not neutral: %v", ct, white)
		}
		// Black patch (index 23) should have low L*.
		if ref[23].Lab[0] > 30 {
			t.Errorf("%s: black L*=%.1f, want <=30", ct, ref[23].Lab[0])
		}
	}
	// The two tabulations should differ somewhere.
	m := mcc.ReferenceRGB(mcc.Macbeth24)
	v := mcc.ReferenceRGB(mcc.Vinyl)
	diff := false
	for i := range m {
		if m[i] != v[i] {
			diff = true
			break
		}
	}
	if !diff {
		t.Error("Macbeth24 and Vinyl reference tables are identical")
	}
}

func TestReferenceChartCopy(t *testing.T) {
	a := mcc.ReferenceChart(mcc.Macbeth24)
	a[0].RGB[0] = 0
	b := mcc.ReferenceChart(mcc.Macbeth24)
	if b[0].RGB[0] == 0 {
		t.Error("ReferenceChart must return an independent copy")
	}
}

func TestReferenceLab(t *testing.T) {
	lab := mcc.ReferenceLab(mcc.Macbeth24)
	if len(lab) != 24 {
		t.Fatalf("ReferenceLab returned %d entries", len(lab))
	}
	// Must match the Lab carried on the Patch structs.
	ref := mcc.ReferenceChart(mcc.Macbeth24)
	for i := range lab {
		if lab[i] != ref[i].Lab {
			t.Errorf("patch %d Lab mismatch: %v vs %v", i, lab[i], ref[i].Lab)
		}
	}
}

func TestDetectWithHintGrayInput(t *testing.T) {
	// A single-channel image exercises the grayscale->RGB promotion path.
	patch, gap := 24, 6
	rgb := mcc.RenderChart(mcc.Macbeth24, patch, gap)
	gray := cv.CvtColor(rgb, cv.ColorRGB2Gray)
	quad := mcc.ChartOuterQuad(mcc.Macbeth24, patch, gap)
	d := mcc.NewCCheckerDetector(mcc.Macbeth24)
	cc, ok := d.DetectWithHint(gray, quad)
	if !ok {
		t.Fatal("DetectWithHint on gray image failed")
	}
	// Measured colors are neutral (equal channels) because the source is gray.
	m := cc.MeasuredRGB()[10]
	if math.Abs(m[0]-m[1]) > 1 || math.Abs(m[1]-m[2]) > 1 {
		t.Errorf("gray sample should be neutral, got %v", m)
	}
}

func TestCheckerTypeString(t *testing.T) {
	if mcc.Macbeth24.String() != "Macbeth24" || mcc.Vinyl.String() != "Vinyl" {
		t.Error("unexpected CheckerType.String()")
	}
	if mcc.CheckerType(99).String() != "Unknown" {
		t.Error("unexpected string for unknown CheckerType")
	}
}

func TestColorMath(t *testing.T) {
	// sRGB<->linear round trip.
	for _, c := range []float64{0, 0.01, 0.04, 0.2, 0.5, 0.9, 1} {
		got := mcc.LinearToSRGB(mcc.SRGBToLinear(c))
		if math.Abs(got-c) > 1e-9 {
			t.Errorf("sRGB round trip %.3f -> %.9f", c, got)
		}
	}
	// White maps to L*~100, a*~0, b*~0.
	w := mcc.RGBToLab(255, 255, 255)
	if math.Abs(w[0]-100) > 0.5 || math.Abs(w[1]) > 1 || math.Abs(w[2]) > 1 {
		t.Errorf("white Lab=%v", w)
	}
	// Black maps to L*~0.
	bl := mcc.RGBToLab(0, 0, 0)
	if math.Abs(bl[0]) > 1e-6 {
		t.Errorf("black L*=%.4f", bl[0])
	}
	// Delta E of identical colors is zero, of different colors positive.
	if mcc.DeltaERGB([3]uint8{100, 120, 130}, [3]uint8{100, 120, 130}) != 0 {
		t.Error("Delta E of identical colors must be 0")
	}
	if mcc.DeltaERGB([3]uint8{255, 0, 0}, [3]uint8{0, 255, 0}) < 50 {
		t.Error("Delta E of red vs green should be large")
	}
	// XYZ of white ~ D65 white point (Y=1).
	xyz := mcc.RGBToXYZ(255, 255, 255)
	if math.Abs(xyz[1]-1) > 1e-3 {
		t.Errorf("white Y=%.4f, want ~1", xyz[1])
	}
}

func TestRenderChartDimensions(t *testing.T) {
	img := mcc.RenderChart(mcc.Macbeth24, 20, 6)
	wantW := 6*20 + 7*6
	wantH := 4*20 + 5*6
	if img.Cols != wantW || img.Rows != wantH {
		t.Fatalf("rendered size %dx%d, want %dx%d", img.Cols, img.Rows, wantW, wantH)
	}
	if img.Channels != 3 {
		t.Fatalf("rendered channels=%d, want 3", img.Channels)
	}
	// The centre of patch (0,0) should equal its reference color.
	ref := mcc.ReferenceChart(mcc.Macbeth24)
	cx := 6 + 10
	cy := 6 + 10
	px := img.AtPixel(cy, cx)
	if px[0] != ref[0].RGB[0] || px[1] != ref[0].RGB[1] || px[2] != ref[0].RGB[2] {
		t.Errorf("patch(0,0) centre=%v, want %v", px, ref[0].RGB)
	}
	// A gap pixel should be black.
	g := img.AtPixel(2, 2)
	if g[0] != 0 || g[1] != 0 || g[2] != 0 {
		t.Errorf("gap pixel=%v, want black", g)
	}
}

func TestRenderChartPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("RenderChart should panic on non-positive params")
		}
	}()
	mcc.RenderChart(mcc.Macbeth24, 0, 4)
}

func TestDetectWithHint(t *testing.T) {
	patch, gap := 30, 8
	img := mcc.RenderChart(mcc.Macbeth24, patch, gap)
	quad := mcc.ChartOuterQuad(mcc.Macbeth24, patch, gap)
	d := mcc.NewCCheckerDetector(mcc.Macbeth24)
	cc, ok := d.DetectWithHint(img, quad)
	if !ok {
		t.Fatal("DetectWithHint failed")
	}
	if cc.MeanError() > 3 {
		t.Errorf("mean Delta E=%.2f, want <=3", cc.MeanError())
	}
	if cc.MaxError() > 8 {
		t.Errorf("max Delta E=%.2f, want <=8", cc.MaxError())
	}
	if len(cc.MeasuredRGB()) != 24 {
		t.Errorf("measured %d patches, want 24", len(cc.MeasuredRGB()))
	}
	if len(cc.PatchErrors()) != 24 {
		t.Errorf("PatchErrors returned %d values", len(cc.PatchErrors()))
	}
	if cc.Type != mcc.Macbeth24 {
		t.Error("wrong checker type recorded")
	}
	if len(cc.Reference()) != 24 {
		t.Error("Reference() should return 24 patches")
	}
}

func TestDetectAutomatic(t *testing.T) {
	patch, gap := 30, 8
	img := mcc.RenderChart(mcc.Macbeth24, patch, gap)
	d := mcc.NewCCheckerDetector(mcc.Macbeth24)
	cc, ok := d.Detect(img)
	if !ok {
		t.Fatal("automatic Detect failed on a clean rendered chart")
	}
	if cc.MeanError() > 4 {
		t.Errorf("automatic mean Delta E=%.2f, want <=4", cc.MeanError())
	}
	// Orientation must be recovered: patch 0 should read as dark skin, not as
	// the white or black neutral of an unrelated corner.
	ref := mcc.ReferenceChart(mcc.Macbeth24)
	m0 := cc.MeasuredRGB()[0]
	if mcc.DeltaE76(mcc.RGBToLab(uint8(m0[0]), uint8(m0[1]), uint8(m0[2])), ref[0].Lab) > 6 {
		t.Errorf("patch 0 mis-oriented: measured %v vs dark-skin ref %v", m0, ref[0].RGB)
	}
}

func TestDetectAutomaticWarped(t *testing.T) {
	patch, gap := 34, 10
	img := mcc.RenderChart(mcc.Macbeth24, patch, gap)
	// Warp the chart by a known mild perspective onto a larger black canvas.
	src := [4]cv.Point{
		{X: 0, Y: 0},
		{X: img.Cols - 1, Y: 0},
		{X: img.Cols - 1, Y: img.Rows - 1},
		{X: 0, Y: img.Rows - 1},
	}
	W, H := img.Cols+120, img.Rows+120
	dst := [4]cv.Point{
		{X: 40, Y: 25},
		{X: W - 20, Y: 45},
		{X: W - 45, Y: H - 20},
		{X: 25, Y: H - 40},
	}
	hmat := cv.GetPerspectiveTransform(src, dst)
	warped := cv.WarpPerspective(img, hmat, W, H, cv.InterLinear)

	d := mcc.NewCCheckerDetector(mcc.Macbeth24)
	cc, ok := d.Detect(warped)
	if !ok {
		t.Fatal("automatic Detect failed on a warped chart")
	}
	if cc.MeanError() > 6 {
		t.Errorf("warped mean Delta E=%.2f, want <=6", cc.MeanError())
	}
}

func TestDetectEmpty(t *testing.T) {
	d := mcc.NewCCheckerDetector(mcc.Macbeth24)
	if _, ok := d.Detect(nil); ok {
		t.Error("Detect(nil) should fail")
	}
	if _, ok := d.DetectWithHint(nil, [4]cv.Point{}); ok {
		t.Error("DetectWithHint(nil) should fail")
	}
	// An image with no chart-like structure should not detect.
	blank := cv.NewMat(60, 60, 3)
	blank.SetTo(120)
	if _, ok := d.Detect(blank); ok {
		t.Error("Detect on a flat image should fail")
	}
}

func TestCCMReducesError(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := applyDistortion(ref)
	before := mcc.MeanDeltaE(measured, ref)
	if before < 2 {
		t.Fatalf("distortion too weak, before Delta E=%.2f", before)
	}

	types := []struct {
		name string
		cfg  mcc.CCMConfig
	}{
		{"linear3x3", mcc.CCMConfig{Type: mcc.CCMLinear3x3}},
		{"affine3x4", mcc.CCMConfig{Type: mcc.CCMAffine3x4}},
		{"affine-linearized", mcc.CCMConfig{Type: mcc.CCMAffine3x4, Linearize: true}},
		{"poly", mcc.CCMConfig{Type: mcc.CCMPolynomial}},
		{"poly-gamma", mcc.CCMConfig{Type: mcc.CCMPolynomial, Linearize: true, Gamma: 2.2}},
	}
	for _, tc := range types {
		model, err := mcc.TrainCCM(measured, ref, tc.cfg)
		if err != nil {
			t.Fatalf("%s: TrainCCM error: %v", tc.name, err)
		}
		after := model.MeanError(measured, ref)
		if after >= before {
			t.Errorf("%s: after Delta E=%.2f not < before %.2f", tc.name, after, before)
		}
		if after > 4 {
			t.Errorf("%s: residual Delta E=%.2f, want <=4", tc.name, after)
		}
	}
}

func TestCCMApplyImage(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := applyDistortion(ref)
	model, err := mcc.TrainCCM(measured, ref, mcc.CCMConfig{Type: mcc.CCMAffine3x4})
	if err != nil {
		t.Fatal(err)
	}
	// Build a distorted chart image and correct it, then re-detect and check the
	// corrected patches are close to reference.
	dist := cv.NewMat(4, 6, 3)
	for r := 0; r < 4; r++ {
		for c := 0; c < 6; c++ {
			m := measured[r*6+c]
			dist.SetPixel(r, c, []uint8{uint8(m[0]), uint8(m[1]), uint8(m[2])})
		}
	}
	corrected := model.Apply(dist)
	if corrected.Rows != 4 || corrected.Cols != 6 || corrected.Channels != 3 {
		t.Fatalf("corrected size wrong: %dx%dx%d", corrected.Rows, corrected.Cols, corrected.Channels)
	}
	var got [][3]float64
	for r := 0; r < 4; r++ {
		for c := 0; c < 6; c++ {
			p := corrected.AtPixel(r, c)
			got = append(got, [3]float64{float64(p[0]), float64(p[1]), float64(p[2])})
		}
	}
	if e := mcc.MeanDeltaE(got, ref); e > 3 {
		t.Errorf("corrected image mean Delta E=%.2f, want <=3", e)
	}
	// Matrix has the expected shape.
	if m := model.Matrix(); len(m) != 4 {
		t.Errorf("affine matrix has %d rows, want 4", len(m))
	}
	if model.Type() != mcc.CCMAffine3x4 {
		t.Error("wrong model type reported")
	}
}

func TestCCMApplyPanicsOnGray(t *testing.T) {
	model, _ := mcc.TrainCCM(mcc.ReferenceRGB(mcc.Macbeth24), mcc.ReferenceRGB(mcc.Macbeth24), mcc.CCMConfig{})
	defer func() {
		if recover() == nil {
			t.Error("Apply on a 1-channel image should panic")
		}
	}()
	model.Apply(cv.NewMat(3, 3, 1))
}

func TestCCMIdentityFit(t *testing.T) {
	// Fitting reference->reference should give near-zero residual.
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	model, err := mcc.TrainCCM(ref, ref, mcc.CCMConfig{Type: mcc.CCMLinear3x3})
	if err != nil {
		t.Fatal(err)
	}
	if e := model.MeanError(ref, ref); e > 2 {
		t.Errorf("identity fit residual Delta E=%.2f, want small", e)
	}
}

func TestTrainCCMErrors(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	if _, err := mcc.TrainCCM(ref, ref[:10], mcc.CCMConfig{}); err == nil {
		t.Error("length mismatch should error")
	}
	if _, err := mcc.TrainCCM(ref[:2], ref[:2], mcc.CCMConfig{Type: mcc.CCMPolynomial}); err == nil {
		t.Error("too few samples should error")
	}
	if _, err := mcc.TrainCCM(ref, ref, mcc.CCMConfig{Type: mcc.CCMType(99)}); err == nil {
		t.Error("unknown type should error")
	}
}

func TestEndToEnd(t *testing.T) {
	// Render a chart, distort its colors, detect, train a CCM from the detection,
	// and confirm the CCM lowers the detection's error toward the reference.
	patch, gap := 30, 8
	clean := mcc.RenderChart(mcc.Macbeth24, patch, gap)
	// Distort every pixel by a fixed camera color model.
	distorted := distortImage(clean)

	d := mcc.NewCCheckerDetector(mcc.Macbeth24)
	cc, ok := d.Detect(distorted)
	if !ok {
		t.Fatal("detection on distorted chart failed")
	}
	measured := cc.MeasuredRGB()
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	before := mcc.MeanDeltaE(measured, ref)

	model, err := mcc.TrainCCM(measured, ref, mcc.CCMConfig{Type: mcc.CCMAffine3x4, Linearize: true})
	if err != nil {
		t.Fatal(err)
	}
	after := model.MeanError(measured, ref)
	if after >= before {
		t.Errorf("CCM did not help: before=%.2f after=%.2f", before, after)
	}
}

// distortImage applies a fixed per-pixel color distortion to an RGB image,
// simulating a miscalibrated camera. Used by the end-to-end test.
func distortImage(img *cv.Mat) *cv.Mat {
	out := cv.NewMat(img.Rows, img.Cols, 3)
	for i := 0; i < img.Total(); i++ {
		b := i * 3
		r := float64(img.Data[b+0]) / 255
		g := float64(img.Data[b+1]) / 255
		bl := float64(img.Data[b+2]) / 255
		nr := 0.90*r + 0.10*g + 0.05*bl
		ng := 0.08*r + 0.85*g + 0.07*bl
		nb := 0.04*r + 0.12*g + 0.95*bl
		out.Data[b+0] = clampU8(nr*255 + 6)
		out.Data[b+1] = clampU8(ng*255 + 4)
		out.Data[b+2] = clampU8(nb*255 + 8)
	}
	return out
}

func clampU8(v float64) uint8 {
	v = math.Round(v)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

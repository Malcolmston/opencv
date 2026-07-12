package mcc_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/mcc"
)

func TestDigitalSGGeometry(t *testing.T) {
	if mcc.DigitalSGRows() != 10 || mcc.DigitalSGCols() != 14 {
		t.Errorf("SG geometry rows=%d cols=%d, want 10x14", mcc.DigitalSGRows(), mcc.DigitalSGCols())
	}
	if mcc.DigitalSGNumPatches() != 140 {
		t.Errorf("SG patches=%d, want 140", mcc.DigitalSGNumPatches())
	}
	ref := mcc.DigitalSGReference()
	if len(ref) != 140 {
		t.Fatalf("DigitalSGReference len=%d, want 140", len(ref))
	}
	if len(mcc.DigitalSGReferenceRGB()) != 140 || len(mcc.DigitalSGReferenceLab()) != 140 {
		t.Error("SG reference RGB/Lab lengths must be 140")
	}
	// Every Lab must be consistent with its RGB.
	for i, p := range ref {
		if p.Lab != mcc.RGBToLab(p.RGB[0], p.RGB[1], p.RGB[2]) {
			t.Errorf("patch %d Lab inconsistent with RGB", i)
		}
	}
}

func TestDigitalSGStructure(t *testing.T) {
	ref := mcc.DigitalSGReference()
	// The four physical corners are white.
	corners := []int{0, 13, 9 * 14, 10*14 - 1}
	for _, idx := range corners {
		p := ref[idx]
		if p.RGB[0] < 240 || p.RGB[1] < 240 || p.RGB[2] < 240 {
			t.Errorf("corner %d not white: %v", idx, p.RGB)
		}
	}
	// The border frame patches are neutral (equal channels).
	for r := 0; r < 10; r++ {
		for c := 0; c < 14; c++ {
			if r == 0 || r == 9 || c == 0 || c == 13 {
				p := ref[r*14+c]
				if p.RGB[0] != p.RGB[1] || p.RGB[1] != p.RGB[2] {
					t.Errorf("frame patch (%d,%d) not neutral: %v", r, c, p.RGB)
				}
			}
		}
	}
	// The embedded classic 24-patch block (rows 3..6, cols 4..9) must match the
	// Macbeth24 reference exactly.
	macbeth := mcc.ReferenceChart(mcc.Macbeth24)
	for i := 0; i < 24; i++ {
		rr := 3 + i/6
		cc := 4 + i%6
		if ref[rr*14+cc].RGB != macbeth[i].RGB {
			t.Errorf("embedded classic patch %d = %v, want %v", i, ref[rr*14+cc].RGB, macbeth[i].RGB)
		}
	}
}

func TestDigitalSGReferenceCopy(t *testing.T) {
	a := mcc.DigitalSGReference()
	a[0].RGB[0] = 0
	b := mcc.DigitalSGReference()
	if b[0].RGB[0] == 0 {
		t.Error("DigitalSGReference must return an independent copy")
	}
}

func TestRenderSGChartDimensions(t *testing.T) {
	patch, gap := 12, 4
	img := mcc.RenderSGChart(patch, gap)
	wantW := 14*patch + 15*gap
	wantH := 10*patch + 11*gap
	if img.Cols != wantW || img.Rows != wantH {
		t.Fatalf("SG render size %dx%d, want %dx%d", img.Cols, img.Rows, wantW, wantH)
	}
	if img.Channels != 3 {
		t.Fatalf("SG render channels=%d, want 3", img.Channels)
	}
	// Centre of patch (0,0) equals its reference color.
	ref := mcc.DigitalSGReference()
	px := img.AtPixel(gap+patch/2, gap+patch/2)
	if px[0] != ref[0].RGB[0] || px[1] != ref[0].RGB[1] || px[2] != ref[0].RGB[2] {
		t.Errorf("SG patch(0,0) centre=%v, want %v", px, ref[0].RGB)
	}
}

func TestRenderSGChartPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("RenderSGChart should panic on non-positive params")
		}
	}()
	mcc.RenderSGChart(10, 0)
}

func TestSampleSGWithHint(t *testing.T) {
	patch, gap := 14, 5
	img := mcc.RenderSGChart(patch, gap)
	quad := mcc.DigitalSGOuterQuad(patch, gap)
	cc, ok := mcc.SampleSGWithHint(img, quad)
	if !ok {
		t.Fatal("SampleSGWithHint failed")
	}
	if len(cc.MeasuredRGB()) != 140 {
		t.Fatalf("measured %d patches, want 140", len(cc.MeasuredRGB()))
	}
	if len(cc.Reference()) != 140 {
		t.Error("Reference() should return 140 patches")
	}
	if cc.MeanError() > 2 {
		t.Errorf("SG sampled mean dE76=%.2f, want <=2", cc.MeanError())
	}
	if cc.MaxError() > 8 {
		t.Errorf("SG sampled max dE76=%.2f, want <=8", cc.MaxError())
	}
	if len(cc.PatchErrors()) != 140 {
		t.Errorf("PatchErrors len=%d, want 140", len(cc.PatchErrors()))
	}
}

func TestSampleSGWithHintWarped(t *testing.T) {
	patch, gap := 16, 6
	img := mcc.RenderSGChart(patch, gap)
	src := [4]cv.Point{
		{X: 0, Y: 0},
		{X: img.Cols - 1, Y: 0},
		{X: img.Cols - 1, Y: img.Rows - 1},
		{X: 0, Y: img.Rows - 1},
	}
	W, H := img.Cols+140, img.Rows+140
	dst := [4]cv.Point{
		{X: 45, Y: 30},
		{X: W - 25, Y: 50},
		{X: W - 50, Y: H - 25},
		{X: 30, Y: H - 45},
	}
	hmat := cv.GetPerspectiveTransform(src, dst)
	warped := cv.WarpPerspective(img, hmat, W, H, cv.InterLinear)

	cc, ok := mcc.SampleSGWithHint(warped, dst)
	if !ok {
		t.Fatal("SampleSGWithHint on warped chart failed")
	}
	if cc.MeanError() > 6 {
		t.Errorf("warped SG mean dE76=%.2f, want <=6", cc.MeanError())
	}
}

func TestSampleSGWithHintEmpty(t *testing.T) {
	if _, ok := mcc.SampleSGWithHint(nil, [4]cv.Point{}); ok {
		t.Error("SampleSGWithHint(nil) should fail")
	}
}

func TestDetectorParameters(t *testing.T) {
	p := mcc.NewDetectorParameters()
	if err := p.Validate(); err != nil {
		t.Fatalf("default params invalid: %v", err)
	}
	// Build a detector and confirm it works on a rendered chart.
	patch, gap := 30, 8
	img := mcc.RenderChart(mcc.Macbeth24, patch, gap)
	d := p.NewDetector(mcc.Macbeth24)
	if d.Type != mcc.Macbeth24 {
		t.Error("NewDetector did not carry the chart type")
	}
	if d.MinPatchAreaFrac != p.MinPatchAreaFrac || d.MaxPatchAreaFrac != p.MaxPatchAreaFrac || d.ApproxEpsilonFrac != p.ApproxEpsilonFrac {
		t.Error("NewDetector did not carry the parameters")
	}
	cc, ok := d.Detect(img)
	if !ok || cc.MeanError() > 4 {
		t.Errorf("detector from params failed: ok=%v", ok)
	}
	// Editing the params after building must not affect the detector.
	p.MinPatchAreaFrac = 0.5
	if d.MinPatchAreaFrac == 0.5 {
		t.Error("detector should be independent of later param edits")
	}
	// Validation rejects bad values.
	for _, bad := range []*mcc.DetectorParameters{
		{MinPatchAreaFrac: -1, MaxPatchAreaFrac: 0.2, ApproxEpsilonFrac: 0.08},
		{MinPatchAreaFrac: 0.3, MaxPatchAreaFrac: 0.2, ApproxEpsilonFrac: 0.08},
		{MinPatchAreaFrac: 0.001, MaxPatchAreaFrac: 0.2, ApproxEpsilonFrac: 2},
		{MinPatchAreaFrac: 0.001, MaxPatchAreaFrac: 5, ApproxEpsilonFrac: 0.08},
	} {
		if err := bad.Validate(); err == nil {
			t.Errorf("Validate should reject %+v", bad)
		}
	}
}

func TestSGEndToEndCorrection(t *testing.T) {
	// Sample a distorted SG chart, fit a model from it, and confirm the model
	// lowers the perceptual error toward the reference.
	patch, gap := 12, 4
	clean := mcc.RenderSGChart(patch, gap)
	distorted := cv.NewMat(clean.Rows, clean.Cols, 3)
	for i := 0; i < clean.Total(); i++ {
		b := i * 3
		r := float64(clean.Data[b+0]) / 255
		g := float64(clean.Data[b+1]) / 255
		bl := float64(clean.Data[b+2]) / 255
		distorted.Data[b+0] = u8(255 * (0.9*r + 0.08*g + 0.04*bl))
		distorted.Data[b+1] = u8(255 * (0.06*r + 0.88*g + 0.06*bl))
		distorted.Data[b+2] = u8(255 * (0.03*r + 0.1*g + 0.92*bl))
	}
	quad := mcc.DigitalSGOuterQuad(patch, gap)
	cc, ok := mcc.SampleSGWithHint(distorted, quad)
	if !ok {
		t.Fatal("SG sample on distorted chart failed")
	}
	measured := cc.MeasuredRGB()
	ref := mcc.DigitalSGReferenceRGB()
	before := mcc.MeanDeltaE(measured, ref)
	model, err := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{Model: mcc.ModelAffine})
	if err != nil {
		t.Fatal(err)
	}
	after := model.MeanDeltaE76(measured, ref)
	if after >= before {
		t.Errorf("SG CCM did not help: before=%.2f after=%.2f", before, after)
	}
}

func u8(v float64) uint8 {
	v = math.Round(v)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

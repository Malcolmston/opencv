package bioinspired

import (
	"math"
	"strings"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- Bayer mosaic / demosaic ---

func TestBayerRoundTripOnSmoothImage(t *testing.T) {
	rows, cols := 32, 32
	// Linear (hence bilinear-exact in the interior) RGB planes, all in-range.
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8(40+x/2))
			img.Set(y, x, 1, uint8(50+y/2))
			img.Set(y, x, 2, uint8(60+(x+y)/4))
		}
	}
	for _, pat := range []BayerPattern{BayerRGGB, BayerBGGR, BayerGRBG, BayerGBRG} {
		mosaic := MosaicBayer(img, pat)
		if mosaic.Channels != 1 {
			t.Fatalf("%s: mosaic should be single-channel, got %d", pat, mosaic.Channels)
		}
		back := DemosaicBayer(mosaic, pat)
		if back.Channels != 3 {
			t.Fatalf("%s: demosaic should be 3-channel, got %d", pat, back.Channels)
		}
		var maxErr int
		for y := 2; y < rows-2; y++ {
			for x := 2; x < cols-2; x++ {
				for c := 0; c < 3; c++ {
					d := int(img.At(y, x, c)) - int(back.At(y, x, c))
					if d < 0 {
						d = -d
					}
					if d > maxErr {
						maxErr = d
					}
				}
			}
		}
		if maxErr > 2 {
			t.Errorf("%s: demosaic∘mosaic interior error too large: %d", pat, maxErr)
		}
	}
}

func TestBayerPatternString(t *testing.T) {
	if BayerRGGB.String() != "RGGB" || BayerGBRG.String() != "GBRG" {
		t.Fatal("unexpected pattern names")
	}
	if got := BayerPattern(99).String(); !strings.Contains(got, "99") {
		t.Fatalf("unknown pattern string = %q", got)
	}
}

func TestMosaicBayerRejectsNonRGB(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on non-3-channel input")
		}
	}()
	MosaicBayer(cv.NewMat(4, 4, 1), BayerRGGB)
}

func TestDemosaicBayerRejectsMultiChannel(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on multi-channel input")
		}
	}()
	DemosaicBayer(cv.NewMat(4, 4, 3), BayerRGGB)
}

// --- Transient areas segmentation ---

func TestSegmentationFlagsMovingBlobNotStatic(t *testing.T) {
	rows, cols := 48, 48

	// A static (all-zero transient) sequence must yield an empty segmentation.
	still := NewTransientAreasSegmentationModule(rows, cols)
	zero := cv.NewFloatMat(rows, cols)
	for i := 0; i < 6; i++ {
		still.RunFloat(zero)
	}
	for _, v := range still.GetSegmentationPicture().Data {
		if v != 0 {
			t.Fatal("static frame should produce no segmentation")
		}
	}

	// A compact, persistent transient blob must be flagged.
	moving := NewTransientAreasSegmentationModule(rows, cols)
	blob := cv.NewFloatMat(rows, cols)
	for y := 22; y < 26; y++ {
		for x := 22; x < 26; x++ {
			blob.Data[y*cols+x] = 200
		}
	}
	for i := 0; i < 8; i++ {
		moving.RunFloat(blob)
	}
	mask := moving.GetSegmentationMask()
	if !mask[24*cols+24] {
		t.Error("centre of the moving blob was not segmented")
	}
	if mask[0] || mask[rows*cols-1] {
		t.Error("static corners were wrongly segmented")
	}
	pic := moving.GetSegmentationPicture()
	if pic.At(24, 24, 0) != 255 {
		t.Errorf("segmentation picture centre = %d, want 255", pic.At(24, 24, 0))
	}
}

func TestSegmentationRunMatAndClear(t *testing.T) {
	rows, cols := 24, 24
	m := NewTransientAreasSegmentationModule(rows, cols)
	mat := cv.NewMat(rows, cols, 1)
	for y := 10; y < 14; y++ {
		for x := 10; x < 14; x++ {
			mat.Set(y, x, 0, 220)
		}
	}
	for i := 0; i < 8; i++ {
		m.Run(mat)
	}
	if !m.GetSegmentationMask()[12*cols+12] {
		t.Fatal("expected blob centre flagged via Run(Mat)")
	}
	m.ClearAllBuffers()
	if m.hasOutput {
		t.Fatal("ClearAllBuffers should discard output")
	}
	for _, v := range m.localTemporal.data {
		if v != 0 {
			t.Fatal("ClearAllBuffers should zero energy state")
		}
	}
	ir, ic := m.GetInputSize()
	or, oc := m.GetOutputSize()
	if ir != rows || ic != cols || or != rows || oc != cols {
		t.Fatal("input/output size mismatch")
	}
	if !strings.Contains(m.PrintSetup(), "TransientAreasSegmentationModule") {
		t.Fatal("PrintSetup should describe the module")
	}
}

func TestSegmentationParameterValidation(t *testing.T) {
	bad := DefaultSegmentationParameters()
	bad.LocalEnergySpatialConstant = -1
	if bad.Validate() == nil {
		t.Fatal("negative constant should fail validation")
	}
	bad = DefaultSegmentationParameters()
	bad.ThresholdOFF = bad.ThresholdON + 1
	if bad.Validate() == nil {
		t.Fatal("OFF>ON should fail validation")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic constructing with invalid params")
		}
	}()
	NewTransientAreasSegmentationModuleWithParams(8, 8, bad)
}

func TestSegmentationRunSizeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on size mismatch")
		}
	}()
	NewTransientAreasSegmentationModule(10, 10).Run(cv.NewMat(4, 4, 1))
}

func TestSegmentationGetBeforeRunPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic reading before Run")
		}
	}()
	NewTransientAreasSegmentationModule(8, 8).GetSegmentationPicture()
}

// --- Parameter text round-trip and validation ---

func TestParameterTextRoundTrip(t *testing.T) {
	p := DefaultRetinaParameters()
	p.OPLandIplParvo.ColorMode = false
	p.OPLandIplParvo.HorizontalCellsGain = 0.42
	p.IplMagno.MagnoGain = 7.5
	text := WriteRetinaParameters(p)
	got, err := ReadRetinaParameters(text)
	if err != nil {
		t.Fatalf("round-trip parse failed: %v", err)
	}
	if got != p {
		t.Fatalf("round-trip mismatch:\n got %+v\nwant %+v", got, p)
	}
}

func TestReadParametersDefaultsAndComments(t *testing.T) {
	text := "# only override one field\nIplMagno.MagnoGain 4\n\n"
	got, err := ReadRetinaParameters(text)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.IplMagno.MagnoGain != 4 {
		t.Fatalf("override not applied: %v", got.IplMagno.MagnoGain)
	}
	if got.OPLandIplParvo.HorizontalCellsGain != DefaultRetinaParameters().OPLandIplParvo.HorizontalCellsGain {
		t.Fatal("omitted key should keep its default")
	}
}

func TestReadParametersErrors(t *testing.T) {
	if _, err := ReadRetinaParameters("bogus.key 1"); err == nil {
		t.Fatal("unknown key should error")
	}
	if _, err := ReadRetinaParameters("IplMagno.MagnoGain notanumber"); err == nil {
		t.Fatal("bad number should error")
	}
	if _, err := ReadRetinaParameters("too many fields here"); err == nil {
		t.Fatal("malformed line should error")
	}
	// Out-of-range value must fail validation on read.
	if _, err := ReadRetinaParameters("OPLandIplParvo.GanglionCellsSensitivity 2"); err == nil {
		t.Fatal("out-of-range value should fail validation")
	}
}

func TestParameterValidateRanges(t *testing.T) {
	if DefaultRetinaParameters().Validate() != nil {
		t.Fatal("defaults should validate")
	}
	p := DefaultRetinaParameters()
	p.IplMagno.MagnoGain = -1
	if p.Validate() == nil {
		t.Fatal("negative gain should fail")
	}
	p = DefaultRetinaParameters()
	p.OPLandIplParvo.PhotoreceptorsTemporalConstant = -1
	if p.Validate() == nil {
		t.Fatal("negative constant should fail")
	}
}

func TestRetinaWriteSetupPrint(t *testing.T) {
	r := NewRetina(8, 8)
	text := r.Write()
	// Change a field via text and re-apply.
	text = strings.Replace(text, "IplMagno.MagnoGain 3", "IplMagno.MagnoGain 6", 1)
	if err := r.SetupFromText(text); err != nil {
		t.Fatalf("SetupFromText failed: %v", err)
	}
	if r.GetParameters().IplMagno.MagnoGain != 6 {
		t.Fatalf("SetupFromText did not apply: %v", r.GetParameters().IplMagno.MagnoGain)
	}
	if err := r.SetupFromText("nonsense line line"); err == nil {
		t.Fatal("bad text should error")
	}
	if !strings.Contains(r.PrintSetup(), "Retina setup") {
		t.Fatal("PrintSetup missing header")
	}
	ir, ic := r.GetInputSize()
	or, oc := r.GetOutputSize()
	if ir != 8 || ic != 8 || or != 8 || oc != 8 {
		t.Fatal("Get*Size mismatch")
	}
}

// --- ON/OFF ganglion split ---

func TestOnOffSplit(t *testing.T) {
	rows, cols := 32, 32
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(30)
	// Small bright spot at the centre.
	for y := 15; y <= 17; y++ {
		for x := 15; x <= 17; x++ {
			img.Set(y, x, 0, 220)
		}
	}
	oo := SplitOnOffChannels(img, 3, 1)
	if oo.On.At(16, 16, 0) == 0 {
		t.Error("ON channel should fire at a bright centre")
	}
	if oo.Off.At(16, 16, 0) != 0 {
		t.Error("OFF channel should be zero at the brightest point")
	}
	var maxOff uint8
	for _, v := range oo.Off.Data {
		if v > maxOff {
			maxOff = v
		}
	}
	if maxOff == 0 {
		t.Error("OFF channel should fire in the darker surround")
	}

	// A flat field yields no ON/OFF response.
	flat := cv.NewMat(rows, cols, 1)
	flat.SetTo(100)
	fo := SplitOnOffChannels(flat, 3, 1)
	for i := range fo.On.Data {
		if fo.On.Data[i] != 0 || fo.Off.Data[i] != 0 {
			t.Fatal("flat field should give zero ON/OFF")
		}
	}
}

func TestOnOffRejectsNegativeConstant(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on negative spatialConstant")
		}
	}()
	SplitOnOffChannels(cv.NewMat(4, 4, 1), -1, 1)
}

// --- Log-polar photoreceptor sampling ---

func TestLogSamplerUniformPreserved(t *testing.T) {
	rows, cols := 32, 32
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(120)
	s := NewRetinaLogSampler(rows, cols, 16, 32)
	lp := s.Sample(img)
	for _, v := range lp.Data {
		if v < 118 || v > 122 {
			t.Fatalf("uniform input should map to ~120, got %d", v)
		}
	}
	back := s.InverseSample(lp)
	if d := int(back.At(16, 16, 0)) - 120; d < -3 || d > 3 {
		t.Fatalf("inverse of uniform should be ~120 at centre, got %d", back.At(16, 16, 0))
	}
	if ir, ic := s.InputSize(); ir != rows || ic != cols {
		t.Fatal("InputSize mismatch")
	}
	if rr, ss := s.OutputSize(); rr != 16 || ss != 32 {
		t.Fatal("OutputSize mismatch")
	}
}

func TestLogSamplerFovealMagnification(t *testing.T) {
	rows, cols := 40, 40
	img := cv.NewMat(rows, cols, 1)
	cx, cy := 19.5, 19.5
	cartBright := 0
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if math.Hypot(float64(x)-cx, float64(y)-cy) <= 4 {
				img.Set(y, x, 0, 255)
				cartBright++
			}
		}
	}
	s := NewRetinaLogSampler(rows, cols, 16, 32)
	lp := s.Sample(img)
	lpBright := 0
	for _, v := range lp.Data {
		if v > 127 {
			lpBright++
		}
	}
	cartFrac := float64(cartBright) / float64(rows*cols)
	lpFrac := float64(lpBright) / float64(len(lp.Data))
	if lpFrac <= cartFrac {
		t.Fatalf("log sampling should magnify the fovea: lpFrac=%.3f cartFrac=%.3f", lpFrac, cartFrac)
	}
}

func TestLogSamplerSizeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on size mismatch")
		}
	}()
	NewRetinaLogSampler(16, 16, 8, 8).Sample(cv.NewMat(8, 8, 1))
}

// --- Standalone fast tone mapping ---

func TestApplyFastToneMappingLiftsShadows(t *testing.T) {
	rows, cols := 16, 24
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8(2+x/3)) // faint deep-shadow gradient
		}
	}
	out := ApplyFastToneMapping(img)
	in := grayFloatMat(img)
	of := grayFloatMat(out)
	inMean, _ := regionMeanVar(in, 2, rows-2, 2, cols-2)
	outMean, _ := regionMeanVar(of, 2, rows-2, 2, cols-2)
	if outMean <= inMean {
		t.Fatalf("standalone tone mapping did not lift shadows: %.2f -> %.2f", inMean, outMean)
	}
}

func TestRetinaApplyFastToneMapping(t *testing.T) {
	rows, cols := 16, 16
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			img.Set(y, x, 0, uint8(3+x))
		}
	}
	r := NewRetina(rows, cols)
	out := r.ApplyFastToneMapping(img)
	if out.At(0, cols-1, 0) <= img.At(0, cols-1, 0) {
		t.Fatal("retina tone mapping should lift the shadow gradient")
	}
	// Interleaving must not disturb a running retina.
	r.Run(img)
	if r.GetParvo().Channels != 1 {
		t.Fatal("retina still usable after tone mapping")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on size mismatch")
		}
	}()
	r.ApplyFastToneMapping(cv.NewMat(4, 4, 1))
}

// --- Retina processor toggles ---

func TestRetinaProcessorChannelToggles(t *testing.T) {
	rows, cols := 32, 32
	edgeAt := func(edge int) *cv.Mat {
		m := cv.NewMat(rows, cols, 1)
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				v := uint8(40)
				if x >= edge {
					v = 200
				}
				m.Set(y, x, 0, v)
			}
		}
		return m
	}
	rp := NewRetinaProcessor(rows, cols)
	for i := 0; i < 12; i++ {
		rp.Run(edgeAt(12))
	}
	rp.Run(edgeAt(18)) // motion

	if sum(rp.GetParvo().Data) == 0 {
		t.Fatal("active parvo should be non-zero")
	}
	if sum(rp.GetMagno().Data) == 0 {
		t.Fatal("active magno should respond to motion")
	}

	rp.ActivateMovingContoursProcessing(false)
	if sum(rp.GetMagno().Data) != 0 {
		t.Fatal("disabled magno should be all zero")
	}
	if rp.MovingContoursProcessingActive() {
		t.Fatal("moving contours should report inactive")
	}
	// Parvo still active.
	if sum(rp.GetParvo().Data) == 0 {
		t.Fatal("parvo should still be active")
	}

	rp.ActivateContoursProcessing(false)
	if sum(rp.GetParvo().Data) != 0 {
		t.Fatal("disabled parvo should be all zero")
	}
	if rp.ContoursProcessingActive() {
		t.Fatal("contours should report inactive")
	}

	rp.ActivateMovingContoursProcessing(true)
	if sum(rp.GetMagno().Data) == 0 {
		t.Fatal("re-enabled magno should be non-zero again")
	}

	if ir, ic := rp.GetInputSize(); ir != rows || ic != cols {
		t.Fatal("processor input size mismatch")
	}
	if or, oc := rp.GetOutputSize(); or != rows || oc != cols {
		t.Fatal("processor output size mismatch")
	}
	if rp.Retina() == nil {
		t.Fatal("Retina() should expose the underlying model")
	}
}

func TestRetinaProcessorClearAndPanic(t *testing.T) {
	rp := NewRetinaProcessor(8, 8)
	img := cv.NewMat(8, 8, 1)
	img.SetTo(80)
	rp.Run(img)
	rp.ClearBuffers()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic reading after ClearBuffers")
		}
	}()
	rp.GetParvo()
}

func sum(data []uint8) int {
	s := 0
	for _, v := range data {
		s += int(v)
	}
	return s
}

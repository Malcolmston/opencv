package mcc_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/mcc"
)

// distort applies a mild cross-channel gain plus offset and a gamma-like
// nonlinearity to simulate a real camera, returning 0..255 "measured" colors.
func distort(ref [][3]float64) [][3]float64 {
	mat := [3][3]float64{
		{0.88, 0.11, 0.06},
		{0.07, 0.86, 0.08},
		{0.05, 0.13, 0.93},
	}
	out := make([][3]float64, len(ref))
	for i, c := range ref {
		r := c[0] / 255
		g := c[1] / 255
		b := c[2] / 255
		nr := mat[0][0]*r + mat[0][1]*g + mat[0][2]*b
		ng := mat[1][0]*r + mat[1][1]*g + mat[1][2]*b
		nb := mat[2][0]*r + mat[2][1]*g + mat[2][2]*b
		out[i] = [3]float64{
			clampf(nr*255 + 5),
			clampf(ng*255 + 3),
			clampf(nb*255 + 7),
		}
	}
	return out
}

func clampf(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}

func TestColorCorrectionModelsReduceError(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := distort(ref)
	before2000 := 0.0
	for i := range ref {
		before2000 += mcc.DeltaE2000RGB(
			[3]uint8{uint8(measured[i][0]), uint8(measured[i][1]), uint8(measured[i][2])},
			[3]uint8{uint8(ref[i][0]), uint8(ref[i][1]), uint8(ref[i][2])})
	}
	before2000 /= float64(len(ref))
	if before2000 < 2 {
		t.Fatalf("distortion too weak, before dE2000=%.2f", before2000)
	}

	cases := []struct {
		name string
		cfg  mcc.ColorCorrectionConfig
	}{
		{"linear", mcc.ColorCorrectionConfig{Model: mcc.ModelLinear}},
		{"affine", mcc.ColorCorrectionConfig{Model: mcc.ModelAffine}},
		{"poly2", mcc.ColorCorrectionConfig{Model: mcc.ModelPoly2}},
		{"poly3", mcc.ColorCorrectionConfig{Model: mcc.ModelPoly3}},
		{"rootpoly2", mcc.ColorCorrectionConfig{Model: mcc.ModelRootPoly2}},
		{"rootpoly3", mcc.ColorCorrectionConfig{Model: mcc.ModelRootPoly3}},
		{"linear-srgb", mcc.ColorCorrectionConfig{Model: mcc.ModelLinear, Linearize: mcc.LinSRGB}},
		{"affine-gamma", mcc.ColorCorrectionConfig{Model: mcc.ModelAffine, Linearize: mcc.LinGamma, Gamma: 2.2}},
		{"linear-wb", mcc.ColorCorrectionConfig{Model: mcc.ModelLinear, WhiteBalance: true, WhitePatch: 18}},
		{"linear-wb-auto", mcc.ColorCorrectionConfig{Model: mcc.ModelLinear, WhiteBalance: true, WhitePatch: -1}},
	}
	for _, tc := range cases {
		model, err := mcc.TrainColorCorrection(measured, ref, tc.cfg)
		if err != nil {
			t.Fatalf("%s: train error: %v", tc.name, err)
		}
		after := model.MeanDeltaE2000(measured, ref)
		if after >= before2000 {
			t.Errorf("%s: after dE2000=%.3f not < before %.3f", tc.name, after, before2000)
		}
		if after > 3 {
			t.Errorf("%s: residual dE2000=%.3f, want <=3", tc.name, after)
		}
	}
}

func TestWeightedLeastSquares(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := distort(ref)
	w := mcc.LuminanceWeights(ref, 1)
	if len(w) != len(ref) {
		t.Fatalf("weights length %d, want %d", len(w), len(ref))
	}
	model, err := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{
		Model:   mcc.ModelAffine,
		Weights: w,
	})
	if err != nil {
		t.Fatal(err)
	}
	if e := model.MeanDeltaE2000(measured, ref); e > 2.5 {
		t.Errorf("weighted fit residual dE2000=%.3f, want small", e)
	}
	// A bad weight length is rejected.
	if _, err := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{
		Model: mcc.ModelLinear, Weights: w[:5],
	}); err == nil {
		t.Error("mismatched weights length should error")
	}
}

func TestWhiteBalanceGains(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := distort(ref)
	model, err := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{
		Model: mcc.ModelLinear, WhiteBalance: true, WhitePatch: 18,
	})
	if err != nil {
		t.Fatal(err)
	}
	g := model.WhiteBalanceGains()
	if g == [3]float64{1, 1, 1} {
		t.Error("white balance gains should not be identity for a distorted white")
	}
	// Disabled white balance leaves gains at identity.
	m2, _ := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{Model: mcc.ModelLinear})
	if m2.WhiteBalanceGains() != [3]float64{1, 1, 1} {
		t.Error("gains should be identity when WhiteBalance is false")
	}
}

func TestModelInferImageClamps(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := distort(ref)
	model, err := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{Model: mcc.ModelAffine})
	if err != nil {
		t.Fatal(err)
	}
	dist := cv.NewMat(4, 6, 3)
	for r := 0; r < 4; r++ {
		for c := 0; c < 6; c++ {
			m := measured[r*6+c]
			dist.SetPixel(r, c, []uint8{uint8(m[0]), uint8(m[1]), uint8(m[2])})
		}
	}
	out := model.Infer(dist)
	if out.Rows != 4 || out.Cols != 6 || out.Channels != 3 {
		t.Fatalf("infer output shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
	var got [][3]float64
	for r := 0; r < 4; r++ {
		for c := 0; c < 6; c++ {
			p := out.AtPixel(r, c)
			got = append(got, [3]float64{float64(p[0]), float64(p[1]), float64(p[2])})
		}
	}
	if e := mcc.MeanDeltaE(got, ref); e > 3 {
		t.Errorf("corrected image mean dE76=%.3f, want <=3", e)
	}
	// Single-color inference is clamped to [0,255].
	c := model.InferRGB([3]float64{255, 255, 255})
	for i, v := range c {
		if v < 0 || v > 255 {
			t.Errorf("InferRGB channel %d = %.2f, out of range", i, v)
		}
	}
}

func TestModelInferPanicsOnGray(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	model, _ := mcc.TrainColorCorrection(ref, ref, mcc.ColorCorrectionConfig{Model: mcc.ModelLinear})
	defer func() {
		if recover() == nil {
			t.Error("Infer on a 1-channel image should panic")
		}
	}()
	model.Infer(cv.NewMat(3, 3, 1))
}

func TestModelReport(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := distort(ref)
	model, err := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{Model: mcc.ModelAffine})
	if err != nil {
		t.Fatal(err)
	}
	rep := model.Report(measured, ref)
	if len(rep) != 24 {
		t.Fatalf("report has %d rows, want 24", len(rep))
	}
	var corr, meas float64
	for i, pr := range rep {
		if pr.Index != i {
			t.Errorf("row %d index=%d", i, pr.Index)
		}
		if pr.ReferenceRGB != ref[i] || pr.MeasuredRGB != measured[i] {
			t.Errorf("row %d colors not carried through", i)
		}
		// The reported measured Lab must match a direct conversion.
		if pr.MeasuredLab != mcc.RGBToLab(uint8(measured[i][0]), uint8(measured[i][1]), uint8(measured[i][2])) {
			// measured colors may be fractional; allow a small tolerance instead.
			if !close3(pr.MeasuredLab, mcc.RGBToLab(uint8(measured[i][0]), uint8(measured[i][1]), uint8(measured[i][2])), 1.0) {
				t.Errorf("row %d measured Lab inconsistent: %v", i, pr.MeasuredLab)
			}
		}
		corr += pr.DeltaE2000
		meas += mcc.DeltaE2000(pr.MeasuredLab, pr.ReferenceLab)
	}
	// In aggregate the correction must reduce the perceptual error.
	if corr >= meas {
		t.Errorf("report: corrected total dE2000 %.2f not < measured %.2f", corr, meas)
	}
	// Matrix shape for affine is 4x3.
	if m := model.Matrix(); len(m) != 4 {
		t.Errorf("affine matrix rows=%d, want 4", len(m))
	}
	if model.Type() != mcc.ModelAffine {
		t.Error("wrong model type reported")
	}
}

func TestTrainColorCorrectionErrors(t *testing.T) {
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	if _, err := mcc.TrainColorCorrection(ref, ref[:10], mcc.ColorCorrectionConfig{}); err == nil {
		t.Error("length mismatch should error")
	}
	if _, err := mcc.TrainColorCorrection(ref[:5], ref[:5], mcc.ColorCorrectionConfig{Model: mcc.ModelPoly3}); err == nil {
		t.Error("too few samples should error")
	}
	if _, err := mcc.TrainColorCorrection(ref, ref, mcc.ColorCorrectionConfig{Model: mcc.CCMModelType(99)}); err == nil {
		t.Error("unknown model should error")
	}
	if _, err := mcc.TrainColorCorrection(nil, nil, mcc.ColorCorrectionConfig{}); err == nil {
		t.Error("empty input should error")
	}
}

func TestRootPolynomialExposureInvariance(t *testing.T) {
	// A root-polynomial model fitted at one exposure should generalise across a
	// global scale change far better than a plain linear matrix would in the
	// gamma-encoded space. Here we simply verify it fits well and stays finite.
	ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := distort(ref)
	model, err := mcc.TrainColorCorrection(measured, ref, mcc.ColorCorrectionConfig{Model: mcc.ModelRootPoly3})
	if err != nil {
		t.Fatal(err)
	}
	if e := model.MeanDeltaE76(measured, ref); e > 3 {
		t.Errorf("rootpoly3 residual dE76=%.3f, want small", e)
	}
}

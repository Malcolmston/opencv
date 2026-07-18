package superres

import (
	"math"
	"testing"
)

func TestMetricsIdentical(t *testing.T) {
	a := constMat(8, 8, 3, 60)
	b := constMat(8, 8, 3, 60)
	if MSE(a, b) != 0 {
		t.Errorf("MSE identical = %v, want 0", MSE(a, b))
	}
	if MAE(a, b) != 0 {
		t.Errorf("MAE identical = %v, want 0", MAE(a, b))
	}
	if !math.IsInf(PSNR(a, b), 1) {
		t.Errorf("PSNR identical = %v, want +Inf", PSNR(a, b))
	}
	if !approx(SSIM(a, b, 8), 1, 1e-9) {
		t.Errorf("SSIM identical = %v, want 1", SSIM(a, b, 8))
	}
}

func TestMetricsKnownAnswer(t *testing.T) {
	a := gray1(1, 1, []uint8{0})
	b := gray1(1, 1, []uint8{10})
	if !approx(MSE(a, b), 100, 1e-9) {
		t.Errorf("MSE = %v, want 100", MSE(a, b))
	}
	if !approx(MAE(a, b), 10, 1e-9) {
		t.Errorf("MAE = %v, want 10", MAE(a, b))
	}
	wantPSNR := 10 * math.Log10(255*255/100.0)
	if !approx(PSNR(a, b), wantPSNR, 1e-9) {
		t.Errorf("PSNR = %v, want %v", PSNR(a, b), wantPSNR)
	}
}

func TestSSIMRange(t *testing.T) {
	a := gray1(4, 4, []uint8{
		0, 255, 0, 255,
		255, 0, 255, 0,
		0, 255, 0, 255,
		255, 0, 255, 0,
	})
	// Inverting the image should give a low SSIM (structure preserved but
	// contrast inverted yields negative correlation).
	b := gray1(4, 4, []uint8{
		255, 0, 255, 0,
		0, 255, 0, 255,
		255, 0, 255, 0,
		0, 255, 0, 255,
	})
	s := SSIM(a, b, 4)
	if s > 0.5 {
		t.Errorf("SSIM of inverted image = %v, want <= 0.5", s)
	}
	if s < -1.001 || s > 1.001 {
		t.Errorf("SSIM out of range: %v", s)
	}
}

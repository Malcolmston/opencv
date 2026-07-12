package quality

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- deterministic test fixtures ---------------------------------------------

// gradientImage builds a channels-channel image whose gray value ramps with a
// smooth diagonal gradient, giving structure without abrupt edges.
func gradientImage(rows, cols, channels int) *cv.Mat {
	m := cv.NewMat(rows, cols, channels)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := uint8((x*255/cols + y*255/rows) / 2)
			for c := 0; c < channels; c++ {
				m.Set(y, x, c, v)
			}
		}
	}
	return m
}

// checkerImage builds a single-channel sharp checkerboard: maximal
// high-frequency content for the focus-measure tests.
func checkerImage(rows, cols, block int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if ((x/block)+(y/block))%2 == 0 {
				m.Set(y, x, 0, 255)
			}
		}
	}
	return m
}

// addNoise returns a copy of src with a deterministic, bounded pseudo-random
// perturbation of the given amplitude added to every sample.
func addNoise(src *cv.Mat, amplitude int) *cv.Mat {
	out := src.Clone()
	// A small linear-congruential generator keeps the test self-contained and
	// reproducible without importing math/rand.
	state := uint32(12345)
	next := func() int {
		state = state*1664525 + 1013904223
		return int(state>>8) % (2*amplitude + 1)
	}
	for i := range out.Data {
		v := int(out.Data[i]) + next() - amplitude
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		out.Data[i] = uint8(v)
	}
	return out
}

const tol = 1e-9

// --- MSE / MAE / PSNR ---------------------------------------------------------

func TestMSEIdenticalIsZero(t *testing.T) {
	img := gradientImage(32, 40, 3)
	for c, v := range MSE(img, img.Clone()) {
		if v != 0 {
			t.Fatalf("MSE channel %d = %v, want 0", c, v)
		}
	}
	for c, v := range MAE(img, img.Clone()) {
		if v != 0 {
			t.Fatalf("MAE channel %d = %v, want 0", c, v)
		}
	}
}

func TestPSNRIdenticalIsInf(t *testing.T) {
	img := gradientImage(24, 24, 1)
	if p := PSNR(img, img.Clone()); !math.IsInf(p, 1) {
		t.Fatalf("PSNR of identical images = %v, want +Inf", p)
	}
}

func TestMSEKnownValue(t *testing.T) {
	a := cv.NewMat(1, 2, 1)
	b := cv.NewMat(1, 2, 1)
	a.Set(0, 0, 0, 10)
	a.Set(0, 1, 0, 20)
	b.Set(0, 0, 0, 12) // diff 2 -> 4
	b.Set(0, 1, 0, 24) // diff 4 -> 16
	got := MSE(a, b)[0]
	if math.Abs(got-10) > tol { // (4+16)/2
		t.Fatalf("MSE = %v, want 10", got)
	}
	if mae := MAE(a, b)[0]; math.Abs(mae-3) > tol { // (2+4)/2
		t.Fatalf("MAE = %v, want 3", mae)
	}
}

func TestPSNRDecreasesWithNoise(t *testing.T) {
	img := gradientImage(40, 40, 3)
	low := addNoise(img, 5)
	high := addNoise(img, 40)
	pLow := PSNR(img, low)
	pHigh := PSNR(img, high)
	if !(pLow > pHigh) {
		t.Fatalf("PSNR should fall as noise grows: low-noise=%v high-noise=%v", pLow, pHigh)
	}
	// Sanity: both finite and positive for genuinely noisy candidates.
	if math.IsInf(pLow, 1) || math.IsInf(pHigh, 1) {
		t.Fatalf("noisy PSNR unexpectedly +Inf: %v, %v", pLow, pHigh)
	}
}

// --- SSIM ---------------------------------------------------------------------

func TestSSIMIdenticalIsOne(t *testing.T) {
	img := gradientImage(48, 48, 3)
	mean, m := SSIM(img, img.Clone())
	if math.Abs(mean-1) > 1e-6 {
		t.Fatalf("SSIM of identical images = %v, want 1", mean)
	}
	if m == nil || m.Channels != 1 || m.Rows != img.Rows || m.Cols != img.Cols {
		t.Fatalf("SSIM map has wrong shape: %+v", m)
	}
}

func TestSSIMDecreasesWithDistortion(t *testing.T) {
	img := gradientImage(48, 48, 1)
	mild := addNoise(img, 8)
	strong := addNoise(img, 60)
	sMild, _ := SSIM(img, mild)
	sStrong, _ := SSIM(img, strong)
	if !(sMild > sStrong) {
		t.Fatalf("SSIM should fall with distortion: mild=%v strong=%v", sMild, sStrong)
	}
	if !(sMild < 1) {
		t.Fatalf("distorted SSIM should be < 1, got %v", sMild)
	}
}

func TestSSIMIsSymmetric(t *testing.T) {
	a := gradientImage(40, 40, 1)
	b := addNoise(a, 30)
	sab, _ := SSIM(a, b)
	sba, _ := SSIM(b, a)
	if math.Abs(sab-sba) > 1e-12 {
		t.Fatalf("SSIM not symmetric: SSIM(a,b)=%v SSIM(b,a)=%v", sab, sba)
	}
}

// --- MS-SSIM ------------------------------------------------------------------

func TestMSSSIMIdenticalIsOne(t *testing.T) {
	img := gradientImage(64, 64, 3)
	if v := MSSSIM(img, img.Clone()); math.Abs(v-1) > 1e-6 {
		t.Fatalf("MSSSIM of identical images = %v, want 1", v)
	}
}

func TestMSSSIMDecreasesWithDistortion(t *testing.T) {
	img := gradientImage(64, 64, 1)
	mild := addNoise(img, 8)
	strong := addNoise(img, 60)
	if !(MSSSIM(img, mild) > MSSSIM(img, strong)) {
		t.Fatalf("MSSSIM should fall with distortion")
	}
}

// --- GMSD ---------------------------------------------------------------------

func TestGMSDIdenticalIsZero(t *testing.T) {
	img := gradientImage(48, 48, 3)
	dev, m := GMSD(img, img.Clone())
	if math.Abs(dev) > 1e-9 {
		t.Fatalf("GMSD of identical images = %v, want 0", dev)
	}
	if m == nil || m.Channels != 1 {
		t.Fatalf("GMSD map has wrong shape: %+v", m)
	}
}

func TestGMSDIncreasesWithDistortion(t *testing.T) {
	img := gradientImage(48, 48, 1)
	mild := addNoise(img, 8)
	strong := addNoise(img, 60)
	dMild, _ := GMSD(img, mild)
	dStrong, _ := GMSD(img, strong)
	if !(dStrong > dMild) {
		t.Fatalf("GMSD should rise with distortion: mild=%v strong=%v", dMild, dStrong)
	}
}

// --- UQI ----------------------------------------------------------------------

func TestUQIIdenticalIsOne(t *testing.T) {
	img := gradientImage(32, 32, 3)
	if v := UQI(img, img.Clone()); math.Abs(v-1) > 1e-6 {
		t.Fatalf("UQI of identical images = %v, want 1", v)
	}
}

func TestUQIDecreasesWithDistortion(t *testing.T) {
	img := gradientImage(32, 32, 1)
	mild := addNoise(img, 8)
	strong := addNoise(img, 60)
	if !(UQI(img, mild) > UQI(img, strong)) {
		t.Fatalf("UQI should fall with distortion")
	}
}

// --- No-reference focus measures ---------------------------------------------

func TestSharpnessBlurredLessThanSharp(t *testing.T) {
	sharp := checkerImage(48, 48, 4)
	blurred := cv.GaussianBlur(sharp, 7, 0)
	if !(Sharpness(blurred) < Sharpness(sharp)) {
		t.Fatalf("blurred sharpness (%v) should be < sharp (%v)",
			Sharpness(blurred), Sharpness(sharp))
	}
	// Sharpness is defined as the variance of the Laplacian.
	if Sharpness(sharp) != LaplacianVariance(sharp) {
		t.Fatalf("Sharpness and LaplacianVariance must agree")
	}
}

func TestTenengradBlurredLessThanSharp(t *testing.T) {
	sharp := checkerImage(48, 48, 4)
	blurred := cv.GaussianBlur(sharp, 7, 0)
	if !(Tenengrad(blurred) < Tenengrad(sharp)) {
		t.Fatalf("blurred Tenengrad (%v) should be < sharp (%v)",
			Tenengrad(blurred), Tenengrad(sharp))
	}
}

func TestBrisqueScoreBlurredLessThanSharp(t *testing.T) {
	sharp := checkerImage(48, 48, 4)
	blurred := cv.GaussianBlur(sharp, 7, 0)
	bs := BrisqueScore(sharp)
	bb := BrisqueScore(blurred)
	if !(bb < bs) {
		t.Fatalf("blurred BRISQUE-lite (%v) should be < sharp (%v)", bb, bs)
	}
	if math.IsNaN(bs) || math.IsInf(bs, 0) {
		t.Fatalf("BrisqueScore must be finite, got %v", bs)
	}
}

// --- QualityBase object forms -------------------------------------------------

func TestQualityBaseObjects(t *testing.T) {
	ref := gradientImage(40, 40, 3)
	cmp := addNoise(ref, 20)

	mse := NewQualityMSE(ref)
	if mse.QualityMap() != nil {
		t.Fatalf("QualityMap should be nil before Compute")
	}
	got := mse.Compute(cmp)
	want := MSE(ref, cmp)
	if len(got) != len(want) {
		t.Fatalf("QualityMSE channel count %d, want %d", len(got), len(want))
	}
	for c := range got {
		if math.Abs(got[c]-want[c]) > tol {
			t.Fatalf("QualityMSE channel %d = %v, want %v", c, got[c], want[c])
		}
	}
	if m := mse.QualityMap(); m == nil || m.Rows != ref.Rows {
		t.Fatalf("QualityMSE map missing after Compute")
	}

	psnr := NewQualityPSNR(ref).Compute(cmp)
	if len(psnr) != 1 || math.Abs(psnr[0]-PSNR(ref, cmp)) > tol {
		t.Fatalf("QualityPSNR = %v, want %v", psnr, PSNR(ref, cmp))
	}

	ssimObj := NewQualitySSIM(ref)
	sMean, _ := SSIM(ref, cmp)
	if s := ssimObj.Compute(cmp); len(s) != 1 || math.Abs(s[0]-sMean) > tol {
		t.Fatalf("QualitySSIM = %v, want %v", s, sMean)
	}
	if ssimObj.QualityMap() == nil {
		t.Fatalf("QualitySSIM map missing after Compute")
	}

	gmsdObj := NewQualityGMSD(ref)
	gDev, _ := GMSD(ref, cmp)
	if g := gmsdObj.Compute(cmp); len(g) != 1 || math.Abs(g[0]-gDev) > tol {
		t.Fatalf("QualityGMSD = %v, want %v", g, gDev)
	}
	if gmsdObj.QualityMap() == nil {
		t.Fatalf("QualityGMSD map missing after Compute")
	}
}

func TestEmptyImagePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on empty image")
		}
	}()
	Sharpness(&cv.Mat{})
}

// --- input validation ---------------------------------------------------------

func TestMismatchedSizePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on size mismatch")
		}
	}()
	MSE(gradientImage(10, 10, 1), gradientImage(10, 12, 1))
}

func TestMismatchedChannelsPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on channel mismatch")
		}
	}()
	SSIM(gradientImage(16, 16, 1), gradientImage(16, 16, 3))
}

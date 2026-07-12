package quality

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// colorGradient builds a three-channel image whose R, G and B channels ramp in
// different directions, giving genuine chrominance for the colour-aware metrics.
func colorGradient(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Set(y, x, 0, uint8(x*255/cols))
			m.Set(y, x, 1, uint8(y*255/rows))
			m.Set(y, x, 2, uint8((x+y)*255/(rows+cols)))
		}
	}
	return m
}

// naturalLike builds a deterministic smooth, gently textured single-channel
// image resembling natural-scene statistics, used to exercise the no-reference
// naturalness metrics.
func naturalLike(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := 128 + 60*math.Sin(float64(x)/18.0) + 40*math.Cos(float64(y)/23.0)
			v := base + 12*math.Sin(float64(x+y)/5.0)
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			m.Set(y, x, 0, uint8(v))
		}
	}
	return m
}

// --- RMSE / NRMSE / SNR -------------------------------------------------------

func TestRMSEAndNRMSE(t *testing.T) {
	a := cv.NewMat(1, 2, 1)
	b := cv.NewMat(1, 2, 1)
	a.Set(0, 0, 0, 10)
	a.Set(0, 1, 0, 20)
	b.Set(0, 0, 0, 12) // diff 2 -> 4
	b.Set(0, 1, 0, 24) // diff 4 -> 16
	if got := RMSE(a, b)[0]; math.Abs(got-math.Sqrt(10)) > tol {
		t.Fatalf("RMSE = %v, want sqrt(10)", got)
	}
	// Reference range is 20-10 = 10.
	if got := NRMSE(a, b)[0]; math.Abs(got-math.Sqrt(10)/10) > tol {
		t.Fatalf("NRMSE = %v, want sqrt(10)/10", got)
	}
}

func TestRMSEIdenticalZeroAndMonotonic(t *testing.T) {
	img := gradientImage(40, 40, 3)
	for _, v := range RMSE(img, img.Clone()) {
		if v != 0 {
			t.Fatalf("RMSE identical = %v, want 0", v)
		}
	}
	for _, v := range NRMSE(img, img.Clone()) {
		if v != 0 {
			t.Fatalf("NRMSE identical = %v, want 0", v)
		}
	}
	mild := poolMean(RMSE(img, addNoise(img, 8)))
	strong := poolMean(RMSE(img, addNoise(img, 60)))
	if !(mild < strong) {
		t.Fatalf("RMSE should rise with noise: mild=%v strong=%v", mild, strong)
	}
}

func TestSNR(t *testing.T) {
	img := gradientImage(40, 40, 3)
	if s := SNR(img, img.Clone()); !math.IsInf(s, 1) {
		t.Fatalf("SNR identical = %v, want +Inf", s)
	}
	mild := SNR(img, addNoise(img, 8))
	strong := SNR(img, addNoise(img, 60))
	if !(mild > strong) {
		t.Fatalf("SNR should fall with noise: mild=%v strong=%v", mild, strong)
	}
}

func poolMean(xs []float64) float64 { return meanOf(xs) }

// --- Entropy / EdgePreservationRatio -----------------------------------------

func TestEntropyDiff(t *testing.T) {
	img := gradientImage(48, 48, 1)
	if d := EntropyDiff(img, img.Clone()); d != 0 {
		t.Fatalf("EntropyDiff identical = %v, want 0", d)
	}
	if d := EntropyDiff(img, addNoise(img, 40)); !(d > 0) {
		t.Fatalf("EntropyDiff of distorted should be > 0, got %v", d)
	}
	// Symmetry.
	a := gradientImage(32, 32, 1)
	b := addNoise(a, 25)
	if math.Abs(EntropyDiff(a, b)-EntropyDiff(b, a)) > tol {
		t.Fatalf("EntropyDiff not symmetric")
	}
	if e := Entropy(img); e <= 0 || e > 8 {
		t.Fatalf("Entropy out of range: %v", e)
	}
}

func TestEdgePreservationRatio(t *testing.T) {
	img := checkerImage(48, 48, 4)
	if r := EdgePreservationRatio(img, img.Clone()); math.Abs(r-1) > 1e-9 {
		t.Fatalf("EPR identical = %v, want 1", r)
	}
	sharp := EdgePreservationRatio(img, cv.GaussianBlur(img, 3, 0))
	blurred := EdgePreservationRatio(img, cv.GaussianBlur(img, 9, 0))
	if !(sharp > blurred) {
		t.Fatalf("EPR should fall with heavier blur: mild=%v heavy=%v", sharp, blurred)
	}
	if !(sharp < 1) {
		t.Fatalf("EPR of blurred should be < 1, got %v", sharp)
	}
}

// --- VIF / VIFP ---------------------------------------------------------------

func TestVIFPIdenticalAndMonotonic(t *testing.T) {
	img := gradientImage(64, 64, 3)
	if v := VIFP(img, img.Clone()); math.Abs(v-1) > 1e-3 {
		t.Fatalf("VIFP identical = %v, want ~1", v)
	}
	mild := VIFP(img, addNoise(img, 8))
	strong := VIFP(img, addNoise(img, 60))
	if !(mild > strong) {
		t.Fatalf("VIFP should fall with noise: mild=%v strong=%v", mild, strong)
	}
	if !(mild < 1) {
		t.Fatalf("distorted VIFP should be < 1, got %v", mild)
	}
}

func TestVIFIdenticalAndMonotonic(t *testing.T) {
	img := gradientImage(64, 64, 1)
	if v := VIF(img, img.Clone()); math.Abs(v-1) > 1e-3 {
		t.Fatalf("VIF identical = %v, want ~1", v)
	}
	mild := VIF(img, addNoise(img, 8))
	strong := VIF(img, addNoise(img, 60))
	if !(mild > strong) {
		t.Fatalf("VIF should fall with noise: mild=%v strong=%v", mild, strong)
	}
}

// --- FSIM / FSIMc -------------------------------------------------------------

func TestFSIMIdenticalAndMonotonic(t *testing.T) {
	img := gradientImage(64, 64, 1)
	if v := FSIM(img, img.Clone()); math.Abs(v-1) > 1e-9 {
		t.Fatalf("FSIM identical = %v, want 1", v)
	}
	mild := FSIM(img, addNoise(img, 8))
	strong := FSIM(img, addNoise(img, 60))
	if !(mild > strong) {
		t.Fatalf("FSIM should fall with noise: mild=%v strong=%v", mild, strong)
	}
	// Blur monotonicity.
	b3 := FSIM(img, cv.GaussianBlur(img, 3, 0))
	b9 := FSIM(img, cv.GaussianBlur(img, 9, 0))
	if !(b3 > b9) {
		t.Fatalf("FSIM should fall with heavier blur: %v vs %v", b3, b9)
	}
}

func TestFSIMcColour(t *testing.T) {
	img := colorGradient(64, 64)
	if v := FSIMc(img, img.Clone()); math.Abs(v-1) > 1e-9 {
		t.Fatalf("FSIMc identical = %v, want 1", v)
	}
	mild := FSIMc(img, addNoise(img, 8))
	strong := FSIMc(img, addNoise(img, 60))
	if !(mild > strong) {
		t.Fatalf("FSIMc should fall with noise: mild=%v strong=%v", mild, strong)
	}
	// On grey input FSIMc reduces to FSIM.
	g := gradientImage(48, 48, 1)
	if math.Abs(FSIMc(g, addNoise(g, 20))-FSIM(g, addNoise(g, 20))) > tol {
		t.Fatalf("FSIMc must equal FSIM on single-channel input")
	}
}

// --- IWSSIM -------------------------------------------------------------------

func TestIWSSIMIdenticalAndMonotonic(t *testing.T) {
	img := gradientImage(64, 64, 3)
	if v := IWSSIM(img, img.Clone()); math.Abs(v-1) > 1e-6 {
		t.Fatalf("IWSSIM identical = %v, want 1", v)
	}
	mild := IWSSIM(img, addNoise(img, 8))
	strong := IWSSIM(img, addNoise(img, 60))
	if !(mild > strong) {
		t.Fatalf("IWSSIM should fall with noise: mild=%v strong=%v", mild, strong)
	}
}

// --- VSI ----------------------------------------------------------------------

func TestVSIIdenticalAndMonotonic(t *testing.T) {
	img := colorGradient(64, 64)
	if v := VSI(img, img.Clone()); math.Abs(v-1) > 1e-9 {
		t.Fatalf("VSI identical = %v, want 1", v)
	}
	mild := VSI(img, addNoise(img, 8))
	strong := VSI(img, addNoise(img, 60))
	if !(mild > strong) {
		t.Fatalf("VSI should fall with noise: mild=%v strong=%v", mild, strong)
	}
}

// --- CWSSIM -------------------------------------------------------------------

func TestCWSSIMIdenticalAndMonotonic(t *testing.T) {
	img := gradientImage(64, 64, 1)
	if v := CWSSIM(img, img.Clone()); math.Abs(v-1) > 1e-6 {
		t.Fatalf("CWSSIM identical = %v, want 1", v)
	}
	mild := CWSSIM(img, addNoise(img, 8))
	strong := CWSSIM(img, addNoise(img, 60))
	if !(mild > strong) {
		t.Fatalf("CWSSIM should fall with noise: mild=%v strong=%v", mild, strong)
	}
}

// --- SSIMMap ------------------------------------------------------------------

func TestSSIMMap(t *testing.T) {
	img := gradientImage(48, 48, 1)
	m := SSIMMap(img, img.Clone())
	if m == nil || m.Channels != 1 || m.Rows != 48 || m.Cols != 48 {
		t.Fatalf("SSIMMap wrong shape: %+v", m)
	}
	for _, v := range m.Data {
		if v != 255 {
			t.Fatalf("SSIMMap of identical images should be all 255, got %d", v)
		}
	}
}

// --- NIQE / PIQE --------------------------------------------------------------

func TestNIQE(t *testing.T) {
	clean := naturalLike(96, 96)
	if v := NIQE(clean); v > 0.5 {
		t.Fatalf("NIQE of pristine reference = %v, want near 0", v)
	}
	mild := NIQE(addNoise(clean, 8))
	strong := NIQE(addNoise(clean, 40))
	if !(NIQE(clean) < mild && mild < strong) {
		t.Fatalf("NIQE should rise with noise: clean=%v mild=%v strong=%v",
			NIQE(clean), mild, strong)
	}
	if !(NIQE(clean) < NIQE(cv.GaussianBlur(clean, 9, 0))) {
		t.Fatalf("NIQE should rise with blur")
	}
}

func TestPIQE(t *testing.T) {
	clean := naturalLike(96, 96)
	c := PIQE(clean)
	mild := PIQE(addNoise(clean, 8))
	strong := PIQE(addNoise(clean, 40))
	if math.IsNaN(c) || math.IsInf(c, 0) {
		t.Fatalf("PIQE must be finite, got %v", c)
	}
	if !(c < mild && mild < strong) {
		t.Fatalf("PIQE should rise with noise: clean=%v mild=%v strong=%v", c, mild, strong)
	}
}

// --- object forms -------------------------------------------------------------

func TestNewObjectForms(t *testing.T) {
	ref := gradientImage(64, 64, 3)
	cmp := addNoise(ref, 20)

	rmse := NewQualityRMSE(ref)
	if rmse.QualityMap() != nil {
		t.Fatalf("QualityMap should be nil before Compute")
	}
	got := rmse.Compute(cmp)
	want := RMSE(ref, cmp)
	for c := range got {
		if math.Abs(got[c]-want[c]) > tol {
			t.Fatalf("QualityRMSE channel %d = %v, want %v", c, got[c], want[c])
		}
	}
	if rmse.QualityMap() == nil {
		t.Fatalf("QualityRMSE map missing after Compute")
	}

	fsim := NewQualityFSIM(ref)
	if s := fsim.Compute(cmp); len(s) != 1 || math.Abs(s[0]-FSIM(ref, cmp)) > tol {
		t.Fatalf("QualityFSIM = %v, want %v", s, FSIM(ref, cmp))
	}
	if fsim.QualityMap() == nil {
		t.Fatalf("QualityFSIM map missing after Compute")
	}

	fsimc := NewQualityFSIMc(ref)
	if s := fsimc.Compute(cmp); len(s) != 1 || math.Abs(s[0]-FSIMc(ref, cmp)) > tol {
		t.Fatalf("QualityFSIMc = %v, want %v", s, FSIMc(ref, cmp))
	}

	vifp := NewQualityVIFP(ref)
	if s := vifp.Compute(cmp); len(s) != 1 || math.Abs(s[0]-VIFP(ref, cmp)) > tol {
		t.Fatalf("QualityVIFP = %v, want %v", s, VIFP(ref, cmp))
	}
	if vifp.QualityMap() == nil {
		t.Fatalf("QualityVIFP map missing after Compute")
	}
}

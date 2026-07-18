package texture_test

import (
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/texture"
)

func TestLawsMasks(t *testing.T) {
	if got := texture.LawsL5(); !equalVec(got, []float64{1, 4, 6, 4, 1}) {
		t.Errorf("LawsL5 = %v", got)
	}
	if got := texture.LawsE5(); !equalVec(got, []float64{-1, -2, 0, 2, 1}) {
		t.Errorf("LawsE5 = %v", got)
	}
	if got := texture.LawsS5(); !equalVec(got, []float64{-1, 0, 2, 0, -1}) {
		t.Errorf("LawsS5 = %v", got)
	}
	if got := texture.LawsW5(); !equalVec(got, []float64{-1, 2, 0, -2, 1}) {
		t.Errorf("LawsW5 = %v", got)
	}
	if got := texture.LawsR5(); !equalVec(got, []float64{1, -4, 6, -4, 1}) {
		t.Errorf("LawsR5 = %v", got)
	}
}

func TestLawsKernel2D(t *testing.T) {
	k := texture.LawsKernel2D(texture.LawsL5(), texture.LawsL5())
	if len(k) != 25 {
		t.Fatalf("kernel length = %d, want 25", len(k))
	}
	// Centre element is 6*6 = 36.
	if k[2*5+2] != 36 {
		t.Errorf("L5L5 centre = %v, want 36", k[2*5+2])
	}
	// Sum of the outer product equals product of the vector sums (16*16).
	var sum float64
	for _, v := range k {
		sum += v
	}
	if sum != 256 {
		t.Errorf("L5L5 sum = %v, want 256", sum)
	}
}

func TestLawsConstantImageZeroEnergy(t *testing.T) {
	// After local-mean normalisation a flat image is all zeros, so every
	// (non-L5L5) energy map is zero; LawsFeatures excludes L5L5, so all zero.
	feats := texture.LawsFeatures(fill(12, 12, 130), 5)
	for i, v := range feats {
		if !approx(v, 0, 1e-9) {
			t.Errorf("feature %d on flat image = %v, want 0", i, v)
		}
	}
}

func TestLawsEnergyMapsKeys(t *testing.T) {
	le := texture.LawsEnergyMaps(fill(16, 16, 100), 5)
	if len(le.Maps) != 25 {
		t.Fatalf("expected 25 maps, got %d", len(le.Maps))
	}
	if le.Map("S5S5") == nil {
		t.Error("missing S5S5 map")
	}
	// Combined of a symmetric pair equals the map itself.
	c := le.Combined("S5", "S5")
	if len(c) != 16*16 {
		t.Fatalf("combined map size = %d", len(c))
	}
}

func TestLawsFeaturesDetectTexture(t *testing.T) {
	// A busy random image has more high-frequency energy than a flat one.
	rng := rand.New(rand.NewSource(1))
	busy := cv.NewMat(24, 24, 1)
	for i := range busy.Data {
		busy.Data[i] = uint8(rng.Intn(256))
	}
	fb := texture.LawsFeatures(busy, 5)
	ff := texture.LawsFeatures(fill(24, 24, 100), 5)
	var sb, sf float64
	for i := range fb {
		sb += fb[i]
		sf += ff[i]
	}
	if !(sb > sf) {
		t.Errorf("busy energy %.3f should exceed flat energy %.3f", sb, sf)
	}
}

func equalVec(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// BenchmarkLawsEnergyMaps exercises the heaviest routine: 25 separable 5x5
// convolutions plus box-mean energy pooling over a moderately sized image.
func BenchmarkLawsEnergyMaps(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	img := cv.NewMat(128, 128, 1)
	for i := range img.Data {
		img.Data[i] = uint8(rng.Intn(256))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = texture.LawsEnergyMaps(img, 15)
	}
}

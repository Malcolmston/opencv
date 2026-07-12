package photo

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// meanVar returns the sample mean and variance of a single-channel Mat.
func meanVar(m *cv.Mat) (mean, variance float64) {
	n := float64(len(m.Data))
	for _, v := range m.Data {
		mean += float64(v)
	}
	mean /= n
	for _, v := range m.Data {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= n
	return mean, variance
}

// noisyGray returns a flat single-channel image of the given value with additive
// Gaussian noise from a seeded generator.
func noisyGray(rows, cols int, value float64, sigma float64, seed int64) *cv.Mat {
	rng := rand.New(rand.NewSource(seed))
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = clampU8(value + rng.NormFloat64()*sigma)
	}
	return m
}

func TestFastNlMeansDenoisingReducesVariance(t *testing.T) {
	const value = 128.0
	noisy := noisyGray(24, 24, value, 20, 1)
	_, varNoisy := meanVar(noisy)

	den := FastNlMeansDenoising(noisy, 12, 3, 5)
	meanDen, varDen := meanVar(den)

	if den.Channels != 1 || den.Rows != 24 || den.Cols != 24 {
		t.Fatalf("unexpected output shape %dx%dx%d", den.Rows, den.Cols, den.Channels)
	}
	if varDen >= varNoisy {
		t.Errorf("variance did not drop: noisy=%.1f denoised=%.1f", varNoisy, varDen)
	}
	// The flat region's mean must be preserved.
	if math.Abs(meanDen-value) > 4 {
		t.Errorf("mean drifted: got %.2f want ~%.0f", meanDen, value)
	}
	t.Logf("variance %.1f -> %.1f, mean %.2f", varNoisy, varDen, meanDen)
}

func TestFastNlMeansDenoisingSaltPepper(t *testing.T) {
	const value = 100.0
	rng := rand.New(rand.NewSource(7))
	noisy := cv.NewMat(20, 20, 1)
	noisy.SetTo(uint8(value))
	// Sprinkle salt-and-pepper.
	for i := range noisy.Data {
		switch p := rng.Float64(); {
		case p < 0.05:
			noisy.Data[i] = 0
		case p > 0.95:
			noisy.Data[i] = 255
		}
	}
	_, varNoisy := meanVar(noisy)
	den := FastNlMeansDenoising(noisy, 20, 3, 7)
	_, varDen := meanVar(den)
	if varDen >= varNoisy {
		t.Errorf("salt-and-pepper variance did not drop: %.1f -> %.1f", varNoisy, varDen)
	}
}

func TestFastNlMeansDenoisingColored(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	img := cv.NewMat(16, 16, 3)
	base := []uint8{120, 90, 60}
	for p := 0; p < img.Total(); p++ {
		for c := 0; c < 3; c++ {
			img.Data[p*3+c] = clampU8(float64(base[c]) + rng.NormFloat64()*18)
		}
	}
	den := FastNlMeansDenoisingColored(img, 10, 10, 3, 5)
	if den.Channels != 3 {
		t.Fatalf("expected 3 channels, got %d", den.Channels)
	}
	// Per-channel variance should drop for each channel.
	for c := 0; c < 3; c++ {
		_, vn := planeVar(img, c)
		_, vd := planeVar(den, c)
		if vd >= vn {
			t.Errorf("channel %d variance did not drop: %.1f -> %.1f", c, vn, vd)
		}
	}
}

// planeVar returns the mean and variance of one channel of a multi-channel Mat.
func planeVar(m *cv.Mat, c int) (mean, variance float64) {
	n := float64(m.Total())
	for p := 0; p < m.Total(); p++ {
		mean += float64(m.Data[p*m.Channels+c])
	}
	mean /= n
	for p := 0; p < m.Total(); p++ {
		d := float64(m.Data[p*m.Channels+c]) - mean
		variance += d * d
	}
	variance /= n
	return mean, variance
}

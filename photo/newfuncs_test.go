package photo

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// regionVar returns the variance of a single channel over a rectangular region.
func regionVar(m *cv.Mat, y0, x0, h, w, c int) float64 {
	var mean float64
	n := float64(h * w)
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			mean += float64(m.At(y, x, c))
		}
	}
	mean /= n
	var v float64
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			d := float64(m.At(y, x, c)) - mean
			v += d * d
		}
	}
	return v / n
}

// noisyGray3 builds a 3-channel flat image with additive Gaussian noise.
func noisyGray3(rows, cols int, base [3]float64, sigma float64, seed int64) *cv.Mat {
	rng := rand.New(rand.NewSource(seed))
	m := cv.NewMat(rows, cols, 3)
	for i := 0; i < rows*cols; i++ {
		for c := 0; c < 3; c++ {
			m.Data[i*3+c] = clampU8(base[c] + rng.NormFloat64()*sigma)
		}
	}
	return m
}

func TestDomainTransformFilterKeepsEdge(t *testing.T) {
	for _, mode := range []EdgePreservingFlag{RecursFilter, NormconvFilter} {
		img := stepImage(16, 16, 30, 220)
		// Add a little texture on each side so smoothing has work to do.
		rng := rand.New(rand.NewSource(9))
		for i := range img.Data {
			img.Data[i] = clampU8(float64(img.Data[i]) + rng.NormFloat64()*6)
		}
		out := DomainTransformFilter(img, mode, 40, 0.2)
		if out.Rows != 16 || out.Cols != 16 || out.Channels != 1 {
			t.Fatalf("mode %d: bad shape %dx%dx%d", mode, out.Rows, out.Cols, out.Channels)
		}
		// Edge preserved: left dark, right bright.
		if out.At(8, 1, 0) > 90 {
			t.Errorf("mode %d: left side too bright: %d", mode, out.At(8, 1, 0))
		}
		if out.At(8, 14, 0) < 160 {
			t.Errorf("mode %d: right side too dark: %d", mode, out.At(8, 14, 0))
		}
		// Flat regions smoothed: interior variance drops on each side.
		if vd, vo := regionVar(out, 4, 1, 8, 5, 0), regionVar(img, 4, 1, 8, 5, 0); vd >= vo {
			t.Errorf("mode %d: left not smoothed: %.1f -> %.1f", mode, vo, vd)
		}
	}
}

func TestFastNlMeansDenoisingMultiReducesVariance(t *testing.T) {
	const value = 130.0
	var frames []*cv.Mat
	for i := 0; i < 5; i++ {
		frames = append(frames, noisyGray(20, 20, value, 22, int64(100+i)))
	}
	_, varNoisy := meanVar(frames[2])
	den := FastNlMeansDenoisingMulti(frames, 2, 3, 12, 3, 5)
	meanDen, varDen := meanVar(den)
	if den.Rows != 20 || den.Cols != 20 || den.Channels != 1 {
		t.Fatalf("bad shape %dx%dx%d", den.Rows, den.Cols, den.Channels)
	}
	if varDen >= varNoisy {
		t.Errorf("variance did not drop: %.1f -> %.1f", varNoisy, varDen)
	}
	if math.Abs(meanDen-value) > 4 {
		t.Errorf("mean drifted: got %.2f want ~%.0f", meanDen, value)
	}
	t.Logf("multi variance %.1f -> %.1f", varNoisy, varDen)
}

func TestFastNlMeansDenoisingColoredMulti(t *testing.T) {
	base := [3]float64{120, 90, 60}
	var frames []*cv.Mat
	for i := 0; i < 3; i++ {
		frames = append(frames, noisyGray3(14, 14, base, 18, int64(200+i)))
	}
	den := FastNlMeansDenoisingColoredMulti(frames, 1, 3, 10, 10, 3, 5)
	if den.Channels != 3 {
		t.Fatalf("expected 3 channels, got %d", den.Channels)
	}
	for c := 0; c < 3; c++ {
		_, vn := planeVar(frames[1], c)
		_, vd := planeVar(den, c)
		if vd >= vn {
			t.Errorf("channel %d variance did not drop: %.1f -> %.1f", c, vn, vd)
		}
	}
}

func TestDenoiseTVL1(t *testing.T) {
	const value = 128.0
	noisy := noisyGray(24, 24, value, 20, 5)
	_, varNoisy := meanVar(noisy)
	den := DenoiseTVL1([]*cv.Mat{noisy}, 1.0, 40)
	meanDen, varDen := meanVar(den)
	if varDen >= varNoisy {
		t.Errorf("TV-L1 variance did not drop: %.1f -> %.1f", varNoisy, varDen)
	}
	if math.Abs(meanDen-value) > 6 {
		t.Errorf("TV-L1 mean drifted: got %.2f want ~%.0f", meanDen, value)
	}

	// Edge preservation: a clean step must keep its two levels.
	step := stepImage(20, 20, 40, 210)
	out := DenoiseTVL1([]*cv.Mat{step}, 1.5, 40)
	if out.At(10, 2, 0) > 70 {
		t.Errorf("TV-L1 blurred the dark side: %d", out.At(10, 2, 0))
	}
	if out.At(10, 17, 0) < 180 {
		t.Errorf("TV-L1 blurred the bright side: %d", out.At(10, 17, 0))
	}
	t.Logf("TV-L1 variance %.1f -> %.1f", varNoisy, varDen)
}

func TestColorChangeIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	img := cv.NewMat(16, 16, 3)
	for i := range img.Data {
		img.Data[i] = clampU8(120 + rng.NormFloat64()*30)
	}
	mask := cv.NewMat(16, 16, 1)
	for y := 4; y < 12; y++ {
		for x := 4; x < 12; x++ {
			mask.Set(y, x, 0, 255)
		}
	}
	// Unit multipliers reproduce the source (guidance == source gradient).
	same := ColorChange(mask3Src(img), mask, 1, 1, 1)
	var maxDiff int
	for y := 5; y < 11; y++ {
		for x := 5; x < 11; x++ {
			for c := 0; c < 3; c++ {
				d := int(same.At(y, x, c)) - int(img.At(y, x, c))
				if d < 0 {
					d = -d
				}
				if d > maxDiff {
					maxDiff = d
				}
			}
		}
	}
	if maxDiff > 3 {
		t.Errorf("identity ColorChange drifted by %d", maxDiff)
	}

	// Boosting red must change the red channel inside the region.
	boosted := ColorChange(img, mask, 2.0, 1, 1)
	if boosted.At(8, 8, 0) == img.At(8, 8, 0) {
		t.Errorf("red boost had no effect at centre")
	}
}

// mask3Src is a helper returning a clone so the original is untouched.
func mask3Src(img *cv.Mat) *cv.Mat { return img.Clone() }

func TestIlluminationChange(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	img := cv.NewMat(14, 14, 3)
	for i := range img.Data {
		img.Data[i] = clampU8(100 + rng.NormFloat64()*25)
	}
	mask := cv.NewMat(14, 14, 1)
	for y := 3; y < 11; y++ {
		for x := 3; x < 11; x++ {
			mask.Set(y, x, 0, 255)
		}
	}
	out := IlluminationChange(img, mask, 0.2, 0.4)
	if out.Rows != 14 || out.Channels != 3 {
		t.Fatalf("bad shape")
	}
	// The masked interior should be flatter (illumination flattens gradients).
	if vo, vd := regionVar(img, 4, 4, 6, 6, 0), regionVar(out, 4, 4, 6, 6, 0); vd >= vo {
		t.Errorf("illumination did not flatten interior: %.1f -> %.1f", vo, vd)
	}
}

func TestTextureFlattening(t *testing.T) {
	// A flat field with fine checker texture, no strong edges.
	img := cv.NewMat(18, 18, 3)
	for y := 0; y < 18; y++ {
		for x := 0; x < 18; x++ {
			v := uint8(100)
			if (x+y)%2 == 0 {
				v = 116
			}
			for c := 0; c < 3; c++ {
				img.Set(y, x, c, v)
			}
		}
	}
	mask := cv.NewMat(18, 18, 1)
	for y := 3; y < 15; y++ {
		for x := 3; x < 15; x++ {
			mask.Set(y, x, 0, 255)
		}
	}
	out := TextureFlattening(img, mask, 40, 120, 3)
	// Fine texture (gradient ~16 < 40 threshold) is removed, so interior flattens.
	vo := regionVar(img, 5, 5, 8, 8, 0)
	vd := regionVar(out, 5, 5, 8, 8, 0)
	if vd >= vo {
		t.Errorf("texture flattening did not reduce interior variance: %.1f -> %.1f", vo, vd)
	}
	t.Logf("texture variance %.1f -> %.1f", vo, vd)
}

func TestPencilSketch(t *testing.T) {
	// Gradient image so the sketch is non-trivial.
	img := cv.NewMat(20, 20, 3)
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			for c := 0; c < 3; c++ {
				img.Set(y, x, c, clampU8(float64(x)*12))
			}
		}
	}
	gray, color := PencilSketch(img, 40, 0.07, 0.02)
	if gray.Channels != 1 || color.Channels != 3 {
		t.Fatalf("bad channel counts: %d %d", gray.Channels, color.Channels)
	}
	if _, v := meanVar(gray); v < 1 {
		t.Errorf("gray sketch is nearly constant (var %.3f)", v)
	}

	// A flat image dodges to near-white.
	flat := cv.NewMat(8, 8, 1)
	flat.SetTo(120)
	fg, _ := PencilSketch(flat, 40, 0.07, 0.02)
	if fg.At(4, 4, 0) < 230 {
		t.Errorf("flat sketch not near white: %d", fg.At(4, 4, 0))
	}
}

func TestOilPaintingReducesNoise(t *testing.T) {
	noisy := noisyGray(20, 20, 128, 25, 21)
	_, vn := meanVar(noisy)
	out := OilPainting(noisy, 3, 16)
	if out.Rows != 20 || out.Channels != 1 {
		t.Fatalf("bad shape")
	}
	_, vd := meanVar(out)
	if vd >= vn {
		t.Errorf("oil painting did not reduce variance: %.1f -> %.1f", vn, vd)
	}
}

func TestCartoonify(t *testing.T) {
	img := cv.NewMat(16, 16, 3)
	for i := range img.Data {
		img.Data[i] = uint8((i * 5) % 256)
	}
	out := Cartoonify(img, 40, 0.3)
	if out.Rows != 16 || out.Cols != 16 || out.Channels != 3 {
		t.Fatalf("bad shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
	if _, v := planeVar(out, 0); v < 1 {
		t.Errorf("cartoon output is nearly constant (var %.3f)", v)
	}
}

func TestGammaCorrection(t *testing.T) {
	img := cv.NewMat(4, 4, 1)
	img.SetTo(100)
	// gamma < 1 brightens.
	bright := GammaCorrection(img, 0.5)
	if bright.At(0, 0, 0) <= 100 {
		t.Errorf("gamma<1 did not brighten: %d", bright.At(0, 0, 0))
	}
	// gamma == 1 is identity.
	same := GammaCorrection(img, 1)
	if same.At(0, 0, 0) != 100 {
		t.Errorf("gamma=1 not identity: %d", same.At(0, 0, 0))
	}
	// gamma > 1 darkens.
	dark := GammaCorrection(img, 2)
	if dark.At(0, 0, 0) >= 100 {
		t.Errorf("gamma>1 did not darken: %d", dark.At(0, 0, 0))
	}
}

func TestUnsharpMaskIncreasesEdgeContrast(t *testing.T) {
	img := stepImage(12, 12, 90, 150)
	_, vin := meanVar(img)
	out := UnsharpMask(img, 5, 1.0, 1.5)
	_, vout := meanVar(out)
	if vout <= vin {
		t.Errorf("unsharp mask did not increase contrast: %.1f -> %.1f", vin, vout)
	}
}

func TestHistogramStretch(t *testing.T) {
	img := cv.NewMat(1, 4, 1)
	img.Set(0, 0, 0, 80)
	img.Set(0, 1, 0, 100)
	img.Set(0, 2, 0, 140)
	img.Set(0, 3, 0, 180)
	out := HistogramStretch(img)
	if out.At(0, 0, 0) != 0 {
		t.Errorf("min did not map to 0: %d", out.At(0, 0, 0))
	}
	if out.At(0, 3, 0) != 255 {
		t.Errorf("max did not map to 255: %d", out.At(0, 3, 0))
	}
}

func TestGrayWorldWhiteBalance(t *testing.T) {
	// Red-cast image: red mean far above green/blue.
	img := noisyGray3(16, 16, [3]float64{200, 90, 60}, 5, 31)
	out := GrayWorldWhiteBalance(img)
	means := func(m *cv.Mat) [3]float64 {
		var s [3]float64
		for i := 0; i < m.Total(); i++ {
			for c := 0; c < 3; c++ {
				s[c] += float64(m.Data[i*3+c])
			}
		}
		for c := 0; c < 3; c++ {
			s[c] /= float64(m.Total())
		}
		return s
	}
	spread := func(m [3]float64) float64 {
		lo, hi := m[0], m[0]
		for _, v := range m {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		return hi - lo
	}
	before := spread(means(img))
	after := spread(means(out))
	if after >= before {
		t.Errorf("gray-world did not balance channel means: spread %.1f -> %.1f", before, after)
	}
}

func TestSimpleWhiteBalance(t *testing.T) {
	// Values confined to [60,180]; stretching should reach the extremes.
	img := cv.NewMat(1, 13, 1)
	for x := 0; x < 13; x++ {
		img.Set(0, x, 0, clampU8(60+float64(x)*10))
	}
	out := SimpleWhiteBalance(img, 0.02, 0.02)
	var lo, hi uint8 = 255, 0
	for x := 0; x < 13; x++ {
		v := out.At(0, x, 0)
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if lo > 20 || hi < 235 {
		t.Errorf("simple WB did not stretch range: lo=%d hi=%d", lo, hi)
	}
}

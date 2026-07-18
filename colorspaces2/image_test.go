package colorspaces2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// solid builds a rows x cols three-channel Mat filled with the given RGB bytes.
func solid(rows, cols int, r, g, b uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for i := 0; i < rows*cols; i++ {
		m.Data[i*3] = r
		m.Data[i*3+1] = g
		m.Data[i*3+2] = b
	}
	return m
}

// fromPixels builds a 1 x len(px) Mat from a list of RGB byte triples.
func fromPixels(px [][3]uint8) *cv.Mat {
	m := cv.NewMat(1, len(px), 3)
	for i, p := range px {
		m.Data[i*3] = p[0]
		m.Data[i*3+1] = p[1]
		m.Data[i*3+2] = p[2]
	}
	return m
}

func TestGammaLUTKnown(t *testing.T) {
	lut := BuildGammaLUT(2.0)
	if lut[0] != 0 || lut[255] != 255 || lut[64] != 128 {
		t.Errorf("gamma LUT endpoints/mid = %d,%d,%d", lut[0], lut[255], lut[64])
	}
	// gamma == 1 is identity.
	id := BuildGammaLUT(1.0)
	for i := 0; i < 256; i++ {
		if id[i] != uint8(i) {
			t.Fatalf("gamma 1 not identity at %d: %d", i, id[i])
		}
	}
}

func TestGammaCorrectMat(t *testing.T) {
	src := solid(1, 1, 64, 128, 192)
	out := GammaCorrectMat(src, 2.0)
	if out.Data[0] != 128 {
		t.Errorf("gamma corrected R = %d want 128", out.Data[0])
	}
	// Source is unmodified.
	if src.Data[0] != 64 {
		t.Errorf("source mutated")
	}
}

func TestLinearizeRoundTrip(t *testing.T) {
	src := solid(2, 2, 10, 128, 240)
	back := DelinearizeMat(LinearizeMat(src))
	// The intermediate linear image is stored in 8 bits, so dark samples (whose
	// linear values are heavily compressed) round-trip only to a few counts.
	for c := 0; c < 3; c++ {
		if d := int(back.Data[c]) - int(src.Data[c]); d < -4 || d > 4 {
			t.Errorf("linearize round trip channel %d off by %d", c, d)
		}
	}
}

func TestGrayWorldWhiteBalance(t *testing.T) {
	src := solid(2, 2, 100, 150, 200)
	gains := GrayWorldGains(src)
	if !approx(gains[0], 1.5, 1e-9) || !approx(gains[1], 1.0, 1e-9) || !approx(gains[2], 0.75, 1e-9) {
		t.Errorf("gray-world gains = %+v", gains)
	}
	out := GrayWorldWhiteBalance(src)
	// Every channel mean should now be ~150.
	m := ChannelMeans(out)
	for c := 0; c < 3; c++ {
		if !approx(m[c], 150, 1.0) {
			t.Errorf("balanced mean channel %d = %v", c, m[c])
		}
	}
}

func TestWhitePatchWhiteBalance(t *testing.T) {
	src := fromPixels([][3]uint8{{100, 150, 200}, {200, 100, 50}})
	out := WhitePatchWhiteBalance(src)
	max := ChannelMaxima(out)
	for c := 0; c < 3; c++ {
		if max[c] != 255 {
			t.Errorf("white-patch channel %d max = %d want 255", c, max[c])
		}
	}
}

func TestApplyChannelGainsClamp(t *testing.T) {
	src := solid(1, 1, 200, 200, 200)
	out := ApplyChannelGains(src, [3]float64{2, 1, 0.1})
	if out.Data[0] != 255 || out.Data[1] != 200 || out.Data[2] != 20 {
		t.Errorf("channel gains = %d,%d,%d", out.Data[0], out.Data[1], out.Data[2])
	}
}

func TestQuantizeUniform(t *testing.T) {
	src := fromPixels([][3]uint8{{10, 130, 250}})
	out := QuantizeUniform(src, 2)
	// <128 -> bin 0 -> 64; >=128 -> bin 1 -> 191.
	if out.Data[0] != 64 || out.Data[1] != 191 || out.Data[2] != 191 {
		t.Errorf("uniform quantise = %d,%d,%d", out.Data[0], out.Data[1], out.Data[2])
	}
}

func TestPosterize(t *testing.T) {
	src := fromPixels([][3]uint8{{0, 127, 128}, {255, 200, 60}})
	out := Posterize(src, 2)
	want := []uint8{0, 0, 255, 255, 255, 0}
	for i, w := range want {
		if out.Data[i] != w {
			t.Errorf("posterize[%d] = %d want %d", i, out.Data[i], w)
		}
	}
}

func TestMedianCutTwoColors(t *testing.T) {
	src := fromPixels([][3]uint8{{0, 0, 0}, {0, 0, 0}, {255, 255, 255}, {255, 255, 255}})
	p := MedianCut(src, 2)
	if len(p) != 2 {
		t.Fatalf("median cut palette size = %d", len(p))
	}
	// One entry near black, one near white.
	var haveBlack, haveWhite bool
	for _, c := range p {
		if c.R < 0.01 && c.G < 0.01 && c.B < 0.01 {
			haveBlack = true
		}
		if c.R > 0.99 && c.G > 0.99 && c.B > 0.99 {
			haveWhite = true
		}
	}
	if !haveBlack || !haveWhite {
		t.Errorf("median cut palette = %+v", p)
	}
}

func TestKMeansDeterministicTwoColors(t *testing.T) {
	src := fromPixels([][3]uint8{{10, 10, 10}, {12, 8, 11}, {240, 250, 245}, {250, 245, 255}})
	p1, l1 := KMeansQuantize(src, 2, 10, 42)
	p2, l2 := KMeansQuantize(src, 2, 10, 42)
	if len(p1) != 2 {
		t.Fatalf("kmeans palette size = %d", len(p1))
	}
	// Deterministic for the same seed.
	for i := range l1 {
		if l1[i] != l2[i] {
			t.Fatalf("kmeans labels not deterministic at %d", i)
		}
	}
	if !rgbApprox(p1[0], p2[0], 1e-12) || !rgbApprox(p1[1], p2[1], 1e-12) {
		t.Errorf("kmeans centres not deterministic")
	}
	// The two dark pixels share a label distinct from the two light pixels.
	if l1[0] != l1[1] || l1[2] != l1[3] || l1[0] == l1[2] {
		t.Errorf("kmeans clustering wrong: %+v", l1)
	}
}

func TestApplyPaletteNearest(t *testing.T) {
	p := Palette{{0, 0, 0}, {1, 1, 1}}
	src := fromPixels([][3]uint8{{10, 10, 10}, {240, 240, 240}})
	out := ApplyPalette(src, p)
	if out.Data[0] != 0 || out.Data[3] != 255 {
		t.Errorf("apply palette = %d ... %d", out.Data[0], out.Data[3])
	}
	if NearestColorIndex(p, RGB{0.9, 0.9, 0.9}) != 1 {
		t.Errorf("nearest color index wrong")
	}
}

func TestExtractPalette(t *testing.T) {
	// Three light pixels, one dark: light cluster must come first (more common).
	src := fromPixels([][3]uint8{{250, 250, 250}, {248, 250, 252}, {245, 246, 244}, {5, 5, 5}})
	p := ExtractPalette(src, 2)
	if len(p) != 2 {
		t.Fatalf("extract palette size = %d", len(p))
	}
	if p[0].R < 0.5 {
		t.Errorf("most common colour should be light, got %+v", p[0])
	}
}

func TestChromaticAdaptIdentity(t *testing.T) {
	src := solid(1, 1, 120, 90, 200)
	out := ChromaticAdaptMat(src, WhitePointD65, WhitePointD65)
	for c := 0; c < 3; c++ {
		if d := int(out.Data[c]) - int(src.Data[c]); d < -1 || d > 1 {
			t.Errorf("D65->D65 adapt changed channel %d by %d", c, d)
		}
	}
}

func TestMatrix3Inverse(t *testing.T) {
	inv, ok := rgb2xyzMat.Inverse()
	if !ok {
		t.Fatal("rgb2xyz not invertible")
	}
	prod := rgb2xyzMat.Mul(inv)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			want := 0.0
			if i == j {
				want = 1.0
			}
			if !approx(prod[i][j], want, 1e-9) {
				t.Errorf("M*M^-1[%d][%d] = %v", i, j, prod[i][j])
			}
		}
	}
	if _, ok := (Matrix3{}).Inverse(); ok {
		t.Error("zero matrix reported invertible")
	}
}

func TestColorTemperature(t *testing.T) {
	// D65 correlated colour temperature is ~6504K.
	cct := CorrelatedColorTemperature(WhitePointD65)
	if !approx(cct, 6504, 20) {
		t.Errorf("CCT(D65) = %v want ~6504", cct)
	}
	// A warm temperature has more red than blue.
	warm := KelvinToRGB(3000)
	if warm.R <= warm.B {
		t.Errorf("3000K should be warm: %+v", warm)
	}
	// A cool temperature has more blue than a warm one.
	cool := KelvinToRGB(9000)
	if cool.B <= warm.B {
		t.Errorf("9000K should be cooler (more blue) than 3000K")
	}
}

func TestPlanckianCCTConsistency(t *testing.T) {
	// The CCT of the Planckian locus point at 5000K should be close to 5000K.
	xyz := PlanckianXYZ(5000)
	cct := CorrelatedColorTemperature(xyz)
	if math.Abs(cct-5000) > 150 {
		t.Errorf("Planckian/CCT round trip at 5000K = %v", cct)
	}
}

func TestAdjustColorTemperatureNeutral(t *testing.T) {
	src := solid(2, 2, 120, 120, 120)
	out := AdjustColorTemperatureMat(src, 6500)
	// 6500K is (near) neutral, so the image barely changes.
	for c := 0; c < 3; c++ {
		if d := int(out.Data[c]) - int(src.Data[c]); d < -6 || d > 6 {
			t.Errorf("neutral temperature changed channel %d by %d", c, d)
		}
	}
}

func BenchmarkKMeansQuantize(b *testing.B) {
	// A 64x64 gradient image is the heaviest routine's workload.
	m := cv.NewMat(64, 64, 3)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			i := (y*64 + x) * 3
			m.Data[i] = uint8(x * 4)
			m.Data[i+1] = uint8(y * 4)
			m.Data[i+2] = uint8((x + y) * 2)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		KMeansQuantize(m, 8, 10, 1)
	}
}

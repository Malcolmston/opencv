package imghash

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- deterministic fixtures --------------------------------------------------

// gradientImage builds a smooth diagonal ramp with values in [0, rows+cols),
// kept well below 255 so a brightness shift does not clamp.
func gradientImage(rows, cols, channels int) *cv.Mat {
	m := cv.NewMat(rows, cols, channels)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := uint8(x + y)
			for c := 0; c < channels; c++ {
				m.Set(y, x, c, v)
			}
		}
	}
	return m
}

// structuredImage builds a smooth but broadband "photo-like" image from a few
// sinusoids, with per-channel phase offsets so colour images carry real chroma.
// Values stay within [20, 200] so a moderate brightness shift never clamps, and
// the spectral energy is spread across many DCT coefficients so frequency-domain
// hashes are well conditioned (unlike a pure ramp, whose coefficients are nearly
// all zero).
func structuredImage(rows, cols, channels int) *cv.Mat {
	m := cv.NewMat(rows, cols, channels)
	for y := 0; y < rows; y++ {
		fy := float64(y) / float64(rows)
		for x := 0; x < cols; x++ {
			fx := float64(x) / float64(cols)
			for c := 0; c < channels; c++ {
				ph := float64(c)
				v := 110 +
					45*math.Sin(2*math.Pi*2*fx+ph) +
					35*math.Sin(2*math.Pi*3*fy+0.7*ph) +
					25*math.Sin(2*math.Pi*(fx+fy)+0.3*ph)
				if v < 20 {
					v = 20
				} else if v > 200 {
					v = 200
				}
				m.Set(y, x, c, uint8(v))
			}
		}
	}
	return m
}

// checkerImage builds a high-frequency checkerboard: structurally very
// different from a smooth gradient.
func checkerImage(rows, cols, block, channels int) *cv.Mat {
	m := cv.NewMat(rows, cols, channels)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var v uint8
			if ((x/block)+(y/block))%2 == 0 {
				v = 255
			}
			for c := 0; c < channels; c++ {
				m.Set(y, x, c, v)
			}
		}
	}
	return m
}

// shiftBrightness returns a copy of src with delta added to every sample,
// clamped to the 8-bit range.
func shiftBrightness(src *cv.Mat, delta int) *cv.Mat {
	out := src.Clone()
	for i := range out.Data {
		v := int(out.Data[i]) + delta
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		out.Data[i] = uint8(v)
	}
	return out
}

// binaryHashers lists every Hamming-distance hasher under test.
func binaryHashers() map[string]ImgHash {
	return map[string]ImgHash{
		"AverageHash":      AverageHash{},
		"PHash":            PHash{},
		"DHash":            DHash{},
		"BlockMeanHash":    BlockMeanHash{},
		"MarrHildrethHash": MarrHildrethHash{},
	}
}

// bitLen returns the number of bits in a hash.
func bitLen(h []byte) int { return len(h) * 8 }

// --- identical / similar / different (binary hashes) -------------------------

func TestIdenticalImagesHaveZeroDistance(t *testing.T) {
	img := structuredImage(64, 64, 3)
	for name, h := range binaryHashers() {
		a := h.Compute(img)
		b := h.Compute(img.Clone())
		if d := h.Compare(a, b); d != 0 {
			t.Errorf("%s: identical images distance = %v, want 0", name, d)
		}
	}
}

func TestSimilarImagesHaveSmallDistance(t *testing.T) {
	base := structuredImage(64, 64, 1)
	blurred := cv.GaussianBlur(base, 5, 0)
	bright := shiftBrightness(base, 40)
	different := checkerImage(64, 64, 8, 1)

	for name, h := range binaryHashers() {
		hb := h.Compute(base)
		dBlur := h.Compare(hb, h.Compute(blurred))
		dBright := h.Compare(hb, h.Compute(bright))
		dDiff := h.Compare(hb, h.Compute(different))

		similarLimit := float64(bitLen(hb)) / 4 // "similar" cutoff
		if dBlur > similarLimit {
			t.Errorf("%s: blurred distance %v exceeds similar limit %v", name, dBlur, similarLimit)
		}
		if dBright > similarLimit {
			t.Errorf("%s: brightness-shifted distance %v exceeds similar limit %v", name, dBright, similarLimit)
		}
		if !(dDiff > dBlur) {
			t.Errorf("%s: different distance %v should exceed blurred %v", name, dDiff, dBlur)
		}
		if dDiff <= float64(bitLen(hb))/8 {
			t.Errorf("%s: different distance %v unexpectedly small", name, dDiff)
		}
	}
}

// --- aHash hand-computed bits ------------------------------------------------

func TestAverageHashKnownBits(t *testing.T) {
	// An 8×8 image whose columns ramp 0,32,...,224 (constant down each column).
	// Resizing 8×8 to 8×8 is the identity, so the hash is computed on exactly
	// these values. The block mean is (0+32+...+224)/8 = 112, so a pixel is
	// "on" iff its column value exceeds 112: columns 4..7. Every row therefore
	// yields bits 0 0 0 0 1 1 1 1 = 0x0F.
	img := cv.NewMat(8, 8, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			img.Set(y, x, 0, uint8(x*32))
		}
	}
	got := AverageHash{}.Compute(img)
	want := []byte{0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F}
	if len(got) != len(want) {
		t.Fatalf("hash length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("byte %d = %#02x, want %#02x (full hash %#v)", i, got[i], want[i], got)
		}
	}
}

// --- PHash DCT verified on a constant image ----------------------------------

func TestDCT2DConstantIsAllDC(t *testing.T) {
	const n = 8
	const c = 100.0
	in := make([]float64, n*n)
	for i := range in {
		in[i] = c
	}
	out := dct2D(in, n)
	// DC term equals c*N for the orthonormal transform; every AC term is zero.
	if math.Abs(out[0]-c*n) > 1e-9 {
		t.Errorf("DC coefficient = %v, want %v", out[0], c*n)
	}
	for i := 1; i < n*n; i++ {
		if math.Abs(out[i]) > 1e-9 {
			t.Errorf("AC coefficient %d = %v, want 0", i, out[i])
		}
	}
}

func TestPHashConstantImageIsAllDC(t *testing.T) {
	// A constant image has an all-DC spectrum: after the 32×32 DCT the only
	// non-zero low-frequency coefficient the perceptual hash inspects is the DC
	// term at index 0; every alternating-current coefficient is zero (to within
	// floating-point tolerance). This is the frequency-domain property PHash
	// relies on.
	img := cv.NewMat(40, 40, 1)
	img.SetTo(123)
	low := pHashLowFreq(img)
	if low[0] <= 0 {
		t.Fatalf("DC coefficient = %v, want > 0", low[0])
	}
	for i := 1; i < len(low); i++ {
		if math.Abs(low[i]) > 1e-6 {
			t.Errorf("AC coefficient %d = %v, want ~0", i, low[i])
		}
	}
}

// --- real-valued hashes ------------------------------------------------------

func TestRealValuedHashesRankSimilarity(t *testing.T) {
	base := structuredImage(64, 64, 3)
	blurred := cv.GaussianBlur(base, 5, 0)
	different := checkerImage(64, 64, 8, 3)

	realHashers := map[string]ImgHash{
		"RadialVarianceHash": RadialVarianceHash{},
		"ColorMomentHash":    ColorMomentHash{},
	}
	for name, h := range realHashers {
		hb := h.Compute(base)
		if d := h.Compare(hb, h.Compute(base.Clone())); d != 0 {
			t.Errorf("%s: identical images distance = %v, want 0", name, d)
		}
		dBlur := h.Compare(hb, h.Compute(blurred))
		dDiff := h.Compare(hb, h.Compute(different))
		if !(dBlur < dDiff) {
			t.Errorf("%s: blurred distance %v should be < different %v", name, dBlur, dDiff)
		}
	}
}

// --- determinism -------------------------------------------------------------

func TestDeterministic(t *testing.T) {
	img := gradientImage(50, 70, 3)
	all := []ImgHash{
		AverageHash{}, PHash{}, DHash{}, BlockMeanHash{},
		MarrHildrethHash{}, RadialVarianceHash{}, ColorMomentHash{},
	}
	for _, h := range all {
		a := h.Compute(img)
		b := h.Compute(img)
		if len(a) != len(b) {
			t.Fatalf("%T: length changed between runs", h)
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%T: byte %d differs between runs (%d vs %d)", h, i, a[i], b[i])
			}
		}
	}
}

// --- hash lengths ------------------------------------------------------------

func TestHashLengths(t *testing.T) {
	img := gradientImage(48, 48, 3)
	cases := []struct {
		name string
		hash []byte
		want int
	}{
		{"AverageHash", Average(img), 8},
		{"PHash", Perceptual(img), 8},
		{"DHash", Difference(img), 8},
		{"BlockMeanHash", BlockMean(img), 32},
		{"MarrHildrethHash", MarrHildreth(img), 8},
		{"RadialVarianceHash", RadialVariance(img), rvAngles},
		{"ColorMomentHash", ColorMoment(img), 42 * 8},
	}
	for _, c := range cases {
		if len(c.hash) != c.want {
			t.Errorf("%s length = %d, want %d", c.name, len(c.hash), c.want)
		}
	}
}

// --- panics ------------------------------------------------------------------

func TestEmptyImagePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on empty image")
		}
	}()
	PHash{}.Compute(&cv.Mat{})
}

func TestCompareLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on length mismatch")
		}
	}()
	AverageHash{}.Compare([]byte{0, 0}, []byte{0})
}

func TestNewBlockMeanHashInvalidPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on non-positive block count")
		}
	}()
	NewBlockMeanHash(0)
}

package imghash

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// newBinaryHashers lists every added Hamming-distance hasher under test. Each is
// exercised with the same identical/similar/different battery as the built-in
// hashers.
func newBinaryHashers() map[string]ImgHash {
	return map[string]ImgHash{
		"MedianHash":         MedianHash{},
		"WaveletHash":        WaveletHash{},
		"AverageHashN":       NewAverageHashN(16),
		"PHashN":             NewPHashN(8),
		"DHashVertical":      DHashVertical{},
		"DHashCombined":      DHashCombined{},
		"BlockMeanModeHash0": NewBlockMeanModeHash(16, BlockMeanMode0),
		"BlockMeanModeHash1": NewBlockMeanModeHash(16, BlockMeanMode1),
		"MarrHildrethHash72": MarrHildrethHash72{},
	}
}

// noisyImage returns a copy of src with deterministic per-sample noise of at
// most amp added, using a seeded PRNG so the test is fully reproducible.
func noisyImage(src *cv.Mat, amp int, seed int64) *cv.Mat {
	rng := rand.New(rand.NewSource(seed))
	out := src.Clone()
	for i := range out.Data {
		v := int(out.Data[i]) + rng.Intn(2*amp+1) - amp
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		out.Data[i] = uint8(v)
	}
	return out
}

func TestNewBinaryHashersIdenticalSimilarDifferent(t *testing.T) {
	base := structuredImage(64, 64, 1)
	blurred := cv.GaussianBlur(base, 5, 0)
	bright := shiftBrightness(base, 40)
	noisy := noisyImage(base, 8, 20260712)
	different := checkerImage(64, 64, 8, 1)

	for name, h := range newBinaryHashers() {
		hb := h.Compute(base)

		if d := h.Compare(hb, h.Compute(base.Clone())); d != 0 {
			t.Errorf("%s: identical images distance = %v, want 0", name, d)
		}

		dBlur := h.Compare(hb, h.Compute(blurred))
		dBright := h.Compare(hb, h.Compute(bright))
		dNoise := h.Compare(hb, h.Compute(noisy))
		dDiff := h.Compare(hb, h.Compute(different))

		similarLimit := float64(bitLen(hb)) / 4
		// Additive noise scatters high-frequency coefficients, which the
		// frequency-domain hashes feel more than blur or brightness, so it gets a
		// looser bound.
		noiseLimit := float64(bitLen(hb)) / 3
		if dBlur > similarLimit {
			t.Errorf("%s: blurred distance %v exceeds similar limit %v", name, dBlur, similarLimit)
		}
		if dBright > similarLimit {
			t.Errorf("%s: brightness-shifted distance %v exceeds similar limit %v", name, dBright, similarLimit)
		}
		if dNoise > noiseLimit {
			t.Errorf("%s: noisy distance %v exceeds noise limit %v", name, dNoise, noiseLimit)
		}
		if dDiff <= dBlur {
			t.Errorf("%s: different distance %v should exceed blurred %v", name, dDiff, dBlur)
		}
	}
}

func TestNewHashersDeterministic(t *testing.T) {
	img := gradientImage(50, 70, 3)
	all := []ImgHash{
		MedianHash{}, WaveletHash{}, NewAverageHashN(16), NewPHashN(12),
		DHashVertical{}, DHashCombined{}, NewBlockMeanModeHash(8, BlockMeanMode0),
		NewBlockMeanModeHash(8, BlockMeanMode1), MarrHildrethHash72{},
		RadialVarianceCorrHash{}, ColorMomentL2Hash{},
	}
	for _, h := range all {
		a := h.Compute(img)
		b := h.Compute(img)
		if len(a) != len(b) {
			t.Fatalf("%T: length changed between runs", h)
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%T: byte %d differs between runs", h, i)
			}
		}
	}
}

func TestNewHashLengths(t *testing.T) {
	img := gradientImage(48, 48, 3)
	cases := []struct {
		name string
		hash []byte
		want int
	}{
		{"MedianHash", Median(img), 8},
		{"WaveletHash", Wavelet(img), 8},
		{"AverageHashN(16)", AverageN(img, 16), 32},
		{"PHashN(8)", PerceptualN(img, 8), 8},
		{"PHashN(16)", PerceptualN(img, 16), 32},
		{"DHashVertical", DifferenceVertical(img), 8},
		{"DHashCombined", DifferenceCombined(img), 16},
		{"BlockMeanMode0", BlockMeanMode(img, BlockMeanMode0), 32},
		{"BlockMeanMode1", BlockMeanMode(img, BlockMeanMode1), (31*31 + 7) / 8},
		{"MarrHildrethHash72", MarrHildreth72(img), 9},
		{"RadialVarianceCorr", RadialVarianceCorr(img), rvAngles},
		{"ColorMomentL2", ColorMomentL2(img), 42 * 8},
	}
	for _, c := range cases {
		if len(c.hash) != c.want {
			t.Errorf("%s length = %d, want %d", c.name, len(c.hash), c.want)
		}
	}
}

// --- Haar DWT: exact round-trip ----------------------------------------------

func TestHaarDWTRoundTrip(t *testing.T) {
	const n = 8
	rng := rand.New(rand.NewSource(42))
	in := make([]float64, n*n)
	for i := range in {
		in[i] = rng.Float64()*200 - 100
	}
	fwd := HaarDWT2D(in, n)
	back := HaarIDWT2D(fwd, n)
	for i := range in {
		if math.Abs(back[i]-in[i]) > 1e-9 {
			t.Fatalf("round-trip mismatch at %d: got %v want %v", i, back[i], in[i])
		}
	}
	// The forward transform must actually change the data (it is not identity).
	changed := false
	for i := range in {
		if math.Abs(fwd[i]-in[i]) > 1e-9 {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("forward Haar transform left data unchanged")
	}
}

func TestHaarDWTConstantIsPureLL(t *testing.T) {
	// One level of the 2-D Haar transform of a constant block puts all its energy
	// in the LL quadrant (the top-left n/2×n/2 sub matrix), each equal to 2c for
	// the orthonormal normalisation, and zeroes every detail coefficient.
	const n = 4
	const c = 50.0
	const half = n / 2
	in := make([]float64, n*n)
	for i := range in {
		in[i] = c
	}
	out := HaarDWT2D(in, n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			want := 0.0
			if y < half && x < half {
				want = 2 * c // LL quadrant
			}
			if math.Abs(out[y*n+x]-want) > 1e-9 {
				t.Errorf("coefficient (%d,%d) = %v, want %v", y, x, out[y*n+x], want)
			}
		}
	}
}

// --- peak cross-correlation --------------------------------------------------

func TestPeakCrossCorrelationShiftInvariant(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	n := 40
	a := make([]float64, n)
	for i := range a {
		a[i] = rng.NormFloat64()
	}
	// Identical vectors correlate at 1.
	if c := PeakCrossCorrelation(a, a); math.Abs(c-1) > 1e-9 {
		t.Errorf("identical correlation = %v, want 1", c)
	}
	// A circular shift of a still peaks at 1 (rotation invariance).
	shift := 11
	b := make([]float64, n)
	for i := range b {
		b[i] = a[(i+shift)%n]
	}
	if c := PeakCrossCorrelation(a, b); math.Abs(c-1) > 1e-9 {
		t.Errorf("shifted correlation peak = %v, want 1", c)
	}
}

func TestRadialVarianceCorrRanksSimilarity(t *testing.T) {
	base := structuredImage(64, 64, 1)
	blurred := cv.GaussianBlur(base, 5, 0)
	different := checkerImage(64, 64, 8, 1)
	h := RadialVarianceCorrHash{}
	hb := h.Compute(base)
	if d := h.Compare(hb, h.Compute(base.Clone())); math.Abs(d) > 1e-9 {
		t.Errorf("identical distance = %v, want 0", d)
	}
	dBlur := h.Compare(hb, h.Compute(blurred))
	dDiff := h.Compare(hb, h.Compute(different))
	if !(dBlur < dDiff) {
		t.Errorf("blur distance %v should be < different %v", dBlur, dDiff)
	}
}

func TestColorMomentL2RanksSimilarity(t *testing.T) {
	base := structuredImage(64, 64, 3)
	blurred := cv.GaussianBlur(base, 5, 0)
	different := checkerImage(64, 64, 8, 3)
	h := ColorMomentL2Hash{}
	hb := h.Compute(base)
	if d := h.Compare(hb, h.Compute(base.Clone())); d != 0 {
		t.Errorf("identical distance = %v, want 0", d)
	}
	dBlur := h.Compare(hb, h.Compute(blurred))
	dDiff := h.Compare(hb, h.Compute(different))
	if !(dBlur < dDiff) {
		t.Errorf("blur distance %v should be < different %v", dBlur, dDiff)
	}
}

// --- hex round-trip ----------------------------------------------------------

func TestHexRoundTrip(t *testing.T) {
	img := structuredImage(48, 48, 1)
	h := Perceptual(img)
	s := HexEncode(h)
	if len(s) != len(h)*2 {
		t.Fatalf("hex length = %d, want %d", len(s), len(h)*2)
	}
	back, err := HexDecode(s)
	if err != nil {
		t.Fatalf("HexDecode: %v", err)
	}
	if len(back) != len(h) {
		t.Fatalf("decoded length = %d, want %d", len(back), len(h))
	}
	for i := range h {
		if back[i] != h[i] {
			t.Fatalf("byte %d differs after hex round-trip", i)
		}
	}
	if _, err := HexDecode("xyz"); err == nil {
		t.Errorf("expected error decoding invalid hex")
	}
}

// --- normalised distance / similarity / duplicate ----------------------------

func TestNormalizedSimilarityDuplicate(t *testing.T) {
	base := structuredImage(64, 64, 1)
	blurred := cv.GaussianBlur(base, 5, 0)
	different := checkerImage(64, 64, 8, 1)

	hb := Perceptual(base)
	hBlur := Perceptual(blurred)
	hDiff := Perceptual(different)

	if nd := HammingNormalized(hb, hb); nd != 0 {
		t.Errorf("self normalized distance = %v, want 0", nd)
	}
	if s := Similarity(hb, hb); s != 1 {
		t.Errorf("self similarity = %v, want 1", s)
	}
	ndBlur := HammingNormalized(hb, hBlur)
	ndDiff := HammingNormalized(hb, hDiff)
	if !(ndBlur >= 0 && ndBlur <= 1) {
		t.Errorf("normalized distance %v out of range", ndBlur)
	}
	if !(ndBlur < ndDiff) {
		t.Errorf("blur normalized distance %v should be < different %v", ndBlur, ndDiff)
	}
	if math.Abs(Similarity(hb, hBlur)-(1-ndBlur)) > 1e-12 {
		t.Errorf("Similarity inconsistent with HammingNormalized")
	}
	if !IsDuplicate(hb, hBlur, 0.25) {
		t.Errorf("blurred copy should be a duplicate within 0.25")
	}
	if IsDuplicate(hb, hDiff, 0.25) {
		t.Errorf("checkerboard should not be a duplicate within 0.25")
	}
}

// --- constructor panics ------------------------------------------------------

func TestNewConstructorsPanic(t *testing.T) {
	cases := map[string]func(){
		"AverageHashN(0)":        func() { NewAverageHashN(0) },
		"PHashN(0)":              func() { NewPHashN(0) },
		"PHashN(tooBig)":         func() { NewPHashN(pHashSize + 1) },
		"BlockMeanMode(0)":       func() { NewBlockMeanModeHash(0, BlockMeanMode0) },
		"BlockMeanMode(badMode)": func() { NewBlockMeanModeHash(8, 9) },
		"HaarDWT2D(odd)":         func() { HaarDWT2D(make([]float64, 9), 3) },
	}
	for name, fn := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: expected panic", name)
				}
			}()
			fn()
		}()
	}
}

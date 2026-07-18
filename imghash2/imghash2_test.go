package imghash2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- image builders -------------------------------------------------------

// grayMat builds a single-channel Mat of the given size from a value function.
func grayMat(rows, cols int, f func(y, x int) uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = f(y, x)
		}
	}
	return m
}

// halfHalf8 is an 8x8 image whose left four columns are 0 and right four 255.
func halfHalf8() *cv.Mat {
	return grayMat(8, 8, func(y, x int) uint8 {
		if x >= 4 {
			return 255
		}
		return 0
	})
}

// hRamp builds a rows×cols horizontal ramp increasing left to right in [0,255].
func hRamp(rows, cols int) *cv.Mat {
	return grayMat(rows, cols, func(y, x int) uint8 {
		return uint8((x * 255) / (cols - 1))
	})
}

// --- distance / hash type -------------------------------------------------

func TestHammingAndBits(t *testing.T) {
	a := Hash{0x00}
	b := Hash{0xFF}
	if d := a.Hamming(b); d != 8 {
		t.Fatalf("Hamming = %d, want 8", d)
	}
	if a.Hamming(a) != 0 {
		t.Fatalf("self Hamming must be 0")
	}
	if got := (Hash{0xF0}).OnesCount(); got != 4 {
		t.Fatalf("OnesCount = %d, want 4", got)
	}
	if !(Hash{0x80}).Bit(0) || (Hash{0x80}).Bit(7) {
		t.Fatalf("Bit MSB-first packing wrong")
	}
	if got := (Hash{0x00, 0xFF}).NormalizedHamming(Hash{0xFF, 0xFF}); got != 0.5 {
		t.Fatalf("NormalizedHamming = %v, want 0.5", got)
	}
	if got := Similarity(Hash{0x00}, Hash{0x0F}); got != 0.5 {
		t.Fatalf("Similarity = %v, want 0.5", got)
	}
	if PopCount([]byte{0xFF, 0x01}) != 9 {
		t.Fatalf("PopCount wrong")
	}
}

func TestHashHexRoundTrip(t *testing.T) {
	h := Hash{0x0f, 0xa1, 0xff, 0x00}
	got, err := ParseHash(h.String())
	if err != nil {
		t.Fatal(err)
	}
	if !got.Equal(h) {
		t.Fatalf("hex round trip: got %s want %s", got, h)
	}
	if _, err := ParseHash("xyz"); err == nil {
		t.Fatalf("ParseHash should reject invalid hex")
	}
}

func TestFloatHashDistances(t *testing.T) {
	a := FloatHash{0, 0, 0}
	b := FloatHash{3, 4, 0}
	if got := a.L1(b); got != 7 {
		t.Fatalf("L1 = %v, want 7", got)
	}
	if got := a.L2(b); got != 5 {
		t.Fatalf("L2 = %v, want 5", got)
	}
	rt := ParseFloatHash(b.Bytes())
	if rt.L1(b) != 0 {
		t.Fatalf("FloatHash byte round trip failed: %v", rt)
	}
}

func TestStats(t *testing.T) {
	v := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	if Mean(v) != 5 {
		t.Fatalf("Mean = %v, want 5", Mean(v))
	}
	if Variance(v) != 4 {
		t.Fatalf("Variance = %v, want 4", Variance(v))
	}
	if StdDev(v) != 2 {
		t.Fatalf("StdDev = %v, want 2", StdDev(v))
	}
	if Median([]float64{3, 1, 2}) != 2 {
		t.Fatalf("odd median wrong")
	}
	if Median([]float64{1, 2, 3, 4}) != 2.5 {
		t.Fatalf("even median wrong")
	}
}

// --- transforms -----------------------------------------------------------

func TestDCTConstant(t *testing.T) {
	// Constant block: only the DC term is non-zero.
	n := 8
	in := make([]float64, n*n)
	for i := range in {
		in[i] = 100
	}
	c := DCT2D(in, n)
	if math.Abs(c[0]-100*float64(n)) > 1e-9 {
		t.Fatalf("DC = %v, want %v", c[0], 100*float64(n))
	}
	for i := 1; i < len(c); i++ {
		if math.Abs(c[i]) > 1e-9 {
			t.Fatalf("AC[%d] = %v, want ~0", i, c[i])
		}
	}
}

func TestDCTRoundTrip(t *testing.T) {
	in := []float64{10, 20, 33, 47, 5, 99, 1, 200}
	rt := IDCT1D(DCT1D(in))
	for i := range in {
		if math.Abs(rt[i]-in[i]) > 1e-9 {
			t.Fatalf("IDCT1D∘DCT1D[%d] = %v, want %v", i, rt[i], in[i])
		}
	}
	n := 4
	blk := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	rt2 := IDCT2D(DCT2D(blk, n), n)
	for i := range blk {
		if math.Abs(rt2[i]-blk[i]) > 1e-9 {
			t.Fatalf("IDCT2D∘DCT2D[%d] = %v, want %v", i, rt2[i], blk[i])
		}
	}
}

func TestHaar2x2KnownAnswer(t *testing.T) {
	got := HaarDWT2D([]float64{1, 2, 3, 4}, 2)
	want := []float64{5, -1, -2, 0}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("HaarDWT2D[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	rt := HaarIDWT2D(got, 2)
	for i, v := range []float64{1, 2, 3, 4} {
		if math.Abs(rt[i]-v) > 1e-12 {
			t.Fatalf("Haar round trip[%d] = %v, want %v", i, rt[i], v)
		}
	}
}

// --- binary hashers: hand-verifiable known answers ------------------------

func TestAverageHashKnownAnswer(t *testing.T) {
	// 8x8 (resize is identity): left half < mean, right half > mean.
	// Each row -> 0b00001111 = 0x0F.
	h := Average(halfHalf8())
	want := Hash{0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f}
	if !h.Equal(want) {
		t.Fatalf("aHash = %s, want %s", h, want)
	}
	if NewAverageHash().Bits() != 64 || NewAverageHash().Name() != "ahash" {
		t.Fatalf("aHash metadata wrong")
	}
}

func TestDifferenceHashKnownAnswer(t *testing.T) {
	// 8 rows x 9 cols strictly increasing left->right: every right>left -> all 1s.
	hor := grayMat(8, 9, func(y, x int) uint8 { return uint8(x * 20) })
	if got := Difference(hor); !got.Equal(Hash{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}) {
		t.Fatalf("horizontal dHash = %s, want all ones", got)
	}
	// 9 rows x 8 cols strictly increasing top->bottom: every below>above -> all 1s.
	ver := grayMat(9, 8, func(y, x int) uint8 { return uint8(y * 20) })
	if got := NewVerticalDifferenceHash().Compute(ver); !got.Equal(Hash{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}) {
		t.Fatalf("vertical dHash = %s, want all ones", got)
	}
	if NewVerticalDifferenceHash().Name() != "dhash-v" {
		t.Fatalf("vertical name wrong")
	}
}

func TestWaveletHashKnownAnswer(t *testing.T) {
	// A horizontal ramp: the 8x8 LL band is itself a horizontal ramp, so each
	// row splits into a low half (below median) and high half (above) -> 0x0F.
	h := Wavelet(hRamp(32, 32))
	want := Hash{0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f}
	if !h.Equal(want) {
		t.Fatalf("wHash ramp = %s, want %s", h, want)
	}
}

func TestBlockMeanHashKnownAnswer(t *testing.T) {
	// Horizontal ramp on a 16x16 grid: in every grid row the left 8 cells fall
	// below the global mean and the right 8 above -> 0x00,0xFF per row.
	h := BlockMean(hRamp(128, 128))
	if h.Bits() != 256 {
		t.Fatalf("blockmean bits = %d, want 256", h.Bits())
	}
	for row := 0; row < 16; row++ {
		if h[row*2] != 0x00 || h[row*2+1] != 0xff {
			t.Fatalf("blockmean row %d = %02x%02x, want 00ff", row, h[row*2], h[row*2+1])
		}
	}
}

func TestPHashRampRegression(t *testing.T) {
	// Golden value produced by this implementation on a fixed 32x32 ramp; guards
	// against unintended changes to the DCT front end.
	h := NewPHash().Compute(hRamp(32, 32))
	want, _ := ParseHash("aaa8472afcaaaaaa")
	if !h.Equal(want) {
		t.Fatalf("pHash ramp = %s, want %s", h, want)
	}
	// Self distance is always zero and the hash is 64 bits.
	if h.Hamming(h) != 0 || h.Bits() != 64 {
		t.Fatalf("pHash self/bits wrong")
	}
}

func TestMarrHildrethRampRegression(t *testing.T) {
	h := MarrHildreth(hRamp(32, 32))
	want, _ := ParseHash("7b7b7b7b7b7b7b7b")
	if !h.Equal(want) {
		t.Fatalf("marr ramp = %s, want %s", h, want)
	}
}

// --- hashers respond to change --------------------------------------------

func TestHashersDistinguishImages(t *testing.T) {
	ramp := hRamp(64, 64)
	flat := grayMat(64, 64, func(y, x int) uint8 { return 128 })
	hashers := []Hasher{
		NewAverageHash(), NewPHash(), NewDifferenceHash(),
		NewWaveletHash(), NewBlockMeanHash(8), NewMarrHildrethHash(),
	}
	for _, hh := range hashers {
		a := hh.Compute(ramp)
		b := hh.Compute(flat)
		if a.Hamming(b) == 0 {
			t.Fatalf("%s failed to distinguish ramp from flat", hh.Name())
		}
		if a.Hamming(hh.Compute(ramp)) != 0 {
			t.Fatalf("%s not deterministic", hh.Name())
		}
		if a.Bits() != hh.Bits() {
			t.Fatalf("%s bit length mismatch", hh.Name())
		}
	}
}

// --- radial variance / rotation robustness --------------------------------

func TestRadialCrossCorrelation(t *testing.T) {
	rv := RadialVariance(hRamp(48, 48))
	if rv.Dims() != 40 {
		t.Fatalf("radial dims = %d, want 40", rv.Dims())
	}
	if c := RadialCrossCorrelation(rv, rv); math.Abs(c-1) > 1e-12 {
		t.Fatalf("self cross-correlation = %v, want 1", c)
	}
	// A cyclically shifted profile still correlates at ~1 under the best shift.
	shifted := make(FloatHash, len(rv))
	for i := range rv {
		shifted[i] = rv[(i+7)%len(rv)]
	}
	if c := RadialCrossCorrelation(rv, shifted); c < 0.999 {
		t.Fatalf("shifted cross-correlation = %v, want ~1", c)
	}
}

// --- colour moments -------------------------------------------------------

func TestColorMomentAndHu(t *testing.T) {
	rgb := cv.NewMat(24, 24, 3)
	for i := range rgb.Data {
		rgb.Data[i] = uint8((i * 7) % 256)
	}
	cm := ColorMoment(rgb)
	if cm.Dims() != 42 {
		t.Fatalf("colour moment dims = %d, want 42", cm.Dims())
	}
	if cm.L1(ColorMoment(rgb)) != 0 {
		t.Fatalf("colour moment not deterministic")
	}
	// Hu[0] of a non-empty intensity plane is strictly positive.
	if hu := HuMoments(rgb, 0); hu[0] <= 0 {
		t.Fatalf("Hu[0] = %v, want > 0", hu[0])
	}
}

// --- near-duplicate detection ---------------------------------------------

func TestIsDuplicateThresholds(t *testing.T) {
	a := Hash{0x00, 0x00}
	b := Hash{0x00, 0x03} // 2 bits differ
	if !IsDuplicate(a, b, 2) || IsDuplicate(a, b, 1) {
		t.Fatalf("IsDuplicate threshold wrong")
	}
	if !NearDuplicate(a, b, 2.0/16) || NearDuplicate(a, b, 1.0/16) {
		t.Fatalf("NearDuplicate threshold wrong")
	}
}

func TestFindDuplicates(t *testing.T) {
	hashes := []Hash{
		{0x00}, // 0
		{0x01}, // 1  (1 bit from 0)
		{0xF0}, // 2  (far)
		{0xF1}, // 3  (1 bit from 2)
	}
	groups := FindDuplicates(hashes, 1)
	if len(groups) != 2 {
		t.Fatalf("groups = %v, want 2 clusters", groups)
	}
	// First group starts at index 0 and contains {0,1}.
	if len(groups[0]) != 2 || groups[0][0] != 0 || groups[0][1] != 1 {
		t.Fatalf("group0 = %v, want [0 1]", groups[0])
	}
	if len(groups[1]) != 2 || groups[1][0] != 2 || groups[1][1] != 3 {
		t.Fatalf("group1 = %v, want [2 3]", groups[1])
	}
}

func TestDuplicatePairs(t *testing.T) {
	entries := []Entry{
		{ID: "a", Hash: Hash{0x00}},
		{ID: "b", Hash: Hash{0x01}},
		{ID: "c", Hash: Hash{0xFF}},
	}
	pairs := DuplicatePairs(entries, 1)
	if len(pairs) != 1 || pairs[0].A != "a" || pairs[0].B != "b" || pairs[0].Distance != 1 {
		t.Fatalf("pairs = %v, want single a-b pair", pairs)
	}
}

func TestIndexNearestAndWithin(t *testing.T) {
	data := map[string]Hash{
		"a": {0x00, 0x00},
		"b": {0x00, 0x01},
		"c": {0x0F, 0xF0},
		"d": {0xFF, 0xFF},
	}
	idx := NewIndex()
	for id, h := range data {
		idx.Add(id, h)
	}
	if idx.Len() != 4 || idx.Bits() != 16 {
		t.Fatalf("index Len/Bits wrong: %d %d", idx.Len(), idx.Bits())
	}

	// Nearest to {0x00,0x00} is "a" at distance 0.
	q := Hash{0x00, 0x00}
	m, ok := idx.Nearest(q)
	if !ok || m.ID != "a" || m.Distance != 0 {
		t.Fatalf("Nearest = %+v ok=%v, want a@0", m, ok)
	}

	// Within radius 1 of q: a (0) and b (1), sorted by distance.
	within := idx.Within(q, 1)
	if len(within) != 2 || within[0].ID != "a" || within[1].ID != "b" {
		t.Fatalf("Within = %+v, want [a b]", within)
	}

	// Cross-check Nearest for a random query against brute force.
	query := Hash{0x0F, 0x0F}
	bestID, bestD := bruteNearest(data, query)
	gm, _ := idx.Nearest(query)
	if gm.Distance != bestD {
		t.Fatalf("Nearest distance = %d, brute = %d (id %s)", gm.Distance, bestD, bestID)
	}
}

func bruteNearest(data map[string]Hash, q Hash) (string, int) {
	bestID, bestD := "", 1<<30
	for id, h := range data {
		if d := q.Hamming(h); d < bestD {
			bestID, bestD = id, d
		}
	}
	return bestID, bestD
}

func TestIndexEmpty(t *testing.T) {
	idx := NewIndex()
	if _, ok := idx.Nearest(Hash{0x00}); ok {
		t.Fatalf("empty index Nearest should report not found")
	}
	if idx.Within(Hash{0x00}, 3) != nil {
		t.Fatalf("empty index Within should be nil")
	}
}

// --- benchmark: colour-moment hash is the heaviest routine ----------------

func BenchmarkColorMomentHash(b *testing.B) {
	img := cv.NewMat(256, 256, 3)
	for i := range img.Data {
		img.Data[i] = uint8((i * 13) % 256)
	}
	h := NewColorMomentHash()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.ComputeFloat(img)
	}
}

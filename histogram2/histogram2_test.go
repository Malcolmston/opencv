package histogram2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// grayFrom builds a single-channel cv.Mat of the given dimensions from a flat
// row-major slice of samples.
func grayFrom(rows, cols int, data []uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	copy(m.Data, data)
	return m
}

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestCalcHist1D(t *testing.T) {
	m := grayFrom(1, 6, []uint8{10, 10, 20, 20, 20, 30})
	h, err := CalcHist1D(m, 0, 256)
	if err != nil {
		t.Fatal(err)
	}
	if h.Counts[10] != 2 || h.Counts[20] != 3 || h.Counts[30] != 1 {
		t.Fatalf("bad counts: %v %v %v", h.Counts[10], h.Counts[20], h.Counts[30])
	}
	if h.Total() != 6 {
		t.Fatalf("total = %v want 6", h.Total())
	}
	if h.Peak() != 20 {
		t.Fatalf("peak = %d want 20", h.Peak())
	}
}

func TestCalcHist1DBinning(t *testing.T) {
	// Values 0..255 spread over 4 bins => 64 samples per bin.
	data := make([]uint8, 256)
	for i := range data {
		data[i] = uint8(i)
	}
	h, _ := CalcHist1D(grayFrom(1, 256, data), 0, 4)
	for i := 0; i < 4; i++ {
		if h.Counts[i] != 64 {
			t.Fatalf("bin %d = %v want 64", i, h.Counts[i])
		}
	}
}

func TestCalcHist1DMasked(t *testing.T) {
	m := grayFrom(1, 4, []uint8{5, 5, 9, 9})
	mask := grayFrom(1, 4, []uint8{255, 0, 255, 0})
	h, err := CalcHist1DMasked(m, 0, 256, mask)
	if err != nil {
		t.Fatal(err)
	}
	if h.Counts[5] != 1 || h.Counts[9] != 1 || h.Total() != 2 {
		t.Fatalf("masked counts wrong: total=%v", h.Total())
	}
}

func TestHistogramStats(t *testing.T) {
	// Uniform over 4 bins: entropy = 2 bits.
	h := NewHistogram1D(4, 0, 256)
	for i := range h.Counts {
		h.Counts[i] = 10
	}
	if e := h.Entropy(); !approx(e, 2.0, 1e-9) {
		t.Fatalf("entropy = %v want 2", e)
	}
	// Cumulative distribution reaches 1 at the last bin.
	cdf := CumulativeDistribution(h)
	if !approx(cdf[3], 1.0, 1e-12) {
		t.Fatalf("cdf end = %v want 1", cdf[3])
	}
	cum := h.Cumulative()
	if cum.Counts[3] != 40 {
		t.Fatalf("cumulative end = %v want 40", cum.Counts[3])
	}
}

func TestEqualizeHistKnown(t *testing.T) {
	m := grayFrom(1, 6, []uint8{10, 10, 20, 20, 20, 30})
	out, err := EqualizeHist(m)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{0, 0, 191, 191, 191, 255}
	for i, w := range want {
		if out.Data[i] != w {
			t.Fatalf("equalize[%d] = %d want %d", i, out.Data[i], w)
		}
	}
}

func TestEqualizeHistIdentitySpread(t *testing.T) {
	m := grayFrom(2, 2, []uint8{0, 85, 170, 255})
	out, _ := EqualizeHist(m)
	want := []uint8{0, 85, 170, 255}
	for i, w := range want {
		if out.Data[i] != w {
			t.Fatalf("equalize[%d] = %d want %d", i, out.Data[i], w)
		}
	}
}

func TestEqualizeHistConstant(t *testing.T) {
	m := grayFrom(1, 4, []uint8{100, 100, 100, 100})
	out, _ := EqualizeHist(m)
	for i, v := range out.Data {
		if v != 100 {
			t.Fatalf("constant equalize[%d] = %d want 100", i, v)
		}
	}
}

func TestEqualizeHistLuminanceGray(t *testing.T) {
	// A pure-gray RGB image should behave like grayscale equalisation.
	rgb := cv.NewMat(1, 6, 3)
	grays := []uint8{10, 10, 20, 20, 20, 30}
	for i, g := range grays {
		rgb.Data[i*3] = g
		rgb.Data[i*3+1] = g
		rgb.Data[i*3+2] = g
	}
	out, err := EqualizeHistLuminance(rgb)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{0, 0, 191, 191, 191, 255}
	for i, w := range want {
		if out.Data[i*3] != w {
			t.Fatalf("lum equalize[%d] = %d want %d", i, out.Data[i*3], w)
		}
	}
}

func TestCLAHEConstantNoClip(t *testing.T) {
	// With clipping disabled a constant non-zero image maps every pixel to 255
	// (its single populated bin sits at the top of the CDF).
	m := grayFrom(4, 4, nil)
	m.SetTo(120)
	c := NewCLAHE(0, 2, 2)
	out, err := c.Apply(m)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out.Data {
		if v != 255 {
			t.Fatalf("clahe const[%d] = %d want 255", i, v)
		}
	}
}

func TestCLAHEShapeAndRange(t *testing.T) {
	m := grayFrom(8, 8, nil)
	for i := range m.Data {
		m.Data[i] = uint8((i * 4) % 256)
	}
	c := NewCLAHE(4, 4, 4)
	out, err := c.Apply(m)
	if err != nil {
		t.Fatal(err)
	}
	if out.Rows != 8 || out.Cols != 8 || out.Channels != 1 {
		t.Fatalf("clahe shape wrong: %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
}

func TestCompareMetrics(t *testing.T) {
	a := []float64{1, 2, 3}
	if c := Correlation(a, a); !approx(c, 1, 1e-12) {
		t.Fatalf("corr self = %v want 1", c)
	}
	if c := Correlation(a, []float64{3, 2, 1}); !approx(c, -1, 1e-12) {
		t.Fatalf("corr rev = %v want -1", c)
	}
	if v := Intersection(a, []float64{2, 2, 2}); v != 5 {
		t.Fatalf("intersect = %v want 5", v)
	}
	if v := ChiSquare([]float64{4, 0, 2}, []float64{4, 0, 2}); v != 0 {
		t.Fatalf("chi self = %v want 0", v)
	}
	if v := Bhattacharyya(a, a); !approx(v, 0, 1e-9) {
		t.Fatalf("bhatt self = %v want 0", v)
	}
	if v := KLDivergence(a, a); !approx(v, 0, 1e-12) {
		t.Fatalf("kl self = %v want 0", v)
	}
}

func TestEMD1DKnown(t *testing.T) {
	// Moving all mass three bins to the right costs 3 units of work.
	a := []float64{1, 0, 0, 0}
	b := []float64{0, 0, 0, 1}
	if v := EMD1D(a, b); !approx(v, 3, 1e-12) {
		t.Fatalf("emd = %v want 3", v)
	}
}

func TestCompareHistDispatch(t *testing.T) {
	a := NewHistogram1D(3, 0, 3)
	b := NewHistogram1D(3, 0, 3)
	copy(a.Counts, []float64{1, 2, 3})
	copy(b.Counts, []float64{1, 2, 3})
	v, err := CompareHist1D(a, b, CompareCorrelation)
	if err != nil || !approx(v, 1, 1e-12) {
		t.Fatalf("compare corr = %v err=%v", v, err)
	}
	if _, err := CompareHist1D(a, NewHistogram1D(4, 0, 4), CompareCorrelation); err != ErrSizeMismatch {
		t.Fatalf("want size mismatch, got %v", err)
	}
	if _, err := CompareHist1D(a, b, CompareMethod(99)); err != ErrInvalidArgument {
		t.Fatalf("want invalid arg, got %v", err)
	}
}

func TestMatchHistogramsIdentity(t *testing.T) {
	// Matching an image to itself must be a near-identity mapping.
	data := []uint8{0, 40, 40, 120, 200, 255}
	src := grayFrom(1, 6, data)
	out, err := MatchHistograms(src, src)
	if err != nil {
		t.Fatal(err)
	}
	for i := range data {
		if out.Data[i] != data[i] {
			t.Fatalf("match[%d] = %d want %d", i, out.Data[i], data[i])
		}
	}
}

func TestSpecifyAndBuildLUT(t *testing.T) {
	src := grayFrom(1, 4, []uint8{0, 0, 0, 0})
	target := NewHistogram1D(256, 0, 256)
	target.Counts[255] = 1
	out, err := SpecifyHistogram(src, target)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out.Data {
		if v != 255 {
			t.Fatalf("specify[%d] = %d want 255", i, v)
		}
	}
}

func TestBackProject1D(t *testing.T) {
	src := grayFrom(1, 4, []uint8{10, 20, 20, 30})
	h, _ := CalcHist1D(src, 0, 256)
	bp, err := CalcBackProject1D(src, 0, h)
	if err != nil {
		t.Fatal(err)
	}
	// Bin 20 is the max (count 2) -> 255; bins 10 and 30 (count 1) -> ~128.
	if bp.Data[1] != 255 || bp.Data[2] != 255 {
		t.Fatalf("backproject peak wrong: %d %d", bp.Data[1], bp.Data[2])
	}
	if bp.Data[0] == 0 || bp.Data[0] >= 255 {
		t.Fatalf("backproject non-peak = %d", bp.Data[0])
	}
}

func TestBackProject2D(t *testing.T) {
	rgb := cv.NewMat(1, 2, 3)
	rgb.SetPixel(0, 0, []uint8{10, 200, 0})
	rgb.SetPixel(0, 1, []uint8{10, 200, 0})
	h, err := CalcHist2D(rgb, 0, 1, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	bp, err := CalcBackProject2D(rgb, 0, 1, h)
	if err != nil {
		t.Fatal(err)
	}
	if bp.Data[0] != 255 || bp.Data[1] != 255 {
		t.Fatalf("2d backproject = %d %d want 255", bp.Data[0], bp.Data[1])
	}
}

func TestHist2D3DTotals(t *testing.T) {
	rgb := cv.NewMat(2, 2, 3)
	for p := 0; p < 4; p++ {
		rgb.Data[p*3] = uint8(p * 60)
		rgb.Data[p*3+1] = uint8(p * 60)
		rgb.Data[p*3+2] = uint8(p * 60)
	}
	h2, _ := CalcHist2D(rgb, 0, 1, 4, 4)
	if h2.Total() != 4 {
		t.Fatalf("hist2d total = %v want 4", h2.Total())
	}
	h3, _ := CalcHist3D(rgb, 4)
	if h3.Total() != 4 {
		t.Fatalf("hist3d total = %v want 4", h3.Total())
	}
	h3.Normalize()
	if !approx(h3.Total(), 1, 1e-12) {
		t.Fatalf("hist3d norm total = %v want 1", h3.Total())
	}
}

func TestGammaIdentity(t *testing.T) {
	m := grayFrom(1, 4, []uint8{0, 50, 128, 255})
	out, err := GammaCorrect(m, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range m.Data {
		if out.Data[i] != m.Data[i] {
			t.Fatalf("gamma1[%d] = %d want %d", i, out.Data[i], m.Data[i])
		}
	}
}

func TestContrastStretchRange(t *testing.T) {
	// Range width 51 gives an exact scale of 5 (255/51), so the midpoint maps
	// cleanly without floating-point rounding ambiguity.
	m := grayFrom(1, 3, []uint8{100, 125, 151})
	out, err := ContrastStretchRange(m, 100, 151)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{0, 125, 255}
	for i, w := range want {
		if out.Data[i] != w {
			t.Fatalf("stretch[%d] = %d want %d", i, out.Data[i], w)
		}
	}
}

func TestMinMaxStretch(t *testing.T) {
	m := grayFrom(1, 3, []uint8{40, 60, 80})
	out, err := MinMaxStretch(m)
	if err != nil {
		t.Fatal(err)
	}
	if out.Data[0] != 0 || out.Data[2] != 255 {
		t.Fatalf("minmax = %d..%d want 0..255", out.Data[0], out.Data[2])
	}
}

func TestContrastStretchPercentile(t *testing.T) {
	data := make([]uint8, 100)
	for i := range data {
		data[i] = uint8(i + 50) // 50..149
	}
	out, err := ContrastStretch(grayFrom(1, 100, data), 0, 100)
	if err != nil {
		t.Fatal(err)
	}
	if out.Data[0] != 0 {
		t.Fatalf("stretch low = %d want 0", out.Data[0])
	}
	if out.Data[99] != 255 {
		t.Fatalf("stretch high = %d want 255", out.Data[99])
	}
}

func TestHOGShapeAndNorm(t *testing.T) {
	// Vertical edge: left half dark, right half bright.
	m := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			if x >= 8 {
				m.Data[y*16+x] = 255
			}
		}
	}
	d := NewHOGDescriptor(8, 2, 9)
	feat, err := d.Compute(m)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(feat), d.FeatureLength(16, 16); got != want {
		t.Fatalf("feature len = %d want %d", got, want)
	}
	// The single block's normalised vector must have unit L2 norm.
	blockLen := 2 * 2 * 9
	for b := 0; b*blockLen < len(feat); b++ {
		var norm float64
		for i := 0; i < blockLen; i++ {
			v := feat[b*blockLen+i]
			norm += v * v
		}
		if norm > 0 && !approx(math.Sqrt(norm), 1, 1e-3) {
			t.Fatalf("block %d norm = %v want ~1", b, math.Sqrt(norm))
		}
	}
}

func TestHOGConvenienceAndTooSmall(t *testing.T) {
	m := cv.NewMat(16, 16, 1)
	m.SetTo(30)
	if _, err := HOG(m, 8, 9); err != nil {
		t.Fatal(err)
	}
	// 8x8 with 8px cells => 1 cell each way, cannot fit a 2x2 block.
	if _, err := HOG(cv.NewMat(8, 8, 1), 8, 9); err != ErrInvalidArgument {
		t.Fatalf("want invalid argument for tiny image, got %v", err)
	}
}

func TestErrorPaths(t *testing.T) {
	if _, err := EqualizeHist(cv.NewMat(2, 2, 3)); err != ErrChannelRange {
		t.Fatalf("want channel range, got %v", err)
	}
	if _, err := CalcHist1D(&cv.Mat{}, 0, 8); err != ErrEmptyImage {
		t.Fatalf("want empty image, got %v", err)
	}
	if _, err := CalcHist1D(cv.NewMat(2, 2, 1), 5, 8); err != ErrChannelRange {
		t.Fatalf("want channel range, got %v", err)
	}
	if _, err := CalcHist1D(cv.NewMat(2, 2, 1), 0, 0); err != ErrBinCount {
		t.Fatalf("want bin count, got %v", err)
	}
}

func BenchmarkCLAHEApply(b *testing.B) {
	m := cv.NewMat(256, 256, 1)
	for i := range m.Data {
		m.Data[i] = uint8((i * 7) % 256)
	}
	c := NewCLAHE(4, 8, 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.Apply(m); err != nil {
			b.Fatal(err)
		}
	}
}

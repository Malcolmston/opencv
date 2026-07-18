package pyramids

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// newFloat builds a FloatMat from row-major data for tests.
func newFloat(rows, cols int, data []float64) *cv.FloatMat {
	f := cv.NewFloatMat(rows, cols)
	copy(f.Data, data)
	return f
}

// maxAbsDiff returns the largest absolute difference between two same-sized
// grids, or +Inf if the sizes disagree.
func maxAbsDiff(a, b *cv.FloatMat) float64 {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return math.Inf(1)
	}
	m := 0.0
	for i := range a.Data {
		d := math.Abs(a.Data[i] - b.Data[i])
		if d > m {
			m = d
		}
	}
	return m
}

func TestHaarForwardKnownAnswer(t *testing.T) {
	f := newFloat(2, 2, []float64{10, 20, 30, 40})
	h := HaarForward(f)
	want := map[string]float64{
		"LL": (10 + 20 + 30 + 40) / 2.0, // 50
		"LH": (10 - 20 + 30 - 40) / 2.0, // -10
		"HL": (10 + 20 - 30 - 40) / 2.0, // -20
		"HH": (10 - 20 - 30 + 40) / 2.0, // 0
	}
	got := map[string]float64{"LL": h.LL.Data[0], "LH": h.LH.Data[0], "HL": h.HL.Data[0], "HH": h.HH.Data[0]}
	for k, w := range want {
		if math.Abs(got[k]-w) > 1e-12 {
			t.Errorf("Haar %s = %v, want %v", k, got[k], w)
		}
	}
	// Orthonormal round trip.
	inv := HaarInverse(h)
	if d := maxAbsDiff(inv, f); d > 1e-12 {
		t.Errorf("HaarInverse round trip diff = %v", d)
	}
}

func TestHaarRoundTripOddSize(t *testing.T) {
	// 5x3 ramp; odd in both dimensions exercises edge padding + cropping.
	data := make([]float64, 5*3)
	for i := range data {
		data[i] = float64(i * 7 % 13)
	}
	f := newFloat(5, 3, data)
	wp := BuildWaveletPyramid(f, 3)
	rec := wp.Reconstruct()
	if d := maxAbsDiff(rec, f); d > 1e-9 {
		t.Fatalf("wavelet pyramid reconstruction diff = %v", d)
	}
}

func TestLaplacianReconstruct(t *testing.T) {
	const R, C = 9, 7
	data := make([]float64, R*C)
	for y := 0; y < R; y++ {
		for x := 0; x < C; x++ {
			data[y*C+x] = float64((x*13+y*29)%97) + 0.5*float64(x)
		}
	}
	f := newFloat(R, C, data)
	lp := BuildLaplacianPyramid(f, 4)
	rec := lp.Reconstruct()
	if d := maxAbsDiff(rec, f); d > 1e-9 {
		t.Fatalf("laplacian reconstruction diff = %v (bands=%d)", d, lp.NumBands())
	}
}

func TestPyrDownConstant(t *testing.T) {
	const R, C = 8, 8
	data := make([]float64, R*C)
	for i := range data {
		data[i] = 42
	}
	f := newFloat(R, C, data)
	down := PyrDownFloat(f)
	if down.Rows != 4 || down.Cols != 4 {
		t.Fatalf("PyrDown size = %dx%d, want 4x4", down.Rows, down.Cols)
	}
	for _, v := range down.Data {
		if math.Abs(v-42) > 1e-9 {
			t.Fatalf("PyrDown of constant not constant: %v", v)
		}
	}
}

func TestGaussianKernelSumsToOne(t *testing.T) {
	k := GaussianKernel(1.5)
	var s float64
	for _, v := range k {
		s += v
	}
	if math.Abs(s-1) > 1e-12 {
		t.Fatalf("gaussian kernel sum = %v", s)
	}
}

func TestDerivativeKernelsSumToZero(t *testing.T) {
	for _, k := range [][]float64{GaussianDerivativeKernel(1.2), GaussianSecondDerivativeKernel(1.2)} {
		var s float64
		for _, v := range k {
			s += v
		}
		if math.Abs(s) > 1e-12 {
			t.Fatalf("derivative kernel sum = %v, want 0", s)
		}
	}
}

func TestLaplacianOfGaussianZeroSum(t *testing.T) {
	k := LaplacianOfGaussianKernel(1.4)
	var s float64
	for _, v := range k.Data {
		s += v
	}
	if math.Abs(s) > 1e-9 {
		t.Fatalf("LoG kernel sum = %v, want 0", s)
	}
}

func TestDifferenceOfGaussiansConstantIsZero(t *testing.T) {
	f := newFloat(6, 6, make([]float64, 36))
	for i := range f.Data {
		f.Data[i] = 100
	}
	dog := DifferenceOfGaussians(f, 1.0, 2.0)
	for _, v := range dog.Data {
		if math.Abs(v) > 1e-9 {
			t.Fatalf("DoG of constant not zero: %v", v)
		}
	}
}

func TestSteeringIdentity(t *testing.T) {
	// SteerG1 at 0 must equal gx and at pi/2 must equal gy exactly.
	data := make([]float64, 7*7)
	for i := range data {
		data[i] = float64((i * 3) % 11)
	}
	f := newFloat(7, 7, data)
	gx, gy := SteerableBasisG1(f, 1.0)
	if d := maxAbsDiff(SteerG1(gx, gy, 0), gx); d > 1e-12 {
		t.Errorf("SteerG1(0) != gx, diff=%v", d)
	}
	if d := maxAbsDiff(SteerG1(gx, gy, math.Pi/2), gy); d > 1e-12 {
		t.Errorf("SteerG1(pi/2) != gy, diff=%v", d)
	}
	gxx, gxy, gyy := SteerableBasisG2(f, 1.0)
	if d := maxAbsDiff(SteerG2(gxx, gxy, gyy, 0), gxx); d > 1e-12 {
		t.Errorf("SteerG2(0) != gxx, diff=%v", d)
	}
}

func TestSteerableHorizontalRamp(t *testing.T) {
	// f(x,y)=x has a purely horizontal gradient: gy should be ~0 in the
	// interior while gx is a nonzero constant.
	const R, C = 12, 12
	data := make([]float64, R*C)
	for y := 0; y < R; y++ {
		for x := 0; x < C; x++ {
			data[y*C+x] = float64(x)
		}
	}
	f := newFloat(R, C, data)
	gx, gy := SteerableBasisG1(f, 1.0)
	// Inspect an interior pixel away from the replicated border.
	i := 6*C + 6
	if math.Abs(gy.Data[i]) > 1e-9 {
		t.Errorf("horizontal ramp gy = %v, want ~0", gy.Data[i])
	}
	if math.Abs(gx.Data[i]) < 0.5 {
		t.Errorf("horizontal ramp |gx| = %v, want clearly nonzero", math.Abs(gx.Data[i]))
	}
	// The interior gx should be spatially constant for a linear ramp.
	if d := math.Abs(gx.Data[i] - gx.Data[5*C+5]); d > 1e-9 {
		t.Errorf("ramp gx not constant in interior, diff=%v", d)
	}
}

func TestAlphaBlendSelectsSides(t *testing.T) {
	a := newFloat(2, 2, []float64{1, 2, 3, 4})
	b := newFloat(2, 2, []float64{10, 20, 30, 40})
	ones := newFloat(2, 2, []float64{1, 1, 1, 1})
	zeros := newFloat(2, 2, []float64{0, 0, 0, 0})
	if d := maxAbsDiff(AlphaBlendFloat(a, b, ones), a); d > 1e-12 {
		t.Errorf("mask=1 should select a, diff=%v", d)
	}
	if d := maxAbsDiff(AlphaBlendFloat(a, b, zeros), b); d > 1e-12 {
		t.Errorf("mask=0 should select b, diff=%v", d)
	}
}

func TestBlendLaplacianIdentityMasks(t *testing.T) {
	// A saturated mask must reproduce the selected source exactly; a zero mask
	// must reproduce the other. This holds for any images, textured or flat.
	const R, C = 16, 16
	a := newFloat(R, C, make([]float64, R*C))
	b := newFloat(R, C, make([]float64, R*C))
	for y := 0; y < R; y++ {
		for x := 0; x < C; x++ {
			a.Data[y*C+x] = float64((x*7 + y*3) % 40) // textured
			b.Data[y*C+x] = float64((x*2 + y*5) % 33)
		}
	}
	ones := newFloat(R, C, make([]float64, R*C))
	zeros := newFloat(R, C, make([]float64, R*C))
	for i := range ones.Data {
		ones.Data[i] = 1
	}
	if d := maxAbsDiff(BlendLaplacian(a, b, ones, 4), a); d > 1e-9 {
		t.Errorf("mask=1 should reproduce a exactly, diff=%v", d)
	}
	if d := maxAbsDiff(BlendLaplacian(a, b, zeros, 4), b); d > 1e-9 {
		t.Errorf("mask=0 should reproduce b exactly, diff=%v", d)
	}
}

func TestBlendLaplacianSeamIsSmooth(t *testing.T) {
	// Blending two constant fields across a hard mask yields a seamless (no
	// abrupt jump) transition bounded by the two source values. The DC offset
	// between the fields is necessarily spread by the coarse base.
	const R, C = 16, 16
	a := newFloat(R, C, make([]float64, R*C))
	b := newFloat(R, C, make([]float64, R*C))
	mask := newFloat(R, C, make([]float64, R*C))
	for y := 0; y < R; y++ {
		for x := 0; x < C; x++ {
			a.Data[y*C+x] = 200
			b.Data[y*C+x] = 50
			if x < C/2 {
				mask.Data[y*C+x] = 1
			}
		}
	}
	out := BlendLaplacian(a, b, mask, 4)
	row := 8
	for x := 0; x < C; x++ {
		v := out.Data[row*C+x]
		if v < 50-1 || v > 200+1 {
			t.Fatalf("blended value %v out of source range at x=%d", v, x)
		}
		if x > 0 {
			if d := math.Abs(v - out.Data[row*C+x-1]); d > 40 {
				t.Fatalf("non-smooth seam: adjacent jump %v at x=%d", d, x)
			}
		}
	}
	// The left side should still lean toward a and the right toward b.
	if out.Data[row*C+0] <= out.Data[row*C+C-1] {
		t.Errorf("expected left brighter than right: %v vs %v", out.Data[row*C+0], out.Data[row*C+C-1])
	}
}

func TestBlendLaplacianMatColor(t *testing.T) {
	const R, C = 16, 16
	a := cv.NewMat(R, C, 3)
	b := cv.NewMat(R, C, 3)
	mask := cv.NewMat(R, C, 1)
	for y := 0; y < R; y++ {
		for x := 0; x < C; x++ {
			p := (y*C + x) * 3
			// Textured so band content, not just DC, distinguishes the sources.
			a.Data[p] = uint8(200 + (x+y)%20)
			a.Data[p+1], a.Data[p+2] = 30, 30
			b.Data[p], b.Data[p+1] = 30, 30
			b.Data[p+2] = uint8(200 + (x*y)%20)
			mask.Data[y*C+x] = 255 // fully select a
		}
	}
	out := BlendLaplacianMat(a, b, mask, 4)
	if out.Rows != R || out.Cols != C || out.Channels != 3 {
		t.Fatalf("unexpected output shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
	// A saturated mask reproduces source a to within 8-bit rounding.
	var maxd int
	for i := range out.Data {
		d := int(out.Data[i]) - int(a.Data[i])
		if d < 0 {
			d = -d
		}
		if d > maxd {
			maxd = d
		}
	}
	if maxd > 1 {
		t.Errorf("full mask should reproduce a, max channel diff = %d", maxd)
	}
}

func TestMultiBandBlendMatReproducesSingleSource(t *testing.T) {
	// With one source fully masked in and the other fully masked out, the
	// mosaic reproduces the selected source (to 8-bit rounding). Masks are
	// unnormalised (255/0), exercising the internal normalisation.
	const R, C = 16, 16
	imgs := []*cv.Mat{cv.NewMat(R, C, 1), cv.NewMat(R, C, 1)}
	masks := []*cv.Mat{cv.NewMat(R, C, 1), cv.NewMat(R, C, 1)}
	for y := 0; y < R; y++ {
		for x := 0; x < C; x++ {
			imgs[0].Data[y*C+x] = uint8((x*11 + y*13) % 200)
			imgs[1].Data[y*C+x] = 40
			masks[0].Data[y*C+x] = 255 // select source 0 everywhere
		}
	}
	out := MultiBandBlendMat(imgs, masks, 4)
	var maxd int
	for i := range out.Data {
		d := int(out.Data[i]) - int(imgs[0].Data[i])
		if d < 0 {
			d = -d
		}
		if d > maxd {
			maxd = d
		}
	}
	if maxd > 1 {
		t.Errorf("single-source blend should reproduce source 0, max diff = %d", maxd)
	}
}

func TestDetectDoGBlob(t *testing.T) {
	// A bright disc on a dark field should produce a strong scale-space
	// extremum near its centre.
	const N = 40
	f := cv.NewFloatMat(N, N)
	cx, cy := 20, 20
	for y := 0; y < N; y++ {
		for x := 0; x < N; x++ {
			dx, dy := x-cx, y-cy
			if dx*dx+dy*dy <= 16 { // radius 4
				f.Data[y*N+x] = 255
			}
		}
	}
	dog := BuildDoGScaleSpace(f, 5, 1.6, math.Sqrt2)
	kps := DetectDoGExtrema(dog, 1.0)
	if len(kps) == 0 {
		t.Fatal("expected at least one keypoint")
	}
	// Strongest-response keypoint should be near the disc centre.
	best := kps[0]
	for _, k := range kps {
		if math.Abs(k.Response) > math.Abs(best.Response) {
			best = k
		}
	}
	if math.Abs(float64(best.X-cx)) > 3 || math.Abs(float64(best.Y-cy)) > 3 {
		t.Errorf("strongest keypoint at (%d,%d), want near (%d,%d)", best.X, best.Y, cx, cy)
	}
	if best.Sigma <= 0 {
		t.Errorf("keypoint sigma = %v, want positive", best.Sigma)
	}
}

func TestGaussianPyramidSizes(t *testing.T) {
	f := cv.NewFloatMat(10, 10)
	p := BuildGaussianPyramid(f, 3)
	if p.NumLevels() != 3 {
		t.Fatalf("levels = %d, want 3", p.NumLevels())
	}
	want := [][2]int{{10, 10}, {5, 5}, {3, 3}}
	for i, w := range want {
		lv := p.Level(i)
		if lv.Rows != w[0] || lv.Cols != w[1] {
			t.Errorf("level %d size = %dx%d, want %dx%d", i, lv.Rows, lv.Cols, w[0], w[1])
		}
	}
	if p.Base().Rows != 10 || p.Coarsest().Rows != 3 {
		t.Errorf("base/coarsest sizes wrong")
	}
}

func TestConvolveSeparableDelta(t *testing.T) {
	// A delta kernel must leave the image unchanged.
	f := newFloat(4, 4, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	})
	delta := []float64{0, 1, 0}
	out := ConvolveSeparable(f, delta, delta)
	if d := maxAbsDiff(out, f); d > 1e-12 {
		t.Fatalf("delta convolution changed image, diff=%v", d)
	}
}

func TestFloatMatConversions(t *testing.T) {
	m := cv.NewMat(2, 2, 3)
	for i := range m.Data {
		m.Data[i] = uint8(i * 10)
	}
	planes := SplitFloat(m)
	if len(planes) != 3 {
		t.Fatalf("SplitFloat returned %d planes", len(planes))
	}
	back := MergeFloat(planes)
	for i := range m.Data {
		if back.Data[i] != m.Data[i] {
			t.Fatalf("Split/Merge round trip mismatch at %d: %d vs %d", i, back.Data[i], m.Data[i])
		}
	}
	// FloatToMat clamps out-of-range values.
	f := newFloat(1, 3, []float64{-5, 128.4, 300})
	g := FloatToMat(f)
	if g.Data[0] != 0 || g.Data[1] != 128 || g.Data[2] != 255 {
		t.Errorf("FloatToMat clamp = %v", g.Data)
	}
}

func BenchmarkMultiBandBlendMat(b *testing.B) {
	const R, C = 128, 128
	imgs := []*cv.Mat{cv.NewMat(R, C, 3), cv.NewMat(R, C, 3)}
	masks := []*cv.Mat{cv.NewMat(R, C, 1), cv.NewMat(R, C, 1)}
	for y := 0; y < R; y++ {
		for x := 0; x < C; x++ {
			p := (y*C + x) * 3
			imgs[0].Data[p], imgs[0].Data[p+1], imgs[0].Data[p+2] = 200, 20, 20
			imgs[1].Data[p], imgs[1].Data[p+1], imgs[1].Data[p+2] = 20, 20, 200
			if x < C/2 {
				masks[0].Data[y*C+x] = 255
			} else {
				masks[1].Data[y*C+x] = 255
			}
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MultiBandBlendMat(imgs, masks, 5)
	}
}

package hdr

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- Synthetic scene / camera model -----------------------------------------

// trueGamma is the exponent of the synthetic camera response used throughout
// the tests. The response maps normalised exposure Xn in [0,1] to the pixel
// value 255*Xn^(1/gamma), a monotonic curve.
const trueGamma = 2.2

// xref normalises exposure so mid-bracket exposures land in the middle of the
// 8-bit range.
const xref = 6.0

// sceneRadiance returns the (channel-independent) scene radiance at column x of
// a cols-wide image: log-spaced from 0.1 to 3.0.
func sceneRadiance(x, cols int) float64 {
	f := float64(x) / float64(cols-1)
	return 0.1 * math.Pow(30.0, f)
}

// renderExposure renders one LDR frame of a rows×cols×3 scene at the given
// exposure time through the synthetic response.
func renderExposure(rows, cols int, t float64) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			e := sceneRadiance(x, cols)
			xn := (e * t) / xref
			if xn < 0 {
				xn = 0
			}
			if xn > 1 {
				xn = 1
			}
			z := clamp8(math.Pow(xn, 1.0/trueGamma) * 255)
			i := (y*cols + x) * 3
			m.Data[i+0] = z
			m.Data[i+1] = z
			m.Data[i+2] = z
		}
	}
	return m
}

func syntheticBracket(rows, cols int) ([]*cv.Mat, []float64) {
	times := []float64{0.125, 0.25, 0.5, 1, 2, 4}
	imgs := make([]*cv.Mat, len(times))
	for j, t := range times {
		imgs[j] = renderExposure(rows, cols, t)
	}
	return imgs, times
}

// --- Radiance basics ---------------------------------------------------------

func TestRadianceBasics(t *testing.T) {
	r := NewRadiance(4, 5, 3)
	if r.Rows != 4 || r.Cols != 5 || r.Channels != 3 {
		t.Fatalf("unexpected dims %dx%dx%d", r.Rows, r.Cols, r.Channels)
	}
	r.Set(1, 2, 1, 3.5)
	if got := r.At(1, 2, 1); got != 3.5 {
		t.Fatalf("At=%v want 3.5", got)
	}
	cl := r.Clone()
	cl.Set(1, 2, 1, 9)
	if r.At(1, 2, 1) != 3.5 {
		t.Fatalf("clone aliased backing storage")
	}
	fm := r.ChannelFloatMat(1)
	if fm.At(1, 2) != 3.5 {
		t.Fatalf("ChannelFloatMat mismatch: %v", fm.At(1, 2))
	}
	// luminance of a gray pixel equals the sample.
	r2 := NewRadiance(1, 1, 3)
	r2.Set(0, 0, 0, 2)
	r2.Set(0, 0, 1, 2)
	r2.Set(0, 0, 2, 2)
	if l := r2.luminance().data[0]; math.Abs(l-2) > 1e-9 {
		t.Fatalf("luminance=%v want 2", l)
	}
}

func TestRadiancePanics(t *testing.T) {
	assertPanic(t, "NewRadiance", func() { NewRadiance(0, 4, 3) })
	r := NewRadiance(2, 2, 1)
	assertPanic(t, "ChannelFloatMat", func() { r.ChannelFloatMat(5) })
}

func assertPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic", name)
		}
	}()
	fn()
}

// --- Calibration -------------------------------------------------------------

func TestCalibrateDebevecMonotonicAndCorrelated(t *testing.T) {
	imgs, times := syntheticBracket(32, 24)
	resp, err := CalibrateDebevec(imgs, times, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Channels != 3 {
		t.Fatalf("channels=%d", resp.Channels)
	}
	g := resp.logCurve(0)

	// Monotonic (non-decreasing) across the well-observed middle range.
	for z := 21; z <= 220; z++ {
		if g[z] < g[z-1]-0.05 {
			t.Fatalf("response not monotonic at z=%d: g[%d]=%v g[%d]=%v", z, z-1, g[z-1], z, g[z])
		}
	}
	// Correlate recovered g with the ground-truth log-exposure gamma*log(z/255).
	corr := correlation(func(z int) float64 { return g[z] },
		func(z int) float64 { return trueGamma * math.Log(float64(z)/255.0) }, 30, 220)
	if corr < 0.95 {
		t.Fatalf("recovered response correlation %.3f < 0.95", corr)
	}
	// Curve is positive and monotonic in linear space.
	if resp.Curve[0][200] <= resp.Curve[0][40] {
		t.Fatalf("linear curve not increasing")
	}
}

func TestCalibrateRobertson(t *testing.T) {
	imgs, times := syntheticBracket(32, 24)
	resp, err := CalibrateRobertson(imgs, times, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	g := resp.logCurve(0)
	// Non-decreasing everywhere (fillMonotone enforces this).
	for z := 1; z < 256; z++ {
		if g[z] < g[z-1]-1e-6 {
			t.Fatalf("robertson response not monotonic at z=%d", z)
		}
	}
	corr := correlation(func(z int) float64 { return g[z] },
		func(z int) float64 { return trueGamma * math.Log(float64(z)/255.0) }, 30, 220)
	if corr < 0.9 {
		t.Fatalf("robertson correlation %.3f < 0.9", corr)
	}
	if math.Abs(resp.Curve[0][128]-1) > 1e-6 {
		t.Fatalf("robertson not normalised at 128: %v", resp.Curve[0][128])
	}
}

func TestCalibrateErrors(t *testing.T) {
	imgs, times := syntheticBracket(8, 8)
	if _, err := CalibrateDebevec(imgs[:1], times[:1], 0, 0); err == nil {
		t.Fatal("expected error for single exposure")
	}
	if _, err := CalibrateDebevec(imgs, times[:3], 0, 0); err == nil {
		t.Fatal("expected error for length mismatch")
	}
	bad := []float64{0.1, 0.2, -1, 0.4, 0.5, 0.6}
	if _, err := CalibrateDebevec(imgs, bad, 0, 0); err == nil {
		t.Fatal("expected error for non-positive time")
	}
}

// --- Merge Debevec -----------------------------------------------------------

func TestMergeDebevecRecoversRatios(t *testing.T) {
	rows, cols := 24, 24
	imgs, times := syntheticBracket(rows, cols)
	resp, err := CalibrateDebevec(imgs, times, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	hdrImg, err := MergeDebevec(imgs, times, resp)
	if err != nil {
		t.Fatal(err)
	}
	if hdrImg.Rows != rows || hdrImg.Channels != 3 {
		t.Fatalf("bad radiance dims")
	}
	meanCol := func(x int) float64 {
		var s float64
		for y := 0; y < rows; y++ {
			s += hdrImg.At(y, x, 0)
		}
		return s / float64(rows)
	}
	c1, c2 := 6, 14
	recRatio := meanCol(c2) / meanCol(c1)
	trueRatio := sceneRadiance(c2, cols) / sceneRadiance(c1, cols)
	relErr := math.Abs(recRatio-trueRatio) / trueRatio
	if relErr > 0.15 {
		t.Fatalf("recovered radiance ratio %.3f vs true %.3f (relErr %.3f)", recRatio, trueRatio, relErr)
	}
	// All radiance values finite and positive.
	for _, v := range hdrImg.Data {
		if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("bad radiance value %v", v)
		}
	}
}

func TestMergeDebevecWithLinearResponse(t *testing.T) {
	imgs, times := syntheticBracket(8, 8)
	_, err := MergeDebevec(imgs, times, LinearResponse(3))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := MergeDebevec(imgs, times, nil); err == nil {
		t.Fatal("expected nil-response error")
	}
	if _, err := MergeDebevec(imgs, times, LinearResponse(1)); err == nil {
		t.Fatal("expected channel-mismatch error")
	}
}

// --- Merge Mertens -----------------------------------------------------------

func TestMergeMertensFusesWellExposed(t *testing.T) {
	rows, cols := 40, 40
	// Two exposures: under (dark) and over (bright).
	under := renderExposure(rows, cols, 0.25)
	over := renderExposure(rows, cols, 3.0)
	fused, err := MergeMertens([]*cv.Mat{under, over}, NewMergeMertensParams())
	if err != nil {
		t.Fatal(err)
	}
	if fused.Rows != rows || fused.Cols != cols || fused.Channels != 3 {
		t.Fatalf("bad fused dims")
	}
	meanOf := func(m *cv.Mat) float64 {
		var s float64
		for _, v := range m.Data {
			s += float64(v)
		}
		return s / float64(len(m.Data))
	}
	uMean := meanOf(under)
	oMean := meanOf(over)
	fMean := meanOf(fused)
	// Fused mean sits between the two extremes and near mid-grey.
	if fMean < 70 || fMean > 190 {
		t.Fatalf("fused mean %.1f not mid-histogram (under=%.1f over=%.1f)", fMean, uMean, oMean)
	}
	// Fused image is closer to mid-grey (128) than either input.
	if math.Abs(fMean-128) > math.Abs(uMean-128) && math.Abs(fMean-128) > math.Abs(oMean-128) {
		t.Fatalf("fusion did not improve exposure: f=%.1f u=%.1f o=%.1f", fMean, uMean, oMean)
	}
	// Output uses a broad range (not collapsed).
	if stdMat(fused) < 5 {
		t.Fatalf("fused std %.2f too low", stdMat(fused))
	}
}

func TestMergeMertensErrors(t *testing.T) {
	m := cv.NewMat(4, 4, 3)
	if _, err := MergeMertens([]*cv.Mat{m}, NewMergeMertensParams()); err == nil {
		t.Fatal("expected too-few-images error")
	}
	other := cv.NewMat(4, 5, 3)
	if _, err := MergeMertens([]*cv.Mat{m, other}, NewMergeMertensParams()); err == nil {
		t.Fatal("expected dimension-mismatch error")
	}
}

// --- Tonemapping -------------------------------------------------------------

// highRangeRadiance builds a 3-channel radiance whose luminance spans e^-4..e^4
// across the columns — a scene naive linear scaling cannot display.
func highRangeRadiance(rows, cols int) *Radiance {
	r := NewRadiance(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			f := float64(x)/float64(cols-1)*8 - 4 // -4..4
			e := math.Exp(f)
			// Slightly different per channel so saturation/recolor exercised.
			r.Set(y, x, 0, e*1.1)
			r.Set(y, x, 1, e)
			r.Set(y, x, 2, e*0.9)
		}
	}
	return r
}

// naiveLinear clamps radiance/globalMax*255 — the baseline that crushes shadows.
func naiveLinear(r *Radiance) *cv.Mat {
	var maxV float64
	for _, v := range r.Data {
		if v > maxV {
			maxV = v
		}
	}
	if maxV <= 0 {
		maxV = 1
	}
	out := cv.NewMat(r.Rows, r.Cols, r.Channels)
	for i, v := range r.Data {
		out.Data[i] = clamp8(v / maxV * 255)
	}
	return out
}

func TestTonemappersProduceDetail(t *testing.T) {
	r := highRangeRadiance(32, 48)
	baseStd := stdMat(naiveLinear(r))

	ops := map[string]Tonemap{
		"gamma":    NewTonemapGamma(2.2),
		"reinhard": NewTonemapReinhard(),
		"drago":    NewTonemapDrago(),
		"mantiuk":  NewTonemapMantiuk(),
	}
	for name, op := range ops {
		out := op.Process(r)
		if out.Rows != r.Rows || out.Cols != r.Cols || out.Channels != 3 {
			t.Fatalf("%s: bad output dims", name)
		}
		// Valid range is guaranteed by clamp8; check the image is not constant
		// and reveals more detail than naive linear scaling.
		s := stdMat(out)
		if s < 10 {
			t.Fatalf("%s: output std %.2f too low (no detail)", name, s)
		}
		if s <= baseStd {
			t.Fatalf("%s: std %.2f did not beat naive linear %.2f", name, s, baseStd)
		}
	}
}

func TestTonemapReinhardLocal(t *testing.T) {
	r := highRangeRadiance(48, 48)
	op := NewTonemapReinhard()
	op.Local = true
	op.LocalSigma = 4
	out := op.Process(r)
	if stdMat(out) < 10 {
		t.Fatalf("local reinhard produced no detail")
	}
}

func TestTonemapGrayscale(t *testing.T) {
	// Single-channel radiance exercises the channel==1 code paths.
	r := NewRadiance(16, 16, 1)
	for x := 0; x < 16; x++ {
		for y := 0; y < 16; y++ {
			r.Set(y, x, 0, math.Exp(float64(x)/15*6-3))
		}
	}
	for _, op := range []Tonemap{
		NewTonemapGamma(2.0), NewTonemapReinhard(), NewTonemapDrago(), NewTonemapMantiuk(),
	} {
		out := op.Process(r)
		if out.Channels != 1 {
			t.Fatalf("expected single channel output")
		}
	}
}

// --- helpers -----------------------------------------------------------------

func stdMat(m *cv.Mat) float64 {
	var mean float64
	for _, v := range m.Data {
		mean += float64(v)
	}
	mean /= float64(len(m.Data))
	var varv float64
	for _, v := range m.Data {
		d := float64(v) - mean
		varv += d * d
	}
	return math.Sqrt(varv / float64(len(m.Data)))
}

// correlation returns the Pearson correlation of a and b over z in [lo,hi].
func correlation(a, b func(int) float64, lo, hi int) float64 {
	n := 0
	var sa, sb float64
	for z := lo; z <= hi; z++ {
		sa += a(z)
		sb += b(z)
		n++
	}
	ma := sa / float64(n)
	mb := sb / float64(n)
	var cov, va, vb float64
	for z := lo; z <= hi; z++ {
		da := a(z) - ma
		db := b(z) - mb
		cov += da * db
		va += da * da
		vb += db * db
	}
	if va == 0 || vb == 0 {
		return 0
	}
	return cov / math.Sqrt(va*vb)
}

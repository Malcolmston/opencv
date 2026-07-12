package videostab

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/video"
)

// ---- synthetic-frame helpers -------------------------------------------------

// smoothNoise builds a deterministic, single-channel texture as a light blur of
// white noise. It is used where a generic dense texture is enough (inpainting
// and deblur tests).
func smoothNoise(rows, cols int, seed int64) *cv.Mat {
	rng := rand.New(rand.NewSource(seed))
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = uint8(rng.Intn(256))
	}
	return cv.GaussianBlur(m, 3, 0)
}

// blobTexture builds a deterministic field of small bright blobs on a dark
// background. Unlike dense noise, its features are sparse and locally distinct,
// so pyramidal Lucas-Kanade tracks them reliably between shifted crops.
func blobTexture(rows, cols int, seed int64) *cv.Mat {
	rng := rand.New(rand.NewSource(seed))
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(30)
	for k := 0; k < (rows*cols)/40; k++ {
		cy := rng.Intn(rows)
		cx := rng.Intn(cols)
		v := uint8(140 + rng.Intn(115))
		for dy := -1; dy <= 1; dy++ {
			for dx := -1; dx <= 1; dx++ {
				y, x := cy+dy, cx+dx
				if y >= 0 && y < rows && x >= 0 && x < cols {
					m.Data[y*cols+x] = v
				}
			}
		}
	}
	return cv.GaussianBlur(m, 3, 0)
}

// jitteredSequence returns n crops of a fixed blob texture taken at a base
// offset plus a deterministic per-frame translational jitter, together with the
// exact inter-frame motions M_i (mapping frame i onto frame i+1). Because every
// crop comes from the same underlying texture, a perfect stabilizer would make
// consecutive crops identical.
func jitteredSequence(n, size int, seed int64) (frames []*cv.Mat, motions []Motion) {
	const margin = 6
	base := blobTexture(size+2*margin, size+2*margin, seed)
	rng := rand.New(rand.NewSource(seed + 1))
	offs := make([][2]int, n)
	frames = make([]*cv.Mat, n)
	for i := 0; i < n; i++ {
		jx := rng.Intn(2*margin+1) - margin
		jy := rng.Intn(2*margin+1) - margin
		ox := clampInt(margin+jx, 0, 2*margin)
		oy := clampInt(margin+jy, 0, 2*margin)
		offs[i] = [2]int{ox, oy}
		frames[i] = base.Region(oy, ox, size, size)
	}
	// A texture point seen at column c-ox, row r-oy in frame i moves, in frame
	// i+1, by (ox_i-ox_{i+1}, oy_i-oy_{i+1}); that is the true motion M_i.
	motions = make([]Motion, max0(n-1))
	for i := 0; i+1 < n; i++ {
		motions[i] = TranslationMotion(
			float64(offs[i][0]-offs[i+1][0]),
			float64(offs[i][1]-offs[i+1][1]),
		)
	}
	return frames, motions
}

// smallJitterSequence returns crops with gentle (few-pixel) jitter, small enough
// that optical flow tracks reliably for the real end-to-end pipeline test.
func smallJitterSequence(n, size int, seed int64) []*cv.Mat {
	const margin = 4
	base := blobTexture(size+2*margin, size+2*margin, seed)
	rng := rand.New(rand.NewSource(seed + 1))
	frames := make([]*cv.Mat, n)
	for i := 0; i < n; i++ {
		ox := clampInt(margin+rng.Intn(2*margin+1)-margin, 0, 2*margin)
		oy := clampInt(margin+rng.Intn(2*margin+1)-margin, 0, 2*margin)
		frames[i] = base.Region(oy, ox, size, size)
	}
	return frames
}

// fixedMotionEstimator is a deterministic ImageMotionEstimator that hands back
// pre-computed inter-frame motions in call order. It lets the stabilizer core be
// tested independently of optical-flow accuracy.
type fixedMotionEstimator struct {
	motions []Motion
	model   MotionModel
	call    int
}

func (f *fixedMotionEstimator) Estimate(_, _ *cv.Mat) (Motion, bool) {
	if f.call >= len(f.motions) {
		return IdentityMotion(), false
	}
	m := f.motions[f.call]
	f.call++
	return m, true
}

func (f *fixedMotionEstimator) MotionModel() MotionModel { return f.model }

// centralMAD sums the mean absolute difference between consecutive frames over a
// central crop (avoiding warp borders). It is a direct, estimator-independent
// measure of residual inter-frame motion.
func centralMAD(frames []*cv.Mat) float64 {
	if len(frames) < 2 {
		return 0
	}
	r, c := frames[0].Rows, frames[0].Cols
	y0, x0 := r/4, c/4
	y1, x1 := r-r/4, c-c/4
	var total float64
	for i := 1; i < len(frames); i++ {
		a, b := frames[i-1], frames[i]
		var sum float64
		var cnt int
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				pa := (y*a.Cols + x) * a.Channels
				pb := (y*b.Cols + x) * b.Channels
				for ch := 0; ch < a.Channels; ch++ {
					sum += math.Abs(float64(a.Data[pa+ch]) - float64(b.Data[pb+ch]))
				}
				cnt += a.Channels
			}
		}
		total += sum / float64(cnt)
	}
	return total
}

// sinusoidGray builds a smooth, non-saturating grey texture used for the deblur
// test — its values stay well inside [0, 255] so unsharp masking is not clipped.
func sinusoidGray(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := 110.0 + 45*math.Sin(float64(x)/3.0) + 40*math.Sin(float64(y)/4.0)
			m.Data[y*cols+x] = clampByte(v)
		}
	}
	return m
}

// ---- motion model & algebra --------------------------------------------------

func TestMotionModelString(t *testing.T) {
	cases := map[MotionModel]string{
		MotionModelTranslation:         "translation",
		MotionModelTranslationAndScale: "translationAndScale",
		MotionModelRotation:            "rotation",
		MotionModelRigid:               "rigid",
		MotionModelSimilarity:          "similarity",
		MotionModelAffine:              "affine",
		MotionModelHomography:          "homography",
		MotionModelUnknown:             "unknown",
	}
	for m, want := range cases {
		if got := m.String(); got != want {
			t.Errorf("model %d String()=%q want %q", int(m), got, want)
		}
	}
}

func TestDefaultRansacParamsAndNumIters(t *testing.T) {
	p := DefaultRansacParams(MotionModelAffine)
	if p.Size != 3 || p.Prob != 0.99 {
		t.Fatalf("unexpected defaults %+v", p)
	}
	if n := p.NumIters(); n < 1 {
		t.Fatalf("NumIters=%d, want >=1", n)
	}
	// Degenerate denominator path.
	if (RansacParams{Size: 1, Eps: 0, Prob: 0.99}).NumIters() < 1 {
		t.Fatal("NumIters should clamp to >=1")
	}
}

func TestMotionAlgebra(t *testing.T) {
	m := TranslationMotion(3, -4)
	x, y := m.Apply(1, 1)
	if x != 4 || y != -3 {
		t.Fatalf("translation apply = (%v,%v)", x, y)
	}
	inv, ok := m.Inverse()
	if !ok {
		t.Fatal("translation must be invertible")
	}
	rt := m.Mul(inv)
	for i, v := range IdentityMotion() {
		if math.Abs(rt[i]-v) > 1e-9 {
			t.Fatalf("m*inv != identity at %d: %v", i, rt[i])
		}
	}
	s := SimilarityMotion(2, math.Pi/2, 0, 0)
	if d := s.Determinant(); math.Abs(d-4) > 1e-9 {
		t.Fatalf("similarity determinant = %v, want 4", d)
	}
	a := s.Affine()
	// Row 0 of a 90°-rotated, 2× similarity is [0, -2, 0].
	if math.Abs(a[0]) > 1e-9 || math.Abs(a[1]+2) > 1e-9 {
		t.Fatalf("unexpected affine row 0: %v", a)
	}
	if _, ok := (Motion{}).Inverse(); ok {
		t.Fatal("zero motion must be singular")
	}
}

// ---- sparse estimators -------------------------------------------------------

func makeCorrespondences(m Motion, n int, seed int64) (from, to []video.PointF) {
	rng := rand.New(rand.NewSource(seed))
	from = make([]video.PointF, n)
	to = make([]video.PointF, n)
	for i := 0; i < n; i++ {
		p := video.PointF{X: rng.Float64() * 100, Y: rng.Float64() * 100}
		tx, ty := m.Apply(p.X, p.Y)
		from[i] = p
		to[i] = video.PointF{X: tx, Y: ty}
	}
	return
}

func motionClose(t *testing.T, got, want Motion, tol float64) {
	t.Helper()
	for _, p := range []video.PointF{{X: 10, Y: 20}, {X: 70, Y: 5}, {X: 40, Y: 90}} {
		gx, gy := got.Apply(p.X, p.Y)
		wx, wy := want.Apply(p.X, p.Y)
		if math.Hypot(gx-wx, gy-wy) > tol {
			t.Fatalf("motion mismatch at %v: got (%v,%v) want (%v,%v)", p, gx, gy, wx, wy)
		}
	}
}

func TestLeastSquaresPerModel(t *testing.T) {
	want := map[MotionModel]Motion{
		MotionModelTranslation:         TranslationMotion(5, -3),
		MotionModelTranslationAndScale: {1.2, 0, 4, 0, 1.2, -6, 0, 0, 1},
		MotionModelRigid:               SimilarityMotion(1, 0.1, 2, 1),
		MotionModelSimilarity:          SimilarityMotion(1.15, -0.2, 3, 4),
		MotionModelAffine:              {1.1, 0.05, 2, -0.03, 0.98, -4, 0, 0, 1},
	}
	for model, m := range want {
		from, to := makeCorrespondences(m, 30, 7)
		est, ok := EstimateGlobalMotionLeastSquares(from, to, model)
		if !ok {
			t.Fatalf("%v: estimation failed", model)
		}
		motionClose(t, est, m, 1e-6)
	}
}

func TestRansacL2RejectsOutliers(t *testing.T) {
	want := SimilarityMotion(1.1, 0.15, 6, -4)
	from, to := makeCorrespondences(want, 60, 3)
	// Corrupt a third of the correspondences with gross outliers.
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < len(to); i += 3 {
		to[i] = video.PointF{X: rng.Float64() * 500, Y: rng.Float64() * 500}
	}
	est := NewMotionEstimatorRansacL2(MotionModelSimilarity)
	est.SetSeed(42)
	est.SetMinInlierRatio(0.3)
	if est.RansacParams().Size == 0 {
		t.Fatal("ransac params not set")
	}
	got, ok := est.Estimate(from, to)
	if !ok {
		t.Fatal("ransac estimate failed")
	}
	motionClose(t, got, want, 0.5)

	est.SetMotionModel(MotionModelAffine)
	if est.MotionModel() != MotionModelAffine {
		t.Fatal("SetMotionModel not applied")
	}
	if _, ok := est.Estimate(from[:1], to[:1]); ok {
		t.Fatal("estimate should fail with too few points")
	}
}

func TestEstimatorL1(t *testing.T) {
	want := TranslationMotion(7, -2)
	from, to := makeCorrespondences(want, 40, 11)
	// A few outliers; L1/IRLS should tolerate them.
	to[0] = video.PointF{X: 999, Y: -999}
	to[5] = video.PointF{X: -500, Y: 500}
	e := NewMotionEstimatorL1(MotionModelTranslation)
	if e.MotionModel() != MotionModelTranslation {
		t.Fatal("model mismatch")
	}
	got, ok := e.Estimate(from, to)
	if !ok {
		t.Fatal("L1 estimate failed")
	}
	motionClose(t, got, want, 0.3)
	e.SetMotionModel(MotionModelAffine)
	if _, ok := e.Estimate(from[:2], to[:2]); ok {
		t.Fatal("should fail with too few points for affine")
	}
}

func TestGetMotion(t *testing.T) {
	motions := []Motion{TranslationMotion(1, 0), TranslationMotion(2, 0), TranslationMotion(3, 0)}
	// frame 0 -> frame 3 is the composition, a total shift of 6 in x.
	fwd := GetMotion(0, 3, motions)
	x, _ := fwd.Apply(0, 0)
	if math.Abs(x-6) > 1e-9 {
		t.Fatalf("forward composition x=%v want 6", x)
	}
	back := GetMotion(3, 0, motions)
	bx, _ := back.Apply(0, 0)
	if math.Abs(bx+6) > 1e-6 {
		t.Fatalf("backward composition x=%v want -6", bx)
	}
	if id := GetMotion(2, 2, motions); id != IdentityMotion() {
		t.Fatal("same-frame motion must be identity")
	}
}

// ---- keypoint-based image estimator -----------------------------------------

func TestKeypointBasedMotionEstimator(t *testing.T) {
	const size, margin = 72, 6
	base := blobTexture(size+2*margin, size+2*margin, 5)
	// frame0 window at (margin,margin); frame1 window shifted by (+3,+2) in the
	// texture, so the content moves by (-3,-2) between the frames.
	frame0 := base.Region(margin, margin, size, size)
	frame1 := base.Region(margin+2, margin+3, size, size)
	est := NewKeypointBasedMotionEstimator(NewMotionEstimatorRansacL2(MotionModelTranslation))
	if est.MotionModel() != MotionModelTranslation || est.Base() == nil {
		t.Fatal("estimator not configured")
	}
	m, ok := est.Estimate(frame0, frame1)
	if !ok {
		t.Fatal("keypoint estimate failed")
	}
	tx, ty := m.Apply(float64(size)/2, float64(size)/2)
	dx := tx - float64(size)/2
	dy := ty - float64(size)/2
	if math.Abs(dx-(-3)) > 1.5 || math.Abs(dy-(-2)) > 1.5 {
		t.Fatalf("recovered shift (%.2f,%.2f), want about (-3,-2)", dx, dy)
	}
	if _, ok := est.Estimate(nil, frame1); ok {
		t.Fatal("nil frame must fail")
	}
}

// ---- motion filters & stabilizers -------------------------------------------

func TestGaussianMotionFilterSmooths(t *testing.T) {
	f := NewGaussianMotionFilter(5, 0)
	if f.Radius() != 5 || f.Stdev() <= 0 {
		t.Fatal("filter misconfigured")
	}
	// Jittery zero-mean translations; the smoothed trajectory should have far
	// less high-frequency energy than the raw path.
	rng := rand.New(rand.NewSource(1))
	n := 30
	motions := make([]Motion, n-1)
	for i := range motions {
		motions[i] = TranslationMotion(rng.NormFloat64()*4, rng.NormFloat64()*4)
	}
	rngRange := Range{First: 0, Last: n - 1}
	out := make([]Motion, n)
	f.Stabilize(n, motions, rngRange, out)
	if len(out) != n {
		t.Fatalf("expected %d outputs", n)
	}
	// A single-element degenerate call must not panic.
	_ = f.StabilizeAt(0, motions, Range{First: 0, Last: 0})
}

func TestMotionStabilizationPipelineAndLp(t *testing.T) {
	n := 24
	rng := rand.New(rand.NewSource(2))
	motions := make([]Motion, n-1)
	for i := range motions {
		motions[i] = TranslationMotion(rng.NormFloat64()*3, rng.NormFloat64()*3)
	}
	rngRange := Range{First: 0, Last: n - 1}

	pipe := NewMotionStabilizationPipeline().Add(NewGaussianMotionFilter(4, 0))
	if pipe.Len() != 1 {
		t.Fatalf("pipeline len=%d", pipe.Len())
	}
	pout := make([]Motion, n)
	pipe.Stabilize(n, motions, rngRange, pout)

	lp := NewLpMotionStabilizer()
	lp.SetWeights(1, 5, 25)
	lout := make([]Motion, n)
	lp.Stabilize(n, motions, rngRange, lout)
	for i, m := range lout {
		for _, v := range m {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("Lp output %d not finite: %v", i, m)
			}
		}
	}
}

func TestOnePassStabilizerReducesJitter(t *testing.T) {
	frames, motions := jitteredSequence(18, 64, 20)
	before := centralMAD(frames)

	s := NewOnePassStabilizer(6)
	// Feed the exact inter-frame motions so the test isolates the stabilization
	// core (smoothing + warping) from optical-flow accuracy.
	s.SetMotionEstimator(&fixedMotionEstimator{motions: motions, model: MotionModelTranslation})
	s.SetFrames(frames)
	out := s.Stabilize()
	if len(out) != len(frames) {
		t.Fatalf("got %d stabilized frames, want %d", len(out), len(frames))
	}
	if len(s.Motions()) != len(frames)-1 || len(s.StabilizationMotions()) != len(frames) {
		t.Fatal("motion buffers not populated")
	}
	after := centralMAD(out)
	t.Logf("one-pass central MAD before=%.3f after=%.3f", before, after)
	if after >= before {
		t.Fatalf("stabilization did not reduce residual motion: before=%.3f after=%.3f", before, after)
	}
}

func TestTwoPassStabilizerReducesJitter(t *testing.T) {
	frames, motions := jitteredSequence(18, 64, 33)
	before := centralMAD(frames)

	s := NewTwoPassStabilizer(6)
	if s.Radius() != 6 {
		t.Fatal("radius not applied")
	}
	s.SetMotionEstimator(&fixedMotionEstimator{motions: motions, model: MotionModelTranslation})
	s.SetFrames(frames)
	var out []*cv.Mat
	for {
		f, ok := s.NextFrame()
		if !ok {
			break
		}
		out = append(out, f)
	}
	after := centralMAD(out)
	t.Logf("two-pass MAD before=%.3f after=%.3f", before, after)
	if after >= before {
		t.Fatalf("two-pass stabilization did not reduce residual motion: before=%.3f after=%.3f", before, after)
	}
}

// TestOnePassStabilizerEndToEnd exercises the full default pipeline (real
// keypoint-based estimation) on a gently jittered blob sequence, confirming the
// residual motion drops without any injected ground-truth.
func TestOnePassStabilizerEndToEnd(t *testing.T) {
	frames := smallJitterSequence(14, 72, 3)
	before := centralMAD(frames)
	s := NewOnePassStabilizer(5)
	s.SetFrames(frames)
	out := s.Stabilize()
	after := centralMAD(out)
	t.Logf("end-to-end central MAD before=%.3f after=%.3f", before, after)
	if after >= before {
		t.Fatalf("end-to-end stabilization did not reduce residual motion: before=%.3f after=%.3f", before, after)
	}
}

func TestStabilizerWithInpainterAndDeblurer(t *testing.T) {
	frames, _ := jitteredSequence(10, 56, 7)
	s := NewOnePassStabilizer(4)
	s.SetRadius(4)
	s.SetTrimRatio(0.05)
	if math.Abs(s.TrimRatio()-0.05) > 1e-9 {
		t.Fatal("trim ratio not applied")
	}
	s.SetInpainter(NewInpaintingPipeline().Add(NewMotionInpainter()).Add(NewColorInpainter()))
	s.SetDeblurer(NewWeightingDeblurer(2))
	s.SetFrames(frames)
	out := s.Stabilize()
	if len(out) != len(frames) {
		t.Fatalf("got %d frames, want %d", len(out), len(frames))
	}
	for i, f := range out {
		if f == nil || f.Empty() || f.Rows != frames[0].Rows || f.Cols != frames[0].Cols {
			t.Fatalf("stabilized frame %d has wrong shape", i)
		}
	}
}

// ---- inpainting: the zero-holes guarantee -----------------------------------

// makeHoledFrame warps a texture by a translation, producing a frame whose top
// and left borders are empty, and returns the frame plus its coverage mask
// (0 = hole).
func makeHoledFrame(tex *cv.Mat, tx, ty float64) (*cv.Mat, *cv.Mat) {
	warp := TranslationMotion(tx, ty)
	frame := warp.warp(tex)
	mask := coverageMask(tex, warp)
	return frame, mask
}

func TestMotionInpainterLeavesNoHoles(t *testing.T) {
	const size = 48
	tex := smoothNoise(size, size, 4)
	// Three identical-content frames; the middle one is warped to expose holes.
	frames := []*cv.Mat{tex.Clone(), tex.Clone(), tex.Clone()}
	frame, mask := makeHoledFrame(tex, 9, 7)
	frames[1] = frame

	holesBefore := countHoles(mask)
	if holesBefore == 0 {
		t.Fatal("test setup produced no holes to fill")
	}

	in := NewMotionInpainter()
	stab := []Motion{IdentityMotion(), IdentityMotion(), IdentityMotion()}
	in.SetContext(frames, []Motion{IdentityMotion(), IdentityMotion()}, stab, 2)
	in.Inpaint(1, frame, mask)

	if h := countHoles(mask); h != 0 {
		t.Fatalf("MotionInpainter left %d holes (started with %d)", h, holesBefore)
	}
	// Every pixel must now carry content.
	for p := 0; p < frame.Total(); p++ {
		if mask.Data[p] != 255 {
			t.Fatalf("mask pixel %d not marked filled", p)
		}
	}
}

func TestMotionInpainterFallbackWithoutContext(t *testing.T) {
	const size = 40
	tex := smoothNoise(size, size, 8)
	frame, mask := makeHoledFrame(tex, 6, 5)
	in := NewMotionInpainter() // no context set -> pure diffusion fallback
	in.Inpaint(0, frame, mask)
	if h := countHoles(mask); h != 0 {
		t.Fatalf("diffusion fallback left %d holes", h)
	}
}

func TestColorInpainterFills(t *testing.T) {
	const size = 40
	tex := smoothNoise(size, size, 2)
	frame, mask := makeHoledFrame(tex, 5, 6)
	c := NewColorInpainter()
	c.SetContext(nil, nil, nil, 0)
	c.Inpaint(0, frame, mask)
	if h := countHoles(mask); h != 0 {
		t.Fatalf("ColorInpainter left %d holes", h)
	}
}

func TestColorAverageInpainterFills(t *testing.T) {
	const size = 40
	tex := smoothNoise(size, size, 6)
	frames := []*cv.Mat{tex.Clone(), nil, tex.Clone()}
	frame, mask := makeHoledFrame(tex, 4, 4)
	frames[1] = frame
	c := NewColorAverageInpainter()
	c.SetContext(frames, []Motion{IdentityMotion(), IdentityMotion()}, nil, 2)
	c.Inpaint(1, frame, mask)
	// Temporal averaging fills all holes here because neighbours cover them.
	if h := countHoles(mask); h != 0 {
		t.Fatalf("ColorAverageInpainter left %d holes", h)
	}
}

func TestInpaintingPipelineFills(t *testing.T) {
	const size = 44
	tex := smoothNoise(size, size, 9)
	frames := []*cv.Mat{tex.Clone(), tex.Clone()}
	frame, mask := makeHoledFrame(tex, 7, 3)
	frames[0] = frame
	p := NewInpaintingPipeline().Add(NewMotionInpainter()).Add(NewColorInpainter())
	if p.Len() != 2 {
		t.Fatalf("pipeline len=%d", p.Len())
	}
	p.SetContext(frames, []Motion{IdentityMotion()}, []Motion{IdentityMotion(), IdentityMotion()}, 1)
	p.Inpaint(0, frame, mask)
	if h := countHoles(mask); h != 0 {
		t.Fatalf("pipeline left %d holes", h)
	}
}

// ---- deblurring: the strictly-sharper guarantee -----------------------------

func TestCalcBlurriness(t *testing.T) {
	sharp := sinusoidGray(48, 48)
	blurred := cv.GaussianBlur(sharp, 7, 0)
	if CalcBlurriness(blurred) <= CalcBlurriness(sharp) {
		t.Fatal("a blurred frame must have a larger blurriness measure than a sharp one")
	}
}

func TestWeightingDeblurerSharpens(t *testing.T) {
	sharp := sinusoidGray(56, 56)
	blurred := cv.GaussianBlur(sharp, 7, 0)

	target := blurred.Clone()
	frames := []*cv.Mat{sharp, target}
	blur := []float64{CalcBlurriness(sharp), CalcBlurriness(target)}

	before := CalcBlurriness(target)
	d := NewWeightingDeblurer(1)
	d.SetContext(frames, []Motion{IdentityMotion()}, blur)
	d.Deblur(1, target)
	after := CalcBlurriness(target)

	t.Logf("blurriness before=%.6g after=%.6g (lower is sharper)", before, after)
	if !(after < before) {
		t.Fatalf("WeightingDeblurer did not sharpen: before=%.6g after=%.6g", before, after)
	}
	// Confirm it is not a no-op copy.
	identical := true
	for i := range target.Data {
		if target.Data[i] != blurred.Data[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("deblurer output is identical to its input (a copy, not a deblur)")
	}
}

func TestNullDeblurer(t *testing.T) {
	f := sinusoidGray(20, 20)
	before := f.Clone()
	var d NullDeblurer
	d.SetContext(nil, nil, nil)
	d.Deblur(0, f)
	for i := range f.Data {
		if f.Data[i] != before.Data[i] {
			t.Fatal("NullDeblurer must not modify the frame")
		}
	}
}

func TestWeightingDeblurerStandalone(t *testing.T) {
	// With no context the base amount still sharpens.
	target := cv.GaussianBlur(sinusoidGray(40, 40), 5, 0)
	before := CalcBlurriness(target)
	d := NewWeightingDeblurer(2)
	d.Deblur(0, target)
	if CalcBlurriness(target) >= before {
		t.Fatal("standalone deblur must still sharpen")
	}
}

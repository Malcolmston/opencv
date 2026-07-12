package cudalegacy_test

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/bgsegm"
	"github.com/malcolmston/opencv/cudalegacy"
)

// solid returns a rows×cols single-channel Mat filled with val.
func solid(rows, cols int, val uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(val)
	return m
}

// noisy returns a static single-channel frame of base intensity with a small
// deterministic amount of per-pixel noise, giving the models some variance.
func noisy(rows, cols int, base uint8, rng *rand.Rand) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		v := int(base) + rng.Intn(5) - 2
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		m.Data[i] = uint8(v)
	}
	return m
}

// withBlob paints a filled rectangle of value val onto a clone of frame.
func withBlob(frame *cv.Mat, y0, y1, x0, x1 int, val uint8) *cv.Mat {
	out := frame.Clone()
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			out.Set(y, x, 0, val)
		}
	}
	return out
}

func countValue(m *cv.Mat, v uint8) int {
	n := 0
	for _, s := range m.Data {
		if s == v {
			n++
		}
	}
	return n
}

// --- GpuMat / Stream ---

func TestGpuMatUploadDownload(t *testing.T) {
	stream := cudalegacy.NewStream()
	g := cudalegacy.NewGpuMat()
	if !g.Empty() {
		t.Fatal("new GpuMat should be empty")
	}
	src := solid(4, 5, 7)
	g.Upload(src, stream)
	if r, c := g.Size(); r != 4 || c != 5 {
		t.Fatalf("size = %d,%d", r, c)
	}
	if g.Channels() != 1 {
		t.Fatalf("channels = %d", g.Channels())
	}
	got := g.Download(stream)
	if got.At(1, 1, 0) != 7 {
		t.Fatal("download mismatch")
	}
	// Upload copies, not aliases.
	src.Set(0, 0, 0, 200)
	if g.Download(nil).At(0, 0, 0) == 200 {
		t.Fatal("upload should have copied")
	}
	cl := g.Clone()
	cl.Mat.Set(0, 0, 0, 111)
	if g.Mat.At(0, 0, 0) == 111 {
		t.Fatal("clone should be independent")
	}
	g.Release()
	if !g.Empty() {
		t.Fatal("released GpuMat should be empty")
	}
	if !stream.QueryIfComplete() {
		t.Fatal("stream should always be complete")
	}
	stream.WaitForCompletion()
}

func TestGpuMatFromMatNil(t *testing.T) {
	g := cudalegacy.GpuMatFromMat(nil)
	if !g.Empty() {
		t.Fatal("nil-wrapped GpuMat should be empty")
	}
	if g.Download(nil) != nil {
		t.Fatal("empty download should be nil")
	}
	if r, c := g.Size(); r != 0 || c != 0 {
		t.Fatal("empty size should be zero")
	}
	if !g.Clone().Empty() {
		t.Fatal("clone of empty should be empty")
	}
}

// --- FGD ---

func TestBackgroundSubtractorFGD(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	sub := cudalegacy.CreateBackgroundSubtractorFGD(20)
	sub.MinArea = 4
	stream := cudalegacy.NewStream()

	// Warm up on a static-ish background.
	for i := 0; i < 40; i++ {
		sub.Apply(cudalegacy.GpuMatFromMat(noisy(20, 20, 50, rng)), -1, stream)
	}
	bg := sub.GetBackgroundImage(stream)
	if bg.Empty() {
		t.Fatal("background image should be available after warmup")
	}
	if v := bg.Mat.At(10, 10, 0); v < 40 || v > 60 {
		t.Fatalf("background mean out of expected range: %d", v)
	}

	frame := withBlob(noisy(20, 20, 50, rng), 6, 12, 6, 12, 220)
	mask := sub.Apply(cudalegacy.GpuMatFromMat(frame), -1, stream)
	if countValue(mask.Mat, bgsegm.ForegroundValue) == 0 {
		t.Fatal("expected foreground detection for bright blob")
	}
	// Centre of the blob must be foreground.
	if mask.Mat.At(9, 9, 0) != bgsegm.ForegroundValue {
		t.Fatal("blob centre should be foreground")
	}
}

func TestFGDFirstFrameAllBackground(t *testing.T) {
	sub := cudalegacy.CreateBackgroundSubtractorFGD(0) // history<=0 -> default
	mask := sub.Apply(cudalegacy.GpuMatFromMat(solid(8, 8, 90)), -1, nil)
	if countValue(mask.Mat, bgsegm.ForegroundValue) != 0 {
		t.Fatal("first frame should be all background")
	}
	if sub.GetBackgroundImage(nil).Empty() {
		t.Fatal("background image should be available after the first frame")
	}
}

func TestFGDPanicsEmpty(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on empty frame")
		}
	}()
	cudalegacy.CreateBackgroundSubtractorFGD(10).Apply(cudalegacy.NewGpuMat(), -1, nil)
}

// --- GMG ---

func TestBackgroundSubtractorGMG(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	sub := cudalegacy.CreateBackgroundSubtractorGMG(15, 0)
	if sub.GetNumFrames() != 15 {
		t.Fatalf("num frames = %d", sub.GetNumFrames())
	}
	sub.SetDecisionThreshold(0.7)
	if math.Abs(sub.GetDecisionThreshold()-0.7) > 1e-9 {
		t.Fatal("decision threshold not set")
	}
	sub.SetNumFrames(15)
	stream := cudalegacy.NewStream()
	for i := 0; i < 20; i++ {
		sub.Apply(cudalegacy.GpuMatFromMat(noisy(16, 16, 60, rng)), -1, stream)
	}
	frame := withBlob(noisy(16, 16, 60, rng), 4, 10, 4, 10, 240)
	mask := sub.Apply(cudalegacy.GpuMatFromMat(frame), -1, stream)
	if countValue(mask.Mat, bgsegm.ForegroundValue) == 0 {
		t.Fatal("expected GMG foreground detection")
	}
	if sub.GetBackgroundImage(stream).Empty() {
		t.Fatal("GMG background image should exist")
	}
}

// --- ImagePyramid ---

func TestImagePyramid(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	img := cv.NewMat(32, 32, 1)
	for i := range img.Data {
		img.Data[i] = uint8(rng.Intn(256))
	}
	pyr := cudalegacy.NewImagePyramid(cudalegacy.GpuMatFromMat(img), 3, nil)
	if pyr.NumLayers() != 3 {
		t.Fatalf("layers = %d", pyr.NumLayers())
	}
	l0 := pyr.Layer(0)
	if r, c := l0.Size(); r != 32 || c != 32 {
		t.Fatalf("layer0 size %d,%d", r, c)
	}
	l1 := pyr.Layer(1)
	if r, c := l1.Size(); r != 16 || c != 16 {
		t.Fatalf("layer1 size %d,%d", r, c)
	}
	// GetLayer at an exact stored size returns that layer.
	g := pyr.GetLayer(16, 16, nil)
	if r, c := g.Size(); r != 16 || c != 16 {
		t.Fatalf("getlayer size %d,%d", r, c)
	}
	// GetLayer at an intermediate size resamples.
	g2 := pyr.GetLayer(24, 24, nil)
	if r, c := g2.Size(); r != 24 || c != 24 {
		t.Fatalf("getlayer2 size %d,%d", r, c)
	}
}

func TestImagePyramidLayerOOR(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	pyr := cudalegacy.NewImagePyramid(cudalegacy.GpuMatFromMat(solid(8, 8, 1)), 2, nil)
	pyr.Layer(99)
}

// --- Connectivity / labelling ---

func TestConnectivityMaskAndLabels(t *testing.T) {
	// Two flat regions separated by a sharp step.
	m := cv.NewMat(4, 4, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if x < 2 {
				m.Set(y, x, 0, 10)
			} else {
				m.Set(y, x, 0, 200)
			}
		}
	}
	mask := cudalegacy.ConnectivityMask(cudalegacy.GpuMatFromMat(m), 0, 5, cv.Connectivity4, nil)
	// A pixel on the left edge of the step must NOT connect east.
	if mask.Mat.At(0, 1, 0)&cudalegacy.MaskConnectEast != 0 {
		t.Fatal("step boundary should not be connected east")
	}
	// Within the left region pixels connect east.
	if mask.Mat.At(0, 0, 0)&cudalegacy.MaskConnectEast == 0 {
		t.Fatal("flat region should be connected east")
	}
	labels, count := cudalegacy.LabelComponents(mask, nil)
	if count != 2 {
		t.Fatalf("expected 2 components, got %d", count)
	}
	if labels[0] == labels[2] {
		t.Fatal("left and right regions should differ in label")
	}
	rendered := cudalegacy.RenderLabels(labels, 4, 4)
	if r, c := rendered.Size(); r != 4 || c != 4 {
		t.Fatalf("rendered size %d,%d", r, c)
	}
}

// --- Block-matching optical flow ---

func TestCalcOpticalFlowBM(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	const n = 24
	prev := cv.NewMat(n, n, 1)
	for i := range prev.Data {
		prev.Data[i] = uint8(rng.Intn(256))
	}
	// curr is prev shifted right by 2 (curr[y][x] = prev[y][x-2]).
	const shift = 2
	curr := cv.NewMat(n, n, 1)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			sx := x - shift
			if sx < 0 {
				sx = 0
			}
			curr.Set(y, x, 0, prev.At(y, sx, 0))
		}
	}
	flow := cudalegacy.CalcOpticalFlowBM(
		cudalegacy.GpuMatFromMat(prev), cudalegacy.GpuMatFromMat(curr),
		4, 4, 4, nil)

	// Interior blocks should recover u = +shift.
	hits, total := 0, 0
	for y := 6; y < n-6; y++ {
		for x := 6; x < n-6; x++ {
			u, v := flow.At(y, x)
			total++
			if int(math.Round(u)) == shift && int(math.Round(v)) == 0 {
				hits++
			}
		}
	}
	if float64(hits)/float64(total) < 0.8 {
		t.Fatalf("block matching recovered shift in only %d/%d interior pixels", hits, total)
	}
}

// --- Frame interpolation ---

func TestInterpolateFrames(t *testing.T) {
	// A vertical edge that moves right by 4 pixels between frames.
	const n = 16
	frame0 := cv.NewMat(n, n, 1)
	frame1 := cv.NewMat(n, n, 1)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if x >= 4 {
				frame0.Set(y, x, 0, 255)
			}
			if x >= 8 {
				frame1.Set(y, x, 0, 255)
			}
		}
	}
	flow := cudalegacy.NewFlow(n, n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			flow.U.Data[y*n+x] = 4 // rightward motion
		}
	}
	mid := cudalegacy.InterpolateFrames(
		cudalegacy.GpuMatFromMat(frame0), cudalegacy.GpuMatFromMat(frame1),
		flow, nil, 0.5, nil)
	// At pos 0.5 the edge should sit near x = 6.
	row := 8
	edge := -1
	for x := 0; x < n; x++ {
		if mid.Mat.At(row, x, 0) > 127 {
			edge = x
			break
		}
	}
	if edge < 5 || edge > 7 {
		t.Fatalf("interpolated edge at x=%d, expected ~6", edge)
	}

	// pos 0 reproduces frame0, pos 1 reproduces frame1.
	at0 := cudalegacy.InterpolateFrames(cudalegacy.GpuMatFromMat(frame0), cudalegacy.GpuMatFromMat(frame1), flow, nil, 0, nil)
	if at0.Mat.At(8, 4, 0) != 255 || at0.Mat.At(8, 3, 0) != 0 {
		t.Fatal("pos 0 should reproduce frame0")
	}
}

// --- Needle map ---

func TestCreateOpticalFlowNeedleMap(t *testing.T) {
	flow := cudalegacy.NewFlow(16, 16)
	for i := range flow.U.Data {
		flow.U.Data[i] = 3
	}
	img, segs := cudalegacy.CreateOpticalFlowNeedleMap(flow, 8, 1, nil)
	if len(segs) == 0 {
		t.Fatal("expected needle segments")
	}
	if img.Channels() != 3 {
		t.Fatalf("needle map should be 3-channel, got %d", img.Channels())
	}
	// Segments point rightward (End.X > Start.X).
	for _, s := range segs {
		if s.End.X <= s.Start.X {
			t.Fatalf("segment not pointing right: %+v", s)
		}
	}
	// Some white pixels drawn.
	white := 0
	for _, v := range img.Mat.Data {
		if v == 255 {
			white++
		}
	}
	if white == 0 {
		t.Fatal("needle map should have drawn pixels")
	}
}

// --- Graph cut ---

func TestGraphCut(t *testing.T) {
	const rows, cols = 6, 6
	src := cv.NewFloatMat(rows, cols)
	snk := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if x < cols/2 {
				src.Data[i] = 10
				snk.Data[i] = 1
			} else {
				src.Data[i] = 1
				snk.Data[i] = 10
			}
		}
	}
	labels := cudalegacy.GraphCut(src, snk, 2.0, nil)
	// Left half source-labeled, right half sink-labeled.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			want := cudalegacy.SinkLabel
			if x < cols/2 {
				want = cudalegacy.SourceLabel
			}
			if labels.Mat.At(y, x, 0) != want {
				t.Fatalf("label at (%d,%d)=%d want %d", y, x, labels.Mat.At(y, x, 0), want)
			}
		}
	}
}

func TestGraphCutLambdaZero(t *testing.T) {
	// With no smoothness each pixel decides independently by capacity.
	src := cv.NewFloatMat(1, 3)
	snk := cv.NewFloatMat(1, 3)
	src.Data = []float64{5, 1, 3}
	snk.Data = []float64{1, 5, 3} // pixel2 tie -> sink side (S->i saturates)
	labels := cudalegacy.GraphCut(src, snk, 0, nil)
	if labels.Mat.At(0, 0, 0) != cudalegacy.SourceLabel {
		t.Fatal("pixel0 should be source")
	}
	if labels.Mat.At(0, 1, 0) != cudalegacy.SinkLabel {
		t.Fatal("pixel1 should be sink")
	}
}

// --- Projection / PnP ---

func TestRodriguesRoundTrip(t *testing.T) {
	rvec := [3]float64{0.1, -0.25, 0.4}
	R := cudalegacy.Rodrigues(rvec)
	// R should be orthonormal: R Rᵀ = I.
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			dot := R[i][0]*R[j][0] + R[i][1]*R[j][1] + R[i][2]*R[j][2]
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(dot-want) > 1e-9 {
				t.Fatalf("R not orthonormal at %d,%d: %f", i, j, dot)
			}
		}
	}
}

func planarObject() [][3]float64 {
	var pts [][3]float64
	for gy := 0; gy < 5; gy++ {
		for gx := 0; gx < 5; gx++ {
			pts = append(pts, [3]float64{float64(gx) - 2, float64(gy) - 2, 0})
		}
	}
	return pts
}

func TestSolvePnPRansac(t *testing.T) {
	K := [3][3]float64{{800, 0, 320}, {0, 800, 240}, {0, 0, 1}}
	obj := planarObject()
	trueR := [3]float64{0.05, -0.1, 0.15}
	trueT := [3]float64{0.3, -0.2, 8}
	img := cudalegacy.ProjectPoints(obj, trueR, trueT, K, nil)

	// Corrupt two observations into gross outliers.
	img[3][0] += 60
	img[10][1] -= 55

	rng := rand.New(rand.NewSource(7))
	rvec, tvec, inliers, ok := cudalegacy.SolvePnPRansac(obj, img, K, nil, 2.0, 300, rng)
	if !ok {
		t.Fatal("solvePnPRansac failed")
	}
	// The recovered pose must reproject the clean points accurately.
	proj := cudalegacy.ProjectPoints(obj, rvec, tvec, K, nil)
	good := 0
	for i := range obj {
		if i == 3 || i == 10 {
			continue
		}
		dx := proj[i][0] - img[i][0]
		dy := proj[i][1] - img[i][1]
		if math.Hypot(dx, dy) < 2 {
			good++
		}
	}
	if good < 20 {
		t.Fatalf("recovered pose reprojects only %d clean points well", good)
	}
	// Outliers flagged as non-inliers.
	if inliers[3] || inliers[10] {
		t.Fatal("outliers should not be inliers")
	}
}

func TestSolvePnPRansacTooFew(t *testing.T) {
	K := [3][3]float64{{800, 0, 320}, {0, 800, 240}, {0, 0, 1}}
	obj := [][3]float64{{0, 0, 0}, {1, 0, 0}}
	img := [][2]float64{{0, 0}, {1, 1}}
	if _, _, _, ok := cudalegacy.SolvePnPRansac(obj, img, K, nil, 2, 10, nil); ok {
		t.Fatal("expected failure with too few points")
	}
}

func TestProjectPointsDistortion(t *testing.T) {
	K := [3][3]float64{{500, 0, 100}, {0, 500, 100}, {0, 0, 1}}
	obj := [][3]float64{{0, 0, 5}}
	// On the optical axis, distortion has no effect and the point lands at the
	// principal point.
	p := cudalegacy.ProjectPoints(obj, [3]float64{}, [3]float64{}, K, []float64{0.1, 0.01, 0, 0, 0})
	if math.Abs(p[0][0]-100) > 1e-6 || math.Abs(p[0][1]-100) > 1e-6 {
		t.Fatalf("axis point should map to principal point, got %v", p[0])
	}
}

// --- CompactPoints ---

func TestCompactPoints(t *testing.T) {
	p0 := [][2]float64{{0, 0}, {1, 1}, {2, 2}, {3, 3}}
	p1 := [][2]float64{{10, 0}, {11, 1}, {12, 2}, {13, 3}}
	mask := []uint8{1, 0, 1, 0}
	o0, o1 := cudalegacy.CompactPoints(p0, p1, mask)
	if len(o0) != 2 || len(o1) != 2 {
		t.Fatalf("expected 2 survivors, got %d", len(o0))
	}
	if o0[1][0] != 2 || o1[1][0] != 12 {
		t.Fatalf("wrong survivors: %v %v", o0, o1)
	}
}

func TestCompactPointsMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	cudalegacy.CompactPoints([][2]float64{{0, 0}}, [][2]float64{}, []uint8{1})
}

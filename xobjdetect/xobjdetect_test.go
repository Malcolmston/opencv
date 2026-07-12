package xobjdetect

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- synthetic data ----------------------------------------------------------

// makePositive builds a 24x24 grayscale patch of the target object: a dark
// background with a bright, roughly centred filled block. Small deterministic
// jitter from rng gives the positive class some variety.
func makePositive(rng *rand.Rand) *cv.Mat {
	m := cv.NewMat(24, 24, 1)
	dx := rng.Intn(3) - 1
	dy := rng.Intn(3) - 1
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			v := 30 + rng.Intn(11) // dark background, 30..40
			if x >= 6+dx && x < 18+dx && y >= 6+dy && y < 18+dy {
				v = 200 + rng.Intn(41) // bright central block, 200..240
			}
			m.Set(y, x, 0, uint8(clampInt(v, 0, 255)))
		}
	}
	return m
}

// makeNegative builds a 24x24 grayscale patch that lacks the central bright
// block: either mid-grey noise or a dark background sprinkled with a few small
// bright specks well away from the centre.
func makeNegative(rng *rand.Rand) *cv.Mat {
	m := cv.NewMat(24, 24, 1)
	midGrey := rng.Intn(2) == 0
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			var v int
			if midGrey {
				v = 90 + rng.Intn(80) // 90..170 noise
			} else {
				v = 30 + rng.Intn(11)
			}
			m.Set(y, x, 0, uint8(clampInt(v, 0, 255)))
		}
	}
	if !midGrey {
		for i := 0; i < 5; i++ {
			sx := rng.Intn(20)
			sy := rng.Intn(20)
			// Keep specks out of the central block region.
			if sx >= 5 && sx <= 17 && sy >= 5 && sy <= 17 {
				continue
			}
			m.Set(sy, sx, 0, 220)
		}
	}
	return m
}

// plantObject draws a positive object into bg at (x, y).
func plantObject(bg *cv.Mat, obj *cv.Mat, x, y int) {
	obj.CopyTo(bg, y, x)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func trainDetector(t *testing.T) *WBDetector {
	t.Helper()
	rng := rand.New(rand.NewSource(20260712))
	var pos, neg []*cv.Mat
	for i := 0; i < 40; i++ {
		pos = append(pos, makePositive(rng))
		neg = append(neg, makeNegative(rng))
	}
	d := NewWBDetector()
	d.Rounds = 40
	d.NumFeatures = 200
	d.Seed = 7
	if err := d.Train(pos, neg); err != nil {
		t.Fatalf("Train: %v", err)
	}
	if !d.Trained() {
		t.Fatal("detector reports untrained after Train")
	}
	return d
}

// --- integral channels -------------------------------------------------------

func TestIntegralChannelsRectSum(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	img := cv.NewMat(19, 23, 1)
	for i := range img.Data {
		img.Data[i] = uint8(rng.Intn(256))
	}
	planes := computeChannels(img)
	ic := newIntegralChannels(img)

	rects := [][4]int{{0, 0, 23, 19}, {2, 3, 5, 4}, {10, 7, 8, 9}, {20, 16, 3, 3}}
	for ch := 0; ch < NumChannels; ch++ {
		for _, r := range rects {
			x, y, w, h := r[0], r[1], r[2], r[3]
			var want float64
			for yy := y; yy < y+h; yy++ {
				for xx := x; xx < x+w; xx++ {
					want += planes[ch][yy*img.Cols+xx]
				}
			}
			got := ic.rectSum(ch, x, y, w, h)
			if math.Abs(got-want) > 1e-6 {
				t.Fatalf("ch %d rect %v: rectSum=%.6f want %.6f", ch, r, got, want)
			}
			mean := ic.rectMean(ch, x, y, w, h)
			if math.Abs(mean-want/float64(w*h)) > 1e-6 {
				t.Fatalf("ch %d rect %v: rectMean=%.6f want %.6f", ch, r, mean, want/float64(w*h))
			}
		}
	}
}

func TestRectSumClamping(t *testing.T) {
	img := cv.NewMat(10, 10, 1)
	img.SetTo(100)
	ic := newIntegralChannels(img)
	if got := ic.rectSum(0, -5, -5, 3, 3); got != 0 {
		t.Fatalf("fully out-of-bounds rect: got %.3f want 0", got)
	}
	if got := ic.rectMean(0, 0, 0, 0, 0); got != 0 {
		t.Fatalf("zero-area rect: got %.3f want 0", got)
	}
	// A rect straddling the border should clamp to the in-image part.
	if got := ic.rectSum(0, 8, 8, 5, 5); got <= 0 {
		t.Fatalf("straddling rect should be positive, got %.3f", got)
	}
}

func TestComputeChannelsColor(t *testing.T) {
	img := cv.NewMat(8, 8, 3)
	for i := range img.Data {
		img.Data[i] = uint8((i * 7) % 256)
	}
	planes := computeChannels(img)
	if len(planes) != NumChannels {
		t.Fatalf("channel count = %d want %d", len(planes), NumChannels)
	}
	for c, p := range planes {
		if len(p) != 64 {
			t.Fatalf("channel %d length = %d want 64", c, len(p))
		}
	}
}

// --- feature pool / evaluator ------------------------------------------------

func TestFeaturePoolDeterministic(t *testing.T) {
	a := NewFeaturePool(24, 24, 100, rand.New(rand.NewSource(42)))
	b := NewFeaturePool(24, 24, 100, rand.New(rand.NewSource(42)))
	if a.Len() != 100 || b.Len() != 100 {
		t.Fatalf("pool sizes = %d,%d want 100", a.Len(), b.Len())
	}
	for i := range a.Features {
		if a.Features[i] != b.Features[i] {
			t.Fatalf("feature %d differs for equal seeds: %+v vs %+v", i, a.Features[i], b.Features[i])
		}
	}
	for _, f := range a.Features {
		if f.X < 0 || f.Y < 0 || f.X+f.W > 24 || f.Y+f.H > 24 || f.Channel < 0 || f.Channel >= NumChannels {
			t.Fatalf("feature out of window bounds: %+v", f)
		}
	}
}

func TestACFFeatureEvaluatorSample(t *testing.T) {
	pool := NewFeaturePool(24, 24, 50, rand.New(rand.NewSource(3)))
	e := NewACFFeatureEvaluator(pool)
	if e.Pool() != pool {
		t.Fatal("Pool() did not return the bound pool")
	}
	rng := rand.New(rand.NewSource(9))
	// Sample resizes a differently sized patch to the window.
	big := cv.NewMat(48, 48, 1)
	for i := range big.Data {
		big.Data[i] = uint8(rng.Intn(256))
	}
	v := e.Sample(big)
	if len(v) != 50 {
		t.Fatalf("feature vector length = %d want 50", len(v))
	}
	// EvaluateWindow on the resized window should reproduce Sample.
	e.SetImage(resizeToWindow(big, 24, 24))
	w := e.EvaluateWindow(0, 0)
	for i := range v {
		if math.Abs(v[i]-w[i]) > 1e-9 {
			t.Fatalf("Sample and EvaluateWindow disagree at %d", i)
		}
	}
}

func TestEvaluateWindowBeforeSetImagePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	e := NewACFFeatureEvaluator(NewFeaturePool(24, 24, 10, rand.New(rand.NewSource(1))))
	_ = e.EvaluateWindow(0, 0)
}

// --- WaldBoost ---------------------------------------------------------------

func TestWaldBoostSeparable(t *testing.T) {
	// Two clearly separable clusters on a 3-D feature vector.
	rng := rand.New(rand.NewSource(5))
	var pos, neg [][]float64
	for i := 0; i < 30; i++ {
		pos = append(pos, []float64{0.8 + rng.Float64()*0.1, rng.Float64(), 0.9 + rng.Float64()*0.1})
		neg = append(neg, []float64{0.1 + rng.Float64()*0.1, rng.Float64(), 0.1 + rng.Float64()*0.1})
	}
	wb := NewWaldBoost(20)
	if err := wb.Train(pos, neg); err != nil {
		t.Fatalf("Train: %v", err)
	}
	if len(wb.Stumps) == 0 || len(wb.SPRT) != len(wb.Stumps) {
		t.Fatalf("stumps=%d sprt=%d", len(wb.Stumps), len(wb.SPRT))
	}
	correct := 0
	for _, p := range pos {
		if _, ok := wb.Predict(p); ok {
			correct++
		}
	}
	for _, n := range neg {
		if _, ok := wb.Predict(n); !ok {
			correct++
		}
	}
	if correct < 58 { // out of 60
		t.Fatalf("WaldBoost accuracy too low: %d/60", correct)
	}
	// Positive scores should exceed negative scores on average.
	sp, _ := wb.Predict(pos[0])
	sn, _ := wb.Predict(neg[0])
	if sp <= sn {
		t.Fatalf("positive score %.3f not above negative score %.3f", sp, sn)
	}
}

func TestWaldBoostErrors(t *testing.T) {
	wb := NewWaldBoost(5)
	if err := wb.Train(nil, [][]float64{{1}}); err == nil {
		t.Fatal("expected error for empty positives")
	}
	if err := wb.Train([][]float64{{1}}, nil); err == nil {
		t.Fatal("expected error for empty negatives")
	}
	if err := wb.Train([][]float64{{}}, [][]float64{{}}); err == nil {
		t.Fatal("expected error for zero-length vectors")
	}
	if err := wb.Train([][]float64{{1, 2}}, [][]float64{{1}}); err == nil {
		t.Fatal("expected error for ragged vectors")
	}
}

func TestWaldBoostPredictUntrainedPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	wb := NewWaldBoost(5)
	_, _ = wb.Predict([]float64{1})
}

// --- WBDetector --------------------------------------------------------------

func TestWBDetectorTrainDetect(t *testing.T) {
	d := trainDetector(t)

	bg := cv.NewMat(96, 96, 1)
	rng := rand.New(rand.NewSource(99))
	for i := range bg.Data {
		bg.Data[i] = uint8(30 + rng.Intn(11))
	}
	planted := cv.Rect{X: 36, Y: 28, Width: 24, Height: 24}
	obj := makePositive(rand.New(rand.NewSource(555)))
	plantObject(bg, obj, planted.X, planted.Y)

	rects, confs := d.Detect(bg)
	if len(rects) == 0 {
		t.Fatal("no detections for planted object")
	}
	if len(rects) != len(confs) {
		t.Fatalf("rects/confidences length mismatch %d vs %d", len(rects), len(confs))
	}
	// Confidences must be sorted descending.
	for i := 1; i < len(confs); i++ {
		if confs[i] > confs[i-1] {
			t.Fatalf("confidences not sorted: %v", confs)
		}
	}
	best := 0.0
	for _, r := range rects {
		if iou := rectIoU(r, planted); iou > best {
			best = iou
		}
	}
	if best < 0.4 {
		t.Fatalf("best detection IoU with planted object = %.3f, want >= 0.4 (rects=%v)", best, rects)
	}
}

func TestWBDetectorGobRoundTrip(t *testing.T) {
	d := trainDetector(t)

	bg := cv.NewMat(96, 96, 1)
	rng := rand.New(rand.NewSource(99))
	for i := range bg.Data {
		bg.Data[i] = uint8(30 + rng.Intn(11))
	}
	obj := makePositive(rand.New(rand.NewSource(555)))
	plantObject(bg, obj, 36, 28)

	rects1, confs1 := d.Detect(bg)

	var buf bytes.Buffer
	if err := d.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}
	var d2 WBDetector
	if err := d2.Read(&buf); err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !d2.Trained() {
		t.Fatal("restored detector not trained")
	}
	rects2, confs2 := d2.Detect(bg)

	if len(rects1) != len(rects2) {
		t.Fatalf("detection count changed after round-trip: %d vs %d", len(rects1), len(rects2))
	}
	for i := range rects1 {
		if rects1[i] != rects2[i] {
			t.Fatalf("rect %d changed: %v vs %v", i, rects1[i], rects2[i])
		}
		if math.Abs(confs1[i]-confs2[i]) > 1e-9 {
			t.Fatalf("confidence %d changed: %.6f vs %.6f", i, confs1[i], confs2[i])
		}
	}
}

func TestWBDetectorErrorsAndPanics(t *testing.T) {
	d := NewWBDetector()
	if err := d.Train(nil, nil); err == nil {
		t.Fatal("expected error training with no samples")
	}
	if err := d.Write(&bytes.Buffer{}); err == nil {
		t.Fatal("expected error writing untrained detector")
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic detecting with untrained detector")
			}
		}()
		_, _ = d.Detect(cv.NewMat(32, 32, 1))
	}()
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic from Evaluator on untrained detector")
			}
		}()
		_ = d.Evaluator()
	}()

	// A truncated / empty stream must fail to Read.
	var d2 WBDetector
	if err := d2.Read(bytes.NewReader([]byte{0x01, 0x02})); err == nil {
		t.Fatal("expected error reading garbage")
	}
}

func TestWBDetectorEvaluator(t *testing.T) {
	d := trainDetector(t)
	e := d.Evaluator()
	v := e.Sample(makePositive(rand.New(rand.NewSource(1))))
	if len(v) != d.NumFeatures {
		t.Fatalf("evaluator vector length = %d want %d", len(v), d.NumFeatures)
	}
}

func TestRectIoU(t *testing.T) {
	a := cv.Rect{X: 0, Y: 0, Width: 10, Height: 10}
	if iou := rectIoU(a, a); math.Abs(iou-1) > 1e-9 {
		t.Fatalf("self IoU = %.6f want 1", iou)
	}
	b := cv.Rect{X: 20, Y: 20, Width: 5, Height: 5}
	if iou := rectIoU(a, b); iou != 0 {
		t.Fatalf("disjoint IoU = %.6f want 0", iou)
	}
	c := cv.Rect{X: 5, Y: 0, Width: 10, Height: 10}
	iou := rectIoU(a, c) // intersection 5x10=50, union 200-50=150
	if math.Abs(iou-50.0/150.0) > 1e-9 {
		t.Fatalf("overlap IoU = %.6f want %.6f", iou, 50.0/150.0)
	}
}

func TestNonMaxSuppression(t *testing.T) {
	rects := []cv.Rect{
		{X: 0, Y: 0, Width: 10, Height: 10},
		{X: 1, Y: 1, Width: 10, Height: 10}, // overlaps the first heavily
		{X: 50, Y: 50, Width: 10, Height: 10},
	}
	scores := []float64{0.9, 0.8, 0.7}
	keptR, keptS := nonMaxSuppression(rects, scores, 0.3)
	if len(keptR) != 2 {
		t.Fatalf("kept %d boxes, want 2: %v", len(keptR), keptR)
	}
	if keptS[0] < keptS[1] {
		t.Fatalf("kept scores not descending: %v", keptS)
	}
}

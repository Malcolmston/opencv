package ximgproc_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/ximgproc"
)

// ---- shared local helpers ------------------------------------------------

// rampImageH builds a horizontal intensity ramp (increases with x).
func rampImageH(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := 20 + 3*x
			if v > 255 {
				v = 255
			}
			m.Data[y*cols+x] = uint8(v)
		}
	}
	return m
}

// rampImageV builds a vertical intensity ramp (increases with y).
func rampImageV(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := 20 + 3*y
			if v > 255 {
				v = 255
			}
			m.Data[y*cols+x] = uint8(v)
		}
	}
	return m
}

// texturedStep builds a step edge overlaid with a high-frequency checkerboard
// texture, plus mild noise. Structure (the step) is large-scale; texture is
// small-scale. Deterministic.
func texturedStep(rows, cols, split int, seed uint64) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	var s uint64 = seed
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := 70.0
			if x >= split {
				base = 185.0
			}
			// Checkerboard texture, amplitude 25.
			if (x/2+y/2)%2 == 0 {
				base += 25
			} else {
				base -= 25
			}
			s = s*6364136223846793005 + 1442695040888963407
			base += (float64(s>>40)/float64(1<<24))*6 - 3
			if base < 0 {
				base = 0
			}
			if base > 255 {
				base = 255
			}
			m.Data[y*cols+x] = uint8(base)
		}
	}
	return m
}

func edgeGap(m *cv.Mat, split int) float64 {
	h := m.Rows
	left := 0.0
	right := 0.0
	nl, nr := 0, 0
	for y := 0; y < h; y++ {
		for x := 0; x < split-3; x++ {
			left += float64(m.Data[y*m.Cols+x])
			nl++
		}
		for x := split + 3; x < m.Cols; x++ {
			right += float64(m.Data[y*m.Cols+x])
			nr++
		}
	}
	return right/float64(nr) - left/float64(nl)
}

// ---- DTFilter ------------------------------------------------------------

func TestDTFilterModesEdgePreserving(t *testing.T) {
	rows, cols, split := 40, 60, 30
	src := stepImage(rows, cols, 60, 200, split, 22, 4242)
	modes := []struct {
		name string
		mode ximgproc.DTMode
	}{
		{"NC", ximgproc.DTFilterNC},
		{"IC", ximgproc.DTFilterIC},
		{"RF", ximgproc.DTFilterRF},
	}
	beforeVar := regionVariance(src, 5, 2, 30, 20)
	for _, mc := range modes {
		out := ximgproc.DTFilter(src, nil, 20, 25, mc.mode, 3)
		if out.Rows != rows || out.Cols != cols || out.Channels != 1 {
			t.Fatalf("%s: wrong shape", mc.name)
		}
		afterVar := regionVariance(out, 5, 2, 30, 20)
		if afterVar >= beforeVar {
			t.Errorf("%s: flat-region variance did not drop: %.2f -> %.2f", mc.name, beforeVar, afterVar)
		}
		gap := edgeGap(out, split)
		if gap < 110 {
			t.Errorf("%s: edge not preserved, gap=%.2f", mc.name, gap)
		}
	}
}

func TestDTFilterDeterministic(t *testing.T) {
	src := stepImage(20, 24, 50, 180, 12, 15, 7)
	a := ximgproc.DTFilter(src, nil, 15, 20, ximgproc.DTFilterRF, 3)
	b := ximgproc.DTFilter(src, nil, 15, 20, ximgproc.DTFilterRF, 3)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("non-deterministic DTFilter at %d", i)
		}
	}
}

// ---- JointBilateralFilter ------------------------------------------------

func TestJointBilateralSmoothsAndPreserves(t *testing.T) {
	rows, cols, split := 36, 48, 24
	src := stepImage(rows, cols, 60, 190, split, 20, 88)
	out := ximgproc.JointBilateralFilter(src, src, 7, 40, 6)
	before := regionVariance(src, 4, 2, 26, 16)
	after := regionVariance(out, 4, 2, 26, 16)
	if after >= before {
		t.Errorf("flat variance not reduced: %.2f -> %.2f", before, after)
	}
	if gap := edgeGap(out, split); gap < 110 {
		t.Errorf("edge collapsed, gap=%.2f", gap)
	}
}

func TestJointBilateralEdgeTransfer(t *testing.T) {
	// Clean guide has a sharp step; src is the same step buried in heavy noise.
	rows, cols, split := 30, 40, 20
	guide := stepImage(rows, cols, 60, 200, split, 0, 1)
	src := stepImage(rows, cols, 60, 200, split, 45, 2)
	out := ximgproc.JointBilateralFilter(guide, src, 9, 30, 8)
	// Guided by the clean edge, the flat regions smooth strongly.
	before := regionVariance(src, 3, 2, 24, 14)
	after := regionVariance(out, 3, 2, 24, 14)
	if after >= before*0.7 {
		t.Errorf("cross-guidance did not smooth enough: %.2f -> %.2f", before, after)
	}
}

// ---- RollingGuidanceFilter -----------------------------------------------

func TestRollingGuidanceRemovesTexture(t *testing.T) {
	rows, cols, split := 40, 56, 28
	src := texturedStep(rows, cols, split, 314)
	out := ximgproc.RollingGuidanceFilter(src, 9, 40, 5, 4)
	// Texture variance in a flat (single-side) region must fall substantially.
	before := regionVariance(src, 4, 2, 30, 20)
	after := regionVariance(out, 4, 2, 30, 20)
	if after >= before*0.5 {
		t.Errorf("texture not removed: var %.2f -> %.2f", before, after)
	}
	// The large-scale step survives.
	if gap := edgeGap(out, split); gap < 90 {
		t.Errorf("structure edge lost, gap=%.2f", gap)
	}
}

// ---- FastGlobalSmootherFilter --------------------------------------------

func TestFastGlobalSmootherEdgePreserving(t *testing.T) {
	rows, cols, split := 40, 60, 30
	src := stepImage(rows, cols, 60, 200, split, 22, 55)
	out := ximgproc.FastGlobalSmootherFilter(src, nil, 30, 25, 3)
	before := regionVariance(src, 5, 2, 30, 20)
	after := regionVariance(out, 5, 2, 30, 20)
	if after >= before {
		t.Errorf("FGS flat variance did not drop: %.2f -> %.2f", before, after)
	}
	if gap := edgeGap(out, split); gap < 110 {
		t.Errorf("FGS edge not preserved, gap=%.2f", gap)
	}
}

// ---- BilateralTextureFilter ----------------------------------------------

func TestBilateralTextureFilterRemovesTexture(t *testing.T) {
	rows, cols, split := 34, 48, 24
	src := texturedStep(rows, cols, split, 9001)
	out := ximgproc.BilateralTextureFilter(src, 3, 3)
	before := regionVariance(src, 4, 2, 24, 16)
	after := regionVariance(out, 4, 2, 24, 16)
	if after >= before*0.6 {
		t.Errorf("BTF did not remove texture: %.2f -> %.2f", before, after)
	}
	if gap := edgeGap(out, split); gap < 80 {
		t.Errorf("BTF structure edge lost, gap=%.2f", gap)
	}
}

// ---- WeightedMedianFilter ------------------------------------------------

func TestWeightedMedianRemovesImpulseNoise(t *testing.T) {
	rows, cols, split := 30, 40, 20
	src := stepImage(rows, cols, 60, 190, split, 0, 1)
	// Add salt-and-pepper noise deterministically.
	var s uint64 = 12321
	noisy := src.Clone()
	for i := range noisy.Data {
		s = s*6364136223846793005 + 1442695040888963407
		r := s >> 40
		if r%17 == 0 {
			noisy.Data[i] = 0
		} else if r%17 == 1 {
			noisy.Data[i] = 255
		}
	}
	out := ximgproc.WeightedMedianFilter(noisy, nil, 3, 25)
	// Median should be much closer to the clean image than the noisy input.
	errNoisy := meanAbsDiff(noisy, src)
	errOut := meanAbsDiff(out, src)
	if errOut >= errNoisy {
		t.Errorf("weighted median did not denoise: %.2f -> %.2f", errNoisy, errOut)
	}
	if gap := edgeGap(out, split); gap < 110 {
		t.Errorf("edge not preserved, gap=%.2f", gap)
	}
}

func meanAbsDiff(a, b *cv.Mat) float64 {
	var s float64
	for i := range a.Data {
		s += math.Abs(float64(a.Data[i]) - float64(b.Data[i]))
	}
	return s / float64(len(a.Data))
}

// ---- AdaptiveManifoldFilter ----------------------------------------------

func TestAdaptiveManifoldEdgePreserving(t *testing.T) {
	rows, cols, split := 36, 48, 24
	src := stepImage(rows, cols, 60, 195, split, 20, 771)
	out := ximgproc.AdaptiveManifoldFilter(src, 6, 30)
	if out.Rows != rows || out.Cols != cols {
		t.Fatal("wrong shape")
	}
	before := regionVariance(src, 4, 2, 26, 16)
	after := regionVariance(out, 4, 2, 26, 16)
	if after >= before {
		t.Errorf("AMF flat variance did not drop: %.2f -> %.2f", before, after)
	}
	if gap := edgeGap(out, split); gap < 90 {
		t.Errorf("AMF edge not preserved, gap=%.2f", gap)
	}
}

// ---- Deriche / Paillou gradients -----------------------------------------

func TestGradientDericheSignAndFlatness(t *testing.T) {
	rows, cols := 24, 32
	rh := rampImageH(rows, cols)
	gx := ximgproc.GradientDericheX(rh, 1.0)
	// Interior of a rising horizontal ramp: dI/dx > 0 everywhere.
	pos, neg := 0, 0
	for y := 4; y < rows-4; y++ {
		for x := 4; x < cols-4; x++ {
			if gx.At(y, x) > 0 {
				pos++
			} else {
				neg++
			}
		}
	}
	if pos < 10*neg {
		t.Errorf("DericheX sign wrong on rising ramp: pos=%d neg=%d", pos, neg)
	}
	// The y-gradient of a horizontal ramp is ~0.
	gy := ximgproc.GradientDericheY(rh, 1.0)
	var maxAbs float64
	for y := 4; y < rows-4; y++ {
		for x := 4; x < cols-4; x++ {
			if a := math.Abs(gy.At(y, x)); a > maxAbs {
				maxAbs = a
			}
		}
	}
	if maxAbs > 1.0 {
		t.Errorf("DericheY should be ~0 on horizontal ramp, max=%.3f", maxAbs)
	}
	// Vertical ramp: DericheY positive.
	rv := rampImageV(rows, cols)
	gyv := ximgproc.GradientDericheY(rv, 1.0)
	posv := 0
	for y := 4; y < rows-4; y++ {
		for x := 4; x < cols-4; x++ {
			if gyv.At(y, x) > 0 {
				posv++
			}
		}
	}
	if posv < (rows-8)*(cols-8)*9/10 {
		t.Errorf("DericheY sign wrong on vertical ramp: pos=%d", posv)
	}
}

func TestGradientDericheMatchesFiniteDiffSign(t *testing.T) {
	rows, cols := 28, 28
	src := stepImage(rows, cols, 40, 210, cols/2, 0, 1)
	gx := ximgproc.GradientDericheX(src, 1.2)
	agree, total := 0, 0
	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			fd := float64(src.Data[y*cols+x+1]) - float64(src.Data[y*cols+x-1])
			if fd == 0 {
				continue
			}
			total++
			if (fd > 0) == (gx.At(y, x) > 0) {
				agree++
			}
		}
	}
	if total == 0 || float64(agree)/float64(total) < 0.9 {
		t.Errorf("Deriche gradient sign disagrees with finite differences: %d/%d", agree, total)
	}
}

func TestGradientPaillouSignAndFlatness(t *testing.T) {
	rows, cols := 24, 32
	rh := rampImageH(rows, cols)
	gx := ximgproc.GradientPaillouX(rh, 1.0, 0.3)
	pos, neg := 0, 0
	for y := 4; y < rows-4; y++ {
		for x := 4; x < cols-4; x++ {
			if gx.At(y, x) > 0 {
				pos++
			} else {
				neg++
			}
		}
	}
	if pos < 10*neg {
		t.Errorf("PaillouX sign wrong on rising ramp: pos=%d neg=%d", pos, neg)
	}
	// Flat image -> zero gradient.
	flat := cv.NewMat(rows, cols, 1)
	flat.SetTo(120)
	fy := ximgproc.GradientPaillouY(flat, 1.0, 0.3)
	for i := range fy.Data {
		if math.Abs(fy.Data[i]) > 1e-6 {
			t.Fatalf("Paillou gradient of flat image nonzero: %.6f", fy.Data[i])
		}
	}
	fx := ximgproc.GradientDericheX(flat, 1.0)
	for i := range fx.Data {
		if math.Abs(fx.Data[i]) > 1e-6 {
			t.Fatalf("Deriche gradient of flat image nonzero: %.6f", fx.Data[i])
		}
	}
}

// ---- CovarianceEstimation ------------------------------------------------

func TestCovarianceEstimationSymmetricPSD(t *testing.T) {
	img := blockColorImage(20, 20, 5)
	cov := ximgproc.CovarianceEstimation(img, 3, 3)
	n := 9
	if len(cov) != n {
		t.Fatalf("wrong covariance size: %d", len(cov))
	}
	// Symmetry and non-negative diagonal.
	for i := 0; i < n; i++ {
		if len(cov[i]) != n {
			t.Fatalf("row %d wrong length", i)
		}
		if cov[i][i] < 0 {
			t.Errorf("negative variance on diagonal %d: %.3f", i, cov[i][i])
		}
		for j := 0; j < n; j++ {
			if math.Abs(cov[i][j]-cov[j][i]) > 1e-6 {
				t.Errorf("covariance not symmetric at (%d,%d)", i, j)
			}
		}
	}
	// Positive semidefinite: v^T C v >= 0 for several deterministic vectors.
	var s uint64 = 5
	for trial := 0; trial < 20; trial++ {
		v := make([]float64, n)
		for i := range v {
			s = s*6364136223846793005 + 1442695040888963407
			v[i] = float64(s>>40)/float64(1<<24) - 0.5
		}
		var q float64
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				q += v[i] * cov[i][j] * v[j]
			}
		}
		if q < -1e-6 {
			t.Errorf("covariance not PSD: quadratic form = %.6f", q)
		}
	}
}

// ---- StructuredEdgeDetectionLite -----------------------------------------

func TestStructuredEdgeDetectionResponds(t *testing.T) {
	rows, cols := 40, 40
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(30)
	// A bright square.
	for y := 12; y < 28; y++ {
		for x := 12; x < 28; x++ {
			img.Data[y*cols+x] = 220
		}
	}
	e := ximgproc.StructuredEdgeDetectionLite(img)
	if e.Rows != rows || e.Cols != cols {
		t.Fatal("wrong shape")
	}
	for i, v := range e.Data {
		if v < 0 || v > 1.0001 {
			t.Fatalf("edge value out of [0,1] at %d: %.3f", i, v)
		}
	}
	// Response on a boundary pixel should exceed the flat interior.
	boundary := e.At(12, 20)
	interior := e.At(20, 20)
	if boundary <= interior {
		t.Errorf("boundary response %.3f not greater than interior %.3f", boundary, interior)
	}
}

// ---- EdgeBoxes -----------------------------------------------------------

func TestEdgeBoxesBoundsAndOrdering(t *testing.T) {
	rows, cols := 60, 60
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(20)
	// A clear rectangle object.
	rx, ry, rw, rh := 18, 14, 26, 30
	for y := ry; y < ry+rh; y++ {
		for x := rx; x < rx+rw; x++ {
			img.Data[y*cols+x] = 200
		}
	}
	edges := ximgproc.StructuredEdgeDetectionLite(img)
	boxes := ximgproc.EdgeBoxes(edges, 20)
	if len(boxes) == 0 {
		t.Fatal("no boxes produced")
	}
	for i, b := range boxes {
		if b.X < 0 || b.Y < 0 || b.X+b.W > cols || b.Y+b.H > rows {
			t.Errorf("box %d out of bounds: %+v", i, b)
		}
		if i > 0 && boxes[i-1].Score < b.Score {
			t.Errorf("boxes not sorted by descending score at %d", i)
		}
	}
	// At least one proposal should overlap the object reasonably.
	truth := ximgproc.Box{X: rx, Y: ry, W: rw, H: rh}
	best := 0.0
	for _, b := range boxes {
		if o := boxIoU(b, truth); o > best {
			best = o
		}
	}
	if best < 0.3 {
		t.Errorf("no proposal overlaps the object (best IoU=%.2f)", best)
	}
}

func boxIoU(a, b ximgproc.Box) float64 {
	ax1, ay1 := a.X+a.W, a.Y+a.H
	bx1, by1 := b.X+b.W, b.Y+b.H
	ix0 := maxi(a.X, b.X)
	iy0 := maxi(a.Y, b.Y)
	ix1 := mini(ax1, bx1)
	iy1 := mini(ay1, by1)
	iw, ih := ix1-ix0, iy1-iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	return inter / (float64(a.W*a.H+b.W*b.H) - inter)
}

func maxi(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mini(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ---- SuperpixelLSC -------------------------------------------------------

func TestSuperpixelLSCConnectivity(t *testing.T) {
	rows, cols, region := 48, 48, 12
	img := blockColorImage(rows, cols, 12)
	labels, n := ximgproc.SuperpixelLSC(img, region, 0.075)
	if labels.Rows != rows || labels.Cols != cols || labels.Channels != 1 {
		t.Fatalf("labels wrong shape")
	}
	expected := (rows / region) * (cols / region)
	if n < expected/2 || n > expected*3 {
		t.Errorf("unexpected LSC superpixel count: got %d want ~%d", n, expected)
	}
	assertLabelsConnected(t, labels, n)
}

func TestSuperpixelLSCDeterministic(t *testing.T) {
	img := blockColorImage(32, 32, 8)
	l1, n1 := ximgproc.SuperpixelLSC(img, 8, 0.1)
	l2, n2 := ximgproc.SuperpixelLSC(img, 8, 0.1)
	if n1 != n2 {
		t.Fatalf("non-deterministic LSC count %d vs %d", n1, n2)
	}
	for i := range l1.Data {
		if l1.Data[i] != l2.Data[i] {
			t.Fatalf("non-deterministic LSC labels at %d", i)
		}
	}
}

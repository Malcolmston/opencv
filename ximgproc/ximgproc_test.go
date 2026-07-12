package ximgproc_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/ximgproc"
)

// ---- helpers -------------------------------------------------------------

// deterministic pseudo-random noise: a simple LCG so tests never depend on
// math/rand seeding behaviour.
type lcg struct{ s uint64 }

func (g *lcg) next() float64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return float64(g.s>>40) / float64(1<<24) // in [0,1)
}

// stepImage builds a gray image: left half value lo, right half value hi, with
// additive symmetric noise of the given amplitude. Deterministic.
func stepImage(rows, cols int, lo, hi uint8, split int, noise float64, seed uint64) *cv.Mat {
	g := &lcg{s: seed}
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := float64(lo)
			if x >= split {
				base = float64(hi)
			}
			v := base + (g.next()*2-1)*noise
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			m.Data[y*cols+x] = uint8(v)
		}
	}
	return m
}

// variance of a rectangular region of a single-channel Mat.
func regionVariance(m *cv.Mat, y0, x0, h, w int) float64 {
	var sum, sumSq float64
	n := float64(h * w)
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			v := float64(m.Data[y*m.Cols+x])
			sum += v
			sumSq += v * v
		}
	}
	mean := sum / n
	return sumSq/n - mean*mean
}

// count2x2Blocks returns the number of 2×2 squares that are entirely
// foreground. A one-pixel-wide skeleton contains none: any such block means the
// stroke is at least two pixels thick somewhere, so this is the precise test
// that a skeleton's maximum perpendicular run width is 1.
func count2x2Blocks(m *cv.Mat) int {
	n := 0
	for y := 0; y < m.Rows-1; y++ {
		for x := 0; x < m.Cols-1; x++ {
			if m.Data[y*m.Cols+x] != 0 &&
				m.Data[y*m.Cols+x+1] != 0 &&
				m.Data[(y+1)*m.Cols+x] != 0 &&
				m.Data[(y+1)*m.Cols+x+1] != 0 {
				n++
			}
		}
	}
	return n
}

func countForeground(m *cv.Mat) int {
	n := 0
	for _, v := range m.Data {
		if v != 0 {
			n++
		}
	}
	return n
}

// ---- GuidedFilter --------------------------------------------------------

func TestGuidedFilterSmoothsFlatPreservesEdge(t *testing.T) {
	rows, cols := 40, 60
	split := 30
	src := stepImage(rows, cols, 60, 200, split, 20, 12345)

	// Self-guided filter (guide == src), the usual edge-preserving mode.
	out := ximgproc.GuidedFilter(src, src, 4, 400)

	// Variance in the left flat region must drop substantially.
	before := regionVariance(src, 5, 2, 30, 20)
	after := regionVariance(out, 5, 2, 30, 20)
	if after >= before {
		t.Fatalf("flat-region variance did not drop: before=%.2f after=%.2f", before, after)
	}
	if after > before*0.6 {
		t.Errorf("expected a large variance reduction, before=%.2f after=%.2f", before, after)
	}

	// The step edge must be preserved: mean difference across the boundary
	// stays close to the original ~140 gap.
	leftMean := regionMean(out, 15, 5, 10, 15)
	rightMean := regionMean(out, 15, 40, 10, 15)
	gap := rightMean - leftMean
	if gap < 110 {
		t.Errorf("edge contrast collapsed, gap=%.2f (want >= 110)", gap)
	}
}

func regionMean(m *cv.Mat, y0, x0, h, w int) float64 {
	var sum float64
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			sum += float64(m.Data[y*m.Cols+x])
		}
	}
	return sum / float64(h*w)
}

func TestGuidedFilterDeterministic(t *testing.T) {
	src := stepImage(20, 20, 50, 180, 10, 15, 99)
	a := ximgproc.GuidedFilter(src, src, 3, 200)
	b := ximgproc.GuidedFilter(src, src, 3, 200)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("non-deterministic output at %d: %d vs %d", i, a.Data[i], b.Data[i])
		}
	}
}

// ---- Thinning ------------------------------------------------------------

func TestThinningRectangleYieldsThinSkeleton(t *testing.T) {
	rows, cols := 30, 40
	m := cv.NewMat(rows, cols, 1)
	// A thick filled rectangle.
	for y := 8; y < 22; y++ {
		for x := 6; x < 34; x++ {
			m.Data[y*cols+x] = 255
		}
	}
	sk := ximgproc.Thinning(m)

	if got := count2x2Blocks(sk); got > 0 {
		t.Errorf("skeleton not 1px thin: %d solid 2x2 blocks remain", got)
	}
	if countForeground(sk) == 0 {
		t.Fatal("skeleton is empty")
	}
	if countForeground(sk) >= countForeground(m) {
		t.Error("skeleton should have far fewer pixels than the filled shape")
	}
}

func TestThinningThickLine(t *testing.T) {
	rows, cols := 20, 40
	m := cv.NewMat(rows, cols, 1)
	for y := 8; y < 13; y++ { // 5px thick horizontal bar
		for x := 4; x < 36; x++ {
			m.Data[y*cols+x] = 255
		}
	}
	sk := ximgproc.Thinning(m)
	if got := count2x2Blocks(sk); got > 0 {
		t.Errorf("thick line skeleton not thin: %d solid 2x2 blocks remain", got)
	}
}

// ---- NiBlackThreshold ----------------------------------------------------

// bimodalGradientImage builds a bimodal image (foreground blobs vs background)
// with a strong left-to-right illumination gradient added, so no single global
// threshold separates the classes. Returns the image and the ground-truth mask.
func bimodalGradientImage(rows, cols int) (*cv.Mat, *cv.Mat) {
	img := cv.NewMat(rows, cols, 1)
	truth := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Illumination ramps from ~40 on the left to ~200 on the right.
			illum := 40.0 + 160.0*float64(x)/float64(cols-1)
			fg := false
			// Foreground: thin dark strokes (2px wide every 8px), small
			// relative to the analysis window so the local mean still tracks
			// the background — the classic Niblack/Sauvola scenario.
			if x%8 < 2 {
				fg = true
			}
			v := illum
			if fg {
				v -= 70 // strokes are clearly darker than local background
			}
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			img.Data[y*cols+x] = uint8(v)
			if fg {
				truth.Data[y*cols+x] = 255
			}
		}
	}
	return img, truth
}

// agreement returns the fraction of pixels where a binary result matches truth,
// treating "foreground = darker" (so result 0 means foreground here).
func agreementDarkFg(res, truth *cv.Mat) float64 {
	match := 0
	for i := range res.Data {
		resFg := res.Data[i] == 0 // below-threshold => foreground
		truthFg := truth.Data[i] != 0
		if resFg == truthFg {
			match++
		}
	}
	return float64(match) / float64(len(res.Data))
}

func TestNiBlackBeatsGlobalOnGradient(t *testing.T) {
	img, truth := bimodalGradientImage(48, 48)

	// Global Otsu threshold from the root package.
	global, _ := cv.Threshold(img, 0, 255, cv.ThreshBinary|cv.ThreshOtsu)
	globalAcc := agreementDarkFg(global, truth)

	// Local Sauvola thresholding.
	local := ximgproc.NiBlackThreshold(img, 0.2, 15, int(ximgproc.NiBlackSauvola))
	localAcc := agreementDarkFg(local, truth)

	if localAcc <= globalAcc {
		t.Errorf("local threshold (%.3f) did not beat global (%.3f)", localAcc, globalAcc)
	}
	if localAcc < 0.85 {
		t.Errorf("local threshold accuracy too low: %.3f", localAcc)
	}
}

func TestNiBlackVariantsRun(t *testing.T) {
	img, _ := bimodalGradientImage(24, 24)
	for _, v := range []ximgproc.NiBlackVariant{
		ximgproc.NiBlackNiblack, ximgproc.NiBlackSauvola,
		ximgproc.NiBlackWolf, ximgproc.NiBlackNick,
	} {
		out := ximgproc.NiBlackThreshold(img, -0.2, 11, int(v))
		if out.Rows != 24 || out.Cols != 24 || out.Channels != 1 {
			t.Fatalf("variant %d wrong shape", v)
		}
	}
}

// ---- AnisotropicDiffusion ------------------------------------------------

func TestAnisotropicDiffusionSmoothsButKeepsEdge(t *testing.T) {
	rows, cols := 30, 40
	split := 20
	src := stepImage(rows, cols, 70, 190, split, 18, 555)

	out := ximgproc.AnisotropicDiffusion(src, 0.2, 15, 20)

	before := regionVariance(src, 5, 2, 20, 15)
	after := regionVariance(out, 5, 2, 20, 15)
	if after >= before {
		t.Errorf("diffusion did not reduce flat-region variance: before=%.2f after=%.2f", before, after)
	}
	// Edge preserved.
	leftMean := regionMean(out, 10, 3, 10, 12)
	rightMean := regionMean(out, 10, 25, 10, 12)
	if rightMean-leftMean < 90 {
		t.Errorf("edge not preserved, gap=%.2f", rightMean-leftMean)
	}
}

func TestAnisotropicDiffusionZeroIters(t *testing.T) {
	src := stepImage(10, 10, 20, 200, 5, 0, 1)
	out := ximgproc.AnisotropicDiffusion(src, 0.2, 20, 0)
	for i := range src.Data {
		if out.Data[i] != src.Data[i] {
			t.Fatal("zero iterations must return a copy of the input")
		}
	}
}

// ---- SuperpixelSLIC ------------------------------------------------------

// blockColorImage builds a 3-channel image made of solid colour blocks so that
// superpixels have clear structure.
func blockColorImage(rows, cols, block int) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			bx := x / block
			by := y / block
			i := (y*cols + x) * 3
			m.Data[i+0] = uint8((bx * 53) % 256)
			m.Data[i+1] = uint8((by * 91) % 256)
			m.Data[i+2] = uint8(((bx + by) * 37) % 256)
		}
	}
	return m
}

func TestSuperpixelSLICLabelCountAndConnectivity(t *testing.T) {
	rows, cols := 48, 48
	region := 12
	img := blockColorImage(rows, cols, 12)

	labels, n := ximgproc.SuperpixelSLIC(img, region, 20)
	if labels.Rows != rows || labels.Cols != cols || labels.Channels != 1 {
		t.Fatalf("labels wrong shape: %dx%dx%d", labels.Rows, labels.Cols, labels.Channels)
	}

	// Expected ~ (rows/region)*(cols/region) = 16 superpixels; allow a wide band.
	expected := (rows / region) * (cols / region)
	if n < expected/2 || n > expected*3 {
		t.Errorf("unexpected superpixel count: got %d, want ~%d", n, expected)
	}

	// Every label must be a single 4-connected component.
	assertLabelsConnected(t, labels, n)
}

func TestSuperpixelSLICDeterministic(t *testing.T) {
	img := blockColorImage(32, 32, 8)
	l1, n1 := ximgproc.SuperpixelSLIC(img, 8, 15)
	l2, n2 := ximgproc.SuperpixelSLIC(img, 8, 15)
	if n1 != n2 {
		t.Fatalf("non-deterministic count: %d vs %d", n1, n2)
	}
	for i := range l1.Data {
		if l1.Data[i] != l2.Data[i] {
			t.Fatalf("non-deterministic labels at %d", i)
		}
	}
}

// assertLabelsConnected verifies each label value present forms exactly one
// 4-connected region.
func assertLabelsConnected(t *testing.T, labels *cv.Mat, n int) {
	t.Helper()
	rows, cols := labels.Rows, labels.Cols
	seen := make([]bool, rows*cols)
	// components counts how many separate 4-connected regions share each label.
	components := make(map[uint8]int)
	dx := []int{-1, 1, 0, 0}
	dy := []int{0, 0, -1, 1}
	for start := 0; start < rows*cols; start++ {
		if seen[start] {
			continue
		}
		lab := labels.Data[start]
		components[lab]++
		queue := []int{start}
		seen[start] = true
		for h := 0; h < len(queue); h++ {
			p := queue[h]
			py, px := p/cols, p%cols
			for d := 0; d < 4; d++ {
				ny, nx := py+dy[d], px+dx[d]
				if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
					continue
				}
				q := ny*cols + nx
				if !seen[q] && labels.Data[q] == lab {
					seen[q] = true
					queue = append(queue, q)
				}
			}
		}
	}
	for lab, c := range components {
		if c != 1 {
			t.Errorf("label %d is split into %d disconnected components", lab, c)
		}
	}
}

// ---- PeiLinNormalization -------------------------------------------------

func TestPeiLinMapsCentroidToCenter(t *testing.T) {
	rows, cols := 40, 40
	img := cv.NewMat(rows, cols, 1)
	// An off-centre bright blob.
	for y := 5; y < 15; y++ {
		for x := 8; x < 24; x++ {
			img.Data[y*cols+x] = 255
		}
	}
	// Compute the intensity centroid.
	var m00, m10, m01 float64
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			w := float64(img.Data[y*cols+x])
			m00 += w
			m10 += w * float64(x)
			m01 += w * float64(y)
		}
	}
	xc, yc := m10/m00, m01/m00

	tf := ximgproc.PeiLinNormalization(img)
	gx, gy := tf.Apply(xc, yc)
	if math.Abs(gx-float64(cols)/2) > 1e-6 || math.Abs(gy-float64(rows)/2) > 1e-6 {
		t.Errorf("centroid did not map to image centre: got (%.4f,%.4f) want (%.1f,%.1f)", gx, gy, float64(cols)/2, float64(rows)/2)
	}
	// All entries finite.
	for r := 0; r < 2; r++ {
		for c := 0; c < 3; c++ {
			if math.IsNaN(tf[r][c]) || math.IsInf(tf[r][c], 0) {
				t.Fatalf("non-finite transform entry [%d][%d]", r, c)
			}
		}
	}
}

// ---- FastLineDetector ----------------------------------------------------

func TestFastLineDetectorFindsHorizontalLine(t *testing.T) {
	rows, cols := 40, 80
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(0)
	// A bright horizontal line, 1px, across most of the width.
	y0 := 20
	for x := 5; x < 74; x++ {
		img.Data[y0*cols+x] = 255
	}
	segs := ximgproc.FastLineDetector(img)
	if len(segs) == 0 {
		t.Fatal("no line segments detected")
	}
	// The longest segment should be near-horizontal and reasonably long.
	best := segs[0]
	if best.Length() < 40 {
		t.Errorf("detected segment too short: %.1f", best.Length())
	}
	if math.Abs(best.Y1-best.Y2) > 3 {
		t.Errorf("segment not horizontal: y1=%.1f y2=%.1f", best.Y1, best.Y2)
	}
}

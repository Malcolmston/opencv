package features2d

import (
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// buildScene renders a deterministic corner-rich grayscale scene of the given
// size: a black background scattered with filled squares of varying size and
// intensity. The varied spatial arrangement makes local neighbourhoods (and
// hence descriptors) distinctive rather than repetitive.
func buildScene(size int) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	rng := rand.New(rand.NewSource(12345))
	for i := 0; i < 40; i++ {
		s := 5 + rng.Intn(9)
		x := rng.Intn(size - s)
		y := rng.Intn(size - s)
		val := float64(120 + rng.Intn(136)) // 120..255
		cv.Rectangle(m, cv.Point{X: x, Y: y}, cv.Point{X: x + s, Y: y + s}, cv.NewScalar(val), cv.Filled)
	}
	return m
}

// translatedPair crops two overlapping windows from a shared larger scene so
// that img2 is img1 shifted by (dx, dy): a scene point at img1 pixel (x, y)
// appears at img2 pixel (x-dx, y-dy). The true keypoint displacement kp2-kp1 is
// therefore (-dx, -dy).
func translatedPair(dx, dy int) (img1, img2 *cv.Mat) {
	const big = 160
	const win = 110
	const ox, oy = 20, 20
	scene := buildScene(big)
	img1 = scene.Region(oy, ox, win, win)
	img2 = scene.Region(oy+dy, ox+dx, win, win)
	return img1, img2
}

func TestORBDetectsKeypoints(t *testing.T) {
	img := buildScene(110)
	kps, desc := NewORB(300).DetectAndCompute(img)
	if len(kps) < 20 {
		t.Fatalf("expected many keypoints, got %d", len(kps))
	}
	if len(desc) != len(kps) {
		t.Fatalf("descriptor count %d != keypoint count %d", len(desc), len(kps))
	}
	if len(desc[0]) != defaultNumBits/8 {
		t.Fatalf("expected %d-byte descriptors, got %d", defaultNumBits/8, len(desc[0]))
	}
	for _, kp := range kps {
		if kp.Angle < 0 || kp.Angle >= 360 {
			t.Fatalf("keypoint angle out of range: %v", kp.Angle)
		}
	}
}

func TestORBDeterministic(t *testing.T) {
	img := buildScene(110)
	orb := NewORB(300)
	kA, dA := orb.DetectAndCompute(img)
	kB, dB := orb.DetectAndCompute(img)
	if len(kA) != len(kB) {
		t.Fatalf("non-deterministic keypoint count: %d vs %d", len(kA), len(kB))
	}
	for i := range kA {
		if kA[i] != kB[i] {
			t.Fatalf("keypoint %d differs: %+v vs %+v", i, kA[i], kB[i])
		}
		for j := range dA[i] {
			if dA[i][j] != dB[i][j] {
				t.Fatalf("descriptor %d byte %d differs", i, j)
			}
		}
	}
}

func TestORBMatchTranslated(t *testing.T) {
	dx, dy := 6, 5
	img1, img2 := translatedPair(dx, dy)
	orb := NewORB(300)
	kp1, d1 := orb.DetectAndCompute(img1)
	kp2, d2 := orb.DetectAndCompute(img2)
	if len(kp1) < 20 || len(kp2) < 20 {
		t.Fatalf("too few keypoints: %d and %d", len(kp1), len(kp2))
	}

	matcher := &BFMatcher{Norm: NormHamming, CrossCheck: true}
	matches := matcher.Match(NewBinaryDescriptors(d1), NewBinaryDescriptors(d2))
	if len(matches) < 10 {
		t.Fatalf("too few matches: %d", len(matches))
	}

	// Expected displacement kp2 - kp1 is (-dx, -dy).
	consistent := 0
	for _, m := range matches {
		ddx := kp2[m.TrainIdx].Pt.X - kp1[m.QueryIdx].Pt.X
		ddy := kp2[m.TrainIdx].Pt.Y - kp1[m.QueryIdx].Pt.Y
		if abs(ddx-(-dx)) <= 1 && abs(ddy-(-dy)) <= 1 {
			consistent++
		}
	}
	if consistent*2 <= len(matches) {
		t.Fatalf("expected a majority of geometrically consistent matches, got %d of %d", consistent, len(matches))
	}
	t.Logf("matches=%d consistent=%d kp1=%d kp2=%d", len(matches), consistent, len(kp1), len(kp2))
}

func TestRatioTestFilters(t *testing.T) {
	dx, dy := 4, 3
	img1, img2 := translatedPair(dx, dy)
	orb := NewORB(300)
	kp1, d1 := orb.DetectAndCompute(img1)
	kp2, d2 := orb.DetectAndCompute(img2)

	matcher := NewBFMatcher(NormHamming)
	knn := matcher.KnnMatch(NewBinaryDescriptors(d1), NewBinaryDescriptors(d2), 2)
	good := RatioTest(knn, 0.75)

	// The ratio test must discard some ambiguous matches...
	if len(good) >= len(knn) {
		t.Fatalf("ratio test kept %d of %d; expected it to filter", len(good), len(knn))
	}
	// ...while keeping mostly geometrically consistent ones.
	consistent := 0
	for _, m := range good {
		ddx := kp2[m.TrainIdx].Pt.X - kp1[m.QueryIdx].Pt.X
		ddy := kp2[m.TrainIdx].Pt.Y - kp1[m.QueryIdx].Pt.Y
		if abs(ddx-(-dx)) <= 1 && abs(ddy-(-dy)) <= 1 {
			consistent++
		}
	}
	if len(good) == 0 || consistent*2 <= len(good) {
		t.Fatalf("ratio-test survivors not majority consistent: %d of %d", consistent, len(good))
	}
	t.Logf("knn=%d good=%d consistent=%d", len(knn), len(good), consistent)
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

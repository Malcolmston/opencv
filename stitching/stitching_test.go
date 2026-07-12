package stitching

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// syntheticTexture builds a deterministic single-channel image rich in unique
// corners: a smooth diagonal gradient background stamped with many small
// random-intensity rectangles. The random stream is seeded so the image (and
// therefore every test that uses it) is fully reproducible.
func syntheticTexture(rows, cols int, seed int64) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			g := 40 + (x*80)/cols + (y*40)/rows
			m.Data[y*cols+x] = uint8(g)
		}
	}
	rng := rand.New(rand.NewSource(seed))
	for i := 0; i < 160; i++ {
		rx := rng.Intn(cols - 8)
		ry := rng.Intn(rows - 8)
		rw := 4 + rng.Intn(6)
		rh := 4 + rng.Intn(6)
		val := uint8(rng.Intn(256))
		for y := ry; y < ry+rh && y < rows; y++ {
			for x := rx; x < rx+rw && x < cols; x++ {
				m.Data[y*cols+x] = val
			}
		}
	}
	return m
}

// horizontalCrops returns two overlapping crops of base related by a known
// horizontal translation: cropB is shifted shift pixels to the right of cropA,
// so a point at column xb in cropB corresponds to column xb+shift in cropA. The
// homography mapping cropB into cropA is therefore translation(+shift, 0).
func horizontalCrops(base *cv.Mat, wA, shift, wB int) (a, b *cv.Mat) {
	rows := base.Rows
	a = base.Region(0, 0, rows, wA)
	b = base.Region(0, shift, rows, wB)
	return a, b
}

func TestEstimateTransformTranslation(t *testing.T) {
	base := syntheticTexture(90, 170, 12345)
	shift := 70
	a, b := horizontalCrops(base, 110, shift, 100)

	s := NewStitcher()
	h, err := s.EstimateTransform(a, b)
	if err != nil {
		t.Fatalf("EstimateTransform: %v", err)
	}
	// Expect translation(+shift, 0): h2≈shift, diagonal≈1, off-diagonal≈0.
	checks := []struct {
		name string
		got  float64
		want float64
		tol  float64
	}{
		{"h0", h[0], 1, 0.03},
		{"h1", h[1], 0, 0.03},
		{"h2", h[2], float64(shift), 1.0},
		{"h3", h[3], 0, 0.03},
		{"h4", h[4], 1, 0.03},
		{"h5", h[5], 0, 1.0},
		{"h6", h[6], 0, 1e-3},
		{"h7", h[7], 0, 1e-3},
	}
	for _, c := range checks {
		if math.Abs(c.got-c.want) > c.tol {
			t.Errorf("%s = %.5f, want %.5f ± %.3f", c.name, c.got, c.want, c.tol)
		}
	}
}

func TestStitchReconstruction(t *testing.T) {
	base := syntheticTexture(90, 170, 999)
	shift := 70
	a, b := horizontalCrops(base, 110, shift, 100)

	s := NewStitcher()
	pano, err := s.Stitch([]*cv.Mat{a, b})
	if err != nil {
		t.Fatalf("Stitch: %v", err)
	}
	// Canvas size ≈ union bounds: width ≈ base width, height = base height.
	if pano.Rows != base.Rows {
		t.Errorf("panorama rows = %d, want %d", pano.Rows, base.Rows)
	}
	if math.Abs(float64(pano.Cols-base.Cols)) > 3 {
		t.Errorf("panorama cols = %d, want ≈ %d", pano.Cols, base.Cols)
	}
	// Interior should reconstruct the base image (both inputs equal it there).
	var sum, count float64
	for y := 5; y < base.Rows-5; y++ {
		for x := 5; x < base.Cols-5 && x < pano.Cols-5; x++ {
			d := math.Abs(float64(pano.Data[y*pano.Cols+x]) - float64(base.Data[y*base.Cols+x]))
			sum += d
			count++
		}
	}
	if mean := sum / count; mean > 4 {
		t.Errorf("mean abs reconstruction error = %.3f, want <= 4", mean)
	}
}

func TestFeatherSeamless(t *testing.T) {
	base := syntheticTexture(90, 170, 555)
	shift := 70
	a, b := horizontalCrops(base, 110, shift, 100)
	// Simulate an exposure difference: brighten cropB by a constant. The
	// mean-subtracted descriptors are unaffected, so matching still works, but a
	// naive hard seam would show a step of ~offset at the overlap boundary.
	const offset = 30
	bBright := b.Clone()
	for i := range bBright.Data {
		v := int(bBright.Data[i]) + offset
		if v > 255 {
			v = 255
		}
		bBright.Data[i] = uint8(v)
	}

	s := NewStitcher()
	pano, err := s.Stitch([]*cv.Mat{a, bBright})
	if err != nil {
		t.Fatalf("Stitch: %v", err)
	}
	// Along a mid-height scan line across the overlap [shift, wA], the feather
	// blend must transition smoothly: no single adjacent-pixel jump should be
	// anywhere near the exposure step.
	y := base.Rows / 2
	var maxLocalStep float64
	for x := shift + 2; x < 110-2; x++ {
		d := math.Abs(float64(pano.Data[y*pano.Cols+x]) - float64(pano.Data[y*pano.Cols+x-1]))
		// Ignore texture edges from the base content by comparing to the base's
		// own local step; only the blend-induced excess matters.
		baseStep := math.Abs(float64(base.Data[y*base.Cols+x]) - float64(base.Data[y*base.Cols+x-1]))
		excess := d - baseStep
		if excess > maxLocalStep {
			maxLocalStep = excess
		}
	}
	if maxLocalStep > 8 {
		t.Errorf("feather seam not smooth: max blend-induced local step = %.2f, want <= 8", maxLocalStep)
	}
}

func TestComposePanoramaKnownTransforms(t *testing.T) {
	base := syntheticTexture(60, 120, 7)
	shift := 50
	a, b := horizontalCrops(base, 80, shift, 70)

	s := NewStitcher()
	transforms := []cv.PerspectiveMatrix{
		identityH(),
		translationH(float64(shift), 0),
	}
	pano, err := s.ComposePanorama([]*cv.Mat{a, b}, transforms)
	if err != nil {
		t.Fatalf("ComposePanorama: %v", err)
	}
	// Union: A covers [0,80], B covers [50,120] => width 120, height 60.
	if pano.Rows != 60 || pano.Cols != 120 {
		t.Fatalf("canvas = %dx%d, want 120x60", pano.Cols, pano.Rows)
	}
	var sum, count float64
	for y := 2; y < 58; y++ {
		for x := 2; x < 118; x++ {
			d := math.Abs(float64(pano.Data[y*pano.Cols+x]) - float64(base.Data[y*base.Cols+x]))
			sum += d
			count++
		}
	}
	if mean := sum / count; mean > 3 {
		t.Errorf("mean abs error = %.3f, want <= 3", mean)
	}
}

func TestMultiBandBlend(t *testing.T) {
	base := syntheticTexture(90, 170, 4242)
	shift := 70
	a, b := horizontalCrops(base, 110, shift, 100)

	s := NewStitcher()
	s.Blender = MultiBandBlend{Bands: 4}
	pano, err := s.Stitch([]*cv.Mat{a, b})
	if err != nil {
		t.Fatalf("Stitch (multiband): %v", err)
	}
	if pano.Rows != base.Rows {
		t.Errorf("rows = %d, want %d", pano.Rows, base.Rows)
	}
	if math.Abs(float64(pano.Cols-base.Cols)) > 3 {
		t.Errorf("cols = %d, want ≈ %d", pano.Cols, base.Cols)
	}
	// Interior reconstruction (looser tolerance than feather: pyramid smoothing
	// slightly softens content).
	var sum, count float64
	for y := 10; y < base.Rows-10; y++ {
		for x := 10; x < base.Cols-10 && x < pano.Cols-10; x++ {
			d := math.Abs(float64(pano.Data[y*pano.Cols+x]) - float64(base.Data[y*base.Cols+x]))
			sum += d
			count++
		}
	}
	if mean := sum / count; mean > 12 {
		t.Errorf("multiband mean abs error = %.3f, want <= 12", mean)
	}
}

func TestColorStitch(t *testing.T) {
	gray := syntheticTexture(70, 140, 314)
	// Promote to a 3-channel image so the colour path (gray conversion for
	// detection, 3-channel warp and blend) is exercised.
	base := cv.CvtColor(gray, cv.ColorGray2RGB)
	shift := 55
	a := base.Region(0, 0, base.Rows, 90)
	b := base.Region(0, shift, base.Rows, 85)

	s := NewStitcher()
	pano, err := s.Stitch([]*cv.Mat{a, b})
	if err != nil {
		t.Fatalf("colour Stitch: %v", err)
	}
	if pano.Channels != 3 {
		t.Errorf("channels = %d, want 3", pano.Channels)
	}
	if math.Abs(float64(pano.Cols-base.Cols)) > 3 {
		t.Errorf("cols = %d, want ≈ %d", pano.Cols, base.Cols)
	}
}

func TestStitchErrors(t *testing.T) {
	s := NewStitcher()
	if _, err := s.Stitch(nil); err != ErrNoImages {
		t.Errorf("Stitch(nil) err = %v, want ErrNoImages", err)
	}
	one := syntheticTexture(20, 20, 1)
	out, err := s.Stitch([]*cv.Mat{one})
	if err != nil {
		t.Fatalf("single image: %v", err)
	}
	if out.Cols != 20 || out.Rows != 20 {
		t.Errorf("single-image passthrough size = %dx%d, want 20x20", out.Cols, out.Rows)
	}
	// Channel mismatch.
	g := syntheticTexture(20, 20, 2)
	rgb := cv.CvtColor(syntheticTexture(20, 20, 3), cv.ColorGray2RGB)
	if _, err := s.Stitch([]*cv.Mat{g, rgb}); err != ErrChannelMismatch {
		t.Errorf("mismatch err = %v, want ErrChannelMismatch", err)
	}
}

func TestHomographyDLTExact(t *testing.T) {
	// A known projective transform recovered exactly from clean correspondences.
	want := cv.PerspectiveMatrix{1.1, 0.05, 12, -0.03, 0.98, -7, 0.0005, -0.0003, 1}
	pts := []pointF{{10, 12}, {90, 20}, {80, 95}, {15, 88}, {50, 50}, {30, 70}}
	src := pts
	dst := make([]pointF, len(pts))
	for i, p := range pts {
		x, y, _ := applyH(want, p.x, p.y)
		dst[i] = pointF{x, y}
	}
	got, ok := computeHomographyDLT(src, dst)
	if !ok {
		t.Fatal("computeHomographyDLT failed")
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-6 {
			t.Errorf("h[%d] = %.8f, want %.8f", i, got[i], want[i])
		}
	}
}

func TestMatMulAndApply(t *testing.T) {
	a := translationH(5, -3)
	b := translationH(2, 7)
	c := matMul3(a, b)
	x, y, ok := applyH(c, 10, 10)
	if !ok || math.Abs(x-17) > 1e-9 || math.Abs(y-14) > 1e-9 {
		t.Errorf("composed translation = (%.3f, %.3f), want (17, 14)", x, y)
	}
}

package features2d

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestBRIEFComputeShape(t *testing.T) {
	img := buildScene(60)
	kps := []KeyPoint{
		{Pt: cv.Point{X: 30, Y: 30}, Size: 31, Angle: -1},
		{Pt: cv.Point{X: 20, Y: 25}, Size: 31, Angle: -1},
	}
	outKps, desc := (&BRIEF{}).Compute(img, kps)
	if len(outKps) != len(kps) || len(desc) != len(kps) {
		t.Fatalf("expected %d parallel rows, got kps=%d desc=%d", len(kps), len(outKps), len(desc))
	}
	for i, d := range desc {
		if len(d) != defaultNumBits/8 {
			t.Fatalf("descriptor %d has %d bytes, want %d", i, len(d), defaultNumBits/8)
		}
	}
}

// TestBRIEFTranslationInvariance verifies that the same local neighbourhood,
// shifted, yields an identical unoriented descriptor.
func TestBRIEFTranslationInvariance(t *testing.T) {
	const big = 120
	scene := buildScene(big)
	dx, dy := 7, 5
	img1 := scene.Region(20, 20, 80, 80)
	img2 := scene.Region(20+dy, 20+dx, 80, 80)

	kp1 := []KeyPoint{{Pt: cv.Point{X: 40, Y: 40}, Angle: -1}}
	kp2 := []KeyPoint{{Pt: cv.Point{X: 40 - dx, Y: 40 - dy}, Angle: -1}}

	b := &BRIEF{}
	_, d1 := b.Compute(img1, kp1)
	_, d2 := b.Compute(img2, kp2)
	if HammingDistance(d1[0], d2[0]) != 0 {
		t.Fatalf("translated BRIEF descriptors differ by %d bits, want 0", HammingDistance(d1[0], d2[0]))
	}
}

func TestDrawKeypoints(t *testing.T) {
	img := buildScene(50)
	kps := []KeyPoint{{Pt: cv.Point{X: 25, Y: 25}, Size: 10, Angle: 45}}
	out := DrawKeypoints(img, kps)
	if out.Channels != 3 {
		t.Fatalf("DrawKeypoints output has %d channels, want 3", out.Channels)
	}
	if out.Rows != img.Rows || out.Cols != img.Cols {
		t.Fatalf("DrawKeypoints changed size: %dx%d", out.Rows, out.Cols)
	}
	// The original single-channel image must be untouched.
	if img.Channels != 1 {
		t.Fatalf("input mutated to %d channels", img.Channels)
	}
}

func TestDrawMatches(t *testing.T) {
	img1 := buildScene(50)
	img2 := buildScene(40)
	kp1 := []KeyPoint{{Pt: cv.Point{X: 10, Y: 10}, Size: 8}}
	kp2 := []KeyPoint{{Pt: cv.Point{X: 12, Y: 11}, Size: 8}}
	matches := []DMatch{{QueryIdx: 0, TrainIdx: 0, Distance: 3}}
	out := DrawMatches(img1, kp1, img2, kp2, matches)
	if out.Cols != img1.Cols+img2.Cols {
		t.Fatalf("canvas width %d, want %d", out.Cols, img1.Cols+img2.Cols)
	}
	if out.Rows != 50 { // max(50, 40)
		t.Fatalf("canvas height %d, want 50", out.Rows)
	}
	if out.Channels != 3 {
		t.Fatalf("canvas channels %d, want 3", out.Channels)
	}
}

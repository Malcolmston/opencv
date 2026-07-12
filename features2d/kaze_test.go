package features2d

import "testing"

func TestKAZEDetectsAndDescribes(t *testing.T) {
	img := buildScene(120)
	kps, desc := NewKAZE(0).DetectAndCompute(img)
	if len(kps) < 8 {
		t.Fatalf("expected KAZE keypoints, got %d", len(kps))
	}
	if len(desc) != len(kps) {
		t.Fatalf("desc/kp mismatch: %d vs %d", len(desc), len(kps))
	}
	if len(desc[0]) != 64 {
		t.Fatalf("expected 64-d KAZE descriptors, got %d", len(desc[0]))
	}
	for _, kp := range kps {
		if kp.Angle < 0 || kp.Angle >= 360 {
			t.Fatalf("angle out of range: %v", kp.Angle)
		}
	}
}

func TestAKAZEBinaryDescriptors(t *testing.T) {
	img := buildScene(120)
	kps, desc := NewAKAZE(0).DetectAndCompute(img)
	if len(kps) < 8 {
		t.Fatalf("expected AKAZE keypoints, got %d", len(kps))
	}
	if len(desc) != len(kps) {
		t.Fatalf("desc/kp mismatch: %d vs %d", len(desc), len(kps))
	}
	// All descriptor rows must have the same byte length so Hamming distance is
	// well-defined.
	for i := range desc {
		if len(desc[i]) != len(desc[0]) {
			t.Fatalf("descriptor %d length %d != %d", i, len(desc[i]), len(desc[0]))
		}
	}
	t.Logf("AKAZE descriptor bytes=%d (%d bits)", len(desc[0]), len(desc[0])*8)
}

func TestKAZEDeterministic(t *testing.T) {
	img := buildScene(100)
	k := NewKAZE(30)
	kA, dA := k.DetectAndCompute(img)
	kB, dB := k.DetectAndCompute(img)
	if len(kA) != len(kB) {
		t.Fatalf("non-deterministic count: %d vs %d", len(kA), len(kB))
	}
	for i := range kA {
		if kA[i] != kB[i] {
			t.Fatalf("keypoint %d differs", i)
		}
		for j := range dA[i] {
			if dA[i][j] != dB[i][j] {
				t.Fatalf("descriptor %d differs at %d", i, j)
			}
		}
	}
}

func TestAKAZEMatchTranslated(t *testing.T) {
	dx, dy := 5, 4
	img1, img2 := translatedPair(dx, dy)
	ak := NewAKAZE(0)
	kp1, d1 := ak.DetectAndCompute(img1)
	kp2, d2 := ak.DetectAndCompute(img2)
	if len(kp1) < 6 || len(kp2) < 6 {
		t.Fatalf("too few AKAZE keypoints: %d and %d", len(kp1), len(kp2))
	}
	matcher := &BFMatcher{Norm: NormHamming, CrossCheck: true}
	matches := matcher.Match(NewBinaryDescriptors(d1), NewBinaryDescriptors(d2))
	if len(matches) < 4 {
		t.Fatalf("too few AKAZE matches: %d", len(matches))
	}
	consistent := 0
	for _, m := range matches {
		ddx := kp2[m.TrainIdx].Pt.X - kp1[m.QueryIdx].Pt.X
		ddy := kp2[m.TrainIdx].Pt.Y - kp1[m.QueryIdx].Pt.Y
		if abs(ddx-(-dx)) <= 3 && abs(ddy-(-dy)) <= 3 {
			consistent++
		}
	}
	t.Logf("akaze matches=%d consistent=%d", len(matches), consistent)
	if consistent == 0 {
		t.Fatalf("no geometrically consistent AKAZE matches out of %d", len(matches))
	}
}

package features3

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// texturedImage returns a deterministic non-uniform image so that descriptors at
// different keypoints differ.
func texturedImage(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = uint8((x*7 + y*13 + x*y) % 256)
		}
	}
	return m
}

// horizontalRamp returns an image whose intensity increases with the column.
func horizontalRamp(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = uint8(x % 256)
		}
	}
	return m
}

func TestBRIEFPatternSizes(t *testing.T) {
	p := DefaultBRIEFPattern()
	if p.NumBits() != 256 {
		t.Fatalf("NumBits=%d want 256", p.NumBits())
	}
	if p.NumBytes() != 32 {
		t.Fatalf("NumBytes=%d want 32", p.NumBytes())
	}
	q := GenerateBRIEFPattern(31, 256, 7)
	// Deterministic: same seed yields the same pattern.
	q2 := GenerateBRIEFPattern(31, 256, 7)
	for i := range q.P1x {
		if q.P1x[i] != q2.P1x[i] || q.P2y[i] != q2.P2y[i] {
			t.Fatal("GenerateBRIEFPattern not deterministic for a fixed seed")
		}
	}
}

func TestBRIEFDescriptorIsDeterministicAndMatches(t *testing.T) {
	img := texturedImage(60, 60)
	kps := []KeyPoint{
		NewKeyPoint(20, 20, 1),
		NewKeyPoint(30, 25, 1),
		NewKeyPoint(40, 35, 1),
		NewKeyPoint(25, 40, 1),
	}
	pat := DefaultBRIEFPattern()
	d1 := ComputeBRIEF(img, kps, pat)
	d2 := ComputeBRIEF(img, kps, pat)
	for i := range d1 {
		if HammingDistance(d1[i], d2[i]) != 0 {
			t.Fatalf("BRIEF not deterministic at %d", i)
		}
	}
	matches := MatchBinaryDescriptors(d1, d2, -1)
	if len(matches) != len(kps) {
		t.Fatalf("got %d matches want %d", len(matches), len(kps))
	}
	for _, m := range matches {
		if m.QueryIdx != m.TrainIdx || m.Distance != 0 {
			t.Fatalf("identity match failed: %+v", m)
		}
	}
}

func TestORBEqualsBRIEFAtZeroAngle(t *testing.T) {
	img := texturedImage(60, 60)
	kp := NewKeyPoint(30, 30, 1)
	kp.Angle = 0
	pat := DefaultBRIEFPattern()
	orb := ComputeORB(img, []KeyPoint{kp}, pat)
	brief := ComputeBRIEF(img, []KeyPoint{kp}, pat)
	if HammingDistance(orb[0], brief[0]) != 0 {
		t.Fatal("ORB at angle 0 should equal BRIEF")
	}
}

func TestIntensityCentroidOrientationRamp(t *testing.T) {
	img := horizontalRamp(40, 40)
	ang := IntensityCentroidOrientation(img, 20, 20, 8)
	// Brightness increases to the right, so the centroid points along +x (~0deg).
	if ang > 5 && ang < 355 {
		t.Fatalf("orientation %.2f not near 0 for a rightward ramp", ang)
	}
}

func TestORBRotationChangesDescriptor(t *testing.T) {
	img := texturedImage(60, 60)
	kp0 := NewKeyPoint(30, 30, 1)
	kp0.Angle = 0
	kp90 := NewKeyPoint(30, 30, 1)
	kp90.Angle = 90
	pat := DefaultBRIEFPattern()
	d0 := ComputeORB(img, []KeyPoint{kp0}, pat)
	d90 := ComputeORB(img, []KeyPoint{kp90}, pat)
	if HammingDistance(d0[0], d90[0]) == 0 {
		t.Fatal("steered descriptor should change with orientation")
	}
}

func TestMatchBinaryDescriptorsMaxDistance(t *testing.T) {
	a := [][]byte{{0x00}}
	b := [][]byte{{0xFF}}
	if m := MatchBinaryDescriptors(a, b, 4); len(m) != 0 {
		t.Fatalf("expected no match under maxDistance 4, got %+v", m)
	}
	if m := MatchBinaryDescriptors(a, b, -1); len(m) != 1 || m[0].Distance != 8 {
		t.Fatalf("expected forced match distance 8, got %+v", m)
	}
}

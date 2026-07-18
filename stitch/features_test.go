package stitch

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// textureImage builds a deterministic textured single-channel image.
func textureImage(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := (x*29 + y*47) ^ (x * y * 3)
			m.Data[y*cols+x] = uint8(v & 0xff)
		}
	}
	return m
}

func TestHarrisCornersSquare(t *testing.T) {
	img := cv.NewMat(20, 20, 1)
	for y := 6; y <= 13; y++ {
		for x := 6; x <= 13; x++ {
			img.Data[y*20+x] = 255
		}
	}
	corners := HarrisCorners(img, 10, 0.01, 2)
	if len(corners) < 4 {
		t.Fatalf("found %d corners, want >= 4", len(corners))
	}
	trueCorners := []PointF{{6, 6}, {13, 6}, {6, 13}, {13, 13}}
	hits := 0
	for _, tc := range trueCorners {
		for _, c := range corners {
			if c.Distance(tc) <= 2.0 {
				hits++
				break
			}
		}
	}
	if hits < 3 {
		t.Fatalf("only %d/4 square corners detected near truth: %v", hits, corners)
	}
}

func TestNormalizedCrossCorrelationIdentical(t *testing.T) {
	img := textureImage(20, 20)
	s, ok := NormalizedCrossCorrelation(img, img, 10, 10, 10, 10, 3)
	if !ok {
		t.Fatal("NCC returned not-ok on valid patch")
	}
	if s < 0.999 {
		t.Fatalf("self NCC = %g, want ≈1", s)
	}
}

func TestMatchCornersNCCShift(t *testing.T) {
	const dx, dy = 2, 1
	a := textureImage(30, 30)
	b := cv.NewMat(30, 30, 1)
	for y := 0; y < 30; y++ {
		for x := 0; x < 30; x++ {
			sy, sx := y-dy, x-dx
			if sy >= 0 && sx >= 0 {
				b.Data[y*30+x] = a.Data[sy*30+sx]
			}
		}
	}
	cornersA := []PointF{{8, 8}, {15, 10}, {20, 18}, {12, 22}}
	cornersB := make([]PointF, len(cornersA))
	for i, c := range cornersA {
		cornersB[i] = PointF{c.X + dx, c.Y + dy}
	}
	matches := MatchCornersNCC(a, b, cornersA, cornersB, 3, 4, 0.5)
	if len(matches) != len(cornersA) {
		t.Fatalf("got %d matches, want %d", len(matches), len(cornersA))
	}
	for _, m := range matches {
		if m.Dst.X-m.Src.X != dx || m.Dst.Y-m.Src.Y != dy {
			t.Fatalf("match %v does not recover shift (%d,%d)", m, dx, dy)
		}
	}
}

func TestHarrisResponseSize(t *testing.T) {
	img := textureImage(10, 12)
	r := HarrisResponse(img, 1, 0.04)
	if r.Rows != 10 || r.Cols != 12 {
		t.Fatalf("response size = %dx%d, want 10x12", r.Rows, r.Cols)
	}
}

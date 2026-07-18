package features3

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// grayMat builds a single-channel cv.Mat from row-major uint8 values.
func grayMat(rows, cols int, vals []uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	copy(m.Data, vals)
	return m
}

// filledSquare returns a rows×cols single-channel image with a solid value block
// spanning [x0,x1]×[y0,y1] (inclusive) over a zero background.
func filledSquare(rows, cols, x0, y0, x1, y1 int, value uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			m.Data[y*cols+x] = value
		}
	}
	return m
}

// filledDisc returns a rows×cols single-channel image with a solid disc of the
// given radius centred at (cx, cy) over a zero background.
func filledDisc(rows, cols, cx, cy, radius int, value uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	r2 := radius * radius
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			dx, dy := x-cx, y-cy
			if dx*dx+dy*dy <= r2 {
				m.Data[y*cols+x] = value
			}
		}
	}
	return m
}

func TestBitCountAndHamming(t *testing.T) {
	if BitCount(0) != 0 {
		t.Fatalf("BitCount(0)=%d", BitCount(0))
	}
	if BitCount(0xFF) != 8 {
		t.Fatalf("BitCount(0xFF)=%d", BitCount(0xFF))
	}
	if BitCount(0b1011) != 3 {
		t.Fatalf("BitCount(0b1011)=%d", BitCount(0b1011))
	}
	if got := HammingDistanceUint64(0b1010, 0b0110); got != 2 {
		t.Fatalf("HammingDistanceUint64=%d want 2", got)
	}
	if got := HammingDistance([]byte{0xFF, 0x00}, []byte{0x0F, 0x01}); got != 5 {
		t.Fatalf("HammingDistance=%d want 5", got)
	}
}

func TestHammingDistancePanicsOnMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on unequal lengths")
		}
	}()
	HammingDistance([]byte{1}, []byte{1, 2})
}

func TestCensusTransform3x3KnownAnswer(t *testing.T) {
	img := grayMat(3, 3, []uint8{
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
	})
	code := CensusTransform3x3(img)
	if code[1*3+1] != 15 {
		t.Fatalf("census centre code = %d (%08b), want 15", code[4], code[4])
	}
}

func TestCensusFieldSelfHammingZero(t *testing.T) {
	img := filledSquare(20, 20, 5, 5, 14, 14, 200)
	f := CensusTransform(img, 2)
	if f.Bits != 24 {
		t.Fatalf("Bits=%d want 24", f.Bits)
	}
	// Two interior pixels of the uniform block have identical census codes.
	if d := f.Hamming(8, 8, 9, 9); d != 0 {
		t.Fatalf("interior census Hamming=%d want 0", d)
	}
	if f.At(8, 8) != f.At(9, 9) {
		t.Fatal("expected equal codes for uniform interior")
	}
}

func TestModifiedCensusBits(t *testing.T) {
	img := filledSquare(10, 10, 2, 2, 7, 7, 100)
	f := ModifiedCensusTransform(img, 1)
	if f.Bits != 9 {
		t.Fatalf("Bits=%d want 9", f.Bits)
	}
}

func TestCensusTransformPanicsLargeWindow(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for oversized census window")
		}
	}()
	CensusTransform(filledSquare(20, 20, 0, 0, 19, 19, 5), 4)
}

func TestHarrisCornersOnSquare(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	kps := HarrisCorners(img, 3, 0.04, 0.1, 0)
	if len(kps) == 0 {
		t.Fatal("no Harris corners found on a square")
	}
	corners := [][2]float64{{10, 10}, {30, 10}, {10, 30}, {30, 30}}
	for _, c := range corners {
		if !nearAnyKeyPoint(kps, c[0], c[1], 3) {
			t.Fatalf("no Harris corner near (%v,%v)", c[0], c[1])
		}
	}
}

func TestShiTomasiCornersOnSquare(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	kps := ShiTomasiCorners(img, 3, 0.1, 0)
	if !nearAnyKeyPoint(kps, 10, 10, 3) {
		t.Fatal("Shi-Tomasi missed top-left corner")
	}
}

func TestGoodFeaturesToTrackSpacing(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	kps := GoodFeaturesToTrack(img, 4, 0.1, 8, 3)
	if len(kps) == 0 {
		t.Fatal("GoodFeaturesToTrack found nothing")
	}
	for i := 0; i < len(kps); i++ {
		for j := i + 1; j < len(kps); j++ {
			if kps[i].DistanceTo(kps[j]) < 8 {
				t.Fatalf("keypoints closer than minDistance: %v %v", kps[i].Pt, kps[j].Pt)
			}
		}
	}
}

func TestCornerMinEigenValNonNegative(t *testing.T) {
	img := filledSquare(20, 20, 5, 5, 14, 14, 180)
	resp := CornerMinEigenVal(img, 3)
	for i, v := range resp.Data {
		if v < -1e-6 {
			t.Fatalf("min eigenvalue negative at %d: %v", i, v)
		}
	}
}

func TestCornerEigenValsOrdering(t *testing.T) {
	img := filledSquare(30, 30, 8, 8, 20, 20, 200)
	l1, l2 := CornerEigenVals(img, 3)
	for i := range l1.Data {
		if l1.Data[i]+1e-9 < l2.Data[i] {
			t.Fatalf("lambda1 < lambda2 at %d: %v %v", i, l1.Data[i], l2.Data[i])
		}
	}
}

func TestCornerSubPixStaysNearCorner(t *testing.T) {
	img := filledSquare(41, 41, 10, 10, 30, 30, 255)
	refined := CornerSubPix(img, []cv.Point2f{{X: 10, Y: 10}}, 5, 20, 0.01)
	d := math.Hypot(refined[0].X-10, refined[0].Y-10)
	if d > 2.5 {
		t.Fatalf("sub-pixel refinement drifted %.2f px from corner", d)
	}
}

func TestPreCornerDetectShape(t *testing.T) {
	img := filledSquare(15, 15, 4, 4, 10, 10, 120)
	r := PreCornerDetect(img)
	if r.Rows != 15 || r.Cols != 15 {
		t.Fatalf("PreCornerDetect shape %dx%d", r.Rows, r.Cols)
	}
}

func nearAnyKeyPoint(kps []KeyPoint, x, y, tol float64) bool {
	for _, k := range kps {
		if math.Hypot(k.Pt.X-x, k.Pt.Y-y) <= tol {
			return true
		}
	}
	return false
}

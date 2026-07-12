package stitching

import (
	"image"
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// gainStepPair builds two fully-overlapping single-channel images with the same
// content except that image B is uniformly scaled by factor, simulating an
// exposure step. Both cover every pixel.
func gainStepPair(factor float64) (a, b *cv.Mat, ma, mb *cv.FloatMat) {
	base := syntheticTexture(60, 60, 20)
	a = base.Clone()
	b = base.Clone()
	for i := range b.Data {
		b.Data[i] = clampUint8(float64(b.Data[i]) * factor)
	}
	ma = fullMask(a.Rows, a.Cols)
	mb = fullMask(b.Rows, b.Cols)
	return a, b, ma, mb
}

// overlapMeans returns the mean intensity of two same-size, fully-overlapping
// images.
func overlapMeans(a, b *cv.Mat) (mA, mB float64) {
	for p := 0; p < a.Total(); p++ {
		mA += pixelIntensity(a, p)
		mB += pixelIntensity(b, p)
	}
	n := float64(a.Total())
	return mA / n, mB / n
}

func TestGainCompensatorEqualizes(t *testing.T) {
	a, b, ma, mb := gainStepPair(0.5)
	corners := []image.Point{{X: 0, Y: 0}, {X: 0, Y: 0}}

	mBefore0, mBefore1 := overlapMeans(a, b)
	beforeGap := math.Abs(mBefore0 - mBefore1)

	gc := &GainCompensator{}
	gc.Feed(corners, []*cv.Mat{a, b}, []*cv.FloatMat{ma, mb})
	gains := gc.Gains()
	if len(gains) != 2 {
		t.Fatalf("expected 2 gains, got %d", len(gains))
	}
	// The brighter image (A) should be scaled down relative to the darker (B):
	// gain ratio must move toward the 1:2 intensity ratio.
	gc.Apply(0, corners[0], a)
	gc.Apply(1, corners[1], b)

	mAfter0, mAfter1 := overlapMeans(a, b)
	afterGap := math.Abs(mAfter0 - mAfter1)
	if afterGap >= beforeGap {
		t.Errorf("gain compensation did not reduce intensity gap: before=%.2f after=%.2f", beforeGap, afterGap)
	}
	// After compensation the overlap means should be close.
	if rel := afterGap / mAfter0; rel > 0.05 {
		t.Errorf("compensated means still differ by %.1f%%, want <= 5%%", rel*100)
	}
}

func TestBlocksGainCompensatorEqualizes(t *testing.T) {
	a, b, ma, mb := gainStepPair(0.6)
	corners := []image.Point{{X: 0, Y: 0}, {X: 0, Y: 0}}

	mBefore0, mBefore1 := overlapMeans(a, b)
	beforeGap := math.Abs(mBefore0 - mBefore1)

	bc := &BlocksGainCompensator{BlockWidth: 20, BlockHeight: 20}
	bc.Feed(corners, []*cv.Mat{a, b}, []*cv.FloatMat{ma, mb})
	bc.Apply(0, corners[0], a)
	bc.Apply(1, corners[1], b)

	mAfter0, mAfter1 := overlapMeans(a, b)
	afterGap := math.Abs(mAfter0 - mAfter1)
	if afterGap >= beforeGap {
		t.Errorf("blocks gain compensation did not reduce intensity gap: before=%.2f after=%.2f", beforeGap, afterGap)
	}
}

func TestNoExposureCompensatorIdentity(t *testing.T) {
	a := syntheticTexture(20, 20, 3)
	before := a.Clone()
	ne := NoExposureCompensator{}
	ne.Feed(nil, nil, nil)
	ne.Apply(0, image.Point{}, a)
	for i := range a.Data {
		if a.Data[i] != before.Data[i] {
			t.Fatal("NoExposureCompensator modified the image")
		}
	}
}

package stitching

import (
	"image"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// seamBandPair builds two images placed side by side with a horizontal overlap.
// Inside the overlap the images disagree by a constant intensity step everywhere
// except in a vertical band of global columns [bandX0, bandX1), where they match
// exactly, so the cheapest seam runs down that band. Both images cover every
// pixel of their own footprint.
func seamBandPair(rows, width, offset, bandX0, bandX1 int) (a, b *cv.Mat, corners []image.Point, ma, mb *cv.FloatMat) {
	a = cv.NewMat(rows, width, 1)
	b = cv.NewMat(rows, width, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < width; x++ {
			a.Data[y*width+x] = 100
		}
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < width; x++ {
			gx := offset + x // global column of this B pixel
			if gx >= bandX0 && gx < bandX1 {
				b.Data[y*width+x] = 100 // matches A in the band
			} else {
				b.Data[y*width+x] = 200 // differs from A by 100
			}
		}
	}
	corners = []image.Point{{X: 0, Y: 0}, {X: offset, Y: 0}}
	ma = fullMask(rows, width)
	mb = fullMask(rows, width)
	return a, b, corners, ma, mb
}

// assertOverlapPartition checks that within the horizontal overlap every pixel is
// owned by exactly one of the two images.
func assertOverlapPartition(t *testing.T, a, b *cv.Mat, corners []image.Point, ma, mb *cv.FloatMat) {
	t.Helper()
	x0 := maxInt(corners[0].X, corners[1].X)
	x1 := minInt(corners[0].X+a.Cols, corners[1].X+b.Cols)
	for y := 0; y < a.Rows; y++ {
		for gx := x0; gx < x1; gx++ {
			pa := y*a.Cols + (gx - corners[0].X)
			pb := y*b.Cols + (gx - corners[1].X)
			aOn := ma.Data[pa] > 0
			bOn := mb.Data[pb] > 0
			if aOn == bOn {
				t.Fatalf("overlap pixel (%d,%d) not uniquely owned (a=%v b=%v)", gx, y, aOn, bOn)
			}
		}
	}
}

func TestDpSeamFinderLowCost(t *testing.T) {
	a, b, corners, ma, mb := seamBandPair(60, 40, 25, 30, 34)
	dp := &DpSeamFinder{}
	dp.Find([]*cv.Mat{a, b}, corners, []*cv.FloatMat{ma, mb})
	assertOverlapPartition(t, a, b, corners, ma, mb)
	// A seam threading the matching band has near-zero cost; a straight seam
	// through the mismatched region would cost ~ rows*100.
	if dp.SeamCost() > 5 {
		t.Errorf("DP seam cost = %.3f, want <= 5 (should follow the matching band)", dp.SeamCost())
	}
}

func TestGraphCutSeamFinderLowCost(t *testing.T) {
	a, b, corners, ma, mb := seamBandPair(60, 40, 25, 29, 34)
	gc := &GraphCutSeamFinder{}
	gc.Find([]*cv.Mat{a, b}, corners, []*cv.FloatMat{ma, mb})
	assertOverlapPartition(t, a, b, corners, ma, mb)
	// The min cut runs between two matching columns (zero colour difference), so
	// the cut cost is far below the ~rows*200 of a cut in the mismatched area.
	if gc.CutCost() > 10 {
		t.Errorf("graph-cut cost = %.3f, want <= 10 (should cut inside the matching band)", gc.CutCost())
	}
}

func TestVoronoiSeamFinderPartitions(t *testing.T) {
	a, b, corners, ma, mb := seamBandPair(40, 60, 30, 40, 44)
	VoronoiSeamFinder{}.Find([]*cv.Mat{a, b}, corners, []*cv.FloatMat{ma, mb})
	assertOverlapPartition(t, a, b, corners, ma, mb)
}

func TestNoSeamFinderKeepsMasks(t *testing.T) {
	a, b, corners, ma, mb := seamBandPair(20, 20, 8, 10, 14)
	NoSeamFinder{}.Find([]*cv.Mat{a, b}, corners, []*cv.FloatMat{ma, mb})
	for p := range ma.Data {
		if ma.Data[p] <= 0 || mb.Data[p] <= 0 {
			t.Fatal("NoSeamFinder altered a mask")
		}
	}
}

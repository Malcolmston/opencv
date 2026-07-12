package objdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// TestIntegralImageMatchesDirectSum verifies that integral-image rectangle sums
// (and squared sums) equal the brute-force pixel sums on a synthetic image.
func TestIntegralImageMatchesDirectSum(t *testing.T) {
	rows, cols := 17, 23
	img := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// A non-trivial, deterministic pattern.
			img.Set(y, x, 0, uint8((x*7+y*13)%251))
		}
	}
	ii := NewIntegralImage(img)

	rects := [][4]int{
		{0, 0, cols, rows},
		{2, 3, 5, 4},
		{10, 1, 8, 12},
		{cols - 3, rows - 3, 3, 3},
		{5, 5, 1, 1},
	}
	for _, r := range rects {
		x, y, w, h := r[0], r[1], r[2], r[3]
		var sum, sq float64
		for yy := y; yy < y+h; yy++ {
			for xx := x; xx < x+w; xx++ {
				v := float64(img.At(yy, xx, 0))
				sum += v
				sq += v * v
			}
		}
		if got := ii.Sum(x, y, w, h); got != sum {
			t.Fatalf("Sum(%d,%d,%d,%d) = %.1f, want %.1f", x, y, w, h, got, sum)
		}
		if got := ii.SqSum(x, y, w, h); got != sq {
			t.Fatalf("SqSum(%d,%d,%d,%d) = %.1f, want %.1f", x, y, w, h, got, sq)
		}
	}
}

// A hand-crafted single-stage cascade: one Haar feature that measures
// (top half sum) - (bottom half sum) of an 8x8 window. Its stump outputs +1
// when that difference is at least half the window's normalisation factor and
// -1 otherwise; the stage passes when the output is >= 0.
const oneStageCascadeXML = `<?xml version="1.0"?>
<opencv_storage>
<cascade type_id="opencv-cascade-classifier">
  <stageType>BOOST</stageType>
  <featureType>HAAR</featureType>
  <height>8</height>
  <width>8</width>
  <stageNum>1</stageNum>
  <stages>
    <_>
      <maxWeakCount>1</maxWeakCount>
      <stageThreshold>0.</stageThreshold>
      <weakClassifiers>
        <_>
          <internalNodes>0 -1 0 0.5</internalNodes>
          <leafValues>-1. 1.</leafValues>
        </_>
      </weakClassifiers>
    </_>
  </stages>
  <features>
    <_>
      <rects>
        <_>0 0 8 4 1.</_>
        <_>0 4 8 4 -1.</_>
      </rects>
      <tilted>0</tilted>
    </_>
  </features>
</cascade>
</opencv_storage>`

// TestCascadeLoadAndDetect loads the hand-crafted cascade and checks it fires
// on a top-bright/bottom-dark window and rejects the inverse.
func TestCascadeLoadAndDetect(t *testing.T) {
	var clf CascadeClassifier
	if err := clf.LoadFromString(oneStageCascadeXML); err != nil {
		t.Fatalf("LoadFromString: %v", err)
	}
	if !clf.Loaded() {
		t.Fatal("Loaded() = false after successful load")
	}
	if w, h := clf.WindowSize(); w != 8 || h != 8 {
		t.Fatalf("WindowSize() = %dx%d, want 8x8", w, h)
	}

	// Top half white, bottom half black => feature value strongly positive.
	pos := cv.NewMat(8, 8, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if y < 4 {
				pos.Set(y, x, 0, 255)
			}
		}
	}
	hits := clf.DetectMultiScale(pos)
	if len(hits) != 1 {
		t.Fatalf("expected exactly 1 detection on top-bright window, got %d (%v)", len(hits), hits)
	}
	if hits[0] != (cv.Rect{X: 0, Y: 0, Width: 8, Height: 8}) {
		t.Fatalf("detection rect = %+v, want {0 0 8 8}", hits[0])
	}

	// Inverse image: bottom half white => feature value negative => reject.
	neg := cv.NewMat(8, 8, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if y >= 4 {
				neg.Set(y, x, 0, 255)
			}
		}
	}
	if hits := clf.DetectMultiScale(neg); len(hits) != 0 {
		t.Fatalf("expected no detection on bottom-bright window, got %d (%v)", len(hits), hits)
	}

	// Uniform image => zero variance and zero feature => reject.
	uni := cv.NewMat(8, 8, 1)
	uni.SetTo(200)
	if hits := clf.DetectMultiScale(uni); len(hits) != 0 {
		t.Fatalf("expected no detection on uniform window, got %d", len(hits))
	}
}

// TestCascadeDetectMultiScalePyramid places the pattern in a larger image and
// confirms the scale pyramid finds it.
func TestCascadeDetectMultiScalePyramid(t *testing.T) {
	var clf CascadeClassifier
	if err := clf.LoadFromString(oneStageCascadeXML); err != nil {
		t.Fatalf("LoadFromString: %v", err)
	}
	// 24x24 image: draw a 16x16 top-bright/bottom-dark block at (4,4). The
	// cascade window grows from 8x8 and should fire somewhere over the block.
	img := cv.NewMat(24, 24, 1)
	for y := 4; y < 20; y++ {
		for x := 4; x < 20; x++ {
			if y < 12 {
				img.Set(y, x, 0, 255)
			}
		}
	}
	hits := clf.DetectMultiScale(img)
	if len(hits) == 0 {
		t.Fatal("expected at least one detection over the block")
	}
	// At least one detection should overlap the bright/dark boundary region.
	found := false
	for _, r := range hits {
		cx := r.X + r.Width/2
		cy := r.Y + r.Height/2
		if cx >= 4 && cx <= 20 && cy >= 4 && cy <= 20 {
			found = true
		}
	}
	if !found {
		t.Fatalf("no detection centred over the block, hits=%v", hits)
	}
}

// TestCascadeErrors checks malformed input and unloaded use.
func TestCascadeErrors(t *testing.T) {
	var clf CascadeClassifier
	if err := clf.LoadFromString("<opencv_storage><cascade></cascade></opencv_storage>"); err == nil {
		t.Fatal("expected error on empty cascade")
	}
	if err := clf.LoadFromString("not xml at all <<<"); err == nil {
		t.Fatal("expected error on malformed XML")
	}

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on DetectMultiScale of unloaded classifier")
		}
	}()
	var fresh CascadeClassifier
	fresh.DetectMultiScale(cv.NewMat(8, 8, 1))
}

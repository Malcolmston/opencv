package objdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestComputeLBP(t *testing.T) {
	// A uniform image: every neighbour equals the centre, and the comparison is
	// ">=", so every bit is set -> code 255 everywhere.
	img := cv.NewMat(8, 8, 1)
	img.SetTo(120)
	lbp := ComputeLBP(img)
	if lbp.Rows != 8 || lbp.Cols != 8 || lbp.Channels != 1 {
		t.Fatalf("LBP image shape = %dx%dx%d, want 8x8x1", lbp.Rows, lbp.Cols, lbp.Channels)
	}
	if got := lbp.At(4, 4, 0); got != 255 {
		t.Fatalf("uniform LBP code = %d, want 255", got)
	}

	// A bright centre surrounded by dark: no neighbour >= centre -> code 0.
	img2 := cv.NewMat(3, 3, 1)
	img2.SetTo(0)
	img2.Set(1, 1, 0, 200)
	l2 := ComputeLBP(img2)
	if got := l2.At(1, 1, 0); got != 0 {
		t.Fatalf("bright-centre LBP code = %d, want 0", got)
	}
}

// lbpCascadeXML is a hand-built 12x12 LBP cascade with one 4x4-cell feature and
// one weak classifier. Its 256-bit subset selects only LBP code 255 (all eight
// neighbour cells >= the centre cell), which corresponds to a dark centre cell
// surrounded by bright cells.
const lbpCascadeXML = `<?xml version="1.0"?>
<opencv_storage>
<cascade type_id="opencv-cascade-classifier">
  <stageType>BOOST</stageType>
  <featureType>LBP</featureType>
  <height>12</height>
  <width>12</width>
  <stages>
    <_>
      <maxWeakCount>1</maxWeakCount>
      <stageThreshold>0.5</stageThreshold>
      <weakClassifiers>
        <_>
          <internalNodes>0 -1 0 0 0 0 0 0 0 0 -2147483648</internalNodes>
          <leafValues>1. -1.</leafValues>
        </_>
      </weakClassifiers>
    </_>
  </stages>
  <features>
    <_>
      <rect>0 0 4 4</rect>
    </_>
  </features>
</cascade>
</opencv_storage>`

func TestLBPCascadeLoadAndDetect(t *testing.T) {
	var clf LBPCascadeClassifier
	if err := clf.LoadFromString(lbpCascadeXML); err != nil {
		t.Fatalf("LoadFromString: %v", err)
	}
	if !clf.Loaded() {
		t.Fatal("Loaded() = false after successful load")
	}
	if w, h := clf.WindowSize(); w != 12 || h != 12 {
		t.Fatalf("WindowSize = %dx%d, want 12x12", w, h)
	}

	// Positive: dark 4x4 centre cell, bright surround -> code 255 -> passes.
	pos := cv.NewMat(12, 12, 1)
	pos.SetTo(255)
	for y := 4; y < 8; y++ {
		for x := 4; x < 8; x++ {
			pos.Set(y, x, 0, 0)
		}
	}
	hits := clf.DetectMultiScale(pos)
	if len(hits) != 1 || hits[0] != (cv.Rect{X: 0, Y: 0, Width: 12, Height: 12}) {
		t.Fatalf("expected one detection {0 0 12 12}, got %v", hits)
	}

	// Negative: bright centre, dark surround -> code 0 -> rejected.
	neg := cv.NewMat(12, 12, 1)
	neg.SetTo(0)
	for y := 4; y < 8; y++ {
		for x := 4; x < 8; x++ {
			neg.Set(y, x, 0, 255)
		}
	}
	if hits := clf.DetectMultiScale(neg); len(hits) != 0 {
		t.Fatalf("expected no detection on inverted window, got %v", hits)
	}
}

func TestLBPCascadeErrors(t *testing.T) {
	var clf LBPCascadeClassifier
	// Wrong feature type.
	haar := `<opencv_storage><cascade><featureType>HAAR</featureType><width>8</width><height>8</height>
	<stages><_><stageThreshold>0.</stageThreshold><weakClassifiers><_>
	<internalNodes>0 -1 0 0.5</internalNodes><leafValues>-1. 1.</leafValues></_></weakClassifiers></_></stages>
	<features><_><rect>0 0 3 3</rect></_></features></cascade></opencv_storage>`
	if err := clf.LoadFromString(haar); err == nil {
		t.Fatal("expected error for non-LBP featureType")
	}

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on DetectMultiScale of unloaded classifier")
		}
	}()
	var fresh LBPCascadeClassifier
	fresh.DetectMultiScale(cv.NewMat(12, 12, 1))
}

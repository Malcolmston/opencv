package cudaobjdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// oneStageCascadeXML is a hand-crafted single-stage Haar cascade whose one
// feature measures (top half sum) - (bottom half sum) of an 8x8 window and
// fires when the top half is brighter. It is evaluated over an integral image,
// exercising the same code path as a real cascade.
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

// topBright returns an 8x8 image with its top half white and bottom half black.
func topBright() *cv.Mat {
	m := cv.NewMat(8, 8, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	return m
}

// TestCascadeDetectAndConvert loads the cascade and checks the
// DetectMultiScale/Convert protocol on the integral-image pattern.
func TestCascadeDetectAndConvert(t *testing.T) {
	clf, err := LoadCascadeFromString(oneStageCascadeXML)
	if err != nil {
		t.Fatalf("LoadCascadeFromString: %v", err)
	}
	if w, h := clf.GetClassifierSize(); w != 8 || h != 8 {
		t.Fatalf("GetClassifierSize() = %dx%d, want 8x8", w, h)
	}
	clf.SetMinNeighbors(0) // return raw hits on the single 8x8 window

	gpuObjects := clf.DetectMultiScale(NewGpuMatFromMat(topBright()), NewStream())
	rects := clf.Convert(gpuObjects)
	if len(rects) != 1 {
		t.Fatalf("expected 1 detection on top-bright window, got %d (%v)", len(rects), rects)
	}
	if rects[0] != (cv.Rect{X: 0, Y: 0, Width: 8, Height: 8}) {
		t.Fatalf("detection = %+v, want {0 0 8 8}", rects[0])
	}

	// The inverse pattern must not fire.
	neg := cv.NewMat(8, 8, 1)
	for y := 4; y < 8; y++ {
		for x := 0; x < 8; x++ {
			neg.Set(y, x, 0, 255)
		}
	}
	if got := clf.Convert(clf.DetectMultiScale(NewGpuMatFromMat(neg), nil)); len(got) != 0 {
		t.Fatalf("expected no detection on bottom-bright window, got %v", got)
	}
}

// TestCascadePyramidAndSizeFilter checks the scale pyramid plus min/max object
// size filtering and find-largest.
func TestCascadePyramidAndSizeFilter(t *testing.T) {
	clf, err := LoadCascadeFromString(oneStageCascadeXML)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	clf.SetMinNeighbors(0)

	// 24x24 image with a 16x16 top-bright/bottom-dark block at (4,4).
	img := cv.NewMat(24, 24, 1)
	for y := 4; y < 20; y++ {
		for x := 4; x < 20; x++ {
			if y < 12 {
				img.Set(y, x, 0, 255)
			}
		}
	}
	all := clf.Convert(clf.DetectMultiScale(NewGpuMatFromMat(img), nil))
	if len(all) == 0 {
		t.Fatal("expected detections over the block")
	}

	// Filtering out everything smaller than 12x12 must remove the base 8x8 hits.
	clf.SetMinObjectSize(Size{Width: 12, Height: 12})
	filtered := clf.Convert(clf.DetectMultiScale(NewGpuMatFromMat(img), nil))
	for _, r := range filtered {
		if r.Width < 12 || r.Height < 12 {
			t.Fatalf("min-size filter leaked %+v", r)
		}
	}
	if len(filtered) >= len(all) {
		t.Fatalf("min-size filter did not reduce count (%d vs %d)", len(filtered), len(all))
	}

	// FindLargestObject must return at most one detection.
	clf.SetMinObjectSize(Size{})
	clf.SetFindLargestObject(true)
	largest := clf.Convert(clf.DetectMultiScale(NewGpuMatFromMat(img), nil))
	if len(largest) > 1 {
		t.Fatalf("find-largest returned %d detections", len(largest))
	}
}

// TestCascadeParams round-trips the parameter accessors.
func TestCascadeParams(t *testing.T) {
	clf, err := LoadCascadeFromString(oneStageCascadeXML)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	clf.SetScaleFactor(1.25)
	if clf.GetScaleFactor() != 1.25 {
		t.Fatalf("scale factor = %v", clf.GetScaleFactor())
	}
	clf.SetMinNeighbors(6)
	if clf.GetMinNeighbors() != 6 {
		t.Fatalf("min neighbors = %v", clf.GetMinNeighbors())
	}
	clf.SetMinObjectSize(Size{Width: 4, Height: 4})
	if clf.GetMinObjectSize() != (Size{Width: 4, Height: 4}) {
		t.Fatalf("min size = %+v", clf.GetMinObjectSize())
	}
	clf.SetMaxObjectSize(Size{Width: 40, Height: 40})
	if clf.GetMaxObjectSize() != (Size{Width: 40, Height: 40}) {
		t.Fatalf("max size = %+v", clf.GetMaxObjectSize())
	}
	clf.SetFindLargestObject(true)
	if !clf.GetFindLargestObject() {
		t.Fatal("find largest should be true")
	}
}

// TestCascadeMaxSizeFilter checks the upper size bound.
func TestCascadeMaxSizeFilter(t *testing.T) {
	clf, err := LoadCascadeFromString(oneStageCascadeXML)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	clf.SetMinNeighbors(0)
	clf.SetMaxObjectSize(Size{Width: 8, Height: 8})
	rects := clf.Convert(clf.DetectMultiScale(NewGpuMatFromMat(topBright()), nil))
	for _, r := range rects {
		if r.Width > 8 || r.Height > 8 {
			t.Fatalf("max-size filter leaked %+v", r)
		}
	}
}

// TestCascadeLoadErrors checks error reporting on bad input.
func TestCascadeLoadErrors(t *testing.T) {
	if _, err := LoadCascadeFromString("not xml <<<"); err == nil {
		t.Fatal("expected error on malformed XML")
	}
	if _, err := NewCascadeClassifier("/nonexistent/path/cascade.xml"); err == nil {
		t.Fatal("expected error on missing file")
	}
}

// TestCascadeConvertEmpty verifies Convert on a nil/empty result.
func TestCascadeConvertEmpty(t *testing.T) {
	clf, err := LoadCascadeFromString(oneStageCascadeXML)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := clf.Convert(nil); got != nil {
		t.Fatalf("Convert(nil) = %v, want nil", got)
	}
	if got := clf.Convert(&GpuMat{}); got != nil {
		t.Fatalf("Convert(empty) = %v, want nil", got)
	}
}

package features2d

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestDrawKeypointsFlags(t *testing.T) {
	img := buildScene(80)
	kps := []KeyPoint{
		{Pt: cv.Point{X: 40, Y: 40}, Size: 20, Angle: 45},
		{Pt: cv.Point{X: 20, Y: 20}, Size: 10, Angle: -1},
	}
	// Default flags: fixed markers, custom red colour.
	red := cv.NewScalar(255, 0, 0)
	out := DrawKeypointsFlags(img, kps, red, DrawDefault)
	if out.Channels != 3 {
		t.Fatalf("expected 3-channel output, got %d", out.Channels)
	}
	if out.Rows != img.Rows || out.Cols != img.Cols {
		t.Fatalf("output size changed")
	}
	// Some pixel must now be red.
	foundRed := false
	for i := 0; i+2 < len(out.Data); i += 3 {
		if out.Data[i] == 255 && out.Data[i+1] == 0 && out.Data[i+2] == 0 {
			foundRed = true
			break
		}
	}
	if !foundRed {
		t.Fatal("expected red keypoint markers in output")
	}

	// Rich flag with default colour must not modify the input and must draw.
	before := append([]uint8(nil), img.Data...)
	rich := DrawKeypointsFlags(img, kps, cv.Scalar{}, DrawRichKeypoints)
	if rich.Empty() {
		t.Fatal("rich output empty")
	}
	for i := range img.Data {
		if img.Data[i] != before[i] {
			t.Fatal("input image was modified")
		}
	}
}

func TestDrawKeypointsFlagsRichVsDefaultDiffer(t *testing.T) {
	img := buildScene(80)
	kps := []KeyPoint{{Pt: cv.Point{X: 40, Y: 40}, Size: 30, Angle: 0}}
	def := DrawKeypointsFlags(img, kps, cv.Scalar{}, DrawDefault)
	rich := DrawKeypointsFlags(img, kps, cv.Scalar{}, DrawRichKeypoints)
	// The rich rendering uses the larger Size-derived radius, so the two images
	// cannot be identical.
	same := true
	for i := range def.Data {
		if def.Data[i] != rich.Data[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("rich and default renderings are identical")
	}
}

package cudaobjdetect_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudaobjdetect"
)

// ExampleHOG_GetDescriptorSize reports the descriptor length for the default
// 64×128 people-detector geometry.
func ExampleHOG_GetDescriptorSize() {
	h := cudaobjdetect.NewDefaultHOG()
	fmt.Println(h.GetDescriptorSize())
	// Output: 3780
}

// ExampleHOG_Detect runs the default people detector on a single window and
// reports whether it fired.
func ExampleHOG_Detect() {
	h := cudaobjdetect.NewDefaultHOG()
	h.SetSVMDetector(h.GetDefaultPeopleDetector())

	// A featureless (flat) window never fires.
	flat := cv.NewMat(128, 64, 1)
	flat.SetTo(128)
	locs, _ := h.Detect(cudaobjdetect.NewGpuMatFromMat(flat), nil)
	fmt.Println(len(locs))
	// Output: 0
}

// ExampleCascadeClassifier_DetectMultiScale shows the detect/convert protocol on
// a tiny hand-crafted Haar cascade.
func ExampleCascadeClassifier_DetectMultiScale() {
	const xml = `<?xml version="1.0"?>
<opencv_storage>
<cascade>
  <featureType>HAAR</featureType>
  <height>8</height>
  <width>8</width>
  <stages><_>
    <stageThreshold>0.</stageThreshold>
    <weakClassifiers><_>
      <internalNodes>0 -1 0 0.5</internalNodes>
      <leafValues>-1. 1.</leafValues>
    </_></weakClassifiers>
  </_></stages>
  <features><_>
    <rects><_>0 0 8 4 1.</_><_>0 4 8 4 -1.</_></rects>
    <tilted>0</tilted>
  </_></features>
</cascade>
</opencv_storage>`
	clf, err := cudaobjdetect.LoadCascadeFromString(xml)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	clf.SetMinNeighbors(0)

	img := cv.NewMat(8, 8, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			img.Set(y, x, 0, 255) // top half bright
		}
	}
	objects := clf.DetectMultiScale(cudaobjdetect.NewGpuMatFromMat(img), nil)
	rects := clf.Convert(objects)
	fmt.Println(rects[0])
	// Output: {0 0 8 8}
}

// ExampleNMSBoxes suppresses a redundant overlapping detection.
func ExampleNMSBoxes() {
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 10, Height: 10},
		{X: 1, Y: 1, Width: 10, Height: 10},
	}
	scores := []float64{0.9, 0.8}
	kept := cudaobjdetect.NMSBoxes(boxes, scores, 0.5, 0.5)
	fmt.Println(len(kept))
	// Output: 1
}

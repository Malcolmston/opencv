package objdetect_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/objdetect"
)

// ExampleHOGDescriptor_Compute computes the HOG descriptor for a single
// detection window and reports its length.
func ExampleHOGDescriptor_Compute() {
	h := objdetect.NewHOGDescriptor()
	img := cv.NewMat(h.WinSize.Height, h.WinSize.Width, 1)
	desc := h.Compute(img)
	fmt.Println(len(desc))
	// Output: 3780
}

// ExampleHOGDescriptor_DescriptorSize shows the descriptor length for a custom
// geometry.
func ExampleHOGDescriptor_DescriptorSize() {
	h := &objdetect.HOGDescriptor{
		WinSize:     objdetect.Size{Width: 16, Height: 16},
		BlockSize:   objdetect.Size{Width: 8, Height: 8},
		BlockStride: objdetect.Size{Width: 4, Height: 4},
		CellSize:    objdetect.Size{Width: 4, Height: 4},
		NBins:       6,
	}
	fmt.Println(h.DescriptorSize())
	// Output: 216
}

// ExampleCascadeClassifier_LoadFromString parses a tiny cascade and reports its
// window size.
func ExampleCascadeClassifier_LoadFromString() {
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
	var clf objdetect.CascadeClassifier
	if err := clf.LoadFromString(xml); err != nil {
		fmt.Println("error:", err)
		return
	}
	w, h := clf.WindowSize()
	fmt.Printf("%dx%d\n", w, h)
	// Output: 8x8
}

// ExampleIntegralImage shows constant-time rectangle sums.
func ExampleIntegralImage() {
	img := cv.NewMat(4, 4, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.Set(y, x, 0, 10)
		}
	}
	ii := objdetect.NewIntegralImage(img)
	fmt.Println(ii.Sum(1, 1, 2, 2))
	// Output: 40
}

// ExampleQRCodeDetector_Detect locates the finder patterns of a synthetic
// QR-like image.
func ExampleQRCodeDetector_Detect() {
	const size, module = 140, 4
	img := cv.NewMat(size, size, 1)
	img.SetTo(255)
	draw := func(ox, oy int) {
		fill := func(x0, y0, w, h int, v uint8) {
			for y := y0; y < y0+h; y++ {
				for x := x0; x < x0+w; x++ {
					img.Set(y, x, 0, v)
				}
			}
		}
		fill(ox, oy, 7*module, 7*module, 0)
		fill(ox+module, oy+module, 5*module, 5*module, 255)
		fill(ox+2*module, oy+2*module, 3*module, 3*module, 0)
	}
	draw(10, 10)
	draw(size-7*module-10, 10)
	draw(10, size-7*module-10)

	d := objdetect.NewQRCodeDetector()
	_, found := d.Detect(img)
	fmt.Println(found)
	// Output: true
}

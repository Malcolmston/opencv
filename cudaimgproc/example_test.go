package cudaimgproc_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudaimgproc"
)

// ExampleCvtColor uploads an RGB image, converts it to grayscale on the
// (CPU-backed) "device", and downloads the result.
func ExampleCvtColor() {
	img := cv.NewMat(2, 2, 3)
	img.SetTo(200)

	var d cudaimgproc.GpuMat
	d.Upload(img)
	gray := cudaimgproc.CvtColor(d, cv.ColorRGB2Gray)
	out := gray.Download()

	fmt.Printf("%dx%dx%d value=%d\n", out.Rows, out.Cols, out.Channels, out.Data[0])
	// Output: 2x2x1 value=200
}

// ExampleCreateCLAHE runs Contrast-Limited Adaptive Histogram Equalisation via
// the algorithm object.
func ExampleCreateCLAHE() {
	img := cv.NewMat(8, 8, 1)
	for i := range img.Data {
		img.Data[i] = uint8(i * 3)
	}
	var d cudaimgproc.GpuMat
	d.Upload(img)

	clahe := cudaimgproc.CreateCLAHE(2.0, 2)
	out := clahe.Apply(d).Download()

	fmt.Printf("%dx%d clip=%.1f\n", out.Rows, out.Cols, clahe.GetClipLimit())
	// Output: 8x8 clip=2.0
}

// ExampleCreateTemplateMatching locates a patch inside a larger image.
func ExampleCreateTemplateMatching() {
	src := cv.NewMat(10, 10, 1)
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			src.Data[y*10+x] = uint8((x*5 + y*9) % 256)
		}
	}
	templ := src.Region(3, 4, 3, 3) // 3x3 patch at (y=3, x=4)

	var gs, gt cudaimgproc.GpuMat
	gs.Upload(src)
	gt.Upload(templ)

	tm := cudaimgproc.CreateTemplateMatching(cv.TmSqdiff)
	res := tm.Match(gs, gt)
	_, _, minX, minY, _, _ := cv.MinMaxLoc(res)

	fmt.Printf("best match at (%d,%d)\n", minX, minY)
	// Output: best match at (4,3)
}

// ExampleCreateCannyEdgeDetector detects the edges of a filled rectangle.
func ExampleCreateCannyEdgeDetector() {
	img := cv.NewMat(20, 20, 1)
	cv.Rectangle(img, cv.Point{X: 5, Y: 5}, cv.Point{X: 15, Y: 15}, cv.NewScalar(255), -1)

	var d cudaimgproc.GpuMat
	d.Upload(img)

	det := cudaimgproc.CreateCannyEdgeDetector(50, 150)
	edges := det.Detect(d).Download()

	count := 0
	for _, v := range edges.Data {
		if v == 255 {
			count++
		}
	}
	fmt.Printf("edge pixels: %t\n", count > 0)
	// Output: edge pixels: true
}

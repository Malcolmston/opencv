package cudaimgproc_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudaimgproc"
)

// Example runs a representative cudaimgproc pipeline: upload an RGB image,
// convert it to grayscale on the (CPU-backed) device, equalize its histogram,
// then download the result to host memory.
func Example() {
	img := cv.NewMat(4, 4, 3)
	for i := range img.Data {
		img.Data[i] = uint8(i * 5)
	}

	var d cudaimgproc.GpuMat
	d.Upload(img)
	gray := cudaimgproc.CvtColor(d, cv.ColorRGB2Gray)
	out := cudaimgproc.EqualizeHist(gray).Download()

	fmt.Printf("%dx%dx%d\n", out.Rows, out.Cols, out.Channels)
	// Output: 4x4x1
}

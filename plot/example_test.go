package plot_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/plot"
)

// ExampleCreatePlot renders a simple line chart to an in-memory image.
func ExampleCreatePlot() {
	xs := []float64{0, 1, 2, 3, 4}
	ys := []float64{0, 1, 4, 9, 16}
	img := plot.CreatePlot(xs, ys).
		SetSize(320, 240).
		SetLineColor(cv.NewScalar(255, 0, 0)).
		Render()
	fmt.Printf("%dx%d, %d channels\n", img.Cols, img.Rows, img.Channels)
	// Output: 320x240, 3 channels
}

// ExampleHistogramPlot bins data and reports the resulting bar chart size.
func ExampleHistogramPlot() {
	data := []float64{1, 2, 2, 3, 3, 3, 4, 4, 4, 4}
	img := plot.HistogramPlot(data, 4).SetSize(400, 300).Render()
	fmt.Printf("%dx%d\n", img.Cols, img.Rows)
	// Output: 400x300
}

// ExampleApplyColorMap false-colours a grayscale ramp and prints the endpoints.
func ExampleApplyColorMap() {
	gray := cv.NewMat(1, 256, 1)
	for i := 0; i < 256; i++ {
		gray.Data[i] = uint8(i)
	}
	color := plot.ApplyColorMap(gray, plot.ColormapJet)
	lo := color.AtPixel(0, 0)
	hi := color.AtPixel(0, 255)
	fmt.Printf("0 -> %v, 255 -> %v\n", lo, hi)
	// Output: 0 -> [0 0 128], 255 -> [128 0 0]
}

// ExampleLUT applies an inverting lookup table to an image.
func ExampleLUT() {
	table := make([]uint8, 256)
	for i := range table {
		table[i] = uint8(255 - i)
	}
	img := cv.NewMat(1, 1, 1)
	img.Data[0] = 40
	out := plot.LUT(img, table)
	fmt.Println(out.Data[0])
	// Output: 215
}

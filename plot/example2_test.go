package plot_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/plot"
)

// ExampleStemPlot renders a stem (lollipop) chart.
func ExampleStemPlot() {
	img := plot.NewStemPlot([]float64{0, 1, 2, 3}, []float64{1, 3, 2, 4}).
		SetSize(320, 240).Render()
	fmt.Printf("%dx%d\n", img.Cols, img.Rows)
	// Output: 320x240
}

// ExampleHeatmapPlot renders a 2-D field with an attached colorbar.
func ExampleHeatmapPlot() {
	data := [][]float64{{0, 1, 2}, {3, 4, 5}}
	img := plot.NewHeatmapPlot(data).SetCellSize(20).SetColormap(plot.ColormapInferno).Render()
	fmt.Printf("%dx%d\n", img.Cols, img.Rows)
	// Output: 104x40
}

// ExampleColorbar renders a standalone vertical colour scale and prints its
// endpoint colours.
func ExampleColorbar() {
	img := plot.NewColorbar(plot.ColormapMagma, 16, 64).Render()
	table := plot.Table(plot.ColormapMagma)
	fmt.Printf("bottom=%v top=%v size=%dx%d\n", table[0], table[255], img.Cols, img.Rows)
	// Output: bottom=[0 0 4] top=[252 253 191] size=16x64
}

// ExampleMultiSeriesPlot overlays two series with a legend.
func ExampleMultiSeriesPlot() {
	img := plot.NewMultiSeriesPlot().
		Add("linear", []float64{0, 1, 2}, []float64{0, 1, 2}, plot.KindLine).
		Add("square", []float64{0, 1, 2}, []float64{0, 1, 4}, plot.KindScatter).
		SetSize(400, 300).Render()
	fmt.Printf("%dx%d\n", img.Cols, img.Rows)
	// Output: 400x300
}

// ExampleColorize false-colours a ramp through one of the additional colormaps.
func ExampleColorize() {
	gray := cv.NewMat(1, 256, 1)
	for i := 0; i < 256; i++ {
		gray.Data[i] = uint8(i)
	}
	out := plot.Colorize(gray, plot.ColormapTurbo)
	fmt.Printf("0 -> %v, 255 -> %v\n", out.AtPixel(0, 0), out.AtPixel(0, 255))
	// Output: 0 -> [48 18 59], 255 -> [122 4 3]
}

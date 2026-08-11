# plot — provides 2-D data plotting and intensity colormaps for the stdlib-only OpenCV port

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/plot.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/plot)

Package plot provides 2-D data plotting and intensity colormaps for the
stdlib-only OpenCV port github.com/malcolmston/opencv (imported here as cv).

- the cv::plot module — turning a series of numbers into a line, scatter,
bar or histogram chart drawn onto an image canvas; and - cv2.applyColorMap —
mapping the intensities of a single-channel image through a 256-entry colour
lookup table to produce a false-colour image.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/plot"
```

## Usage

This is the package's own `ExampleApplyColorMap`, so it compiles and its output is
asserted on every `go test ./plot/`.

```go
gray := cv.NewMat(1, 256, 1)
	for i := 0; i < 256; i++ {
		gray.Data[i] = uint8(i)
	}
	color := plot.ApplyColorMap(gray, plot.ColormapJet)
	lo := color.AtPixel(0, 0)
	hi := color.AtPixel(0, 255)
	fmt.Printf("0 -> %v, 255 -> %v\n", lo, hi)
```

```
0 -> [0 0 128], 255 -> [128 0 0]
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyColorMap(gray *cv.Mat, cm Colormap) *cv.Mat` | ApplyColorMap recolours a single-channel image through the given built-in colormap and returns a fresh three-channel RGB `cv.Mat`: output pixel (y,x)… |
| `func ApplyCustomColorMap(gray *cv.Mat, table [][3]uint8) *cv.Mat` | ApplyCustomColorMap recolours a single-channel image through a caller-supplied 256-entry RGB table and returns a fresh three-channel `cv.Mat`. |
| `func Colorize(gray *cv.Mat, cm Colormap) *cv.Mat` | Colorize recolours a single-channel image through any built-in `Colormap` (original or additional) and returns a fresh three-channel RGB `cv.Mat`. |
| `func ColormapTable(cm Colormap) [][3]uint8` | ColormapTable returns the 256-entry lookup table for the given built-in colormap. |
| `func DrawLegend(m *cv.Mat, entries []LegendEntry, x, y, scale int, textColor, backgroundColor cv.Scalar)` | DrawLegend paints a boxed legend onto m with its top-left corner at (x, y). |
| `func LUT(img *cv.Mat, table []uint8) *cv.Mat` | LUT applies a per-sample lookup table to every sample of img and returns a fresh `cv.Mat` of the same shape, mirroring cv2.LUT: output sample s… |
| `func Table(cm Colormap) [][3]uint8` | Table returns the 256-entry RGB lookup table for any built-in `Colormap`, including both the original maps served by `ColormapTable` and the… |
| `func TextHeight(scale int) int` | TextHeight returns the pixel height `cv.PutText` uses at the given scale (7*scale). |
| `func TextSize(s string, scale int) (w, h int)` | TextSize returns the pixel width and height that `cv.PutText` uses to render s at the given integer scale, matching the built-in 5×7 bitmap font. |

### Types

| Type | What it is |
| --- | --- |
| `Annotation` | Annotation collects the axis labels, title, numeric tick marks and legend that `Plot2D.RenderAnnotated` paints over a base `Plot2D` chart. |
| `AreaPlot` | AreaPlot draws a filled area chart: the region between the data curve and the y baseline is flooded with FillColor and the curve itself is stroked in… |
| `BoxPlot` | BoxPlot draws a box-and-whisker chart with one box per data group laid out left to right along a shared value axis. |
| `Colorbar` | Colorbar renders a standalone vertical (or horizontal) colour scale for a `Colormap` onto a fresh three-channel RGB `cv.Mat`. |
| `Colormap` | Colormap selects one of the built-in 256-entry colour lookup tables used by `ApplyColorMap`. |
| `ContourPlot` | ContourPlot draws iso-value contour lines of a 2-D scalar field using the marching-squares algorithm. |
| `ErrorBarPlot` | ErrorBarPlot draws a scatter of (x,y) points each capped with a symmetric vertical error bar of half-height YErr. |
| `HeatmapPlot` | HeatmapPlot renders a dense 2-D scalar field as a false-colour image: each matrix cell is normalised to [0,1] over the data range, mapped through a… |
| `LegendEntry` | LegendEntry is a single labelled colour swatch drawn by `DrawLegend`. |
| `MultiSeriesPlot` | MultiSeriesPlot overlays several `Series` on one set of axes with a shared, auto-fitted coordinate range and an optional legend. |
| `PiePlot` | PiePlot draws a pie chart whose wedge angles are proportional to the supplied values. |
| `Plot2D` | Plot2D describes a 2-D chart of one data series and the styling used to draw it. |
| `PlotKind` | PlotKind selects how a `Plot2D` renders its data series. |
| `Series` | Series is one named data series drawn by `MultiSeriesPlot`. |
| `StemPlot` | StemPlot draws a stem (lollipop) chart: a vertical stem rises from the y baseline to each sample and a filled circular marker caps it. |
| `StepPlot` | StepPlot draws a step (staircase) chart: successive samples are connected by a horizontal segment held at the earlier value followed by a vertical… |
| `ViolinPlot` | ViolinPlot draws a violin chart: each group becomes a vertical, symmetric kernel-density silhouette whose half-width is proportional to the estimated… |

<details>
<summary><code>AreaPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAreaPlot(x, y []float64) *AreaPlot` | NewAreaPlot builds an `AreaPlot` from explicit x and y series with default styling. |
| `func (p *AreaPlot) Render() *cv.Mat` | Render draws the area chart onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *AreaPlot) SetFillColor(c cv.Scalar) *AreaPlot` | SetFillColor sets the area fill colour and returns the plot for chaining. |
| `func (p *AreaPlot) SetLineColor(c cv.Scalar) *AreaPlot` | SetLineColor sets the outline colour and returns the plot for chaining. |
| `func (p *AreaPlot) SetShowGrid(on bool) *AreaPlot` | SetShowGrid toggles the grid and returns the plot for chaining. |
| `func (p *AreaPlot) SetSize(w, h int) *AreaPlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

<details>
<summary><code>BoxPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBoxPlot(groups [][]float64) *BoxPlot` | NewBoxPlot builds a `BoxPlot` over the given groups with default styling. |
| `func (p *BoxPlot) Render() *cv.Mat` | Render draws the box plot onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *BoxPlot) SetLabels(labels []string) *BoxPlot` | SetLabels sets the per-group labels and returns the plot for chaining. |
| `func (p *BoxPlot) SetShowGrid(on bool) *BoxPlot` | SetShowGrid toggles the horizontal grid and returns the plot for chaining. |
| `func (p *BoxPlot) SetSize(w, h int) *BoxPlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

<details>
<summary><code>Colorbar</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewColorbar(cm Colormap, width, height int) *Colorbar` | NewColorbar builds a vertical `Colorbar` of the given pixel size for cm with default styling. |
| `func (c *Colorbar) Render() *cv.Mat` | Render draws the colorbar onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (c *Colorbar) SetHorizontal(on bool) *Colorbar` | SetHorizontal orients the colorbar horizontally (low at left) and returns it for chaining. |

</details>

<details>
<summary><code>ContourPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewContourPlot(data [][]float64) *ContourPlot` | NewContourPlot builds a `ContourPlot` over a rectangular field with default styling. |
| `func (p *ContourPlot) Render() *cv.Mat` | Render draws the contour lines onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *ContourPlot) SetFillBelow(on bool) *ContourPlot` | SetFillBelow toggles the colormap background tint and returns the plot for chaining. |
| `func (p *ContourPlot) SetLevels(levels []float64) *ContourPlot` | SetLevels sets explicit iso-values and returns the plot for chaining. |
| `func (p *ContourPlot) SetSize(w, h int) *ContourPlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

<details>
<summary><code>ErrorBarPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewErrorBarPlot(x, y, yErr []float64) *ErrorBarPlot` | NewErrorBarPlot builds an `ErrorBarPlot`. |
| `func (p *ErrorBarPlot) Render() *cv.Mat` | Render draws the error-bar chart onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *ErrorBarPlot) SetRangeY(min, max float64) *ErrorBarPlot` | SetRangeY pins the y axis bounds and returns the plot for chaining. |
| `func (p *ErrorBarPlot) SetShowGrid(on bool) *ErrorBarPlot` | SetShowGrid toggles the grid and returns the plot for chaining. |
| `func (p *ErrorBarPlot) SetSize(w, h int) *ErrorBarPlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

<details>
<summary><code>HeatmapPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHeatmapPlot(data [][]float64) *HeatmapPlot` | NewHeatmapPlot builds a `HeatmapPlot` over a rectangular field with default styling. |
| `func (p *HeatmapPlot) Render() *cv.Mat` | Render draws the heatmap onto a fresh three-channel RGB `cv.Mat` and returns it. |
| `func (p *HeatmapPlot) SetCellSize(s int) *HeatmapPlot` | SetCellSize sets the per-cell pixel size (minimum 1) and returns the plot for chaining. |
| `func (p *HeatmapPlot) SetColormap(cm Colormap) *HeatmapPlot` | SetColormap sets the colour lookup table and returns the plot for chaining. |
| `func (p *HeatmapPlot) SetRange(min, max float64) *HeatmapPlot` | SetRange pins the colour-mapped value range (disabling auto-ranging) and returns the plot for chaining. |
| `func (p *HeatmapPlot) SetShowColorbar(on bool) *HeatmapPlot` | SetShowColorbar toggles the colorbar and returns the plot for chaining. |

</details>

<details>
<summary><code>MultiSeriesPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMultiSeriesPlot() *MultiSeriesPlot` | NewMultiSeriesPlot builds an empty `MultiSeriesPlot` with default styling. |
| `func (p *MultiSeriesPlot) Add(name string, x, y []float64, kind PlotKind) *MultiSeriesPlot` | Add appends a series with the given name, data and kind, assigning it the next colour from the default palette. |
| `func (p *MultiSeriesPlot) AddSeries(s Series) *MultiSeriesPlot` | AddSeries appends a fully specified `Series` (its Color and sizes are used as-is) and returns the plot for chaining. |
| `func (p *MultiSeriesPlot) Render() *cv.Mat` | Render draws every series onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *MultiSeriesPlot) SetRangeX(min, max float64) *MultiSeriesPlot` | SetRangeX pins the x axis bounds and returns the plot for chaining. |
| `func (p *MultiSeriesPlot) SetRangeY(min, max float64) *MultiSeriesPlot` | SetRangeY pins the y axis bounds and returns the plot for chaining. |
| `func (p *MultiSeriesPlot) SetShowGrid(on bool) *MultiSeriesPlot` | SetShowGrid toggles the grid and returns the plot for chaining. |
| `func (p *MultiSeriesPlot) SetShowLegend(on bool) *MultiSeriesPlot` | SetShowLegend toggles the legend and returns the plot for chaining. |
| `func (p *MultiSeriesPlot) SetSize(w, h int) *MultiSeriesPlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

<details>
<summary><code>PiePlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewPiePlot(values []float64) *PiePlot` | NewPiePlot builds a `PiePlot` with default styling. |
| `func (p *PiePlot) Render() *cv.Mat` | Render draws the pie chart onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *PiePlot) SetColors(colors []cv.Scalar) *PiePlot` | SetColors sets the wedge colour cycle and returns the plot for chaining. |
| `func (p *PiePlot) SetLabels(labels []string) *PiePlot` | SetLabels sets the legend labels and returns the plot for chaining. |
| `func (p *PiePlot) SetSize(w, h int) *PiePlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

<details>
<summary><code>Plot2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BarPlot(values []float64) *Plot2D` | BarPlot builds a bar `Plot2D` from a slice of values, one bar per value with x taken as the value index. |
| `func CreatePlot(x, y []float64) *Plot2D` | CreatePlot builds a line `Plot2D` from explicit x and y series. |
| `func CreatePlotY(y []float64) *Plot2D` | CreatePlotY builds a line `Plot2D` from y values alone, using the sample index (0,1,2,...) as x. |
| `func HistogramPlot(data []float64, bins int) *Plot2D` | HistogramPlot bins data into the given number of equal-width bins spanning the data range and returns a bar `Plot2D` whose bar heights are the bin… |
| `func LinePlot(x, y []float64) *Plot2D` | LinePlot builds a `Plot2D` of kind `KindLine` from x and y series. |
| `func ScatterPlot(x, y []float64) *Plot2D` | ScatterPlot builds a `Plot2D` of kind `KindScatter` from x and y series. |
| `func (p *Plot2D) Render() *cv.Mat` | Render draws the configured chart onto a freshly allocated three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *Plot2D) RenderAnnotated(a Annotation) *cv.Mat` | RenderAnnotated renders the base `Plot2D` with `Plot2D.Render` and then paints the annotations from a onto the same canvas, returning it. |
| `func (p *Plot2D) SetAxisColor(c cv.Scalar) *Plot2D` | SetAxisColor sets the axis-line colour and returns the plot for chaining. |
| `func (p *Plot2D) SetBackgroundColor(c cv.Scalar) *Plot2D` | SetBackgroundColor sets the canvas fill colour and returns the plot for chaining. |
| `func (p *Plot2D) SetGridColor(c cv.Scalar) *Plot2D` | SetGridColor sets the gridline colour and returns the plot for chaining. |
| `func (p *Plot2D) SetKind(k PlotKind) *Plot2D` | SetKind sets the rendering style and returns the plot for chaining. |
| `func (p *Plot2D) SetLineColor(c cv.Scalar) *Plot2D` | SetLineColor sets the colour of the plotted line (and bar fill) and returns the plot for chaining. |
| `func (p *Plot2D) SetLineThickness(t int) *Plot2D` | SetLineThickness sets the line/bar-outline width in pixels (minimum 1) and returns the plot for chaining. |
| `func (p *Plot2D) SetPointColor(c cv.Scalar) *Plot2D` | SetPointColor sets the colour of scatter markers and returns the plot for chaining. |
| `func (p *Plot2D) SetPointRadius(r int) *Plot2D` | SetPointRadius sets the scatter marker radius in pixels (minimum 1) and returns the plot for chaining. |
| `func (p *Plot2D) SetRangeX(min, max float64) *Plot2D` | SetRangeX pins the x axis bounds (disabling auto-ranging on x). |
| `func (p *Plot2D) SetRangeY(min, max float64) *Plot2D` | SetRangeY pins the y axis bounds (disabling auto-ranging). |
| `func (p *Plot2D) SetShowGrid(on bool) *Plot2D` | SetShowGrid toggles the gridlines and returns the plot for chaining. |
| `func (p *Plot2D) SetSize(width, height int) *Plot2D` | SetSize sets the canvas dimensions in pixels and returns the plot for chaining. |

</details>

<details>
<summary><code>StemPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStemPlot(x, y []float64) *StemPlot` | NewStemPlot builds a `StemPlot` from explicit x and y series with default styling. |
| `func (p *StemPlot) Render() *cv.Mat` | Render draws the stem chart onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *StemPlot) SetLineColor(c cv.Scalar) *StemPlot` | SetLineColor sets the stem colour and returns the plot for chaining. |
| `func (p *StemPlot) SetRangeY(min, max float64) *StemPlot` | SetRangeY pins the y axis bounds and returns the plot for chaining. |
| `func (p *StemPlot) SetShowGrid(on bool) *StemPlot` | SetShowGrid toggles the grid and returns the plot for chaining. |
| `func (p *StemPlot) SetSize(w, h int) *StemPlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

<details>
<summary><code>StepPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStepPlot(x, y []float64) *StepPlot` | NewStepPlot builds a `StepPlot` from explicit x and y series with default styling. |
| `func (p *StepPlot) Render() *cv.Mat` | Render draws the step chart onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *StepPlot) SetLineColor(c cv.Scalar) *StepPlot` | SetLineColor sets the step-line colour and returns the plot for chaining. |
| `func (p *StepPlot) SetLineThickness(t int) *StepPlot` | SetLineThickness sets the step-line width in pixels (minimum 1) and returns the plot for chaining. |
| `func (p *StepPlot) SetShowGrid(on bool) *StepPlot` | SetShowGrid toggles the grid and returns the plot for chaining. |
| `func (p *StepPlot) SetSize(w, h int) *StepPlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

<details>
<summary><code>ViolinPlot</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewViolinPlot(groups [][]float64) *ViolinPlot` | NewViolinPlot builds a `ViolinPlot` over the given groups with default styling. |
| `func (p *ViolinPlot) Render() *cv.Mat` | Render draws the violin plot onto a fresh three-channel RGB `cv.Mat` of size Height×Width and returns it. |
| `func (p *ViolinPlot) SetLabels(labels []string) *ViolinPlot` | SetLabels sets the per-group labels and returns the plot for chaining. |
| `func (p *ViolinPlot) SetShowGrid(on bool) *ViolinPlot` | SetShowGrid toggles the horizontal grid and returns the plot for chaining. |
| `func (p *ViolinPlot) SetSize(w, h int) *ViolinPlot` | SetSize sets the canvas dimensions and returns the plot for chaining. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/plot).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.

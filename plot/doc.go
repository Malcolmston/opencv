// Package plot provides 2-D data plotting and intensity colormaps for the
// stdlib-only OpenCV port github.com/malcolmston/opencv (imported here as cv).
//
// It mirrors two pieces of OpenCV functionality that both render or recolour a
// [cv.Mat]:
//
//   - the cv::plot module — turning a series of numbers into a line, scatter,
//     bar or histogram chart drawn onto an image canvas; and
//   - cv2.applyColorMap — mapping the intensities of a single-channel image
//     through a 256-entry colour lookup table to produce a false-colour image.
//
// Everything is built on the root package's drawing primitives ([cv.Line],
// [cv.Rectangle], [cv.Circle], [cv.PutText], [cv.Polylines]) and the Go
// standard library (math). The package imports no sibling cv/* subpackages and
// uses no cgo or third-party code.
//
// # Plotting
//
// A chart is described by a [Plot2D] value. Construct one from data with
// [CreatePlot] (explicit x and y), [CreatePlotY] (y only, x is the sample
// index) or one of the kind-specific constructors [LinePlot], [ScatterPlot],
// [BarPlot] and [HistogramPlot]. Each constructor returns a *Plot2D with sane
// defaults that you can further tune with the chainable Set* builder methods,
// then call [Plot2D.Render] to draw the chart onto a freshly allocated
// three-channel [cv.Mat]:
//
//	img := plot.CreatePlot(xs, ys).
//		SetSize(640, 480).
//		SetLineColor(cv.NewScalar(255, 0, 0)).
//		Render()
//
// Data coordinates are mapped into a rectangular plot area inset from the
// canvas edges by the four Margin* fields. The x axis spans [MinX, MaxX] and
// the y axis spans [MinY, MaxY]; by default these bounds are derived from the
// data, but they can be pinned with [Plot2D.SetRangeX] and [Plot2D.SetRangeY].
// The y axis is drawn increasing upward, so larger data values map to smaller
// row indices. The exact mapping is documented on [Plot2D.Render] and is
// deterministic, which makes rendered charts straightforward to test.
//
// [Plot2D.RenderAnnotated] renders a Plot2D and overlays an [Annotation]: a
// centred title, x/y axis labels, numeric tick marks and an optional in-plot
// legend, all painted into the reserved margins.
//
// # Additional chart types
//
// Beyond [Plot2D], the package offers specialised renderers, each constructed
// with its own New* function, tuned with chainable Set* methods and drawn with
// Render onto a fresh three-channel [cv.Mat]:
//
//   - [StemPlot] — a stem (lollipop) chart of vertical stems capped by markers;
//   - [StepPlot] — a staircase line chart;
//   - [AreaPlot] — a filled area between the curve and the baseline;
//   - [BoxPlot] — box-and-whisker summaries (quartiles, Tukey whiskers,
//     outliers) for one or more groups;
//   - [ViolinPlot] — mirrored kernel-density silhouettes per group;
//   - [ErrorBarPlot] — points with symmetric vertical error bars;
//   - [PiePlot] — a pie chart with proportional wedges and a legend;
//   - [MultiSeriesPlot] — several [Series] overlaid on shared axes with a legend;
//   - [HeatmapPlot] — a 2-D scalar field as a false-colour grid with an optional
//     [Colorbar]-style colour scale;
//   - [ContourPlot] — marching-squares iso-value contour lines of a field;
//   - [Colorbar] — a standalone vertical or horizontal colour scale.
//
// Shared helpers [TextSize], [TextHeight], [DrawLegend] and the [LegendEntry]
// type support titles, labels and legends built on [cv.PutText].
//
// # Colormaps
//
// [ApplyColorMap] recolours a single-channel image through one of the original
// built-in [Colormap] tables (JET, HOT, COOL, BONE, HSV, VIRIDIS, PLASMA and
// GRAYSCALE), returning a three-channel image. [ApplyCustomColorMap] does the
// same with a caller-supplied 256-entry table, [ColormapTable] exposes those
// original tables, and [LUT] applies an arbitrary per-sample lookup table to an
// image of any channel count, mirroring cv2.LUT.
//
// The additional OpenCV COLORMAP_* tables — [ColormapAutumn], [ColormapWinter],
// [ColormapSummer], [ColormapSpring], [ColormapOcean], [ColormapRainbow],
// [ColormapPink], [ColormapParula], [ColormapMagma], [ColormapInferno],
// [ColormapCividis], [ColormapTwilight] and [ColormapTurbo] — round out the set.
// Use [Table] to obtain the 256-entry lookup table for any colormap (original or
// additional) and [Colorize] to apply any of them to a single-channel image.
//
// # Colour convention
//
// Consistent with [cv.Mat], a three-channel image produced by this package is
// RGB: channel 0 is red, 1 is green and 2 is blue. (OpenCV's applyColorMap
// instead returns BGR; swap channels with cv.CvtColor if you need that order.)
// A [cv.Scalar] colour passed to the Set*Color methods is likewise interpreted
// as (R, G, B).
//
// # Determinism
//
// Rendering and colormapping are pure functions of their inputs: given the same
// data and configuration they paint identical pixels every time. There is no
// hidden global state and no randomness.
package plot

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
// # Colormaps
//
// [ApplyColorMap] recolours a single-channel image through one of the built-in
// [Colormap] tables (JET, HOT, COOL, BONE, HSV, VIRIDIS, PLASMA and GRAYSCALE),
// returning a three-channel image. [ApplyCustomColorMap] does the same with a
// caller-supplied 256-entry table, [ColormapTable] exposes the built-in tables,
// and [LUT] applies an arbitrary per-sample lookup table to an image of any
// channel count, mirroring cv2.LUT.
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

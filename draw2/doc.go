// Package draw2 provides a self-contained 2-D drawing and rendering toolkit
// for the parent computer-vision library's Mat image type.
//
// It complements the core drawing primitives with higher-quality and higher
// level routines: Wu anti-aliased lines and circles, thick and dashed lines,
// polylines, filled and outlined polygons, midpoint circles, axis-aligned and
// rotated ellipses, elliptical arcs, quadratic and cubic Bezier curves,
// Catmull-Rom splines, a built-in bitmap text renderer, CV-style markers,
// contour overlays, false-colour heatmaps and alpha compositing.
//
// All routines operate on *[github.com/malcolmston/opencv.Mat]. Colours are
// expressed with the parent package's [github.com/malcolmston/opencv.Scalar]
// type (interpreted as R, G, B, A); integer image coordinates use
// [github.com/malcolmston/opencv.Point]. Drawing clips silently to the image
// bounds, so callers never need to pre-check coordinates.
//
// Everything in this package is deterministic and implemented with the Go
// standard library only. Functions do not allocate global state and are safe
// to call concurrently on distinct Mats.
package draw2

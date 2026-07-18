// Package contours2 provides contour extraction and shape analysis for the
// github.com/malcolmston/opencv library.
//
// It operates on the library's existing image type, [cv.Mat], and its geometry
// types [cv.Point] and [cv.Point2f]; it does not introduce a competing image
// representation. Binary or grayscale single-channel Mats are the raster input,
// and contours are represented as ordered slices of [cv.Point] (the [Contour]
// type), matching the parent package's conventions.
//
// The package is pure Go and depends only on the standard library and the
// parent cv package. It is deterministic and CPU-only. The following families
// of routines are implemented:
//
//   - Border following: [FindContours] and [FindExternalContours] implement the
//     Suzuki–Abe border-following algorithm with full parent/child/sibling
//     hierarchy reconstruction ([RetrExternal], [RetrList], [RetrCComp],
//     [RetrTree]) and chain approximation ([ChainApproxNone],
//     [ChainApproxSimple]).
//   - Polyline simplification: [ApproxPolyDP] (Ramer–Douglas–Peucker).
//   - Convexity: [ConvexHull], [ConvexHullIndices], [IsContourConvex] and
//     [ConvexityDefects].
//   - Enclosing shapes: [MinAreaRect] (rotating calipers), [MinEnclosingCircle]
//     (Welzl), [BoundingRect], [FitEllipse] and [FitLine].
//   - Measurement: [ArcLength], [ContourArea], [ContourAreaSigned],
//     [PointPolygonTest] and the [ExtremePoints] helper.
//   - Moments: [ContourMoments] (Green's theorem), [ImageMoments],
//     [Moments.HuMoments], [HuMoments] and [MatchShapes] / [MatchShapesHu].
//   - Descriptors: [AspectRatio], [Extent], [Solidity], [EquivalentDiameter],
//     [Orientation] and [Eccentricity].
//
// Conventions follow OpenCV where practical. Coordinates use image conventions
// (x is the column, y is the row, y increasing downward), so the sign of a
// signed area is negated relative to a mathematical y-up plane; area-returning
// functions here return non-negative magnitudes unless documented otherwise.
package contours2

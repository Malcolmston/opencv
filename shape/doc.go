// Package shape provides shape descriptors, fitting and matching built on top
// of the standard-library-only OpenCV port github.com/malcolmston/opencv
// (imported here as cv). It mirrors a useful subset of OpenCV's shape-analysis
// routines that operate on point sets and contours rather than on raster
// images.
//
// # Overview
//
// The package groups these kinds of operation:
//
//   - Enclosing shapes. [MinEnclosingCircle] finds the smallest circle covering
//     a point set using Welzl's algorithm, and [MinEnclosingTriangle] finds a
//     small enclosing triangle by optimising supporting lines around the convex
//     hull.
//   - Fitting. [FitLine] fits a line to a point set by total least squares (the
//     first principal component of the centred points), [FitLineRobust] refits
//     it with an M-estimator ([DistL1], [DistHuber], [DistFair], [DistWelsch]…)
//     so a few outliers no longer dominate, and [FitEllipse] fits an ellipse
//     with the direct algebraic least-squares method of Fitzgibbon, reformulated
//     for numerical stability by Halir and Flusser.
//   - Moments and invariants. [ContourMoments] computes the geometric moments of
//     a closed polygon via Green's theorem, [HuMoments] derives the seven
//     Hu invariants (invariant to translation, scale and rotation) from a
//     [cv.Moments] value, and [MatchShapes] compares two contours through their
//     Hu moments.
//   - Convexity and predicates. [ConvexityDefects] reports the notches of a
//     contour relative to its convex hull, [ConvexHullIndices] returns the hull
//     as indices into the input, [IsContourConvex] tests convexity,
//     [PointPolygonTest] locates a point relative to a polygon (with an optional
//     signed distance), and [RotatedRectangleIntersection] computes the overlap
//     polygon of two rotated rectangles.
//   - Shape matching. [ShapeContextDistanceExtractor] measures the dissimilarity
//     of two contours with the shape-context pipeline of Belongie, Malik and
//     Puzicha — log-polar [ShapeContext] histograms, optimal correspondence by
//     the Hungarian algorithm ([SolveAssignment]) and a thin-plate-spline
//     bending-energy term — while [HausdorffDistanceExtractor] uses the
//     (partial) Hausdorff distance. [EMDL1] gives the earth mover's distance
//     between one-dimensional histograms.
//   - Geometric transforms. The [ShapeTransformer] interface is implemented by
//     [ThinPlateSplineShapeTransformer] (a minimum-bending-energy warp that
//     interpolates control-point correspondences) and [AffineTransformer]
//     (a full-affine or similarity fit); both estimate, apply and warp images.
//
// # Coordinates and conventions
//
// Points follow the parent package's image convention: x is the column and y is
// the row, with the origin at the top-left and y growing downward. Functions
// that return fractional positions use plain float64 pairs. Angles are reported
// in degrees to match [cv.RotatedRect].
//
// All routines are deterministic: the same input always yields the same result.
// Where an underlying algorithm is classically randomised (Welzl's minimal
// enclosing circle), a fixed deterministic permutation is used so results are
// reproducible.
//
// # Errors and panics
//
// These functions validate their arguments and panic with a descriptive message
// on programmer error (for example an empty point set where at least one point
// is required), mirroring the parent package's style. Degenerate but valid
// inputs — fewer than three points, all-collinear points — are handled by
// returning a sensible degenerate result rather than panicking.
package shape

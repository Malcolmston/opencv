// Package shapefit provides geometric primitive fitting and Hough-transform
// shape detection for point sets and binary edge images.
//
// The package builds on the parent library's core types — it consumes binary
// edge images as [github.com/malcolmston/opencv.Mat] and represents point sets
// as slices of [github.com/malcolmston/opencv.Point2f] (floating-point image
// coordinates, x is the column and y is the row). Fitted results are returned
// as the small value types defined here ([Line], [Circle], [Ellipse]) which
// interoperate with the parent library where natural: an [Ellipse] converts to
// and from [github.com/malcolmston/opencv.RotatedRect].
//
// # What it covers
//
// Least-squares fitting: [FitLine] (total least squares), [FitCircle] (Kåsa
// algebraic fit) and [FitCircleTaubin] (Taubin's bias-corrected fit), and
// [FitEllipse] (the Halíř–Flusser numerically-stable form of Fitzgibbon's
// direct least-squares ellipse fit).
//
// Robust fitting: [RANSACLine], [RANSACCircle] and [RANSACEllipse] fit a single
// primitive in the presence of outliers and report the inlier set;
// [DetectLines] and [DetectCircles] extract multiple primitives by sequential
// RANSAC.
//
// Hough transforms: [HoughLines] and [HoughLinesP] (standard and
// segment-extracting line detection), [HoughCircles] (circle detection over a
// radius range) and the Ballard [GeneralizedHough] transform for arbitrary
// shapes described by a template.
//
// Higher-level detection: [DetectEllipses] (the Xie–Ji accumulator method),
// [DetectRectangles] (corner-pairing rectangle detection),
// [DetectReflectionSymmetry] and [DetectRotationalSymmetry], and
// [FitBestPrimitive] which selects the primitive that best explains a point
// set.
//
// # Conventions
//
// Everything here is written against the Go standard library only — no cgo and
// no third-party dependencies — and is deterministic: the randomized routines
// (RANSAC) draw from a seed carried in [RANSACParams], so repeated calls with
// the same input and seed return identical results. Angles are in radians
// unless a doc comment says otherwise. A [Line] is stored in normalized normal
// form a·x + b·y + c = 0 with a² + b² = 1.
package shapefit

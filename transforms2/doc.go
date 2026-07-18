// Package transforms2 implements pure-Go geometric image transformations for
// the parent cv package, operating on its [cv.Mat] (8-bit, channel-interleaved)
// and [cv.FloatMat] types.
//
// The package deliberately reuses the parent package's image and matrix types,
// its [cv.AffineMatrix] (a 2x3 forward transform stored row-major) and
// [cv.PerspectiveMatrix] (a 3x3 homography stored row-major), and its point
// types [cv.Point] and [cv.Point2f] so that results interoperate with the rest
// of the library. It adds resampling quality (nearest, bilinear and bicubic),
// configurable border handling, and a broad set of geometric operations:
//
//   - Affine and projective warps with builders, composition and inversion.
//   - High-level helpers: resize, rotate (in-place or bound), scale, translate
//     and shear.
//   - Arbitrary remapping via coordinate maps or a mapping function.
//   - Polar and log-polar (un)warping.
//   - Piecewise-affine warping with Delaunay triangulation.
//   - Thin-plate-spline (TPS) warping for smooth non-rigid deformation.
//   - Brown-Conrady lens distortion modelling and (un)distortion.
//   - Enhanced Correlation Coefficient (ECC) image registration.
//
// All transforms use inverse mapping (each destination pixel is sampled from
// the source) and are fully deterministic. Coordinates follow the convention
// (x = column, y = row) with pixel centres at integer coordinates. Every warp
// accepts an [Interpolation] mode and a [BorderMode] with an associated
// constant fill value.
//
// The package depends only on the Go standard library.
package transforms2

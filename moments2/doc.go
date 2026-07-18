// Package moments2 provides image moments and shape descriptors for the
// github.com/malcolmston/opencv library.
//
// The package operates on the library's existing image type, [cv.Mat], and its
// geometry types [cv.Point] and [cv.Point2f]; it does not introduce a competing
// image representation. Functions accept single-channel Mats (grayscale or
// binary masks) and polygonal contours ([]cv.Point) as produced by the parent
// package's contour extraction.
//
// The following families of descriptors are implemented, all in pure Go with
// no external dependencies:
//
//   - Raw, central and normalized central moments of arbitrary order, computed
//     from a raster image ([ImageMoments], [MaskMoments]) or from a polygon via
//     Green's theorem ([ContourMoments]).
//   - Hu's seven rotation, scale and translation invariant moments
//     ([HuMoments]) and OpenCV-compatible shape matching ([MatchShapes]).
//   - Region shape descriptors: eccentricity, elongation, orientation,
//     solidity, extent, circularity, aspect ratio and more.
//   - Zernike and pseudo-Zernike moments on the unit disk.
//   - Legendre moments on the unit square.
//   - Flusser & Suk affine moment invariants.
//   - Fourier descriptors of a closed contour with normalization and
//     reconstruction.
//   - Shape context histograms and their matching cost.
//   - Convex hull indices, convexity defects and convexity tests.
//
// All routines are deterministic and CPU-only.
package moments2

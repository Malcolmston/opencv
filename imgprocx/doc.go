// Package imgprocx provides extended geometric and frequency-domain image
// processing on top of the standard-library-only OpenCV port
// github.com/malcolmston/opencv (imported here as cv). It complements the root
// package's geometric transforms ([cv.WarpAffine], [cv.WarpPerspective],
// [cv.Resize]) and linear filters ([cv.Filter2D], [cv.GaussianBlur]) with
// routines OpenCV groups under imgproc's "geometric image transformations",
// "structural analysis" and "motion analysis" sections.
//
// # Affine estimation
//
// [GetAffineTransform] solves the exact 2×3 affine transform that maps three
// source points to three destinations, mirroring cv2.getAffineTransform. When
// correspondences are noisy and contaminated by outliers, [EstimateAffine2D]
// (six degrees of freedom) and [EstimateAffinePartial2D] (four degrees of
// freedom: rotation, a single uniform scale and translation) fit a robust model
// with a deterministic RANSAC loop followed by a least-squares refit over the
// inliers. All three return a plain [2][3]float64 (or the compatible
// [cv.AffineMatrix]); [ApplyAffine] evaluates such a matrix at a point and
// [ToAffineMatrix]/[FromAffineMatrix] convert between the two representations so
// the result can be handed straight to [cv.WarpAffine].
//
// # Integral images
//
// [IntegralImage] builds the summed-area table (and the table of squared sums)
// of an image, so the sum of pixel intensities over any axis-aligned rectangle
// can be read in constant time from four corner lookups. A pixel's intensity is
// the sum of its channel samples, so the tables are single-channel regardless of
// the input's channel count.
//
// # Sub-pixel refinement
//
// [CornerSubPix] refines integer corner locations to sub-pixel accuracy using
// the standard gradient-orthogonality iteration: the refined corner is the
// point that minimises the weighted squared dot product between the local image
// gradient and the vector from the corner to each neighbour.
//
// # Gabor kernels
//
// [GetGaborKernel] builds a Gabor filter kernel — a sinusoid modulated by a
// Gaussian envelope — as a [cv.Kernel] ready for [cv.Filter2D]. Gabor kernels
// are oriented band-pass filters widely used for texture analysis and edge
// detection at a chosen scale and orientation.
//
// # Phase correlation
//
// [PhaseCorrelate] estimates the translational shift between two same-sized
// images from the phase of their cross-power spectrum, computed with a small
// discrete Fourier transform. It recovers integer shifts exactly on circularly
// shifted inputs and reports a response in [0,1] measuring peak sharpness.
//
// # Polar transforms
//
// [LinearPolar] and [LogPolar] remap an image from Cartesian coordinates into a
// polar (radius, angle) or log-polar (log-radius, angle) grid about a chosen
// centre, the classic pre-processing step that turns rotation and scaling about
// that centre into translation.
//
// # Conventions and determinism
//
// Coordinates follow the cv convention: x is the column and y is the row, with
// the origin at the top-left. Sub-pixel positions are reported as [Point2f].
// Every function in this package is deterministic — the RANSAC estimators seed a
// private pseudo-random generator from a fixed constant and perform no
// concurrent work — so identical inputs always yield identical output. The
// package depends only on the Go standard library and the root cv package; it
// imports no sibling cv/* subpackages.
package imgprocx

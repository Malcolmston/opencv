// Package histogram2 provides a pure-Go, standard-library-only toolkit of
// classic histogram-based computer-vision routines built on top of the parent
// package's [cv.Mat] image type.
//
// The package never defines its own image type. Every routine that produces an
// image returns a [cv.Mat], and every routine that consumes one accepts a
// [cv.Mat], so results interoperate directly with the rest of the module
// (drawing, morphology, thresholding, I/O and so on). The only new types it
// introduces are small value-holders for the histograms themselves —
// [Histogram1D], [Histogram2D] and [Histogram3D] — plus the stateful
// [CLAHE] and [HOGDescriptor] operators.
//
// The functionality is organised in a few groups:
//
//   - Histogram construction: [CalcHist1D], [CalcHist1DMasked], [CalcHist2D]
//     and [CalcHist3D], together with the histogram types and their methods
//     ([Histogram1D.Normalize], [Histogram1D.Cumulative],
//     [Histogram1D.Entropy] and friends).
//   - Equalisation: global [EqualizeHist], per-channel
//     [EqualizeHistPerChannel], luminance-only [EqualizeHistLuminance] and the
//     adaptive [CLAHE] operator.
//   - Matching and specification: [MatchHistograms], [MatchHistogramsColor],
//     [SpecifyHistogram] and [BuildSpecificationLUT].
//   - Back-projection: [CalcBackProject1D] and [CalcBackProject2D].
//   - Comparison metrics: [CompareHist1D], [CompareHist2D] and the standalone
//     [Correlation], [ChiSquare], [ChiSquareAlt], [Intersection],
//     [Bhattacharyya], [KLDivergence] and [EMD1D] measures.
//   - Contrast enhancement: [ContrastStretch], [ContrastStretchRange],
//     [MinMaxStretch] and [GammaCorrect].
//   - Histogram of Oriented Gradients: [HOGDescriptor] and the [HOG]
//     convenience wrapper.
//
// All routines are deterministic, CPU-only and free of third-party
// dependencies.
package histogram2

import "errors"

// ErrEmptyImage is returned by routines that are given a nil or zero-sized
// [cv.Mat].
var ErrEmptyImage = errors.New("histogram2: empty image")

// ErrChannelRange is returned when a channel index is negative or not less
// than the image's channel count.
var ErrChannelRange = errors.New("histogram2: channel index out of range")

// ErrBinCount is returned when a requested bin count is not positive.
var ErrBinCount = errors.New("histogram2: bin count must be positive")

// ErrSizeMismatch is returned when two inputs that must share a shape (image
// sizes, histogram lengths) do not.
var ErrSizeMismatch = errors.New("histogram2: size mismatch")

// ErrInvalidArgument is returned for out-of-domain scalar parameters such as a
// non-positive tile grid or an inverted intensity range.
var ErrInvalidArgument = errors.New("histogram2: invalid argument")

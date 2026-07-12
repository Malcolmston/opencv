// Package bgsegm is a from-scratch, standard-library-only port of a useful
// subset of OpenCV's background-segmentation module (bgsegm plus the two
// subtractors that live in the core video module). It separates the moving
// foreground of a frame sequence from the slowly changing background, producing
// a per-pixel foreground mask for each frame.
//
// Like the parent package [github.com/malcolmston/opencv] (imported here under
// the alias cv), bgsegm is written entirely against the Go standard library
// (only math). It uses no cgo and no third-party dependencies, and it does not
// import any of the sibling cv/* subpackages.
//
// # The model
//
// Background subtraction treats a video as a stream of frames of identical
// size. Each subtractor maintains a statistical model of what the background
// looks like at every pixel and, for each incoming frame, decides pixel by
// pixel whether the observation fits that model (background) or deviates from
// it (foreground). The model is then updated with the new observation so that
// gradual changes — lighting drift, a parked car, a new static object — are
// eventually absorbed into the background, while genuinely moving objects keep
// standing out.
//
// Every subtractor satisfies the [BackgroundSubtractor] interface:
//
//	mask := sub.Apply(frame)      // classify this frame
//	bg   := sub.GetBackgroundImage() // the current background estimate
//
// Apply returns a fresh single-channel [cv.Mat] the same size as the frame
// whose samples take one of three values: [BackgroundValue] (0) for background,
// [ForegroundValue] (255) for foreground, and [ShadowValue] (127) for pixels
// that look like the moving shadow of a background object, emitted only when a
// model's DetectShadows option is enabled. Input frames may be single-channel
// (grayscale) or three-channel (RGB); colour frames are reduced to luma
// internally, so masks and background images are always single-channel.
//
// # Models
//
// Four models are provided, mirroring the OpenCV classes of the same name:
//
//   - [BackgroundSubtractorMOG2] — the adaptive Gaussian-mixture model of
//     Zivkovic. Every pixel keeps a small set of weighted Gaussians; each frame
//     the best-matching Gaussian is updated (or a new one spawned), the mixture
//     weights adapt, and the pixel is background when its match belongs to the
//     high-weight "background" Gaussians. Handles multi-modal backgrounds and
//     optional shadow detection.
//   - [BackgroundSubtractorKNN] — a non-parametric K-nearest-neighbours model.
//     Every pixel remembers a bank of recent samples; a pixel is background when
//     enough stored samples lie within a distance threshold of the observation.
//   - [RunningAverage] — the simplest model: an exponential moving-average
//     background image thresholded against the absolute frame difference.
//   - [BackgroundSubtractorGMG] — the Bayesian per-pixel model of Godbehere,
//     Matsukawa and Goldberg. Each pixel builds a decaying histogram of observed
//     values and flags observations whose posterior background probability is
//     too low, after an initial learning period.
//
// # Morphological cleanup
//
// Raw foreground masks are speckled with isolated false positives. Every model
// exposes an OpenKernel field: when set to a positive odd size k it runs a
// morphological opening (erosion then dilation, via [cv.MorphologyEx] with
// [cv.MorphOpen]) over a k×k rectangular structuring element on the mask before
// returning it, removing specks smaller than the kernel. The same operation is
// available directly as [CleanupMask].
//
// # Determinism
//
// None of the models use randomness: given the same sequence of frames and the
// same configuration they always produce identical masks and background images.
// There is no hidden global state; each subtractor value is self-contained and
// lazily sizes itself to the first frame it is given.
package bgsegm

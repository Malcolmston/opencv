// Package saliency is a from-scratch, standard-library-only port of a useful
// subset of OpenCV's contrib saliency module: algorithms that highlight the
// regions of an image (or video) most likely to draw human visual attention.
//
// The package sits on top of the root module github.com/malcolmston/opencv
// (imported as cv) and the Go standard library (only math and math/cmplx). It
// uses no cgo and no third-party dependencies, and it does not import any of
// the other cv/* subpackages. Every detector operates on the package's central
// image type, [cv.Mat] (8-bit unsigned samples, one or three channels), and
// static detectors return a single-channel saliency map normalised to the
// 8-bit range, where brighter samples mark more salient locations.
//
// OpenCV groups saliency algorithms into three families, all represented here.
//
// # Static saliency
//
// Static detectors score a single still image and implement the common
// [StaticSaliency] interface (one ComputeSaliency method):
//
//   - [StaticSaliencySpectralResidual] — the spectral-residual method of Hou &
//     Zhang (2007). It works in the Fourier domain: the log-amplitude spectrum
//     minus its own local average (the "spectral residual") is recombined with
//     the original phase and inverse-transformed, so structure that departs
//     from the smooth natural-image spectrum stands out. A small radix-2 2-D
//     FFT is implemented locally in this package.
//   - [StaticSaliencyFineGrained] — a multi-scale center-surround detector
//     after Montabone & Soto (2010). Absolute pixel-versus-surround-mean
//     differences are gathered over several octave scales with summed-area
//     tables and averaged, filling the interior of sizeable salient regions
//     rather than only their edges.
//
// [ComputeBinaryMap] turns any such saliency map into a binary foreground mask
// with Otsu thresholding, isolating the salient region.
//
// # Motion saliency
//
//   - [MotionSaliencyBinWangApr2014] — a stateful, per-pixel background-model
//     detector after Wang & Dudek (2014). Fed a sequence of frames, it flags
//     the pixels of moving objects. It is a compact, deterministic rendering of
//     the two-model Bin-Wang scheme (single multi-template model, fixed
//     template replacement).
//
// # Objectness
//
//   - [ObjectnessBING] — a lightweight ("BING-lite") objectness proposer after
//     Cheng et al. (2014). It keeps BING's binarised-normed-gradient front end
//     but scores sliding windows with a fixed boundary-contrast heuristic
//     instead of a learned linear model, returning ranked candidate boxes
//     ([ObjectnessBox]) without any training data.
//
// # Determinism
//
// Every detector is fully deterministic: identical input produces identical
// output, with no randomised sampling. This makes results reproducible and
// unit-testable.
//
// # Relationship to OpenCV and deferred features
//
// The algorithms follow their OpenCV counterparts closely enough to exhibit the
// same qualitative behaviour, but two capabilities that depend on trained model
// data are intentionally out of scope:
//
//   - trained deep-learning saliency models (e.g. DeepGaze-style networks); and
//   - the full BING detector with its offline-learned normed-gradient weights
//     and two-stage cascade.
//
// [ObjectnessBING] therefore returns heuristic objectness cues rather than the
// calibrated scores of the trained detector.
package saliency

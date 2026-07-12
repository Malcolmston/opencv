// Package tracking implements single-object (short-term) visual trackers on top
// of the standard-library-only OpenCV port github.com/malcolmston/opencv
// (imported here as cv). It mirrors a useful subset of OpenCV's tracking and
// video modules: a common [Tracker] interface, a template-matching tracker, a
// KCF-lite correlation tracker, a Median-Flow tracker, and the classic
// histogram-driven mean-shift and CamShift trackers together with the low-level
// [MeanShift] and [CamShift] search routines.
//
// # The Tracker interface
//
// Every tracker satisfies [Tracker]: call [Tracker.Init] once with the first
// frame and the object's bounding box, then call [Tracker.Update] on each
// subsequent frame to obtain the new box and a boolean confidence flag. Boxes
// are [cv.Rect] values (integer top-left corner, width and height); frames are
// [cv.Mat] images. Colour frames (3-channel RGB) are converted internally; the
// appearance trackers work on luma, the histogram trackers on hue.
//
// # Trackers
//
// [TrackerTemplate] is the robust baseline: it stores the initial patch and, on
// each frame, slides it over a search window centred on the last position using
// normalised cross-correlation ([cv.MatchTemplate] with [cv.TmCcoeffNormed]) and
// takes the peak ([cv.MinMaxLoc]). It assumes a roughly constant appearance and
// scale.
//
// [TrackerKCF] is a "KCF-lite" correlation tracker. A faithful KCF learns a
// kernelised ridge-regression correlation filter in the Fourier domain; this
// implementation instead approximates it with an online-adapted normalised
// cross-correlation template — the appearance model is blended toward the newly
// located patch each frame (see [TrackerKCF.LearnRate]). It therefore reuses
// [cv.MatchTemplate] rather than a DFT. The approximation keeps the online
// adaptation that distinguishes a correlation tracker from a fixed template, but
// omits the circulant/kernel machinery and Fourier-domain speed-up.
//
// [TrackerMedianFlow] tracks a grid of points inside the box with a locally
// implemented Lucas-Kanade optical flow, measures the forward-backward
// consistency of each point and its appearance (NCC) error, keeps the reliable
// half, and estimates the box motion as the median point displacement and the
// median pairwise-distance ratio (scale). It reports failure when too few points
// survive, which makes it sensitive to occlusion and drift.
//
// # Histogram trackers and search routines
//
// [MeanShift] iterates a fixed-size window to the local mode (centroid) of a
// back-projection probability image. [CamShift] ("Continuously Adaptive
// Mean-Shift") runs mean-shift and then adapts the window size and orientation
// from the second-order image moments, returning a [cv.RotatedRect].
//
// [MeanShiftTracker] and [CamShiftTracker] wrap these into the [Tracker]
// interface: [Tracker.Init] builds a 256-bin hue histogram of the object region
// with [cv.CalcHist], and each [Tracker.Update] back-projects it over the frame
// with [cv.CalcBackProject] and runs the corresponding search from the last
// window.
//
// # Local Lucas-Kanade flow
//
// [TrackerMedianFlow] uses a small single-level iterative Lucas-Kanade solver
// implemented in this package (see the flow helpers): for each point it solves
// the 2×2 gradient-structure system over a window and refines the displacement
// by Newton steps, with bilinear sub-pixel sampling. It handles small
// inter-frame motion (a few pixels, up to roughly the window radius); larger
// motion would need an image pyramid, which is deferred.
//
// # Conventions and determinism
//
// Coordinates follow the cv convention: x is the column and y is the row, with
// the origin at the top-left. All boxes are clamped to the image. Every function
// in this package is deterministic — no randomness or concurrency is used — so
// the same input frames always yield the same tracks.
//
// # Deferred
//
//   - True KCF: kernelised ridge regression correlation filter with an FFT; this
//     package ships an online-NCC approximation instead.
//   - A Fourier-domain (DFT) correlation filter of any kind.
//   - Pyramidal Lucas-Kanade for large inter-frame motion in [TrackerMedianFlow].
//   - Multi-scale search in [TrackerTemplate] and [TrackerKCF] (both track at a
//     fixed box size; only Median-Flow and CamShift adapt scale).
//   - Long-term re-detection / occlusion recovery (e.g. TLD).
package tracking

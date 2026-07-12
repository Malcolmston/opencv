// Package tracking implements single-object (short-term) visual trackers on top
// of the standard-library-only OpenCV port github.com/malcolmston/opencv
// (imported here as cv). It mirrors a useful subset of OpenCV's tracking and
// video modules: a common [Tracker] interface, a template-matching tracker,
// classic histogram-driven mean-shift and CamShift trackers with the low-level
// [MeanShift] and [CamShift] routines, a Median-Flow tracker, and a family of
// modern trackers built on a genuine Fourier-domain correlation-filter core and
// on online-learning detectors:
//
//   - [TrackerMOSSE] — the MOSSE minimum-output-sum-of-squared-error filter.
//   - [TrackerDCF] — a faithful kernelised correlation filter (KCF) with a
//     Gaussian kernel and multi-scale search.
//   - [TrackerKCFHOG] — KCF driven by multi-channel HOG features ([HOGCells]).
//   - [TrackerCSRT] — a channel-and-spatial-reliability DCF.
//   - [TrackerMIL] — multiple-instance-learning boosting over Haar features.
//   - [TrackerBoosting] — online AdaBoost over Haar features.
//   - [TrackerTLD] — tracking-learning-detection with full-frame re-detection.
//   - [MultiTracker] — drives many trackers on one frame at once.
//
// The Fourier-domain machinery is exported in its own right: [ComplexMat] with
// the radix-2 [FFT2] / [IFFT2] transforms, [HannWindow2D], [GaussianResponse],
// [NextPow2] and the [HOGCells] descriptor.
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
// # Per-frame confidence
//
// The learning trackers additionally satisfy [ConfidenceTracker], whose
// UpdateConfidence returns a continuous score alongside the box instead of only
// a boolean: a peak-to-sidelobe ratio (MOSSE, CSRT), a peak correlation response
// (DCF, KCF-HOG), a classifier margin (MIL, Boosting) or a template similarity
// (TLD). Higher always means more reliable within a given tracker.
//
// # Fourier-domain correlation filters
//
// [TrackerMOSSE], [TrackerDCF], [TrackerKCFHOG] and [TrackerCSRT] all normalise
// the object window to a fixed power-of-two model size, preprocess it (log,
// zero-mean, unit-norm, Hann-windowed) and learn a filter in the frequency
// domain with the package's own [FFT2]. MOSSE learns the closed-form MOSSE
// filter; DCF and KCF-HOG learn Henriques's kernelised ridge-regression filter
// with a Gaussian kernel and search several scales per frame; CSRT runs a
// per-channel filter bank weighted by channel reliability and constrained by a
// spatial-reliability foreground mask.
//
// # Online-learning detectors
//
// [TrackerMIL] and [TrackerBoosting] classify generalised Haar features read
// from an integral image. MIL trains a strong classifier by greedy MILBoost
// selection over positive/negative bags; Boosting runs discrete AdaBoost over
// online weak learners. [TrackerTLD] pairs the Median-Flow tracker with a
// nearest-neighbour template detector scanned across the whole frame, letting it
// re-detect the object after occlusion. All three are deterministic given their
// seeds.
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
// The following remain approximated or unimplemented:
//
//   - Pyramidal Lucas-Kanade for large inter-frame motion in [TrackerMedianFlow]
//     (still single-level).
//   - Multi-scale search in [TrackerTemplate] and the legacy online-NCC
//     [TrackerKCF] (both track at a fixed box size; [TrackerDCF] and
//     [TrackerKCFHOG] add the FFT machinery and scale search the older
//     approximations lacked).
//   - A dedicated 1-D scale filter (DSST-style); the KCF trackers estimate scale
//     from a small discrete set of scaled windows, which handles moderate but not
//     large scale change.
//   - Mixed-radix / Bluestein FFT for arbitrary sizes: [FFT2] is radix-2, so the
//     correlation filters resize the object window to a power-of-two model size.
//   - The full CSRT ADMM filter optimisation and per-channel HOG/colour-names
//     features; [TrackerCSRT] is a lite intensity+gradient variant.
package tracking

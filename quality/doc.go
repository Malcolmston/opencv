// Package quality is a from-scratch, standard-library-only port of a useful
// subset of OpenCV's quality module: objective image-quality assessment (IQA)
// metrics.
//
// The package sits on top of the root module github.com/malcolmston/opencv
// (imported as cv) and the Go standard library (only math). It uses no cgo and
// no third-party dependencies, and it does not import any of the other cv/*
// subpackages. Every metric operates on the package's central image type,
// [cv.Mat] (8-bit unsigned samples, one or three channels).
//
// # Full-reference metrics
//
// Full-reference (FR) metrics compare a distorted image against a pristine
// reference of the same size and channel count. They quantify how far the two
// images are from one another:
//
//   - [MSE] and [MAE] — per-channel mean squared / absolute error. Lower is
//     better; identical images score zero.
//   - [PSNR] — peak signal-to-noise ratio in decibels, derived from the pooled
//     MSE. Higher is better; identical images score +Inf.
//   - [SSIM] — structural similarity index (Wang et al. 2004) computed with an
//     11×11 Gaussian window. Returns the mean score in [-1, 1] and a per-pixel
//     quality map; identical images score 1.
//   - [MSSSIM] — multi-scale SSIM, aggregating contrast/structure over an
//     image pyramid.
//   - [GMSD] — gradient magnitude similarity deviation (Xue et al. 2014). Lower
//     is better; identical images score zero.
//   - [UQI] — the universal quality index (Wang & Bovik 2002), the historical
//     predecessor of SSIM, computed over a sliding uniform window.
//
// The structural metrics (SSIM, MSSSIM, GMSD, UQI) operate on luminance: a
// three-channel image is reduced to gray with the BT.601 weights before the
// metric is evaluated, matching common reference implementations. MSE, MAE and
// PSNR operate channel by channel.
//
// # Additional full-reference metrics
//
// The package adds a broader family of full-reference metrics:
//
//   - [RMSE] / [NRMSE] — the (range-normalised) root-mean-squared error, one
//     value per channel. Lower is better; identical images score zero.
//   - [SNR] — signal-to-noise ratio in decibels, pooled over channels. Higher
//     is better; identical images score +Inf.
//   - [SSIMMap] — the per-pixel [SSIM] similarity image on its own.
//   - [VIFP] / [VIF] — visual information fidelity in the pixel domain and over
//     a bandpass pyramid (Sheikh & Bovik 2006). 1 for identical images.
//   - [FSIM] / [FSIMc] — feature-similarity index (phase congruency + gradient
//     magnitude), grayscale and colour (Zhang et al. 2011). 1 for identical.
//   - [IWSSIM] — information-content-weighted multi-scale SSIM (Wang & Li 2011).
//   - [VSI] — visual-saliency-induced index (Zhang et al. 2014).
//   - [CWSSIM] — complex-wavelet SSIM, robust to small shifts (Wang &
//     Simoncelli 2005), via a steerable-pyramid approximation.
//   - [EdgePreservationRatio] — correlation of the two images' Laplacian
//     responses; 1 when edges are perfectly preserved.
//   - [EntropyDiff] — absolute difference of luminance [Entropy]; 0 for equal
//     histograms.
//
// The phase-congruency term of FSIM/FSIMc, the saliency map of VSI and the
// steerable pyramid of CWSSIM are documented approximations of the transforms
// used in the source papers: they preserve the metrics' defining behaviour
// (identical images score best, distortions rank consistently) without an FFT
// or a log-Gabor filter bank. Every metric here is evaluated on luminance
// (using I/Q chrominance in addition, for FSIMc and colour VSI), except the
// per-channel error metrics RMSE, NRMSE and SNR.
//
// # No-reference metrics
//
// No-reference (NR) metrics score a single image with no pristine original.
// They are focus and naturalness heuristics:
//
//   - [Sharpness] / [LaplacianVariance] — the variance of the Laplacian
//     response, the classic passive auto-focus measure. Blurred images score
//     lower than sharp ones.
//   - [Tenengrad] — the mean squared Sobel gradient magnitude, another focus
//     measure.
//   - [BrisqueScore] — a lightweight BRISQUE-style heuristic built on the
//     variance of mean-subtracted contrast-normalised (MSCN) coefficients.
//     It is not the calibrated OpenCV BRISQUE score (which requires a trained
//     support-vector regressor); it is a deterministic naturalness/activity
//     proxy where higher values indicate more high-frequency structure.
//   - [Entropy] — the Shannon entropy, in bits, of the luminance histogram.
//   - [NIQE] — the Natural Image Quality Evaluator (Mittal et al. 2013), an
//     opinion-unaware naturalness distance from a fixed pristine model built
//     from generalised-Gaussian fits to MSCN coefficients and their paired
//     products at two scales. Lower is better. The pristine mean vector and a
//     diagonal covariance are documented fixed parameters; this is a
//     self-contained variant of NIQE rather than the exact OpenCV model.
//   - [PIQE] — the Perception-based Image Quality Evaluator (Venkatanath et al.
//     2015), a block-based, opinion-unaware measure that scores the spatially
//     active MSCN blocks for noise and blockiness. Lower is better.
//
// # The QualityBase pattern
//
// Mirroring OpenCV's cv::quality::QualityBase, each full-reference metric also
// has an object form that is constructed once with the reference image and then
// applied to many candidates. Construct one with [NewQualityMSE],
// [NewQualityPSNR], [NewQualitySSIM], [NewQualityGMSD], [NewQualityRMSE],
// [NewQualityFSIM], [NewQualityFSIMc] or [NewQualityVIFP]; each satisfies the
// [QualityBase] interface, exposing Compute (returning the score as a slice,
// one element per channel or a single pooled value) and QualityMap (the
// per-pixel error/similarity map produced by the most recent Compute).
//
// # Determinism and errors
//
// Every metric is a pure function of its inputs — there is no randomness and no
// hidden global state, so repeated runs are bit-for-bit identical. Following
// the root package's convention, the functions panic on malformed input (a
// size or channel-count mismatch between the two images, or an empty image)
// rather than returning an error, mirroring a Go slice index-out-of-range.
package quality

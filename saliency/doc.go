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
//   - [StaticSaliencyIttiKochNiebur] — the classical bottom-up attention model
//     of Itti, Koch & Niebur (1998): center-surround contrast on Gaussian
//     pyramids of intensity, colour double-opponency and orientation, combined
//     through the N(·) map-promotion operator.
//   - [MinimumBarrierSaliency] — the Minimum Barrier Distance detector of Zhang
//     et al. (ICCV 2015). Saliency is the barrier distance (path max minus min
//     intensity) from each pixel to the image border, computed with fast
//     alternating raster scans.
//   - [StaticSaliencyFrequencyTuned] — the frequency-tuned method of Achanta et
//     al. (CVPR 2009): the Lab distance between a lightly blurred pixel and the
//     whole-image mean colour.
//   - [StaticSaliencyContextAware] — the context-aware detector of Goferman et
//     al. (CVPR 2010), scoring each pixel by its colour dissimilarity to its
//     most similar (position-discounted) context.
//   - [GMRSaliency] — the graph-based manifold-ranking detector of Yang et al.
//     (CVPR 2013): a two-stage ranking of super-pixel regions against the image
//     borders and then against foreground queries.
//   - [StaticSaliencyBooleanMap] — Boolean Map Saliency (BMS) of Zhang &
//     Sclaroff (ICCV 2013), activating the surrounded (border-disconnected)
//     regions of many thresholded Boolean maps.
//   - [HistogramContrast] (HC) and [RegionContrast] (RC) — the global-contrast
//     detectors of Cheng et al. (CVPR 2011): colour-histogram contrast and
//     spatially-weighted region contrast in Lab space.
//
// [ComputeBinaryMap] turns any such saliency map into a binary foreground mask
// with Otsu thresholding; [AdaptiveBinaryMap] instead thresholds at a multiple
// of the map mean (Achanta's adaptive threshold). [SaliencyToHeatmap] renders a
// map as a jet-coloured image, and [CenterBiasPrior]/[ApplyCenterBias] model
// the human centre-fixation bias.
//
// # Evaluation
//
// The standard saliency-benchmark metrics compare a predicted map against a
// human-fixation record or another map: [AUCJudd] (area under the ROC),
// [NSS] (normalized scanpath saliency), [CC] (linear correlation), [SIM]
// (histogram-intersection similarity) and [KLDiv] (Kullback-Leibler
// divergence).
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
//   - [ObjectnessCascade] — a two-stage proposer in the spirit of the BING
//     cascade: a fast normed-gradient first stage, a richer second stage that
//     re-scores survivors with saliency coverage, boundary contrast and a size
//     prior, followed by non-maximum suppression of overlapping boxes.
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
// same qualitative behaviour. Several detectors substitute regular grid regions
// for a learned colour segmentation or SLIC super-pixels ([GMRSaliency],
// [RegionContrast]) and solve their linear systems iteratively; these
// approximations are documented on each type. The one capability that depends
// on trained model data is intentionally out of scope:
//
//   - trained deep-learning saliency models (e.g. DeepGaze-style networks), and
//     the offline-learned linear weights of the full BING detector.
//
// [ObjectnessBING] and [ObjectnessCascade] therefore return heuristic
// objectness cues rather than the calibrated scores of the trained detector.
package saliency

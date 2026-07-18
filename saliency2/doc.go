// Package saliency2 is a standard-library-only toolkit of visual saliency and
// attention algorithms built on top of the parent cv package's [cv.Mat] image
// type.
//
// Saliency estimation predicts where a human observer's gaze is drawn in an
// image: bright, novel, high-contrast or moving regions "pop out" from their
// surroundings. This package collects the classic static and dynamic saliency
// models, plus an objectness proposal generator and a set of reusable
// post-processing operators, all implemented from scratch against the Go
// standard library (image, math, math/cmplx). There is no cgo, no third-party
// dependency and no GPU requirement; every routine is deterministic and
// CPU-only.
//
// # Detectors
//
// The static single-image detectors implement the [StaticSaliency] interface
// and each return a single-channel saliency map the same size as the input:
//
//   - [StaticSaliencySpectralResidual] — Hou & Zhang's spectral-residual model
//     (CVPR 2007), operating in the Fourier domain.
//   - [StaticSaliencyFineGrained] — Montabone & Soto's multi-scale
//     center-surround contrast (Image and Vision Computing 2010).
//   - [StaticSaliencyFrequencyTuned] — Achanta et al.'s frequency-tuned model
//     in CIE L*a*b* space (CVPR 2009).
//   - [StaticSaliencyIttiKoch] — the Itti-Koch-Niebur center-surround attention
//     model (PAMI 1998) built from intensity, colour-opponency and orientation
//     conspicuity maps.
//
// The [ObjectnessBING] proposal generator ranks candidate windows by their
// normed-gradient boundary energy, a lightweight untrained variant of Cheng et
// al.'s BING (CVPR 2014). The [MotionSaliencyByDifference] and
// [MotionSaliencyRunningAverage] detectors implement the [MotionSaliency]
// interface for streaming video.
//
// # The SaliencyMap type
//
// [SaliencyMap] is a single-channel float64 grid used as the common working and
// result representation. It carries the analysis and post-processing methods
// (normalisation, thresholding, centre of mass, bounding box) and converts to
// and from [cv.Mat] with [SaliencyMap.ToMat] and [SaliencyMapFromMat]. Free
// functions in this package operating on maps — [GaussianSmooth], [IttiNormalize],
// [CenterPrior] and friends — provide the reusable saliency post-processing
// pipeline.
//
// # Conventions
//
// Following the parent package, coordinates are (x, y) with the origin at the
// top-left, x the column and y the row. Neighbourhood operators replicate the
// edge sample at the border. Unexported helper identifiers are prefixed with
// "saliency2" so this package composes cleanly with the rest of the module.
package saliency2

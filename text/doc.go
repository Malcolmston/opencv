// Package text is a from-scratch, standard-library-only port of a useful subset
// of OpenCV's contrib text module: classical scene-text detection on top of the
// grayscale-and-contours machinery in github.com/malcolmston/opencv (imported
// here as cv).
//
// The package does not attempt optical character recognition with a trained
// classifier or a neural network. Instead it implements the geometric,
// pre-recognition half of a scene-text pipeline: finding candidate character
// regions and grouping them into text lines. What you feed to a downstream
// recognizer is left to you.
//
// # Pipeline
//
// A detection pipeline has three stages, each backed by a function or type here:
//
//   - Region proposal. [DetectRegionsMSER] (and the richer [MSERRegions])
//     extract Maximally Stable Extremal Regions — connected components of the
//     image's intensity level sets whose area stays nearly constant across a
//     range of thresholds. Characters, being roughly uniform blobs that contrast
//     with their background, are classic MSERs.
//   - Region filtering. [ERFilter] discards regions that are unlikely to be
//     characters using cheap shape heuristics (area, aspect ratio and fill
//     ratio), a stand-in for OpenCV's trained Extremal Region classifier.
//   - Grouping. [GroupTextRegions] clusters the surviving character boxes into
//     text lines using geometric heuristics: similar height, horizontal
//     alignment and bounded spacing.
//
// [ComputeNMChannels] reproduces the channel decomposition that OpenCV's
// Neumann–Matas detector runs its extremal-region search over, and
// [NearestGlyphClassifier] is a trivial nearest-template recognizer useful for
// tests and simple fixed-font digit reading.
//
// # Types
//
// [Region] describes one detected region: its bounding [cv.Rect], the pixel set
// it covers, the intensity level at which it is maximally stable, its area and
// its stability variation. Grouping and filtering functions work in terms of
// plain [cv.Rect] boxes so they compose with any region source.
//
// # Determinism
//
// Every function here is deterministic: identical input Mats and parameters
// produce byte-identical output. Union-find merges break ties by pixel index and
// all region orderings are sorted, so there is no dependence on Go's map
// iteration order or on any hidden global state.
//
// # Standard library only
//
// Like the parent package, text is written entirely against the Go standard
// library (only math and sort). It uses no cgo and no third-party dependencies,
// and it imports the root cv package but none of the other cv/* subpackages.
//
// # Deferred
//
// Recognition proper — OCR with a trained OCRHMMDecoder/OCRTesseract-style model
// or a CNN word recognizer — is out of scope and deferred. The trained AdaBoost
// Extremal Region classifiers (ERFilter's NM1/NM2 stages) are approximated by
// the heuristic [ERFilter] rather than ported with their learned weights.
package text

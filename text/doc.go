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
//   - Region proposal. [DetectRegionsMSER] (and the richer [MSERRegions],
//     [MSERRegionsWithParams] with min-diversity pruning) extract Maximally
//     Stable Extremal Regions — connected components of the image's intensity
//     level sets whose area stays nearly constant across a range of thresholds.
//     Characters, being roughly uniform blobs that contrast with their
//     background, are classic MSERs.
//   - Region filtering. [ERFilter] applies quick shape heuristics; the two-stage
//     [ERFilterNM1]/[ERFilterNM2] classifier reproduces the Neumann–Matas
//     Extremal Region cascade over the full [ERFeatures] descriptor (aspect,
//     compactness, hole topology, convexity and stroke-width constancy), with
//     documented thresholds standing in for the trained AdaBoost weights.
//   - Grouping. [GroupTextRegions] and [ERGrouping] cluster the surviving
//     character boxes into text lines by geometric linkage (horizontal or
//     slope-tolerant); [ERGroupingBBox] offers a fast bounding-box-only variant.
//
// [DetectRegions] and [DetectTextLines] wire the whole pipeline together, from
// MSER proposal through both filter stages to grouped text-line boxes, controlled
// by [TextDetectorParams].
//
// # Stroke Width Transform
//
// [TextDetectorSWT] and the underlying [StrokeWidthTransform] provide an
// independent detector based on stroke-width constancy (Epshtein–Ofek–Wexler),
// finding connected components whose stroke thickness is nearly uniform.
//
// # Recognition
//
// A built-in 5x7 bitmap font ([RenderText], [FontGlyph], [SupportedChars]) backs
// a nearest-template recognizer, [OCRTemplate] (digits, or the full alphanumeric
// set), with per-character [SegmentChars] segmentation and [SegmentLines] line
// splitting driven by projection profiles ([ProjectionProfile]). [OCRTemplate]
// can also emit per-character score matrices for the lexicon-constrained
// [BeamSearchDecoder], a beam-search word decoder over a [Lexicon].
//
// [ComputeNMChannels] reproduces the channel decomposition that OpenCV's
// Neumann–Matas detector runs its extremal-region search over, and
// [NearestGlyphClassifier] is the trivial nearest-template matcher [OCRTemplate]
// is built on.
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
// Trained-model recognition — a real OCRHMMDecoder/OCRTesseract language model or
// a CNN word recognizer — remains out of scope: [OCRTemplate] reads only the
// fixed built-in font, not arbitrary scene text. The Extremal Region classifier
// stages [ERFilterNM1] and [ERFilterNM2] compute the genuine Neumann–Matas
// features but decide with documented thresholds rather than the original trained
// AdaBoost weights, and [ERGrouping] uses geometric linkage in place of the
// learned pair/triplet classifier.
package text

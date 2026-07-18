// Package morph2 is a standard-library-only toolkit of advanced mathematical
// morphology built on top of the parent package's [github.com/malcolmston/opencv.Mat]
// image type.
//
// It complements the elementary erosion/dilation offered by the root package
// with the harder machinery of grey-scale and binary morphology: compound
// operators (opening, closing, gradients, top-/black-hat), the hit-or-miss
// transform, Zhang-Suen and Guo-Hall thinning, the morphological skeleton,
// grey-scale morphological reconstruction and its many derivatives (hole
// filling, border clearing, opening/closing by reconstruction, regional and
// extended extrema, h-maxima/minima, minima imposition), exact and chamfer
// distance transforms, marker-controlled watershed segmentation and
// granulometry / pattern spectra.
//
// # Conventions
//
// Every routine operates on single-channel (Channels == 1) [cv.Mat] values and
// panics on multi-channel input. Coordinates follow the image convention used
// by the parent package: x is the column, y is the row, origin at the top-left.
//
// Binary routines treat a pixel as foreground when its sample is non-zero and
// as background when it is zero; binary results are written back using 0 for
// background and 255 for foreground. Grey-scale routines interpret samples as
// intensities in the full 0..255 range.
//
// Structuring elements are described by the [Element] type, which carries an
// explicit anchor and can be reflected, cloned and converted to and from a
// [cv.Mat] kernel so it interoperates with the root package's
// GetStructuringElement.
//
// Distance maps and other real-valued results are returned as a [FloatGrid],
// and integer label images (markers, watershed output) as a [LabelMap]; both
// are thin helper containers that convert to and from [cv.Mat] rather than
// duplicating it.
//
// The implementation is pure Go, uses no cgo and no third-party dependencies,
// is CPU-only and fully deterministic.
package morph2

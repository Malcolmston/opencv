// Package objdetect provides classic, learning-free object-detection tooling
// for the parent [github.com/malcolmston/opencv] package: a Histogram of
// Oriented Gradients (HOG) descriptor with a linear-SVM sliding-window
// detector, a Viola–Jones Haar cascade classifier that reads OpenCV's XML
// cascade files, and a QR-code finder-pattern locator. It mirrors a useful
// subset of OpenCV's cv::objdetect module using only the Go standard library
// (including encoding/xml) plus the root cv package.
//
// Everything here operates on the root [cv.Mat] type and returns results in
// the root's [cv.Rect] and [cv.Point] geometry types. Images may be
// single-channel (grayscale) or three-channel; multi-channel input is reduced
// to luma (0.299R + 0.587G + 0.114B, matching the root package's RGB
// convention) before analysis. No cgo, no third-party dependencies.
//
// # Histogram of Oriented Gradients
//
// [HOGDescriptor] computes the Dalal–Triggs HOG feature vector. Construct one
// with [NewHOGDescriptor] for the canonical 64×128 person-detector geometry,
// or set the [Size] fields directly. [HOGDescriptor.Compute] returns the
// descriptor for a single detection window; [HOGDescriptor.DetectMultiScale]
// slides that window over a downscaling image pyramid and scores each position
// with a linear SVM.
//
// The pipeline is the textbook one: per-pixel gradient magnitude and unsigned
// orientation (a centred [-1,0,1] difference), accumulation of magnitude into
// per-cell orientation histograms, grouping of cells into overlapping blocks,
// and L2-Hys normalisation of each block. See [HOGDescriptor.DescriptorSize]
// for the exact length and [HOGDescriptor.Compute] for the byte-for-byte
// component layout.
//
// # Haar cascade classifier
//
// [CascadeClassifier] parses an OpenCV Haar cascade from the modern
// <opencv_storage><cascade> XML layout (stage/weak-classifier/feature/rects
// tree) via encoding/xml and evaluates it with an [IntegralImage] for
// constant-time Haar feature sums. [CascadeClassifier.DetectMultiScale] walks
// a scale pyramid by growing the classifier window (never resampling the
// image) and returns the surviving windows. Load a cascade with
// [CascadeClassifier.Load] (from a file path) or
// [CascadeClassifier.LoadFromString].
//
// The stump evaluator compares each feature's weighted rectangle sum against
// threshold·σ·A, where σ is the window's intensity standard deviation and A
// its area (encoded jointly as sqrt(A·Σx² − (Σx)²)); this is the Viola–Jones
// variance normalisation. The exact numeric threshold conventions of every
// third-party cascade are not guaranteed to match OpenCV bit-for-bit — see the
// deferred notes on [CascadeClassifier].
//
// # QR-code finder patterns
//
// [QRCodeDetector.Detect] locates the three square finder patterns of a QR
// code by scanning rows and columns for the characteristic
// 1:1:3:1:1 dark:light:dark:light:dark run-length ratio and clustering the
// confirmed centres. It returns the three pattern centres as [cv.Point]s.
// Full symbol decoding (alignment, sampling, Reed–Solomon) is deferred; see
// [QRCodeDetector].
//
// # Errors and panics
//
// Following the root package's convention, constructors and I/O that can fail
// return an error, while functions that are handed structurally invalid
// arguments (a mis-proportioned HOG geometry, an SVM weight vector of the
// wrong length, an unloaded classifier) panic with a descriptive message.
package objdetect

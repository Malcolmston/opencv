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
// with a linear SVM. [HOGDescriptor.DetectMultiScaleWeights] additionally
// returns each raw window's SVM score, [HOGDescriptor.ComputeGradient] exposes
// the per-pixel gradient field, and [HOGDescriptor.DefaultPeopleDetector]
// supplies a ready-to-use (approximate) pedestrian weight vector.
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
// # LBP cascades and soft cascades
//
// [LBPCascadeClassifier] parses OpenCV's featureType-LBP cascades and evaluates
// them by reading a 3×3 grid of integral-image cell sums into an 8-bit Local
// Binary Pattern code and looking it up in a per-classifier 256-bit subset;
// [ComputeLBP] exposes the underlying LBP code image as a texture descriptor.
// [SoftCascade] is a soft-cascade detector that flattens the staged ensemble
// into a single additive score with a per-weak rejection trace, so a window can
// be discarded after any weak classifier (early exit); convert a loaded Haar
// cascade with [CascadeClassifier.ToSoftCascade].
//
// # Grouping, non-maximum suppression and tracking
//
// [GroupRectangles] and [GroupRectanglesWeights] cluster overlapping detections
// with OpenCV's eps/minNeighbors rule; [NMSBoxes] and [SoftNMSBoxes] apply
// greedy and Gaussian non-maximum suppression to scored boxes, with [RectIoU]
// as the shared overlap metric. [DetectionBasedTracker] wraps any [Detector]
// (the Haar, LBP and soft cascades all qualify) and follows objects across
// frames with stable identities by intersection-over-union association.
//
// # QR-code finder patterns
//
// [QRCodeDetector.Detect] locates the three square finder patterns of a QR
// code by scanning rows and columns for the characteristic
// 1:1:3:1:1 dark:light:dark:light:dark run-length ratio and clustering the
// confirmed centres. It returns the three pattern centres as [cv.Point]s.
// [QRCodeDetector.DetectFinderPatterns] returns every located finder pattern,
// and [QRCodeDetector.DetectMulti] groups them into the quadrilaterals of
// multiple QR symbols by matching right-angle, equal-leg finder triples.
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

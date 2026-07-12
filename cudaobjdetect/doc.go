// Package cudaobjdetect is a CPU-backed, API-compatible mirror of OpenCV's
// cv::cuda objdetect module (opencv_contrib's cudaobjdetect). It offers the same
// types and method shapes a caller would use with the CUDA classes — a [GpuMat]
// container, a no-op [Stream], a [HOG] descriptor/detector and a Haar
// [CascadeClassifier] — but there is no GPU and no cgo: every operation runs on
// the host in pure Go by delegating to the sibling
// [github.com/malcolmston/opencv/objdetect] package.
//
// # Honest scope
//
// This package exists so code written against the OpenCV CUDA objdetect API can
// compile and run without a GPU, not to accelerate anything. "Gpu" in the type
// names is a naming convention for compatibility only:
//
//   - [GpuMat] wraps a host [github.com/malcolmston/opencv.Mat]. Upload and
//     Download copy between host buffers; they do not cross a device boundary.
//   - [Stream] is a no-op. All work is synchronous, so its methods do nothing
//     and passing nil is always valid.
//   - [HOG.GetDefaultPeopleDetector] returns an approximate matched-filter
//     classifier synthesised from a prototype silhouette, not OpenCV's trained
//     INRIA weights (which cannot be reproduced without the training data).
//   - Detection results that OpenCV packs into a CV_32SC4 GpuMat are carried
//     out-of-band, because the root Mat is 8-bit only; [CascadeClassifier.Convert]
//     decodes them exactly as the CUDA API does.
//
// # HOG
//
// [HOG] mirrors cv::cuda::HOG. Create one with [NewHOG] (custom geometry) or
// [NewDefaultHOG] (the 64×128 person detector), set a classifier with
// [HOG.SetSVMDetector] (commonly [HOG.GetDefaultPeopleDetector]), then call
// [HOG.Detect] for a single-scale scan or [HOG.DetectMultiScale] for a pyramid
// scan; both return per-detection confidences. [HOG.Compute] returns the raw
// descriptor of one window. Geometry and behaviour parameters (hit threshold,
// scale factor, group threshold, window stride, gamma correction, L2-Hys
// threshold, window sigma, number of levels) have matching get/set accessors.
//
// # Cascade classifier
//
// [CascadeClassifier] mirrors cv::cuda::CascadeClassifier. Load a cascade with
// [NewCascadeClassifier] (from a file) or [LoadCascadeFromString], tune it with
// the scale-factor / min-neighbours / min-and-max object size / find-largest
// accessors, then run [CascadeClassifier.DetectMultiScale] to get a result
// [GpuMat] and [CascadeClassifier.Convert] to decode it into [github.com/malcolmston/opencv.Rect]
// values. Only the modern Haar <opencv_storage><cascade> XML layout is
// supported (see [objdetect.CascadeClassifier] for the exact caveats).
//
// # Grouping and non-maximum suppression
//
// [GroupRectangles], [GroupRectanglesWeights], [NMSBoxes], [SoftNMSBoxes] and
// [RectIoU] are convenience pass-throughs to the corresponding objdetect
// helpers for post-processing raw detections.
//
// # Errors and panics
//
// Following the parent package's convention, loaders return an error while
// functions handed structurally invalid arguments (an inconsistent HOG
// geometry, a wrong-length SVM vector, a GpuMat with no image data) panic with
// a descriptive message.
package cudaobjdetect

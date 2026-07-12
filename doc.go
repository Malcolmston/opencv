// Package cv is a from-scratch, standard-library-only port of a useful subset
// of Python's OpenCV (cv2), focused on classic image processing and computer
// vision primitives.
//
// The package is written entirely against the Go standard library — image,
// image/color, image/png, image/jpeg, math and friends. It uses no cgo and no
// third-party dependencies, so it builds and runs anywhere the Go toolchain
// does. The trade-off is scope: heavyweight machine-vision machinery such as
// feature descriptors (SIFT/ORB), full contour hierarchies, camera calibration,
// DNN inference and video I/O are intentionally out of scope. What remains is a
// faithful, genuinely useful image-processing toolkit.
//
// # The Mat type
//
// The central data structure is [Mat], a dense row-major matrix of 8-bit
// unsigned samples backed by a flat []uint8. A Mat has Rows, Cols and Channels;
// one-channel (grayscale) and three-channel (RGB) images are the common cases,
// but any positive channel count is supported. Pixels are stored interleaved:
// the sample for row y, column x, channel c lives at index
// (y*Cols+x)*Channels + c. Convert to and from the standard library with
// [FromImage] and [Mat.ToImage], and read or write PNG/JPEG files with [ImRead]
// and [ImWrite].
//
// Unless documented otherwise, this package treats three-channel data as RGB
// (matching Go's image package), not OpenCV's native BGR. Use [CvtColor] with
// [ColorRGB2BGR] when you need to interoperate with BGR-oriented code or data.
//
// # Conventions
//
// Coordinates follow the image convention: x is the column (horizontal) and y
// is the row (vertical), with the origin at the top-left. Functions that take a
// point-like pair take it as (x, y). Border handling for neighbourhood
// operations replicates the edge sample (OpenCV's BORDER_REPLICATE), which
// avoids the dark halos that zero-padding produces.
//
// # Filtering and analysis
//
// The package provides a generic [Filter2D] convolution plus the common
// specialisations built on it: [Blur]/[BoxFilter], separable [GaussianBlur],
// [MedianBlur], [Sobel], [Scharr] and [Laplacian]. Thresholding covers fixed
// and [Otsu] levels via [Threshold] as well as [AdaptiveThreshold]. Morphology
// offers [Erode], [Dilate] and [MorphologyEx] over structuring elements from
// [GetStructuringElement]. Geometric transforms include [Resize], [Flip],
// [Rotate], [Transpose] and affine warping through [WarpAffine] and
// [GetRotationMatrix2D]. Edge and template tooling provides a full [Canny]
// pipeline and [MatchTemplate] with [MinMaxLoc]. Drawing primitives ([Line],
// [Rectangle], [Circle], [Ellipse], [PutText], [Polylines], [FillPoly]) render
// directly onto a Mat, and [CalcHist]/[EqualizeHist] cover histograms.
//
// # Errors and panics
//
// Constructors and I/O functions that can fail return an error. Pixel-level
// helpers such as [Mat.At] and [Mat.Set] favour speed and panic on
// out-of-range access, mirroring a Go slice index. Processing functions
// validate their arguments and panic with a descriptive message on programmer
// error (for example a mismatched channel count) rather than returning an
// error for every call.
package cv

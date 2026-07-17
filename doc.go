// Package cv is a from-scratch, standard-library-only port of a useful subset
// of Python's OpenCV (cv2), focused on classic image processing and computer
// vision primitives.
//
// The package is written entirely against the Go standard library — image,
// image/color, image/png, image/jpeg, math and friends. It uses no cgo and no
// third-party dependencies, so it builds and runs anywhere the Go toolchain
// does. This root package holds the core imgproc toolkit; the heavier
// machine-vision machinery — dense feature descriptors (SIFT/ORB/AKAZE), camera
// calibration and stereo (calib3d, stereo), DNN inference, optical flow and
// video — lives in importable subpackages under this module (see Subpackages
// below), each also standard-library-only. The remaining unavoidable gaps are
// capabilities that genuinely need a GPU or trained model files, which a
// zero-dependency, cgo-free port cannot provide.
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
// # Colour, arithmetic and shape analysis
//
// [CvtColor] additionally converts between RGB and CIE L*a*b* ([ColorRGB2Lab]),
// Y'CrCb ([ColorRGB2YCrCb]) and HLS ([ColorRGB2HLS]). Element-wise Mat
// arithmetic with saturation is available through [Add], [Subtract], [AbsDiff],
// [AddWeighted], [Multiply], [Divide], the bitwise ops ([BitwiseAnd], [BitwiseOr],
// [BitwiseXor], [BitwiseNot]), [Min], [Max], [Normalize] and [ConvertScaleAbs].
// [BilateralFilter] adds edge-preserving smoothing and [Filter2DSep] exposes
// separable convolution.
//
// Structural analysis covers Suzuki-style [FindContours] with retrieval modes
// and a hierarchy, [DrawContours], [ContourArea], [ArcLength], [BoundingRect],
// [MinAreaRect], [ConvexHull], [ApproxPolyDP] and [ImageMoments], plus
// [ConnectedComponents] and [ConnectedComponentsWithStats]. Feature detection
// provides [CornerHarris], [GoodFeaturesToTrack], [HoughLines], [HoughLinesP],
// [HoughCircles] and [FASTCorners]. Projective geometry adds
// [GetPerspectiveTransform] with [WarpPerspective], [Remap], the [PyrDown] /
// [PyrUp] Gaussian pyramid and [DistanceTransform]. Histogram tooling gains
// [CalcBackProject], [CompareHist] and [CLAHE].
//
// # Linear algebra, signal and geometry
//
// Beyond image processing the root package carries the numerical core OpenCV
// exposes in cv2: dense linear algebra over [FloatMat] — [Invert], [Solve],
// [Determinant], [Trace], [Eigen], [SVDecomp], [Gemm], [PCACompute] /
// [PCAProject] / [PCABackProject], [Mahalanobis] and [CalcCovarMatrix]; array
// utilities [Reduce], [Repeat], [Sort], [SortIdx], [MinMaxIdx], [FindNonZero],
// [Transform] and the channel helpers [ExtractChannel], [InsertChannel] and
// [MixChannels]; element-wise math [Exp], [Log], [Pow], [Sqrt], [Magnitude],
// [Phase], [CartToPolar] and [PolarToCart]; and the frequency-domain routines
// [DFT], [IDFT], [DCT], [IDCT], [MulSpectrums], [CreateHanningWindow] and
// [PhaseCorrelate]. Contour and shape predicates ([PointPolygonTest],
// [IsContourConvex], [MinEnclosingCircle], [FitLine], [MatchShapes],
// [HuMoments]) and extended drawing ([DrawMarker], [ArrowedLine], [GetTextSize],
// [FillConvexPoly], [BoxPoints]) round out the core surface.
//
// # Subpackages
//
// This root package is the core; the wider computer-vision surface lives in
// 58 importable subpackages under the same module, each standard-library-only
// and depending only on this package. They mirror OpenCV's main and contrib
// modules:
//
//   - Features & matching: features2d, xfeatures2d, flann, linedescriptor
//   - Geometry & 3D: calib3d, stereo, rgbd, surface_matching, structured_light,
//     phase_unwrapping, ccalib, rapid
//   - Detection & recognition: objdetect, face, aruco, barcode, datamatrix,
//     text, dnn, saliency, xobjdetect
//   - Motion & tracking: video, optflow, tracking, bgsegm, videostab
//   - Photo & imaging: photo, xphoto, hdr, intensity, fuzzy, bioinspired,
//     dnn_superres
//   - Segmentation, shape & stitching: segmentation, shape, ximgproc,
//     stitching, hfs
//   - Analysis & viz: ml, quality, imghash, plot, videoio, mcc, gapi, imgprocx
//
// A family of cuda* packages (cudaarithm, cudaimgproc, cudafilters,
// cudawarping, cudafeatures2d, cudabgsegm, cudaobjdetect, cudaoptflow,
// cudastereo, cudacodec, cudacore, cudalegacy) mirrors the API shape of
// OpenCV's GPU modules. They are CPU-backed and cgo-free: a GpuMat wraps an
// ordinary host Mat and Stream is a no-op, so code ports naturally from
// OpenCV's cuda modules but runs on the CPU — API parity, not acceleration.
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

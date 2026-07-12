// Package stereo is a standard-library-only implementation of the classic
// stereo-correspondence and disparity routines from OpenCV's calib3d and
// contrib stereo modules, built on top of the root cv package
// (github.com/malcolmston/opencv).
//
// It depends only on the Go standard library and the root cv package: no cgo,
// no third-party code, and it does not import any sibling cv/* subpackage. The
// root package supplies the [github.com/malcolmston/opencv.Mat] container and
// grayscale conversion via [github.com/malcolmston/opencv.CvtColor]; everything
// specific to stereo matching is implemented here from scratch.
//
// # The stereo geometry
//
// The routines assume a rectified stereo pair: the two cameras are coplanar and
// row-aligned, so a scene point projects to the same image row in both views
// and its horizontal offset — the disparity d = xLeft - xRight (in pixels) — is
// inversely proportional to depth. Given a left pixel at column x, its match in
// the right image is searched at columns x-d for d in [0, NumDisparities).
// Larger disparities correspond to nearer surfaces.
//
// If your input is not yet rectified you must rectify it first; [Rectify] is
// only a passthrough stub (see its documentation and the deferred list below).
//
// # Disparity maps
//
// Both matchers return a single-channel 8-bit [github.com/malcolmston/opencv.Mat]
// whose samples are integer disparities in pixels (OpenCV's CV_16S fixed-point
// scaling is not used). The reserved value [InvalidDisparity] (0) marks pixels
// with no reliable match: low-texture (uniform) regions, the left border band
// of width NumDisparities where the full search range is unavailable, and
// ambiguous matches that fail the uniqueness ratio test. Because 0 doubles as
// the invalid marker, a genuine best match of disparity 0 is indistinguishable
// from "no match"; in practice min-disparity is 0 and nearby valid disparities
// are positive.
//
// # Matchers
//
//   - [StereoBM] performs local block matching: for every pixel it minimises the
//     sum of absolute differences (SAD) between a BlockSize×BlockSize window in
//     the left image and the corresponding window in the right image over the
//     horizontal search range, then filters the result with a texture threshold
//     and a uniqueness ratio.
//   - [StereoSGBM] is a semi-global-matching-lite matcher. It builds a pixelwise
//     (BlockSize) SAD cost volume and aggregates it along [NumPaths] (4) cardinal
//     paths — left, right, up and down — using the SGM smoothness penalties P1
//     (small disparity change) and P2 (larger jumps) before taking the
//     winner-take-all disparity. The four-path aggregation makes it markedly more
//     robust than [StereoBM] on weakly textured regions while staying cheap.
//
// # Reconstruction and post-processing
//
//   - [ReprojectImageTo3D] maps a disparity map to metric 3-D coordinates through
//     the 4×4 reprojection matrix Q produced by stereo rectification.
//   - [FilterSpecklesDisparity] removes small, isolated disparity "speckles"
//     (connected blobs smaller than a threshold), a standard cleanup that both
//     OpenCV matchers apply.
//
// # Full semi-global matching
//
// [StereoSGM] is the complete semi-global matcher. It goes beyond the four-path
// [StereoSGBM] with:
//
//   - eight-path aggregation ([ModeHH]) adding the four diagonal directions, the
//     equivalent of OpenCV's MODE_HH;
//   - a selectable matching cost ([CostType]: SAD, SSD, mean-AD, or the
//     illumination-robust census transform);
//   - a Disp12MaxDiff left-right consistency check that rejects occlusions; and
//   - equiangular parabola sub-pixel refinement ([StereoSGM.ComputeFloat],
//     returning a float [DisparityF]).
//
// # Cost volumes and building blocks
//
//   - [MatchingCostVolume] and [CensusCostVolume] build a dense [CostVolume] that
//     [CostVolume.WinnerTakeAll], [StereoSGM.Aggregate], [ComputeConfidence] and
//     [RefineSubpixel] all consume.
//   - [CensusTransform] and [HammingDistance64] provide the non-parametric census
//     descriptor and its matching cost.
//   - [SubpixelParabola] and [RefineSubpixel] add fractional-pixel precision to a
//     winner-take-all disparity.
//   - [ComputeConfidence] scores each pixel by its cost peak ratio.
//
// # Block matching and post-processing
//
//   - [BlockMatcher] is the full local matcher: intensity pre-filtering
//     ([ApplyXSobelPrefilter], [ApplyNormalizedPrefilter] via [PrefilterType]),
//     texture and uniqueness rejection, speckle removal, and a left-right check.
//   - [ValidateDisparity] enforces left-right consistency between a left- and a
//     right-referenced map; [GetValidDisparityROI] reports the reliably matchable
//     rectangle.
//   - [DisparityWLSFilter] is an edge-aware weighted-least-squares post-filter
//     that smooths and hole-fills a disparity map using a guidance image.
//   - [QuasiDenseStereo] grows a dense map from correlation seeds by best-first
//     propagation.
//
// # Deferred
//
// The following are intentionally out of scope for this port:
//
//   - StereoBeliefPropagation and StereoConstantSpaceBP.
//   - GPU / CUDA acceleration.
//   - True calibration-driven rectification ([Rectify] is a stub).
package stereo

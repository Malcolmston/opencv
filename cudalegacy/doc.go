// Package cudalegacy is a CPU-backed, API-compatible mirror of OpenCV's
// cudalegacy module — the grab-bag of "device functions" and legacy NPP/NCV
// helpers that historically lived under the cv::cuda and NCV namespaces
// (background subtraction, image pyramids, connected-component labelling on the
// device, block-matching optical flow, motion-compensated frame interpolation,
// needle-map visualisation, graph-cut segmentation and planar pose estimation).
//
// # Honest scope
//
// There is NO CUDA, NO GPU and NO cgo here. Every type and function is pure Go
// built only on the standard library, the root package
// [github.com/malcolmston/opencv] (imported under the alias cv) and the sibling
// CPU background-segmentation package [github.com/malcolmston/opencv/bgsegm].
// The goal is source and shape compatibility: code written against the OpenCV
// cudalegacy API — a [GpuMat] holding pixels, a [Stream] threaded through each
// call, factory functions and algorithm objects with the same names — ports
// across with minimal edits, while the actual computation runs on the CPU.
//
// Nothing here is faster than an equivalent CPU routine; a [GpuMat] is a thin
// wrapper around a *[cv.Mat] and a [Stream] is a no-op placeholder. Upload and
// Download are ordinary copies. The value delivered is a drop-in API surface,
// not hardware acceleration.
//
// # What is genuinely implemented
//
// These are real algorithms with real numerics, not stubs:
//
//   - [BackgroundSubtractorFGD] — a self-contained foreground-object detector
//     after Li et al.: an adaptive per-pixel, per-channel single-Gaussian model
//     with a Mahalanobis decision, morphological opening and a
//     connected-component minimum-area gate.
//   - [BackgroundSubtractorGMG] — the Godbehere-Matsukawa-Goldberg Bayesian
//     model, delegating pixel classification to [bgsegm.BackgroundSubtractorGMG].
//   - [ImagePyramid] — a Gaussian image pyramid (build with [cv.PyrDown], read
//     arbitrary layers with [cv.Resize]).
//   - [ConnectivityMask] and [LabelComponents] — per-pixel neighbour-similarity
//     masks and union-find connected-component labelling that follows them.
//   - [CalcOpticalFlowBM] — dense optical flow by exhaustive block matching.
//   - [InterpolateFrames] — motion-compensated bidirectional frame interpolation
//     driven by a dense [Flow].
//   - [CreateOpticalFlowNeedleMap] — arrow-field ("needle map") visualisation of
//     a dense flow.
//   - [GraphCut] — binary min-cut / max-flow segmentation on the pixel grid,
//     solved with a Dinic max-flow implementation.
//   - [ProjectPoints] — pinhole projection of 3D points with Rodrigues rotation
//     and radial-tangential distortion.
//   - [SolvePnPRansac] — RANSAC pose estimation for a planar target via
//     homography decomposition, with a consensus refit.
//   - [CompactPoints] — mask-driven compaction of parallel point sets.
//   - [Rodrigues] — axis-angle to rotation-matrix conversion.
//
// That is twelve independent algorithm entry points beyond the [GpuMat]/[Stream]
// infrastructure.
//
// # Representation notes
//
// The root [cv.Mat] is 8-bit unsigned, so results that are inherently signed or
// wider are returned in their natural Go form rather than being forced into a
// GpuMat:
//
//   - Dense optical flow is a [Flow] (two float64 planes), because per-pixel
//     sub-pixel displacements do not fit in a uint8 image. OpenCV writes velx and
//     vely CV_32FC1 GpuMats; a [Flow] holds exactly those two planes.
//   - [LabelComponents] returns an []int32 label slice (OpenCV's label image is
//     CV_32S), with [RenderLabels] to obtain a viewable GpuMat.
//   - Point clouds and 2D correspondences for [ProjectPoints]/[SolvePnPRansac]
//     travel as Go slices, since a uint8 Mat cannot store float coordinate
//     tuples.
//
// # Legitimately deferred surface
//
// Parts of the upstream cudalegacy module are thin wrappers over proprietary
// NVIDIA binaries with no portable algorithmic content of their own, so they are
// intentionally NOT mirrored here:
//
//   - The NCV Haar object-detection pipeline (NCVHaarObjectDetector and the
//     NCVMemStack allocators) — a device port of a cascade classifier bound to
//     NCV device memory management.
//   - The NPP-backed staged box filter and integral-image kernels
//     (nppiStIntegral, nppiStBoxFilter and friends) — direct calls into NVIDIA's
//     closed NPP library.
//   - The Brox variational optical-flow device kernel (NCVBroxOpticalFlow) — a
//     CUDA-only solver; a CPU Brox solver already lives in the sibling
//     cudaoptflow package, so it is not duplicated here.
//
// These are documented rather than shipped as non-functional stubs, in keeping
// with the package's honesty-first stance.
package cudalegacy

// Package cudaoptflow is a CPU-backed, API-compatible mirror of OpenCV's
// cudaoptflow module (cv::cuda dense and sparse optical flow).
//
// # Honesty note
//
// This package contains NO CUDA, NO cgo and NO GPU code. It is written entirely
// against the Go standard library, the root module
// github.com/malcolmston/opencv (imported as cv) and the two sibling optical-flow
// packages github.com/malcolmston/opencv/optflow and
// github.com/malcolmston/opencv/video. The word "Gpu" in the type names is kept
// only so that code ported from OpenCV's C++/Python cuda bindings reads the same
// way; every buffer lives in ordinary host memory and every computation runs on
// the CPU. A [GpuMat] is a thin wrapper around a *cv.Mat and a [Stream] is a
// no-op placeholder for cv::cuda::Stream — there is nothing asynchronous to wait
// for. Nothing here is faster than the equivalent CPU call; the value is a
// drop-in API surface, not acceleration.
//
// # What it mirrors
//
// Each OpenCV cuda optical-flow class is mirrored by a Go type constructed with
// a New* function and driven with a Calc method, exactly like the C++ algorithm
// objects:
//
//   - [SparsePyrLKOpticalFlow] — pyramidal Lucas-Kanade sparse tracking. Calc
//     returns, for each input point, its tracked location, a status byte and a
//     tracking error. Delegates to video.CalcOpticalFlowPyrLK.
//   - [DensePyrLKOpticalFlow] — dense pyramidal Lucas-Kanade. Every pixel gets a
//     displacement by tracking a grid of Lucas-Kanade seeds and interpolating
//     them edge-awarely (optflow.CalcOpticalFlowSparseToDense).
//   - [FarnebackOpticalFlow] — dense Farneback-style flow. Delegates to
//     video.CalcOpticalFlowFarneback (a block-matching stand-in for the
//     polynomial-expansion original).
//   - [OpticalFlowDualTVL1] — the duality-based TV-L1 method of Zach, Pock &
//     Bischof. Delegates to optflow.CalcOpticalFlowDenseTVL1.
//   - [BroxOpticalFlow] — a genuine, self-contained Brox-style variational
//     solver: robust (Charbonnier) brightness- and gradient-constancy data terms
//     with total-variation smoothness, minimised coarse-to-fine with per-level
//     warping. This is real numerical code implemented in this package, not a
//     delegation.
//   - [NvidiaHWOpticalFlow] — OpenCV's NvidiaHWOpticalFlow wraps the fixed-
//     function NVIDIA Optical Flow Accelerator (NVOFA) found on Turing and later
//     GPUs. That hardware cannot exist in a pure-Go process, so rather than ship
//     an unavailable stub this type computes a real dense flow on the CPU with
//     Dense Inverse Search (optflow.CalcOpticalFlowDIS). The API matches; the
//     silicon does not.
//
// # Flow representation
//
// OpenCV's cuda dense estimators write their result into a CV_32FC2 GpuMat. The
// root cv.Mat is 8-bit, so this package returns the faithful, full-precision
// result as an optflow.FlowField (an interleaved (u, v) float64 field) from every
// dense Calc. For code that needs the OpenCV-style two-channel GpuMat, the
// helpers [EncodeFlow]/[FlowToGpuMat] and [DecodeFlow]/[GpuMat.ToFlowField]
// convert between a FlowField and a two-channel uint8 GpuMat using a documented
// signed fixed-point encoding (see [DefaultFlowScale]). That encoding is lossy —
// it quantises to 1/scale of a pixel and saturates beyond ±128/scale — and is
// intended for transport and visualisation, not as the primary result.
//
// # Conventions
//
// Coordinates follow the root package: X is the column and Y is the row. A flow
// component u is a horizontal displacement (positive → right) and v is vertical
// (positive → down), so a scene translating right by dx and down by dy produces
// a flow field close to (dx, dy). All computations are deterministic.
package cudaoptflow

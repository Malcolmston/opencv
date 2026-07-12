// Package dnn_superres is a pure-Go, standard-library-only single-image
// super-resolution toolkit built on top of the OpenCV port
// github.com/malcolmston/opencv (imported here as cv). It mirrors the surface
// of OpenCV's contrib module cv::dnn_superres::DnnSuperResImpl — a small
// stateful object configured with SetModel(algorithm, scale) and driven with
// Upsample — but it does NOT run any trained neural network. Every algorithm
// here is a classical, deterministic, weight-free image-resampling method.
//
// # Conventions
//
// Images are [cv.Mat] values: row-major, channel-interleaved uint8 samples in
// the 0–255 range. Grayscale (1 channel) and RGB (3 channels) inputs are both
// supported; any positive channel count is in fact handled generically, each
// channel resampled independently. The package treats data as RGB (not BGR),
// consistent with the root package. Colour order is irrelevant to the maths:
// every method is a per-channel spatial resample.
//
// All operations are fully deterministic: given the same input Mat and the same
// (algorithm, scale) configuration, the output bytes are identical across runs
// and platforms. No randomness, no goroutine-ordering dependence, no floating
// point reduction that depends on iteration order beyond a fixed left-to-right
// tap sum.
//
// Only integer scale factors 2, 3 and 4 are supported, matching the trained
// OpenCV models. The output dimensions are exactly (Rows*scale, Cols*scale).
//
// # Algorithms
//
// The following methods are REAL — genuine, self-contained super-resolution /
// interpolation that require no external weights:
//
//   - "nearest"  — nearest-neighbour sampling (blocky, exact-preserving).
//   - "bilinear" — bilinear (2-tap separable triangle) interpolation.
//   - "bicubic"  — Keys / Catmull-Rom bicubic convolution (a = -0.5), the
//     4-tap separable cubic used by cv2.INTER_CUBIC.
//   - "lanczos"  — Lanczos-4 windowed-sinc interpolation (8-tap separable),
//     matching cv2.INTER_LANCZOS4.
//   - "edge"     — an edge-directed method (NEDI-lite / edge-guided cubic): a
//     bicubic base pass followed by gradient-guided directional smoothing that
//     suppresses staircase artefacts along strong edges without blurring across
//     them. Fully classical, no learned weights.
//   - "fsrcnn"   — an FSRCNN-STYLE finish: a bicubic base pass followed by a
//     fixed separable unsharp-mask sharpening kernel. This imitates the visual
//     effect of a learned upscaler's high-frequency recovery using hand-built
//     kernels. It is explicitly NOT a trained FSRCNN network (see below).
//
// # Deferred (NOT implemented as real OpenCV behaviour)
//
// OpenCV's dnn_superres ships four trained convolutional models, loaded from
// .pb weight files: EDSR, ESPCN, FSRCNN and LapSRN. This package deliberately
// does NOT reproduce those networks — it contains no trained weights and runs
// no learned inference. The "fsrcnn" algorithm here is a fixed-kernel
// sharpening approximation, honestly named after the family it evokes but not
// equivalent to the real network. If you need the trained-model quality of
// EDSR/ESPCN/FSRCNN/LapSRN, that capability is DEFERRED; use the upstream C++
// module with its .pb files instead.
//
// # Quick start
//
//	sr := dnn_superres.NewDnnSuperResImpl()
//	if err := sr.SetModel("bicubic", 3); err != nil {
//	    // handle unsupported algorithm/scale
//	}
//	big, err := sr.Upsample(small) // big is 3× the size of small
//
// The free functions [UpsampleNearest], [UpsampleBilinear], [UpsampleBicubic],
// [UpsampleLanczos], [UpsampleEdgeDirected] and [UpsampleFSRCNN] expose the same
// algorithms without the stateful wrapper, and [PSNR] measures reconstruction
// quality for tests and benchmarks.
package dnn_superres

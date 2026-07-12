package cudaoptflow

import (
	"github.com/malcolmston/opencv/optflow"
	"github.com/malcolmston/opencv/video"
)

// DensePyrLKOpticalFlow is the CPU-backed mirror of
// cv::cuda::DensePyrLKOpticalFlow: it produces a displacement for every pixel
// with pyramidal Lucas-Kanade. Here that is realised by tracking a regular grid
// of Lucas-Kanade seeds and interpolating them to a dense field with edge-aware
// weighting (optflow.CalcOpticalFlowSparseToDense), which keeps motion
// boundaries crisp.
type DensePyrLKOpticalFlow struct {
	// WinSize, MaxLevel and Iters mirror the OpenCV create() parameters. The
	// underlying sparse-to-dense solver uses its own local Lucas-Kanade schedule,
	// so these are advisory and retained for API compatibility.
	WinSize  int
	MaxLevel int
	Iters    int
}

// NewDensePyrLKOpticalFlow creates a dense pyramidal Lucas-Kanade estimator,
// mirroring cv::cuda::DensePyrLKOpticalFlow::create(winSize, maxLevel, iters).
// winSize must be >= 1 and maxLevel >= 0. OpenCV's defaults are winSize 13,
// maxLevel 3, iters 30.
func NewDensePyrLKOpticalFlow(winSize, maxLevel, iters int) *DensePyrLKOpticalFlow {
	if winSize < 1 {
		panic("cudaoptflow: NewDensePyrLKOpticalFlow requires winSize >= 1")
	}
	if maxLevel < 0 {
		panic("cudaoptflow: NewDensePyrLKOpticalFlow requires maxLevel >= 0")
	}
	return &DensePyrLKOpticalFlow{WinSize: winSize, MaxLevel: maxLevel, Iters: iters}
}

// Calc computes a dense flow field from prev to next, mirroring
// cv::cuda::DensePyrLKOpticalFlow::calc. The result is returned as a full-
// precision [optflow.FlowField]; use [FlowToGpuMat] to obtain the OpenCV-style
// two-channel GpuMat. stream is accepted for API compatibility and ignored.
// prev and next must be non-empty and equally sized.
func (o *DensePyrLKOpticalFlow) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField {
	requireFramePair(prev, next, "DensePyrLKOpticalFlow.Calc")
	_ = stream
	// A nil seed slice lets the sparse-to-dense estimator lay down its own
	// regular grid of Lucas-Kanade seeds.
	return optflow.CalcOpticalFlowSparseToDense(prev.mat, next.mat, nil)
}

// FarnebackOpticalFlow is the CPU-backed mirror of
// cv::cuda::FarnebackOpticalFlow. OpenCV's original fits local quadratic
// polynomials (polynomial expansion); the video package it delegates to uses an
// integer block-matching stand-in with the same coarse behaviour, so the
// polynomial-specific parameters (polyN, polySigma, pyrScale) have no analogue
// here and are omitted in favour of the two knobs the stand-in actually uses.
type FarnebackOpticalFlow struct {
	// WinSize is the half-size of the matching window; the full window is
	// (2*WinSize+1) square. Must be >= 1.
	WinSize int
	// SearchRadius is the maximum per-pixel displacement searched, in pixels.
	// Must be >= 1.
	SearchRadius int
}

// NewFarnebackOpticalFlow creates a dense Farneback-style estimator. winSize and
// searchRadius must be >= 1. It mirrors cv::cuda::FarnebackOpticalFlow::create at
// the API level while exposing the parameters meaningful to the CPU stand-in.
func NewFarnebackOpticalFlow(winSize, searchRadius int) *FarnebackOpticalFlow {
	if winSize < 1 || searchRadius < 1 {
		panic("cudaoptflow: NewFarnebackOpticalFlow requires winSize >= 1 and searchRadius >= 1")
	}
	return &FarnebackOpticalFlow{WinSize: winSize, SearchRadius: searchRadius}
}

// Calc computes a dense flow field from prev to next, mirroring
// cv::cuda::FarnebackOpticalFlow::calc. Delegates to
// video.CalcOpticalFlowFarneback and returns the result as an
// [optflow.FlowField]. stream is accepted for API compatibility and ignored.
// prev and next must be non-empty and equally sized.
func (o *FarnebackOpticalFlow) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField {
	requireFramePair(prev, next, "FarnebackOpticalFlow.Calc")
	_ = stream
	vf := video.CalcOpticalFlowFarneback(prev.mat, next.mat, o.WinSize, o.SearchRadius)
	return videoFlowToOptflow(vf)
}

// OpticalFlowDualTVL1 is the CPU-backed mirror of
// cv::cuda::OpticalFlowDual_TVL1: the duality-based TV-L1 method of Zach, Pock &
// Bischof (an L1 brightness-constancy data term with total-variation
// regularisation, solved by an alternating primal-dual scheme coarse-to-fine).
// Delegates to optflow.CalcOpticalFlowDenseTVL1.
type OpticalFlowDualTVL1 struct {
	// Params holds the TV-L1 solver configuration used by Calc.
	Params optflow.TVL1Params
}

// NewOpticalFlowDualTVL1 creates a TV-L1 estimator with the standard parameters
// (see optflow.DefaultTVL1Params), mirroring
// cv::cuda::OpticalFlowDual_TVL1::create.
func NewOpticalFlowDualTVL1() *OpticalFlowDualTVL1 {
	return &OpticalFlowDualTVL1{Params: optflow.DefaultTVL1Params()}
}

// NewOpticalFlowDualTVL1WithParams creates a TV-L1 estimator with explicit
// parameters, mirroring the fuller create(tau, lambda, theta, nscales, ...)
// overload.
func NewOpticalFlowDualTVL1WithParams(p optflow.TVL1Params) *OpticalFlowDualTVL1 {
	return &OpticalFlowDualTVL1{Params: p}
}

// Calc computes a dense flow field from prev to next, mirroring
// cv::cuda::OpticalFlowDual_TVL1::calc. The result is a full-precision
// [optflow.FlowField]. stream is accepted for API compatibility and ignored.
// prev and next must be non-empty and equally sized.
func (o *OpticalFlowDualTVL1) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField {
	requireFramePair(prev, next, "OpticalFlowDualTVL1.Calc")
	_ = stream
	return optflow.CalcOpticalFlowDenseTVL1(prev.mat, next.mat, o.Params)
}

// NvidiaHWOpticalFlow mirrors cv::cuda::NvidiaHWOpticalFlow, which in OpenCV
// drives the fixed-function NVIDIA Optical Flow Accelerator (NVOFA) present on
// Turing and later GPUs. No such hardware is reachable from a pure-Go process,
// so instead of a permanently-unavailable stub this type computes a real dense
// flow on the CPU using Dense Inverse Search (optflow.CalcOpticalFlowDIS). The
// public API matches OpenCV; only the execution substrate differs.
type NvidiaHWOpticalFlow struct {
	// PatchRadius is the half-size of the matching patch (patch side is
	// 2*PatchRadius+1). Must be >= 1.
	PatchRadius int
	// SearchRadius is the per-level integer search radius. Must be >= 1.
	SearchRadius int
	// Levels is the number of extra coarser pyramid levels above full
	// resolution. Must be >= 0.
	Levels int
}

// NewNvidiaHWOpticalFlow creates the CPU dense-flow estimator standing in for
// the NVIDIA hardware optical flow. patchRadius and searchRadius must be >= 1
// and levels >= 0. Reasonable defaults are patchRadius 4, searchRadius 2,
// levels 3.
func NewNvidiaHWOpticalFlow(patchRadius, searchRadius, levels int) *NvidiaHWOpticalFlow {
	if patchRadius < 1 || searchRadius < 1 {
		panic("cudaoptflow: NewNvidiaHWOpticalFlow requires patchRadius >= 1 and searchRadius >= 1")
	}
	if levels < 0 {
		panic("cudaoptflow: NewNvidiaHWOpticalFlow requires levels >= 0")
	}
	return &NvidiaHWOpticalFlow{PatchRadius: patchRadius, SearchRadius: searchRadius, Levels: levels}
}

// Calc computes a dense flow field from prev to next, mirroring
// cv::cuda::NvidiaHWOpticalFlow::calc. The result is a full-precision
// [optflow.FlowField] computed with Dense Inverse Search on the CPU. stream is
// accepted for API compatibility and ignored. prev and next must be non-empty
// and equally sized.
func (o *NvidiaHWOpticalFlow) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField {
	requireFramePair(prev, next, "NvidiaHWOpticalFlow.Calc")
	_ = stream
	return optflow.CalcOpticalFlowDIS(prev.mat, next.mat, o.PatchRadius, o.SearchRadius, o.Levels)
}

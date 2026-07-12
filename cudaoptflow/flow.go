package cudaoptflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/optflow"
	"github.com/malcolmston/opencv/video"
)

// DefaultFlowScale is the fixed-point scale used by [FlowToGpuMat] and
// [GpuMat.ToFlowField]. A flow component v is stored as the byte
// round(v*DefaultFlowScale)+128, so the representable range is roughly
// ±128/DefaultFlowScale pixels with a quantisation step of 1/DefaultFlowScale.
// At the default value of 8 that is about ±16 pixels at 0.125-pixel precision,
// which comfortably covers typical small-motion optical flow.
const DefaultFlowScale = 8.0

// flowBias centres the signed fixed-point encoding on 128 so that zero flow maps
// to the mid-grey byte value.
const flowBias = 128.0

// EncodeFlow packs a dense [optflow.FlowField] into a two-channel uint8 GpuMat,
// the CPU analogue of OpenCV's CV_32FC2 flow GpuMat. Channel 0 holds the
// horizontal component u and channel 1 the vertical component v, each stored as
// round(value*scale)+128 and saturated to [0,255].
//
// The encoding is lossy: it quantises to 1/scale of a pixel and clips beyond
// ±128/scale. It is meant for transport and visualisation; keep the full-
// precision FlowField for computation. scale must be > 0 and f must be non-nil.
func EncodeFlow(f *optflow.FlowField, scale float64) *GpuMat {
	if f == nil {
		panic("cudaoptflow: EncodeFlow requires a non-nil flow field")
	}
	if scale <= 0 {
		panic("cudaoptflow: EncodeFlow requires scale > 0")
	}
	m := cv.NewMat(f.Rows, f.Cols, 2)
	for y := 0; y < f.Rows; y++ {
		for x := 0; x < f.Cols; x++ {
			u, v := f.At(y, x)
			m.Set(y, x, 0, quantize(u, scale))
			m.Set(y, x, 1, quantize(v, scale))
		}
	}
	return &GpuMat{mat: m}
}

// FlowToGpuMat is EncodeFlow with the [DefaultFlowScale].
func FlowToGpuMat(f *optflow.FlowField) *GpuMat {
	return EncodeFlow(f, DefaultFlowScale)
}

// DecodeFlow unpacks a two-channel uint8 GpuMat produced by [EncodeFlow] back
// into an [optflow.FlowField], inverting the fixed-point encoding as
// (byte-128)/scale. g must be a non-empty two-channel GpuMat and scale must be
// > 0; the same scale passed to EncodeFlow must be used to recover the original
// magnitudes.
func DecodeFlow(g *GpuMat, scale float64) *optflow.FlowField {
	if g.Empty() {
		panic("cudaoptflow: DecodeFlow on an empty GpuMat")
	}
	if g.mat.Channels != 2 {
		panic("cudaoptflow: DecodeFlow requires a two-channel GpuMat")
	}
	if scale <= 0 {
		panic("cudaoptflow: DecodeFlow requires scale > 0")
	}
	m := g.mat
	f := optflow.NewFlowField(m.Rows, m.Cols)
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			u := (float64(m.At(y, x, 0)) - flowBias) / scale
			v := (float64(m.At(y, x, 1)) - flowBias) / scale
			f.Set(y, x, u, v)
		}
	}
	return f
}

// ToFlowField is DecodeFlow with the [DefaultFlowScale], the inverse of
// [FlowToGpuMat].
func (g *GpuMat) ToFlowField() *optflow.FlowField {
	return DecodeFlow(g, DefaultFlowScale)
}

// quantize maps a flow component to its signed fixed-point byte.
func quantize(v, scale float64) uint8 {
	q := math.Round(v*scale) + flowBias
	if q < 0 {
		q = 0
	}
	if q > 255 {
		q = 255
	}
	return uint8(q)
}

// videoFlowToOptflow converts a video.FlowField (produced by the video package's
// Farneback stand-in) into the optflow.FlowField type this package returns from
// its dense estimators. Both store an interleaved (dx, dy) pair per pixel.
func videoFlowToOptflow(vf *video.FlowField) *optflow.FlowField {
	f := optflow.NewFlowField(vf.Rows, vf.Cols)
	for y := 0; y < vf.Rows; y++ {
		for x := 0; x < vf.Cols; x++ {
			dx, dy := vf.At(y, x)
			f.Set(y, x, dx, dy)
		}
	}
	return f
}

// requireFramePair validates that two GpuMats are non-empty and equally sized,
// the common precondition of every dense estimator's Calc.
func requireFramePair(prev, next *GpuMat, fn string) {
	if prev.Empty() || next.Empty() {
		panic("cudaoptflow: " + fn + " requires non-empty frames")
	}
	pr, pc := prev.Size()
	nr, nc := next.Size()
	if pr != nr || pc != nc {
		panic("cudaoptflow: " + fn + " requires equal-sized frames")
	}
}

package videoproc

import (
	cv "github.com/malcolmston/opencv"
)

// BlendFrames returns the linear cross-fade (1-t)*a + t*b of two frames of
// identical size and channel count, rounded and clamped to 8-bit. With t=0 it
// reproduces a, with t=1 it reproduces b, and intermediate t gives a dissolve.
// It is the simplest form of frame interpolation, ignoring motion. t must lie in
// [0,1]. It panics on a size/channel mismatch or out-of-range t.
func BlendFrames(a, b *cv.Mat, t float64) *cv.Mat {
	videoprocRequireSame("BlendFrames", a, b)
	if t < 0 || t > 1 {
		panic("videoproc: BlendFrames requires t in [0,1]")
	}
	out := cv.NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		out.Data[i] = videoprocClampU8((1-t)*float64(a.Data[i]) + t*float64(b.Data[i]) + 0.5)
	}
	return out
}

// CrossFade returns count intermediate frames evenly spaced strictly between a
// and b (exclusive of both endpoints), each a linear blend produced by
// [BlendFrames]. count must be >= 1. The result has length count; frame k
// corresponds to t = (k+1)/(count+1). It panics on a size mismatch or count < 1.
func CrossFade(a, b *cv.Mat, count int) []*cv.Mat {
	videoprocRequireSame("CrossFade", a, b)
	if count < 1 {
		panic("videoproc: CrossFade requires count >= 1")
	}
	out := make([]*cv.Mat, count)
	for k := 0; k < count; k++ {
		t := float64(k+1) / float64(count+1)
		out[k] = BlendFrames(a, b, t)
	}
	return out
}

// WarpByFlow warps src by a dense flow field using backward mapping with
// bilinear sampling: the output pixel at (x, y) is sampled from src at
// (x+scale*dx, y+scale*dy), where (dx, dy) is the flow at (x, y). With scale=1
// and a flow that maps src→dst this reconstructs an approximation of dst from
// src. Out-of-range samples use edge clamping. The flow dimensions must match
// src. It panics on a size mismatch.
func WarpByFlow(src *cv.Mat, flow *FlowField, scale float64) *cv.Mat {
	if src == nil || src.Empty() {
		panic("videoproc: WarpByFlow requires a non-empty frame")
	}
	if flow == nil || flow.X.Rows != src.Rows || flow.X.Cols != src.Cols {
		panic("videoproc: WarpByFlow flow size mismatch")
	}
	out := cv.NewMat(src.Rows, src.Cols, src.Channels)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			dx := flow.X.Data[y*flow.X.Cols+x]
			dy := flow.Y.Data[y*flow.Y.Cols+x]
			sx := float64(x) + scale*dx
			sy := float64(y) + scale*dy
			oi := (y*out.Cols + x) * out.Channels
			for c := 0; c < out.Channels; c++ {
				out.Data[oi+c] = videoprocClampU8(videoprocSampleBilinear(src, sx, sy, c) + 0.5)
			}
		}
	}
	return out
}

// InterpolateFlow synthesises a motion-compensated intermediate frame at time t
// in [0,1] between a and b, given the forward flow field mapping a→b. Frame a is
// warped forward by t*flow and frame b is warped backward by (t-1)*flow (both via
// [WarpByFlow]); the two warped frames are then blended by (1-t)*warpA + t*warpB.
// At t=0 the result is a and at t=1 it is b. Unlike [BlendFrames] this follows
// motion, so a moving object appears at its interpolated position rather than
// ghosting. It panics on a size/channel mismatch or out-of-range t.
func InterpolateFlow(a, b *cv.Mat, flow *FlowField, t float64) *cv.Mat {
	videoprocRequireSame("InterpolateFlow", a, b)
	if t < 0 || t > 1 {
		panic("videoproc: InterpolateFlow requires t in [0,1]")
	}
	if flow == nil || flow.X.Rows != a.Rows || flow.X.Cols != a.Cols {
		panic("videoproc: InterpolateFlow flow size mismatch")
	}
	warpA := WarpByFlow(a, flow, t)
	warpB := WarpByFlow(b, flow, t-1)
	return BlendFrames(warpA, warpB, t)
}

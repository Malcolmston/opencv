package cudalegacy

import (
	cv "github.com/malcolmston/opencv"
)

// InterpolateFrames is a CPU-backed mirror of OpenCV's cv::cuda::interpolateFrames.
// Given two consecutive frames and the forward optical flow from frame0 to
// frame1, it synthesises the intermediate frame at temporal position pos in
// [0,1] (pos = 0 reproduces frame0, pos = 1 reproduces frame1) by
// motion-compensated bidirectional warping.
//
// For each output pixel the forward flow at that location is sampled; frame0 is
// read at the position that pixel came from (shifted back by pos·flow) and
// frame1 at where it is going (shifted forward by (1-pos)·flow, i.e. along the
// negated flow). The two samples are blended with weights (1-pos) and pos. When
// backward is non-nil its flow is used for the frame1 lookup instead of the
// negated forward flow, matching OpenCV's use of both flow directions;
// otherwise the negated forward flow is used.
//
// frame0 and frame1 must be non-empty and identically shaped, and the flow must
// match their size. pos is clamped to [0,1]. It panics on nil/empty/mismatched
// inputs. The stream is a no-op.
func InterpolateFrames(frame0, frame1 *GpuMat, forward, backward *Flow, pos float64, stream *Stream) *GpuMat {
	_ = stream
	m0 := requireMat(frame0, "InterpolateFrames")
	m1 := requireMat(frame1, "InterpolateFrames")
	if m0.Rows != m1.Rows || m0.Cols != m1.Cols || m0.Channels != m1.Channels {
		panic("cudalegacy: InterpolateFrames frames differ in shape")
	}
	if forward == nil {
		panic("cudalegacy: InterpolateFrames requires a forward flow")
	}
	if forward.Rows() != m0.Rows || forward.Cols() != m0.Cols {
		panic("cudalegacy: InterpolateFrames flow size does not match frames")
	}
	if backward != nil && (backward.Rows() != m0.Rows || backward.Cols() != m0.Cols) {
		panic("cudalegacy: InterpolateFrames backward flow size does not match frames")
	}
	if pos < 0 {
		pos = 0
	} else if pos > 1 {
		pos = 1
	}

	rows, cols, ch := m0.Rows, m0.Cols, m0.Channels
	out := cv.NewMat(rows, cols, ch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			fu, fv := forward.At(y, x)
			// Source in frame0: trace back along the forward flow.
			x0 := float64(x) - pos*fu
			y0 := float64(y) - pos*fv
			// Destination in frame1: trace forward along the (backward) flow.
			var bu, bv float64
			if backward != nil {
				bu, bv = backward.At(y, x)
			} else {
				bu, bv = -fu, -fv
			}
			x1 := float64(x) - (1-pos)*bu
			y1 := float64(y) - (1-pos)*bv

			oi := (y*cols + x) * ch
			for c := 0; c < ch; c++ {
				s0 := bilerpMat(m0, x0, y0, c)
				s1 := bilerpMat(m1, x1, y1, c)
				out.Data[oi+c] = clampByte((1-pos)*s0 + pos*s1)
			}
		}
	}
	return GpuMatFromMat(out)
}

package imgprocx

import cv "github.com/malcolmston/opencv"

// checkAccum panics unless src is single-channel and matches dst in size.
func checkAccum(src *cv.Mat, dst *cv.FloatMat, name string) {
	requireSingleChannel(src, name)
	if dst.Rows != src.Rows || dst.Cols != src.Cols {
		panic("imgprocx: " + name + " size mismatch between src and dst")
	}
}

// Accumulate adds the single-channel image src into the running sum dst,
// updating dst in place: dst[y][x] += src[y][x]. It mirrors cv2.accumulate and
// is the building block for background modelling from a stream of frames. It
// panics unless src is single-channel and dst has the same dimensions.
func Accumulate(src *cv.Mat, dst *cv.FloatMat) {
	checkAccum(src, dst, "Accumulate")
	for i, v := range src.Data {
		dst.Data[i] += float64(v)
	}
}

// AccumulateSquare adds the element-wise square of src into dst in place:
// dst[y][x] += src[y][x]². It mirrors cv2.accumulateSquare and is used together
// with [Accumulate] to maintain a running mean and variance. It panics unless
// src is single-channel and dst has the same dimensions.
func AccumulateSquare(src *cv.Mat, dst *cv.FloatMat) {
	checkAccum(src, dst, "AccumulateSquare")
	for i, v := range src.Data {
		f := float64(v)
		dst.Data[i] += f * f
	}
}

// AccumulateProduct adds the element-wise product of src1 and src2 into dst in
// place: dst[y][x] += src1[y][x]·src2[y][x]. It mirrors cv2.accumulateProduct
// and yields the running cross-term needed for a covariance estimate. It panics
// unless both sources are single-channel and share dst's dimensions.
func AccumulateProduct(src1, src2 *cv.Mat, dst *cv.FloatMat) {
	checkAccum(src1, dst, "AccumulateProduct")
	checkAccum(src2, dst, "AccumulateProduct")
	for i := range dst.Data {
		dst.Data[i] += float64(src1.Data[i]) * float64(src2.Data[i])
	}
}

// AccumulateWeighted blends src into dst as a running exponentially-weighted
// average, updating dst in place:
//
//	dst[y][x] = (1-alpha)·dst[y][x] + alpha·src[y][x].
//
// It mirrors cv2.accumulateWeighted; alpha in (0,1] sets how quickly the average
// forgets older frames (larger alpha adapts faster). It panics unless src is
// single-channel and dst has the same dimensions, or if alpha is outside [0,1].
func AccumulateWeighted(src *cv.Mat, dst *cv.FloatMat, alpha float64) {
	checkAccum(src, dst, "AccumulateWeighted")
	if alpha < 0 || alpha > 1 {
		panic("imgprocx: AccumulateWeighted requires alpha in [0,1]")
	}
	beta := 1 - alpha
	for i, v := range src.Data {
		dst.Data[i] = beta*dst.Data[i] + alpha*float64(v)
	}
}

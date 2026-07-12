package imgprocx

import cv "github.com/malcolmston/opencv"

// BlendLinear blends two images with per-pixel weights, mirroring
// cv2.blendLinear. Each output sample is the weighted average
//
//	dst[y][x] = (w1·src1[y][x] + w2·src2[y][x]) / (w1 + w2)
//
// where the weights w1 = weights1[y][x] and w2 = weights2[y][x] are read from
// the single-plane [cv.FloatMat] weight maps; where both weights are (near)
// zero the output is zero. src1 and src2 must have identical dimensions and
// channel counts, and the weight maps must match their spatial size. The result
// is a new [cv.Mat] with samples rounded and clamped to [0,255].
//
// It is the core compositing step of multi-band and feathered image blending,
// where the weight maps encode each source's contribution (for example a
// distance-to-seam ramp).
func BlendLinear(src1, src2 *cv.Mat, weights1, weights2 *cv.FloatMat) *cv.Mat {
	if src1.Rows != src2.Rows || src1.Cols != src2.Cols {
		panic("imgprocx: BlendLinear requires src1 and src2 of equal size")
	}
	if src1.Channels != src2.Channels {
		panic("imgprocx: BlendLinear requires matching channel counts")
	}
	rows, cols, ch := src1.Rows, src1.Cols, src1.Channels
	if weights1.Rows != rows || weights1.Cols != cols ||
		weights2.Rows != rows || weights2.Cols != cols {
		panic("imgprocx: BlendLinear weight maps must match the image size")
	}
	dst := cv.NewMat(rows, cols, ch)
	for p := 0; p < rows*cols; p++ {
		w1 := weights1.Data[p]
		w2 := weights2.Data[p]
		denom := w1 + w2
		base := p * ch
		if denom < 1e-12 {
			continue // leaves the samples at zero
		}
		for c := 0; c < ch; c++ {
			v := (w1*float64(src1.Data[base+c]) + w2*float64(src2.Data[base+c])) / denom
			dst.Data[base+c] = clampUint8(v)
		}
	}
	return dst
}

package dnn

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// BlobFromImage converts a [cv.Mat] into a 4-D NCHW input blob suitable for a
// [Net]. The result has shape [1, C, H, W] where C is the Mat's channel count,
// and each sample is computed as
//
//	blob = scale * (pixel - mean[channel])
//
// matching OpenCV's cv::dnn::blobFromImage (mean subtraction then scaling).
// The Mat is not resized; feed it through [cv.Resize] first if the network
// expects a fixed input size.
//
// mean supplies a per-output-channel offset. A nil or short slice is treated
// as zero for the missing channels; extra entries are ignored. When swapRB is
// true and the Mat has three channels, the red and blue channels are swapped
// (RGB↔BGR) before mean subtraction, so mean is given in the output channel
// order.
func BlobFromImage(img *cv.Mat, scale float64, mean []float64, swapRB bool) *Tensor {
	if img == nil || img.Empty() {
		panic("dnn: BlobFromImage given an empty image")
	}
	c := img.Channels
	h := img.Rows
	w := img.Cols
	out := NewTensor(1, c, h, w)
	hw := h * w
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for ch := 0; ch < c; ch++ {
				src := ch
				if swapRB && c == 3 {
					if ch == 0 {
						src = 2
					} else if ch == 2 {
						src = 0
					}
				}
				var m float64
				if ch < len(mean) {
					m = mean[ch]
				}
				v := scale * (float64(img.At(y, x, src)) - m)
				out.Data[ch*hw+y*w+x] = v
			}
		}
	}
	return out
}

// BlobToImage is the inverse of [BlobFromImage]: it reconstructs a [cv.Mat]
// from one image of an NCHW blob, undoing the normalization with
//
//	pixel = value/scale + mean[channel]
//
// and rounding-and-clamping the result into [0,255]. The blob must be rank 4;
// batch element index selects which image to convert. swapRB and mean mirror
// [BlobFromImage]. scale must be non-zero.
func BlobToImage(t *Tensor, index int, scale float64, mean []float64, swapRB bool) *cv.Mat {
	if t == nil || t.Dims() != 4 {
		panic("dnn: BlobToImage requires a rank-4 NCHW tensor")
	}
	if scale == 0 {
		panic("dnn: BlobToImage scale must be non-zero")
	}
	n, c, h, w := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	if index < 0 || index >= n {
		panic(fmt.Sprintf("dnn: BlobToImage index %d out of range for batch %d", index, n))
	}
	out := cv.NewMat(h, w, c)
	hw := h * w
	imgBase := index * c * hw
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for ch := 0; ch < c; ch++ {
				var m float64
				if ch < len(mean) {
					m = mean[ch]
				}
				v := t.Data[imgBase+ch*hw+y*w+x]/scale + m
				dst := ch
				if swapRB && c == 3 {
					if ch == 0 {
						dst = 2
					} else if ch == 2 {
						dst = 0
					}
				}
				out.Set(y, x, dst, clampU8(v))
			}
		}
	}
	return out
}

// clampU8 rounds v to the nearest integer and clamps it into [0,255]. It
// reimplements the root package's unexported rounding helper locally.
func clampU8(v float64) uint8 {
	v += 0.5
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

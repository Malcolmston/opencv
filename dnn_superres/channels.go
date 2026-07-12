package dnn_superres

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// UpsampleFunc is the signature shared by every free-function upsampler in this
// package: it enlarges src by an integer scale and returns a new Mat or an
// error. It lets [UpsamplePerChannel], [UpsampleLumaOnly] and [Benchmark] treat
// the algorithms uniformly.
type UpsampleFunc func(src *cv.Mat, scale int) (*cv.Mat, error)

// UpsamplePerChannel enlarges src by splitting it into single-channel planes,
// running up on each plane independently and merging the results. Every
// upsampler in this package already handles multi-channel data this way, so the
// function is most useful for applying a method whose behaviour you want to
// verify is genuinely per-channel, or for feeding a method a specific plane
// count. It returns an error for an empty image, from the underlying up call, or
// if up yields planes of differing sizes.
func UpsamplePerChannel(src *cv.Mat, scale int, up UpsampleFunc) (*cv.Mat, error) {
	if src == nil || src.Empty() {
		return nil, fmt.Errorf("dnn_superres: source image is empty")
	}
	if up == nil {
		return nil, fmt.Errorf("dnn_superres: nil UpsampleFunc")
	}
	if src.Channels == 1 {
		return up(src, scale)
	}
	planes := src.Split()
	out := make([]*cv.Mat, len(planes))
	for i, p := range planes {
		u, err := up(p, scale)
		if err != nil {
			return nil, err
		}
		if u.Channels != 1 {
			return nil, fmt.Errorf("dnn_superres: per-channel up produced %d channels", u.Channels)
		}
		if i > 0 && (u.Rows != out[0].Rows || u.Cols != out[0].Cols) {
			return nil, fmt.Errorf("dnn_superres: per-channel size mismatch")
		}
		out[i] = u
	}
	return cv.Merge(out), nil
}

// UpsampleLumaOnly enlarges src by upscaling only the luma channel with the
// high-quality method up, while the chroma channels are enlarged with cheap
// bilinear interpolation. This mirrors standard super-resolution practice: the
// human eye is far more sensitive to luminance detail than to colour, so
// spending the expensive reconstruction on luma alone gives almost all of the
// visible benefit at a fraction of the cost, and it also matches how OpenCV's
// dnn_superres models are typically applied (on the Y plane of YCrCb).
//
// For a single-channel image up is applied directly. For a three-channel image
// src is converted RGB→YCrCb, Y is upscaled with up, Cr and Cb with bilinear,
// and the result is converted back to RGB. Any other channel count falls back to
// [UpsamplePerChannel]. It returns an error for an empty image or from up (which
// must accept the given scale).
func UpsampleLumaOnly(src *cv.Mat, scale int, up UpsampleFunc) (*cv.Mat, error) {
	if src == nil || src.Empty() {
		return nil, fmt.Errorf("dnn_superres: source image is empty")
	}
	if up == nil {
		return nil, fmt.Errorf("dnn_superres: nil UpsampleFunc")
	}
	if scale < 2 {
		return nil, fmt.Errorf("dnn_superres: unsupported scale %d (want >= 2)", scale)
	}
	switch src.Channels {
	case 1:
		return up(src, scale)
	case 3:
		ycrcb := cv.CvtColor(src, cv.ColorRGB2YCrCb)
		planes := ycrcb.Split() // Y, Cr, Cb
		y, err := up(planes[0], scale)
		if err != nil {
			return nil, err
		}
		if y.Channels != 1 || y.Rows != src.Rows*scale || y.Cols != src.Cols*scale {
			return nil, fmt.Errorf("dnn_superres: luma up produced unexpected shape")
		}
		cr := resampleSeparable(planes[1], src.Cols*scale, src.Rows*scale, triangle, 1)
		cb := resampleSeparable(planes[2], src.Cols*scale, src.Rows*scale, triangle, 1)
		merged := cv.Merge([]*cv.Mat{y, cr, cb})
		return cv.CvtColor(merged, cv.ColorYCrCb2RGB), nil
	default:
		return UpsamplePerChannel(src, scale, up)
	}
}

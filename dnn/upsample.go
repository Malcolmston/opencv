package dnn

import (
	"fmt"
	"math"
)

// UpsampleMode selects the interpolation used by [Upsample].
type UpsampleMode int

const (
	// UpsampleNearest repeats the nearest source pixel (no interpolation).
	UpsampleNearest UpsampleMode = iota
	// UpsampleBilinear linearly interpolates the four surrounding source pixels
	// using OpenCV's half-pixel sample centres.
	UpsampleBilinear
)

// Upsample enlarges the spatial dimensions of an NCHW tensor by integer factors
// ScaleH and ScaleW, mapping [N, C, H, W] to [N, C, H*ScaleH, W*ScaleW]. Batch
// and channel axes are untouched. Nearest-neighbour and bilinear interpolation
// are supported. Bilinear uses half-pixel centres — output pixel o samples
// source coordinate (o+0.5)/scale − 0.5 — matching OpenCV's default resize and
// PyTorch's align_corners=false.
type Upsample struct {
	// ScaleH, ScaleW are the vertical and horizontal magnification factors (>= 1).
	ScaleH, ScaleW int
	// Mode selects nearest or bilinear interpolation.
	Mode UpsampleMode
}

// NewUpsampleNearest builds a nearest-neighbour Upsample with a common factor.
func NewUpsampleNearest(scale int) *Upsample {
	return &Upsample{ScaleH: scale, ScaleW: scale, Mode: UpsampleNearest}
}

// NewUpsampleBilinear builds a bilinear Upsample with a common factor.
func NewUpsampleBilinear(scale int) *Upsample {
	return &Upsample{ScaleH: scale, ScaleW: scale, Mode: UpsampleBilinear}
}

// Forward upsamples the single NCHW input tensor.
func (u *Upsample) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Upsample expects 1 input, got %d", len(inputs)))
	}
	if u.ScaleH < 1 || u.ScaleW < 1 {
		panic(fmt.Sprintf("dnn: Upsample scale must be >= 1, got %dx%d", u.ScaleH, u.ScaleW))
	}
	in := inputs[0]
	if in.Dims() != 4 {
		panic(fmt.Sprintf("dnn: Upsample input must be rank-4 NCHW, got %s", in))
	}
	n, ch, h, w := in.Shape[0], in.Shape[1], in.Shape[2], in.Shape[3]
	outH := h * u.ScaleH
	outW := w * u.ScaleW
	out := NewTensor(n, ch, outH, outW)
	inHW := h * w
	outHW := outH * outW
	for nc := 0; nc < n*ch; nc++ {
		src := in.Data[nc*inHW : nc*inHW+inHW]
		dst := out.Data[nc*outHW : nc*outHW+outHW]
		if u.Mode == UpsampleNearest {
			for oy := 0; oy < outH; oy++ {
				iy := oy / u.ScaleH
				for ox := 0; ox < outW; ox++ {
					ix := ox / u.ScaleW
					dst[oy*outW+ox] = src[iy*w+ix]
				}
			}
			continue
		}
		// Bilinear with half-pixel centres.
		for oy := 0; oy < outH; oy++ {
			fy := (float64(oy)+0.5)/float64(u.ScaleH) - 0.5
			y0, y1, wy := interpWeights(fy, h)
			for ox := 0; ox < outW; ox++ {
				fx := (float64(ox)+0.5)/float64(u.ScaleW) - 0.5
				x0, x1, wx := interpWeights(fx, w)
				v00 := src[y0*w+x0]
				v01 := src[y0*w+x1]
				v10 := src[y1*w+x0]
				v11 := src[y1*w+x1]
				top := v00 + (v01-v00)*wx
				bot := v10 + (v11-v10)*wx
				dst[oy*outW+ox] = top + (bot-top)*wy
			}
		}
	}
	return []*Tensor{out}
}

// interpWeights returns the two clamped neighbour indices and the fractional
// weight of the second, for a source coordinate f over a dimension of size dim.
func interpWeights(f float64, dim int) (i0, i1 int, frac float64) {
	if f < 0 {
		f = 0
	}
	i0 = int(math.Floor(f))
	frac = f - float64(i0)
	if i0 >= dim-1 {
		i0 = dim - 1
		i1 = dim - 1
		frac = 0
		return
	}
	i1 = i0 + 1
	return
}

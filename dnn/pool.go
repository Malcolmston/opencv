package dnn

import (
	"fmt"
	"math"
)

// poolParams holds the geometry shared by the pooling layers.
type poolParams struct {
	KernelH, KernelW int
	StrideH, StrideW int
	PadH, PadW       int
}

// outSize returns the pooled height and width for an input of size h×w.
func (p poolParams) outSize(h, w int) (int, int) {
	oh := (h+2*p.PadH-p.KernelH)/p.StrideH + 1
	ow := (w+2*p.PadW-p.KernelW)/p.StrideW + 1
	return oh, ow
}

func (p poolParams) validate(name string) {
	if p.KernelH < 1 || p.KernelW < 1 {
		panic(fmt.Sprintf("dnn: %s kernel must be >= 1, got %dx%d", name, p.KernelH, p.KernelW))
	}
	if p.StrideH < 1 || p.StrideW < 1 {
		panic(fmt.Sprintf("dnn: %s stride must be >= 1, got %dx%d", name, p.StrideH, p.StrideW))
	}
	if p.PadH < 0 || p.PadW < 0 {
		panic(fmt.Sprintf("dnn: %s pad must be >= 0", name))
	}
}

// MaxPool2D downsamples an NCHW input by taking the maximum over each pooling
// window. Padding (if any) is excluded from the maximum — only real input
// samples participate.
type MaxPool2D struct {
	poolParams
}

// NewMaxPool2D builds a max-pooling layer with a square kernel and given
// stride, no padding.
func NewMaxPool2D(kernel, stride int) *MaxPool2D {
	m := &MaxPool2D{poolParams{KernelH: kernel, KernelW: kernel, StrideH: stride, StrideW: stride}}
	m.validate("MaxPool2D")
	return m
}

// Forward max-pools the single NCHW input tensor.
func (m *MaxPool2D) Forward(inputs []*Tensor) []*Tensor {
	return []*Tensor{pool(inputs, m.poolParams, "MaxPool2D", false)}
}

// AvgPool2D downsamples an NCHW input by averaging each pooling window. The
// average divides by the number of real input samples in the window (padding
// is excluded, matching count_include_pad=false).
type AvgPool2D struct {
	poolParams
}

// NewAvgPool2D builds an average-pooling layer with a square kernel and given
// stride, no padding.
func NewAvgPool2D(kernel, stride int) *AvgPool2D {
	a := &AvgPool2D{poolParams{KernelH: kernel, KernelW: kernel, StrideH: stride, StrideW: stride}}
	a.validate("AvgPool2D")
	return a
}

// Forward average-pools the single NCHW input tensor.
func (a *AvgPool2D) Forward(inputs []*Tensor) []*Tensor {
	return []*Tensor{pool(inputs, a.poolParams, "AvgPool2D", true)}
}

// pool implements both pooling modes over an NCHW input.
func pool(inputs []*Tensor, p poolParams, name string, average bool) *Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: %s expects 1 input, got %d", name, len(inputs)))
	}
	in := inputs[0]
	if in.Dims() != 4 {
		panic(fmt.Sprintf("dnn: %s input must be rank-4 NCHW, got %s", name, in))
	}
	n, ch, h, w := in.Shape[0], in.Shape[1], in.Shape[2], in.Shape[3]
	oh, ow := p.outSize(h, w)
	if oh <= 0 || ow <= 0 {
		panic(fmt.Sprintf("dnn: %s produces empty output %dx%d for input %dx%d", name, oh, ow, h, w))
	}
	out := NewTensor(n, ch, oh, ow)
	inHW := h * w
	for ni := 0; ni < n; ni++ {
		for ci := 0; ci < ch; ci++ {
			chanBase := (ni*ch + ci) * inHW
			for oy := 0; oy < oh; oy++ {
				iy0 := oy*p.StrideH - p.PadH
				for ox := 0; ox < ow; ox++ {
					ix0 := ox*p.StrideW - p.PadW
					var acc float64
					count := 0
					best := math.Inf(-1)
					for ky := 0; ky < p.KernelH; ky++ {
						iy := iy0 + ky
						if iy < 0 || iy >= h {
							continue
						}
						row := chanBase + iy*w
						for kx := 0; kx < p.KernelW; kx++ {
							ix := ix0 + kx
							if ix < 0 || ix >= w {
								continue
							}
							v := in.Data[row+ix]
							if average {
								acc += v
							} else if v > best {
								best = v
							}
							count++
						}
					}
					var val float64
					switch {
					case average:
						if count > 0 {
							val = acc / float64(count)
						}
					case count > 0:
						val = best
					default:
						val = 0
					}
					out.Data[((ni*ch+ci)*oh+oy)*ow+ox] = val
				}
			}
		}
	}
	return out
}

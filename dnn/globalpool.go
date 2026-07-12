package dnn

import (
	"fmt"
	"math"
)

// GlobalAvgPool collapses the spatial extent of an NCHW tensor by averaging
// each channel over its entire height and width, mapping [N, C, H, W] to
// [N, C, 1, 1]. It is the standard head of a fully-convolutional classifier,
// replacing a flatten-plus-dense layer. Use a [Flatten] afterwards to obtain a
// plain [N, C] feature matrix.
type GlobalAvgPool struct{}

// Forward global-average-pools the single NCHW input tensor.
func (g *GlobalAvgPool) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: GlobalAvgPool expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	if in.Dims() != 4 {
		panic(fmt.Sprintf("dnn: GlobalAvgPool input must be rank-4 NCHW, got %s", in))
	}
	n, ch, h, w := in.Shape[0], in.Shape[1], in.Shape[2], in.Shape[3]
	hw := h * w
	out := NewTensor(n, ch, 1, 1)
	for nc := 0; nc < n*ch; nc++ {
		base := nc * hw
		var sum float64
		for k := 0; k < hw; k++ {
			sum += in.Data[base+k]
		}
		out.Data[nc] = sum / float64(hw)
	}
	return []*Tensor{out}
}

// LRN performs local response normalization over an NCHW tensor. In the default
// across-channel mode each activation is divided by a term formed from the
// squared activations of the Size neighbouring channels at the same spatial
// position:
//
//	y_c = x_c / (K + a * Σ_{j∈window(c)} x_j²)^Beta
//
// where a is Alpha/Size when NormBySize is true (the default, matching Caffe
// and OpenCV) or Alpha otherwise, and window(c) is the run of Size channels
// centred on c and clamped to the valid range. In within-channel mode the sum
// runs instead over a Size×Size spatial neighbourhood of the same channel.
type LRN struct {
	// Size is the neighbourhood size (number of channels, or the side of the
	// spatial square); it should be positive and is usually odd.
	Size int
	// Alpha scales the squared-sum term.
	Alpha float64
	// Beta is the exponent applied to the normalizer.
	Beta float64
	// K is the additive bias inside the normalizer.
	K float64
	// AcrossChannels selects channel-wise (true) vs spatial (false) pooling of
	// the squared activations.
	AcrossChannels bool
	// NormBySize divides Alpha by Size before use when true.
	NormBySize bool
}

// NewLRN builds an across-channel LRN with the given window size and
// coefficients and NormBySize enabled (the common configuration).
func NewLRN(size int, alpha, beta, k float64) *LRN {
	if size < 1 {
		panic(fmt.Sprintf("dnn: LRN size must be >= 1, got %d", size))
	}
	return &LRN{Size: size, Alpha: alpha, Beta: beta, K: k, AcrossChannels: true, NormBySize: true}
}

// Forward normalizes the single NCHW input tensor, preserving its shape.
func (l *LRN) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: LRN expects 1 input, got %d", len(inputs)))
	}
	if l.Size < 1 {
		panic(fmt.Sprintf("dnn: LRN size must be >= 1, got %d", l.Size))
	}
	in := inputs[0]
	if in.Dims() != 4 {
		panic(fmt.Sprintf("dnn: LRN input must be rank-4 NCHW, got %s", in))
	}
	a := l.Alpha
	if l.NormBySize {
		a /= float64(l.Size)
	}
	if l.AcrossChannels {
		return []*Tensor{l.forwardAcross(in, a)}
	}
	return []*Tensor{l.forwardWithin(in, a)}
}

// forwardAcross normalizes each element using neighbouring channels.
func (l *LRN) forwardAcross(in *Tensor, a float64) *Tensor {
	n, ch, h, w := in.Shape[0], in.Shape[1], in.Shape[2], in.Shape[3]
	hw := h * w
	pre := (l.Size - 1) / 2
	out := NewTensor(in.Shape...)
	for ni := 0; ni < n; ni++ {
		nBase := ni * ch * hw
		for p := 0; p < hw; p++ {
			for c := 0; c < ch; c++ {
				var sum float64
				lo := c - pre
				hi := lo + l.Size // exclusive
				if lo < 0 {
					lo = 0
				}
				if hi > ch {
					hi = ch
				}
				for j := lo; j < hi; j++ {
					v := in.Data[nBase+j*hw+p]
					sum += v * v
				}
				x := in.Data[nBase+c*hw+p]
				out.Data[nBase+c*hw+p] = x / math.Pow(l.K+a*sum, l.Beta)
			}
		}
	}
	return out
}

// forwardWithin normalizes each element using a spatial square of its channel.
func (l *LRN) forwardWithin(in *Tensor, a float64) *Tensor {
	n, ch, h, w := in.Shape[0], in.Shape[1], in.Shape[2], in.Shape[3]
	hw := h * w
	pre := (l.Size - 1) / 2
	out := NewTensor(in.Shape...)
	for nc := 0; nc < n*ch; nc++ {
		base := nc * hw
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var sum float64
				for dy := -pre; dy < -pre+l.Size; dy++ {
					yy := y + dy
					if yy < 0 || yy >= h {
						continue
					}
					for dx := -pre; dx < -pre+l.Size; dx++ {
						xx := x + dx
						if xx < 0 || xx >= w {
							continue
						}
						v := in.Data[base+yy*w+xx]
						sum += v * v
					}
				}
				xv := in.Data[base+y*w+x]
				out.Data[base+y*w+x] = xv / math.Pow(l.K+a*sum, l.Beta)
			}
		}
	}
	return out
}

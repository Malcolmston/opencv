package dnn

import "fmt"

// ConvTranspose2D is a 2-D transposed convolution (also called deconvolution or
// fractionally-strided convolution) over a batched NCHW input. It is the
// gradient-flow adjoint of [Conv2D] and is the layer commonly used to upsample
// feature maps in segmentation and generative networks.
//
// The weight tensor has shape [inChannels, outChannels, kernelH, kernelW] —
// note the input-channel-first layout, matching PyTorch's ConvTranspose2d and
// OpenCV's deconvolution blob. The optional bias has shape [outChannels]. For
// an input of shape [N, inC, H, W] the output has shape [N, outC, outH, outW]
// with
//
//	outH = (H-1)*StrideH - 2*PadH + DilationH*(kH-1) + OutPadH + 1
//	outW = (W-1)*StrideW - 2*PadW + DilationW*(kW-1) + OutPadW + 1
//
// Each input sample is scattered into the output through the flipped kernel and
// accumulated. Padding here removes a border from the (larger) output.
type ConvTranspose2D struct {
	// Weights has shape [inChannels, outChannels, kernelH, kernelW].
	Weights *Tensor
	// Bias has shape [outChannels], or is nil for a bias-free layer.
	Bias *Tensor
	// StrideH, StrideW are the vertical and horizontal strides (>= 1).
	StrideH, StrideW int
	// PadH, PadW are the amounts cropped from each side of the output (>= 0).
	PadH, PadW int
	// DilationH, DilationW are the kernel dilation factors (>= 1).
	DilationH, DilationW int
	// OutPadH, OutPadW add extra rows/columns to one side of the output to
	// disambiguate the output size for strides > 1 (>= 0).
	OutPadH, OutPadW int
}

// NewConvTranspose2D builds a transposed convolution with symmetric stride,
// padding and dilation and no output padding. weights must be rank 4
// ([inC, outC, kH, kW]); bias, if non-nil, must be rank 1 with outC elements.
// For asymmetric geometry set the struct fields directly.
func NewConvTranspose2D(weights, bias *Tensor, stride, pad, dilation int) *ConvTranspose2D {
	if weights == nil || weights.Dims() != 4 {
		panic("dnn: ConvTranspose2D weights must be a rank-4 tensor [inC, outC, kH, kW]")
	}
	if stride < 1 {
		panic(fmt.Sprintf("dnn: ConvTranspose2D stride must be >= 1, got %d", stride))
	}
	if dilation < 1 {
		panic(fmt.Sprintf("dnn: ConvTranspose2D dilation must be >= 1, got %d", dilation))
	}
	if pad < 0 {
		panic(fmt.Sprintf("dnn: ConvTranspose2D pad must be >= 0, got %d", pad))
	}
	if bias != nil && (bias.Dims() != 1 || bias.Shape[0] != weights.Shape[1]) {
		panic(fmt.Sprintf("dnn: ConvTranspose2D bias must have shape [%d], got %v", weights.Shape[1], bias.Shape))
	}
	return &ConvTranspose2D{
		Weights:   weights,
		Bias:      bias,
		StrideH:   stride,
		StrideW:   stride,
		PadH:      pad,
		PadW:      pad,
		DilationH: dilation,
		DilationW: dilation,
	}
}

// Forward transposed-convolves the single NCHW input and returns one NCHW output.
func (c *ConvTranspose2D) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: ConvTranspose2D expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	if in.Dims() != 4 {
		panic(fmt.Sprintf("dnn: ConvTranspose2D input must be rank-4 NCHW, got %s", in))
	}
	wInC := c.Weights.Shape[0]
	outC := c.Weights.Shape[1]
	kH := c.Weights.Shape[2]
	kW := c.Weights.Shape[3]

	n := in.Shape[0]
	inC := in.Shape[1]
	h := in.Shape[2]
	w := in.Shape[3]
	if inC != wInC {
		panic(fmt.Sprintf("dnn: ConvTranspose2D input has %d channels, weights expect %d", inC, wInC))
	}

	outH := (h-1)*c.StrideH - 2*c.PadH + c.DilationH*(kH-1) + c.OutPadH + 1
	outW := (w-1)*c.StrideW - 2*c.PadW + c.DilationW*(kW-1) + c.OutPadW + 1
	if outH <= 0 || outW <= 0 {
		panic(fmt.Sprintf("dnn: ConvTranspose2D produces empty output %dx%d for input %dx%d", outH, outW, h, w))
	}

	out := NewTensor(n, outC, outH, outW)

	// Seed the output with the per-channel bias.
	if c.Bias != nil {
		outHW := outH * outW
		for ni := 0; ni < n; ni++ {
			for oc := 0; oc < outC; oc++ {
				base := (ni*outC + oc) * outHW
				b := c.Bias.Data[oc]
				for k := 0; k < outHW; k++ {
					out.Data[base+k] = b
				}
			}
		}
	}

	inCHW := inC * h * w
	inHW := h * w
	wStrideIn := outC * kH * kW
	wStrideOut := kH * kW
	outCHW := outC * outH * outW
	outHW := outH * outW

	for ni := 0; ni < n; ni++ {
		inBase := ni * inCHW
		outBase := ni * outCHW
		for ic := 0; ic < inC; ic++ {
			inChan := inBase + ic*inHW
			wChanIn := ic * wStrideIn
			for iy := 0; iy < h; iy++ {
				for ix := 0; ix < w; ix++ {
					v := in.Data[inChan+iy*w+ix]
					if v == 0 {
						continue
					}
					for oc := 0; oc < outC; oc++ {
						outChan := outBase + oc*outHW
						wOC := wChanIn + oc*wStrideOut
						for ky := 0; ky < kH; ky++ {
							oy := iy*c.StrideH - c.PadH + ky*c.DilationH
							if oy < 0 || oy >= outH {
								continue
							}
							wRow := wOC + ky*kW
							outRow := outChan + oy*outW
							for kx := 0; kx < kW; kx++ {
								ox := ix*c.StrideW - c.PadW + kx*c.DilationW
								if ox < 0 || ox >= outW {
									continue
								}
								out.Data[outRow+ox] += v * c.Weights.Data[wRow+kx]
							}
						}
					}
				}
			}
		}
	}
	return []*Tensor{out}
}

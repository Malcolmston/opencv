package dnn

import "fmt"

// Conv2D is a 2-D convolution (technically a cross-correlation, matching every
// deep-learning framework and OpenCV's dnn) over a batched NCHW input.
//
// The weight tensor has shape [outChannels, inChannels, kernelH, kernelW]; the
// optional bias has shape [outChannels]. For an input of shape [N, inC, H, W]
// the output has shape [N, outC, outH, outW] with
//
//	outH = (H + 2*PadH - DilationH*(kH-1) - 1)/StrideH + 1
//	outW = (W + 2*PadW - DilationW*(kW-1) - 1)/StrideW + 1
//
// Padding is implicit zero padding. The exported fields may be edited after
// construction to set weights or adjust geometry.
type Conv2D struct {
	// Weights has shape [outChannels, inChannels, kernelH, kernelW].
	Weights *Tensor
	// Bias has shape [outChannels], or is nil for a bias-free convolution.
	Bias *Tensor
	// StrideH, StrideW are the vertical and horizontal strides (>= 1).
	StrideH, StrideW int
	// PadH, PadW are the zero-padding amounts on each side (>= 0).
	PadH, PadW int
	// DilationH, DilationW are the kernel dilation factors (>= 1).
	DilationH, DilationW int
}

// NewConv2D builds a convolution with symmetric stride, padding and dilation.
// weights must be rank 4 ([outC, inC, kH, kW]); bias, if non-nil, must be rank
// 1 with outC elements. Pass dilation = 1 for an ordinary convolution. For
// asymmetric geometry set the struct fields directly.
func NewConv2D(weights, bias *Tensor, stride, pad, dilation int) *Conv2D {
	if weights == nil || weights.Dims() != 4 {
		panic("dnn: Conv2D weights must be a rank-4 tensor [outC, inC, kH, kW]")
	}
	if stride < 1 {
		panic(fmt.Sprintf("dnn: Conv2D stride must be >= 1, got %d", stride))
	}
	if dilation < 1 {
		panic(fmt.Sprintf("dnn: Conv2D dilation must be >= 1, got %d", dilation))
	}
	if pad < 0 {
		panic(fmt.Sprintf("dnn: Conv2D pad must be >= 0, got %d", pad))
	}
	if bias != nil && (bias.Dims() != 1 || bias.Shape[0] != weights.Shape[0]) {
		panic(fmt.Sprintf("dnn: Conv2D bias must have shape [%d], got %v", weights.Shape[0], bias.Shape))
	}
	return &Conv2D{
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

// Forward convolves the single NCHW input tensor and returns one NCHW output.
func (c *Conv2D) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Conv2D expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	if in.Dims() != 4 {
		panic(fmt.Sprintf("dnn: Conv2D input must be rank-4 NCHW, got %s", in))
	}
	outC := c.Weights.Shape[0]
	wInC := c.Weights.Shape[1]
	kH := c.Weights.Shape[2]
	kW := c.Weights.Shape[3]

	n := in.Shape[0]
	inC := in.Shape[1]
	h := in.Shape[2]
	w := in.Shape[3]
	if inC != wInC {
		panic(fmt.Sprintf("dnn: Conv2D input has %d channels, weights expect %d", inC, wInC))
	}

	outH := (h+2*c.PadH-c.DilationH*(kH-1)-1)/c.StrideH + 1
	outW := (w+2*c.PadW-c.DilationW*(kW-1)-1)/c.StrideW + 1
	if outH <= 0 || outW <= 0 {
		panic(fmt.Sprintf("dnn: Conv2D produces empty output %dx%d for input %dx%d", outH, outW, h, w))
	}

	out := NewTensor(n, outC, outH, outW)

	// Precomputed strides for fast flat indexing.
	inCHW := inC * h * w
	inHW := h * w
	wStrideOut := wInC * kH * kW
	wStrideIn := kH * kW

	for ni := 0; ni < n; ni++ {
		inBase := ni * inCHW
		for oc := 0; oc < outC; oc++ {
			var bias float64
			if c.Bias != nil {
				bias = c.Bias.Data[oc]
			}
			wBaseOC := oc * wStrideOut
			for oy := 0; oy < outH; oy++ {
				iy0 := oy*c.StrideH - c.PadH
				for ox := 0; ox < outW; ox++ {
					ix0 := ox*c.StrideW - c.PadW
					sum := bias
					for ic := 0; ic < inC; ic++ {
						inChan := inBase + ic*inHW
						wChan := wBaseOC + ic*wStrideIn
						for ky := 0; ky < kH; ky++ {
							iy := iy0 + ky*c.DilationH
							if iy < 0 || iy >= h {
								continue
							}
							inRow := inChan + iy*w
							wRow := wChan + ky*kW
							for kx := 0; kx < kW; kx++ {
								ix := ix0 + kx*c.DilationW
								if ix < 0 || ix >= w {
									continue
								}
								sum += in.Data[inRow+ix] * c.Weights.Data[wRow+kx]
							}
						}
					}
					out.Data[((ni*outC+oc)*outH+oy)*outW+ox] = sum
				}
			}
		}
	}
	return []*Tensor{out}
}

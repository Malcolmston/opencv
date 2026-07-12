package dnn

import (
	"fmt"
	"math"
)

// PReLU applies the parametric rectified-linear activation
//
//	y = x        for x >= 0
//	y = slope*x  for x < 0
//
// like a [LeakyReLU] whose negative slope is learned rather than fixed. The
// Slope is a rank-1 tensor: with a single element it is shared by every
// element (a channel-independent PReLU), otherwise it holds one slope per
// channel and is broadcast along the channel axis (axis 1) of an input of rank
// two or more (NCHW or [N, C]).
type PReLU struct {
	// Slope holds the negative-side slopes: either one shared value or one
	// value per channel.
	Slope *Tensor
}

// NewPReLU builds a PReLU from a rank-1 slope tensor. It panics if slope is nil
// or not rank 1.
func NewPReLU(slope *Tensor) *PReLU {
	if slope == nil || slope.Dims() != 1 {
		panic("dnn: PReLU slope must be a rank-1 tensor")
	}
	return &PReLU{Slope: slope}
}

// Forward applies the PReLU to the single input tensor, preserving its shape.
func (p *PReLU) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: PReLU expects 1 input, got %d", len(inputs)))
	}
	if p.Slope == nil || p.Slope.Dims() != 1 {
		panic("dnn: PReLU slope must be a rank-1 tensor")
	}
	in := inputs[0]
	slopes := p.Slope.Data
	out := &Tensor{
		Shape: append([]int(nil), in.Shape...),
		Data:  make([]float64, len(in.Data)),
	}
	if len(slopes) == 1 {
		s := slopes[0]
		for i, v := range in.Data {
			if v >= 0 {
				out.Data[i] = v
			} else {
				out.Data[i] = s * v
			}
		}
		return []*Tensor{out}
	}
	if in.Dims() < 2 {
		panic(fmt.Sprintf("dnn: PReLU per-channel slope needs a rank>=2 input, got %s", in))
	}
	ch := in.Shape[1]
	if len(slopes) != ch {
		panic(fmt.Sprintf("dnn: PReLU has %d slopes, input has %d channels", len(slopes), ch))
	}
	inner := 1
	for i := 2; i < in.Dims(); i++ {
		inner *= in.Shape[i]
	}
	n := in.Shape[0]
	for ni := 0; ni < n; ni++ {
		for c := 0; c < ch; c++ {
			base := (ni*ch + c) * inner
			s := slopes[c]
			for k := 0; k < inner; k++ {
				v := in.Data[base+k]
				if v >= 0 {
					out.Data[base+k] = v
				} else {
					out.Data[base+k] = s * v
				}
			}
		}
	}
	return []*Tensor{out}
}

// ELU applies the exponential-linear unit elementwise:
//
//	y = x                for x >= 0
//	y = Alpha*(exp(x)-1) for x < 0
//
// The curve is smooth at the origin and saturates to -Alpha for large negative
// inputs. The common default for Alpha is 1.
type ELU struct {
	// Alpha scales the negative saturation branch.
	Alpha float64
}

// NewELU returns an ELU with the given alpha.
func NewELU(alpha float64) *ELU { return &ELU{Alpha: alpha} }

// Forward applies the ELU to the single input tensor.
func (e *ELU) Forward(inputs []*Tensor) []*Tensor {
	return mapElementwise(inputs, "ELU", func(v float64) float64 {
		if v >= 0 {
			return v
		}
		return e.Alpha * (math.Exp(v) - 1)
	})
}

// Mish applies the self-regularized non-monotonic activation
//
//	y = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
//
// elementwise. It is smooth and, unlike ReLU, has a small negative response.
type Mish struct{}

// Forward applies Mish to the single input tensor.
func (m *Mish) Forward(inputs []*Tensor) []*Tensor {
	return mapElementwise(inputs, "Mish", func(v float64) float64 {
		return v * math.Tanh(softplus(v))
	})
}

// softplus computes ln(1+exp(x)) without overflow for large |x|.
func softplus(x float64) float64 {
	switch {
	case x > 20:
		return x
	case x < -20:
		return math.Exp(x)
	default:
		return math.Log1p(math.Exp(x))
	}
}

// Swish applies the gated activation y = x * sigmoid(Beta*x) elementwise. With
// Beta = 1 it is exactly the SiLU (sigmoid-weighted linear unit); see [NewSiLU].
type Swish struct {
	// Beta scales the sigmoid gate. A Beta of 0 collapses to x/2.
	Beta float64
}

// NewSwish returns a Swish with the given beta.
func NewSwish(beta float64) *Swish { return &Swish{Beta: beta} }

// NewSiLU returns the SiLU activation, a [Swish] with Beta = 1: y = x*sigmoid(x).
func NewSiLU() *Swish { return &Swish{Beta: 1} }

// Forward applies the Swish/SiLU activation to the single input tensor.
func (s *Swish) Forward(inputs []*Tensor) []*Tensor {
	beta := s.Beta
	return mapElementwise(inputs, "Swish", func(v float64) float64 {
		return v * sigmoid(beta*v)
	})
}

// Dropout is an inference-time no-op. During training it would randomly zero a
// fraction Rate of its inputs and rescale the survivors, but inference uses the
// full deterministic signal, so Forward simply passes its input through as a
// fresh copy. The layer exists so that a network transcribed from a training
// definition keeps a one-to-one layer correspondence.
type Dropout struct {
	// Rate is the training-time drop probability in [0,1). It is recorded for
	// documentation only and does not affect inference.
	Rate float64
}

// NewDropout returns a Dropout recording the given training-time rate.
func NewDropout(rate float64) *Dropout { return &Dropout{Rate: rate} }

// Forward returns a copy of the single input tensor unchanged.
func (d *Dropout) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Dropout expects 1 input, got %d", len(inputs)))
	}
	return []*Tensor{inputs[0].Clone()}
}

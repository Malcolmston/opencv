package dnn

import (
	"fmt"
	"math"
)

// BatchNorm applies inference-time batch normalization per channel:
//
//	y = Gamma * (x - Mean)/sqrt(Var + Eps) + Beta
//
// The channel axis is axis 1, so the layer accepts either a 4-D NCHW tensor or
// a 2-D [N, C] matrix. Gamma, Beta, Mean and Var are each rank-1 tensors with
// one element per channel; any of Gamma/Beta may be nil, defaulting to 1 and 0
// respectively. The running Mean and Var are used as-is (this port performs no
// training and computes no batch statistics).
type BatchNorm struct {
	// Gamma is the per-channel scale, or nil for all-ones.
	Gamma *Tensor
	// Beta is the per-channel shift, or nil for all-zeros.
	Beta *Tensor
	// Mean is the per-channel running mean.
	Mean *Tensor
	// Var is the per-channel running variance.
	Var *Tensor
	// Eps is the small constant added to Var for numerical stability.
	Eps float64
}

// NewBatchNorm builds a batch-normalization layer from per-channel mean and
// variance, optional scale (gamma) and shift (beta), and an epsilon. mean and
// var must be non-nil rank-1 tensors of equal length; gamma and beta, if
// non-nil, must match that length.
func NewBatchNorm(gamma, beta, mean, variance *Tensor, eps float64) *BatchNorm {
	if mean == nil || variance == nil || mean.Dims() != 1 || variance.Dims() != 1 {
		panic("dnn: BatchNorm requires rank-1 mean and variance tensors")
	}
	c := mean.Shape[0]
	if variance.Shape[0] != c {
		panic(fmt.Sprintf("dnn: BatchNorm mean has %d channels, variance has %d", c, variance.Shape[0]))
	}
	checkOptional := func(t *Tensor, name string) {
		if t != nil && (t.Dims() != 1 || t.Shape[0] != c) {
			panic(fmt.Sprintf("dnn: BatchNorm %s must have shape [%d], got %v", name, c, t.Shape))
		}
	}
	checkOptional(gamma, "gamma")
	checkOptional(beta, "beta")
	return &BatchNorm{Gamma: gamma, Beta: beta, Mean: mean, Var: variance, Eps: eps}
}

// Forward normalizes the single input tensor along its channel axis (axis 1).
func (b *BatchNorm) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: BatchNorm expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	if in.Dims() != 2 && in.Dims() != 4 {
		panic(fmt.Sprintf("dnn: BatchNorm input must be rank-2 [N,C] or rank-4 NCHW, got %s", in))
	}
	ch := in.Shape[1]
	if b.Mean.Shape[0] != ch {
		panic(fmt.Sprintf("dnn: BatchNorm has %d channels, input has %d", b.Mean.Shape[0], ch))
	}

	// Precompute the affine coefficients per channel: y = scale*x + shift.
	scale := make([]float64, ch)
	shift := make([]float64, ch)
	for c := 0; c < ch; c++ {
		gamma := 1.0
		if b.Gamma != nil {
			gamma = b.Gamma.Data[c]
		}
		beta := 0.0
		if b.Beta != nil {
			beta = b.Beta.Data[c]
		}
		inv := gamma / math.Sqrt(b.Var.Data[c]+b.Eps)
		scale[c] = inv
		shift[c] = beta - inv*b.Mean.Data[c]
	}

	out := in.Clone()
	// inner is the number of elements per (batch, channel) slice.
	inner := 1
	for i := 2; i < in.Dims(); i++ {
		inner *= in.Shape[i]
	}
	n := in.Shape[0]
	for ni := 0; ni < n; ni++ {
		for c := 0; c < ch; c++ {
			base := (ni*ch + c) * inner
			sc, sh := scale[c], shift[c]
			for k := 0; k < inner; k++ {
				out.Data[base+k] = sc*out.Data[base+k] + sh
			}
		}
	}
	return []*Tensor{out}
}

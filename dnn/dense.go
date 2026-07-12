package dnn

import "fmt"

// FullyConnected is a dense (fully-connected / inner-product) layer. It maps a
// batched feature matrix of shape [N, inFeatures] to [N, outFeatures] using
//
//	y = x · Wᵀ + b
//
// The weight tensor has shape [outFeatures, inFeatures] (row per output unit,
// the same layout as a Linear layer in common frameworks); the optional bias
// has shape [outFeatures]. Precede this layer with a [Flatten] to feed it the
// output of a convolutional stack.
type FullyConnected struct {
	// Weights has shape [outFeatures, inFeatures].
	Weights *Tensor
	// Bias has shape [outFeatures], or is nil for a bias-free layer.
	Bias *Tensor
}

// Dense is an alias for [FullyConnected], provided to match the naming used by
// higher-level frameworks.
type Dense = FullyConnected

// NewFullyConnected builds a dense layer. weights must be rank 2
// ([outFeatures, inFeatures]); bias, if non-nil, must be rank 1 with
// outFeatures elements.
func NewFullyConnected(weights, bias *Tensor) *FullyConnected {
	if weights == nil || weights.Dims() != 2 {
		panic("dnn: FullyConnected weights must be a rank-2 tensor [outFeatures, inFeatures]")
	}
	if bias != nil && (bias.Dims() != 1 || bias.Shape[0] != weights.Shape[0]) {
		panic(fmt.Sprintf("dnn: FullyConnected bias must have shape [%d], got %v", weights.Shape[0], bias.Shape))
	}
	return &FullyConnected{Weights: weights, Bias: bias}
}

// NewDense is an alias for [NewFullyConnected].
func NewDense(weights, bias *Tensor) *Dense { return NewFullyConnected(weights, bias) }

// Forward computes y = x·Wᵀ + b for the single input tensor of shape
// [N, inFeatures].
func (f *FullyConnected) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: FullyConnected expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	if in.Dims() != 2 {
		panic(fmt.Sprintf("dnn: FullyConnected input must be rank-2 [N, features], got %s (use a Flatten layer first)", in))
	}
	outF := f.Weights.Shape[0]
	inF := f.Weights.Shape[1]
	n := in.Shape[0]
	if in.Shape[1] != inF {
		panic(fmt.Sprintf("dnn: FullyConnected input has %d features, weights expect %d", in.Shape[1], inF))
	}
	out := NewTensor(n, outF)
	for ni := 0; ni < n; ni++ {
		xBase := ni * inF
		for oc := 0; oc < outF; oc++ {
			wBase := oc * inF
			var sum float64
			if f.Bias != nil {
				sum = f.Bias.Data[oc]
			}
			for k := 0; k < inF; k++ {
				sum += in.Data[xBase+k] * f.Weights.Data[wBase+k]
			}
			out.Data[ni*outF+oc] = sum
		}
	}
	return []*Tensor{out}
}

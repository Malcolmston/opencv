package dnn

import (
	"fmt"
	"math"
)

// Softmax applies a numerically stable softmax along a single axis, turning
// that axis into a probability distribution that sums to 1. The default axis
// (-1) is the last, which is the usual choice for a classifier whose input is
// a [N, classes] logit matrix.
type Softmax struct {
	// Axis is the axis normalized over. Negative values count from the end, so
	// -1 is the last axis.
	Axis int
}

// NewSoftmax returns a Softmax over the last axis.
func NewSoftmax() *Softmax { return &Softmax{Axis: -1} }

// NewSoftmaxAxis returns a Softmax over the given axis.
func NewSoftmaxAxis(axis int) *Softmax { return &Softmax{Axis: axis} }

// Forward applies the softmax to the single input tensor, preserving its shape.
func (s *Softmax) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Softmax expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	axis := s.Axis
	if axis < 0 {
		axis += in.Dims()
	}
	if axis < 0 || axis >= in.Dims() {
		panic(fmt.Sprintf("dnn: Softmax axis %d out of range for %s", s.Axis, in))
	}

	out := in.Clone()
	axisSize := in.Shape[axis]

	// inner is the stride of the softmax axis; outer is the number of
	// independent slices before it. Every group of axisSize elements spaced
	// inner apart forms one distribution.
	inner := 1
	for i := axis + 1; i < in.Dims(); i++ {
		inner *= in.Shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= in.Shape[i]
	}

	for o := 0; o < outer; o++ {
		base := o * axisSize * inner
		for k := 0; k < inner; k++ {
			start := base + k
			// max for numerical stability.
			maxv := math.Inf(-1)
			for a := 0; a < axisSize; a++ {
				if v := out.Data[start+a*inner]; v > maxv {
					maxv = v
				}
			}
			var sum float64
			for a := 0; a < axisSize; a++ {
				e := math.Exp(out.Data[start+a*inner] - maxv)
				out.Data[start+a*inner] = e
				sum += e
			}
			for a := 0; a < axisSize; a++ {
				out.Data[start+a*inner] /= sum
			}
		}
	}
	return []*Tensor{out}
}

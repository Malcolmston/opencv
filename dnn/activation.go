package dnn

import (
	"fmt"
	"math"
)

// mapElementwise applies fn to every element of the single input tensor and
// returns a new tensor of the same shape. It centralizes the input checking
// shared by the activation layers.
func mapElementwise(inputs []*Tensor, name string, fn func(float64) float64) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: %s expects 1 input, got %d", name, len(inputs)))
	}
	in := inputs[0]
	out := &Tensor{
		Shape: append([]int(nil), in.Shape...),
		Data:  make([]float64, len(in.Data)),
	}
	for i, v := range in.Data {
		out.Data[i] = fn(v)
	}
	return []*Tensor{out}
}

// ReLU applies the rectified-linear activation max(0, x) elementwise, leaving
// the shape unchanged.
type ReLU struct{}

// Forward applies ReLU to the single input tensor.
func (r *ReLU) Forward(inputs []*Tensor) []*Tensor {
	return mapElementwise(inputs, "ReLU", func(v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	})
}

// LeakyReLU applies x for x >= 0 and Alpha*x for x < 0, elementwise. An Alpha
// of 0 reduces to a plain ReLU; the common default is 0.01.
type LeakyReLU struct {
	// Alpha is the slope applied to negative inputs.
	Alpha float64
}

// Forward applies the leaky ReLU to the single input tensor.
func (l *LeakyReLU) Forward(inputs []*Tensor) []*Tensor {
	return mapElementwise(inputs, "LeakyReLU", func(v float64) float64 {
		if v >= 0 {
			return v
		}
		return l.Alpha * v
	})
}

// Sigmoid applies the logistic sigmoid 1/(1+exp(-x)) elementwise.
type Sigmoid struct{}

// Forward applies the sigmoid to the single input tensor.
func (s *Sigmoid) Forward(inputs []*Tensor) []*Tensor {
	return mapElementwise(inputs, "Sigmoid", sigmoid)
}

// sigmoid computes the numerically stable logistic function.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// Tanh applies the hyperbolic tangent elementwise.
type Tanh struct{}

// Forward applies tanh to the single input tensor.
func (t *Tanh) Forward(inputs []*Tensor) []*Tensor {
	return mapElementwise(inputs, "Tanh", math.Tanh)
}

package dnn

import (
	"fmt"
	"math"
)

// ArgMax reduces a tensor along one axis to the index of its maximum element,
// returning those indices as float64 samples. With KeepDims the reduced axis is
// retained with size 1 (the ONNX default); otherwise it is removed, lowering
// the rank by one. On ties the smallest index wins. The default axis (-1) is
// the last, which turns a [N, classes] score matrix into the [N, 1] (or [N])
// predicted-class map.
type ArgMax struct {
	// Axis is the axis reduced over. Negative values count from the end.
	Axis int
	// KeepDims retains the reduced axis with size 1 when true.
	KeepDims bool
}

// NewArgMax builds an ArgMax over the given axis that keeps the reduced axis.
func NewArgMax(axis int) *ArgMax { return &ArgMax{Axis: axis, KeepDims: true} }

// Forward computes the argmax of the single input tensor along Axis.
func (a *ArgMax) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: ArgMax expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	axis := a.Axis
	if axis < 0 {
		axis += in.Dims()
	}
	if axis < 0 || axis >= in.Dims() {
		panic(fmt.Sprintf("dnn: ArgMax axis %d out of range for %s", a.Axis, in))
	}
	axisSize := in.Shape[axis]
	inner := 1
	for i := axis + 1; i < in.Dims(); i++ {
		inner *= in.Shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= in.Shape[i]
	}

	outShape := make([]int, 0, in.Dims())
	for i := 0; i < in.Dims(); i++ {
		switch {
		case i != axis:
			outShape = append(outShape, in.Shape[i])
		case a.KeepDims:
			outShape = append(outShape, 1)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	out := NewTensor(outShape...)

	for o := 0; o < outer; o++ {
		base := o * axisSize * inner
		for k := 0; k < inner; k++ {
			start := base + k
			best := math.Inf(-1)
			bestIdx := 0
			for ai := 0; ai < axisSize; ai++ {
				if v := in.Data[start+ai*inner]; v > best {
					best = v
					bestIdx = ai
				}
			}
			out.Data[o*inner+k] = float64(bestIdx)
		}
	}
	return []*Tensor{out}
}

package dnn

import (
	"fmt"
	"math"
)

// EltwiseOp selects the reduction an [Eltwise] layer applies across its inputs.
type EltwiseOp int

const (
	// EltwiseSum adds the inputs (optionally weighted by Coeffs).
	EltwiseSum EltwiseOp = iota
	// EltwiseProd multiplies the inputs.
	EltwiseProd
	// EltwiseMax takes the elementwise maximum of the inputs.
	EltwiseMax
)

// Eltwise combines two or more tensors of identical shape with an elementwise
// reduction — sum, product or maximum — and returns a single tensor of that
// shape. Sum is the generalization of [Add]; with Coeffs each input is scaled
// before summing (a weighted residual). Coeffs applies only to EltwiseSum and,
// when non-nil, must have one entry per input.
type Eltwise struct {
	// Op selects the reduction.
	Op EltwiseOp
	// Coeffs optionally weights each input under EltwiseSum.
	Coeffs []float64
}

// NewEltwise builds an Eltwise with the given operation and no coefficients.
func NewEltwise(op EltwiseOp) *Eltwise { return &Eltwise{Op: op} }

// NewEltwiseSum builds a weighted elementwise sum. A nil coeffs sums with unit
// weights.
func NewEltwiseSum(coeffs []float64) *Eltwise {
	return &Eltwise{Op: EltwiseSum, Coeffs: append([]float64(nil), coeffs...)}
}

// Forward reduces all input tensors elementwise according to Op.
func (e *Eltwise) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) < 1 {
		panic("dnn: Eltwise expects at least 1 input")
	}
	ref := inputs[0]
	for i := 1; i < len(inputs); i++ {
		if !sameShape(inputs[i].Shape, ref.Shape) {
			panic(fmt.Sprintf("dnn: Eltwise input %d shape %v mismatches %v", i, inputs[i].Shape, ref.Shape))
		}
	}
	out := &Tensor{
		Shape: append([]int(nil), ref.Shape...),
		Data:  make([]float64, len(ref.Data)),
	}
	switch e.Op {
	case EltwiseSum:
		if e.Coeffs != nil && len(e.Coeffs) != len(inputs) {
			panic(fmt.Sprintf("dnn: Eltwise has %d coeffs for %d inputs", len(e.Coeffs), len(inputs)))
		}
		for i, in := range inputs {
			c := 1.0
			if e.Coeffs != nil {
				c = e.Coeffs[i]
			}
			for j, v := range in.Data {
				out.Data[j] += c * v
			}
		}
	case EltwiseProd:
		copy(out.Data, ref.Data)
		for i := 1; i < len(inputs); i++ {
			for j, v := range inputs[i].Data {
				out.Data[j] *= v
			}
		}
	case EltwiseMax:
		copy(out.Data, ref.Data)
		for i := 1; i < len(inputs); i++ {
			for j, v := range inputs[i].Data {
				out.Data[j] = math.Max(out.Data[j], v)
			}
		}
	default:
		panic(fmt.Sprintf("dnn: Eltwise unknown op %d", e.Op))
	}
	return []*Tensor{out}
}

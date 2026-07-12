package dnn

import "fmt"

// Reshape reinterprets its input under a new shape without moving data. Exactly
// one target axis may be -1, in which case its size is inferred from the total
// element count. The layer returns a fresh tensor (it does not alias its input),
// so it is safe to reuse. Unlike the [Tensor.Reshape] method, this is a [Layer]
// that can sit inside a [Net].
type Reshape struct {
	// Shape is the requested output shape, with at most one -1 axis.
	Shape []int
}

// NewReshape builds a Reshape to the given target shape.
func NewReshape(shape ...int) *Reshape {
	if len(shape) == 0 {
		panic("dnn: NewReshape requires at least one axis")
	}
	return &Reshape{Shape: append([]int(nil), shape...)}
}

// Forward reshapes the single input tensor.
func (r *Reshape) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Reshape expects 1 input, got %d", len(inputs)))
	}
	view := inputs[0].Reshape(r.Shape...)
	out := &Tensor{
		Shape: append([]int(nil), view.Shape...),
		Data:  make([]float64, len(view.Data)),
	}
	copy(out.Data, view.Data)
	return []*Tensor{out}
}

// Permute reorders the axes of a tensor, moving data so the result is
// contiguous in the new order. Order must be a permutation of 0..rank-1: output
// axis i is input axis Order[i]. For a rank-2 input Order = {1,0} is an ordinary
// matrix transpose; for NCHW→NHWC use Order = {0,2,3,1}.
type Permute struct {
	// Order is the permutation applied to the input axes.
	Order []int
}

// NewPermute builds a Permute with the given axis order.
func NewPermute(order ...int) *Permute {
	if len(order) == 0 {
		panic("dnn: NewPermute requires an order")
	}
	return &Permute{Order: append([]int(nil), order...)}
}

// Forward permutes the axes of the single input tensor.
func (p *Permute) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Permute expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	rank := in.Dims()
	if len(p.Order) != rank {
		panic(fmt.Sprintf("dnn: Permute order %v has %d axes, input has %d", p.Order, len(p.Order), rank))
	}
	seen := make([]bool, rank)
	for _, ax := range p.Order {
		if ax < 0 || ax >= rank || seen[ax] {
			panic(fmt.Sprintf("dnn: Permute order %v is not a permutation of 0..%d", p.Order, rank-1))
		}
		seen[ax] = true
	}
	return []*Tensor{permuteAxes(in, p.Order)}
}

// Transpose swaps two axes of a tensor, moving data. It is the two-axis special
// case of [Permute]; the default (Axis1 = -2, Axis2 = -1) swaps the last two
// axes, transposing an [N, ..., R, C] tensor's trailing matrix.
type Transpose struct {
	// Axis1, Axis2 are the axes to exchange. Negative values count from the end.
	Axis1, Axis2 int
}

// NewTranspose builds a Transpose swapping the two given axes.
func NewTranspose(axis1, axis2 int) *Transpose { return &Transpose{Axis1: axis1, Axis2: axis2} }

// Forward swaps the two configured axes of the single input tensor.
func (t *Transpose) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Transpose expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	rank := in.Dims()
	a1, a2 := t.Axis1, t.Axis2
	if a1 == 0 && a2 == 0 {
		// Zero value: default to swapping the last two axes.
		a1, a2 = -2, -1
	}
	if a1 < 0 {
		a1 += rank
	}
	if a2 < 0 {
		a2 += rank
	}
	if a1 < 0 || a1 >= rank || a2 < 0 || a2 >= rank {
		panic(fmt.Sprintf("dnn: Transpose axes (%d,%d) out of range for %s", t.Axis1, t.Axis2, in))
	}
	order := make([]int, rank)
	for i := range order {
		order[i] = i
	}
	order[a1], order[a2] = order[a2], order[a1]
	return []*Tensor{permuteAxes(in, order)}
}

// permuteAxes returns a new contiguous tensor whose axis i is the input's axis
// order[i]. order is assumed validated.
func permuteAxes(in *Tensor, order []int) *Tensor {
	rank := in.Dims()
	// Input strides in element units.
	inStride := make([]int, rank)
	s := 1
	for i := rank - 1; i >= 0; i-- {
		inStride[i] = s
		s *= in.Shape[i]
	}
	outShape := make([]int, rank)
	for i, ax := range order {
		outShape[i] = in.Shape[ax]
	}
	out := NewTensor(outShape...)
	idx := make([]int, rank) // multi-index into the output
	for flat := 0; flat < len(out.Data); flat++ {
		// Map the output multi-index back to an input flat offset.
		off := 0
		for i := 0; i < rank; i++ {
			off += idx[i] * inStride[order[i]]
		}
		out.Data[flat] = in.Data[off]
		// Odometer increment of the output multi-index.
		for i := rank - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < outShape[i] {
				break
			}
			idx[i] = 0
		}
	}
	return out
}

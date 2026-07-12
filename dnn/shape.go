package dnn

import "fmt"

// Flatten collapses every axis after the batch axis into one, mapping a tensor
// of shape [N, d1, d2, …] to [N, d1*d2*…]. It is the usual bridge between a
// convolutional stack and a [FullyConnected] layer. An input that is already
// rank 2 is returned unchanged (as a copy).
type Flatten struct{}

// Forward flattens the single input tensor to rank 2.
func (f *Flatten) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Flatten expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	if in.Dims() < 1 {
		panic("dnn: Flatten input has no axes")
	}
	n := in.Shape[0]
	rest := 1
	for i := 1; i < in.Dims(); i++ {
		rest *= in.Shape[i]
	}
	out := &Tensor{
		Shape: []int{n, rest},
		Data:  make([]float64, len(in.Data)),
	}
	copy(out.Data, in.Data)
	return []*Tensor{out}
}

// Concat joins several tensors along one axis. Every input must share the same
// shape except on the concatenation axis. The default axis (1) concatenates
// channels of an NCHW tensor.
type Concat struct {
	// Axis is the axis along which inputs are joined. Negative values count
	// from the end.
	Axis int
}

// NewConcat returns a Concat over the given axis.
func NewConcat(axis int) *Concat { return &Concat{Axis: axis} }

// Forward concatenates all input tensors along Axis and returns one tensor.
func (c *Concat) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) == 0 {
		panic("dnn: Concat expects at least 1 input")
	}
	ref := inputs[0]
	axis := c.Axis
	if axis < 0 {
		axis += ref.Dims()
	}
	if axis < 0 || axis >= ref.Dims() {
		panic(fmt.Sprintf("dnn: Concat axis %d out of range for %s", c.Axis, ref))
	}

	// Validate shapes and total the concatenation axis.
	outShape := append([]int(nil), ref.Shape...)
	total := ref.Shape[axis]
	for i := 1; i < len(inputs); i++ {
		t := inputs[i]
		if t.Dims() != ref.Dims() {
			panic(fmt.Sprintf("dnn: Concat input %d rank %d != %d", i, t.Dims(), ref.Dims()))
		}
		for d := 0; d < ref.Dims(); d++ {
			if d == axis {
				continue
			}
			if t.Shape[d] != ref.Shape[d] {
				panic(fmt.Sprintf("dnn: Concat input %d shape %v mismatches %v on axis %d", i, t.Shape, ref.Shape, d))
			}
		}
		total += t.Shape[axis]
	}
	outShape[axis] = total
	out := NewTensor(outShape...)

	// outer slices before the axis; inner elements after it.
	inner := 1
	for i := axis + 1; i < ref.Dims(); i++ {
		inner *= ref.Shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= ref.Shape[i]
	}
	outAxis := total
	offset := 0 // running position along the output axis
	for _, t := range inputs {
		a := t.Shape[axis]
		for o := 0; o < outer; o++ {
			srcBase := o * a * inner
			dstBase := (o*outAxis + offset) * inner
			copy(out.Data[dstBase:dstBase+a*inner], t.Data[srcBase:srcBase+a*inner])
		}
		offset += a
	}
	return []*Tensor{out}
}

// Add sums two or more tensors of identical shape elementwise, returning one
// tensor. It implements the residual/skip connection used by many networks.
type Add struct{}

// Forward returns the elementwise sum of all input tensors.
func (a *Add) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) == 0 {
		panic("dnn: Add expects at least 1 input")
	}
	ref := inputs[0]
	out := ref.Clone()
	for i := 1; i < len(inputs); i++ {
		t := inputs[i]
		if !sameShape(t.Shape, ref.Shape) {
			panic(fmt.Sprintf("dnn: Add input %d shape %v mismatches %v", i, t.Shape, ref.Shape))
		}
		for j := range out.Data {
			out.Data[j] += t.Data[j]
		}
	}
	return []*Tensor{out}
}

// sameShape reports whether two shapes are element-for-element equal.
func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

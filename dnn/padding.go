package dnn

import "fmt"

// Padding enlarges a tensor by adding a constant-valued border. Begin[i] rows
// are prepended and End[i] rows appended along axis i, so axis i grows from
// size d to Begin[i] + d + End[i]. Both Begin and End must have one entry per
// axis. For NCHW spatial padding leave the batch and channel entries at 0 and
// set the height/width entries. All padded samples take the value Value.
type Padding struct {
	// Begin holds the leading pad per axis (each >= 0).
	Begin []int
	// End holds the trailing pad per axis (each >= 0).
	End []int
	// Value is the constant written into the padded region.
	Value float64
}

// NewPadding builds a constant Padding with the given per-axis begin and end
// amounts and fill value.
func NewPadding(begin, end []int, value float64) *Padding {
	return &Padding{
		Begin: append([]int(nil), begin...),
		End:   append([]int(nil), end...),
		Value: value,
	}
}

// NewSpatialPadding builds a Padding for a rank-4 NCHW tensor that pads only the
// height and width axes by the given amounts on each side.
func NewSpatialPadding(top, bottom, left, right int, value float64) *Padding {
	return &Padding{
		Begin: []int{0, 0, top, left},
		End:   []int{0, 0, bottom, right},
		Value: value,
	}
}

// Forward pads the single input tensor.
func (p *Padding) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Padding expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	rank := in.Dims()
	if len(p.Begin) != rank || len(p.End) != rank {
		panic(fmt.Sprintf("dnn: Padding begin/end need %d axes, got %d/%d", rank, len(p.Begin), len(p.End)))
	}
	outShape := make([]int, rank)
	for i := 0; i < rank; i++ {
		if p.Begin[i] < 0 || p.End[i] < 0 {
			panic(fmt.Sprintf("dnn: Padding amounts must be >= 0 on axis %d", i))
		}
		outShape[i] = p.Begin[i] + in.Shape[i] + p.End[i]
	}
	out := NewTensor(outShape...)
	if p.Value != 0 {
		for i := range out.Data {
			out.Data[i] = p.Value
		}
	}

	// Output strides.
	outStride := make([]int, rank)
	s := 1
	for i := rank - 1; i >= 0; i-- {
		outStride[i] = s
		s *= outShape[i]
	}

	idx := make([]int, rank) // multi-index into the input
	for flat := 0; flat < len(in.Data); flat++ {
		off := 0
		for i := 0; i < rank; i++ {
			off += (idx[i] + p.Begin[i]) * outStride[i]
		}
		out.Data[off] = in.Data[flat]
		for i := rank - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < in.Shape[i] {
				break
			}
			idx[i] = 0
		}
	}
	return []*Tensor{out}
}

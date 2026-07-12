package dnn

import "fmt"

// Slice extracts a strided sub-range along a single axis of its input, like the
// Python expression x[..., Start:End:Step, ...] applied to Axis. End is
// exclusive. Negative Axis, Start and End count from the end of their
// dimension. The result keeps the same rank; only the sliced axis shrinks.
type Slice struct {
	// Axis is the axis to slice. Negative values count from the end.
	Axis int
	// Start is the first index taken (inclusive). Negative counts from the end.
	Start int
	// End is one past the last index (exclusive). Negative counts from the end;
	// the zero value means "to the end of the axis".
	End int
	// Step is the stride between taken indices (>= 1).
	Step int
}

// NewSlice builds a Slice over one axis with Step 1. Pass end = 0 to slice to
// the end of the axis.
func NewSlice(axis, start, end int) *Slice {
	return &Slice{Axis: axis, Start: start, End: end, Step: 1}
}

// Forward extracts the configured sub-range from the single input tensor.
func (s *Slice) Forward(inputs []*Tensor) []*Tensor {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("dnn: Slice expects 1 input, got %d", len(inputs)))
	}
	in := inputs[0]
	rank := in.Dims()
	axis := s.Axis
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		panic(fmt.Sprintf("dnn: Slice axis %d out of range for %s", s.Axis, in))
	}
	step := s.Step
	if step < 1 {
		panic(fmt.Sprintf("dnn: Slice step must be >= 1, got %d", step))
	}
	dim := in.Shape[axis]
	start := s.Start
	if start < 0 {
		start += dim
	}
	end := s.End
	if end == 0 {
		end = dim
	} else if end < 0 {
		end += dim
	}
	if start < 0 || start >= dim {
		panic(fmt.Sprintf("dnn: Slice start %d out of range for axis %d size %d", s.Start, axis, dim))
	}
	if end > dim {
		end = dim
	}
	if end <= start {
		panic(fmt.Sprintf("dnn: Slice empty range [%d,%d) on axis %d", start, end, axis))
	}
	outAxis := (end - start + step - 1) / step

	// inner: elements per index along axis; outer: independent slices before it.
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= in.Shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= in.Shape[i]
	}

	outShape := append([]int(nil), in.Shape...)
	outShape[axis] = outAxis
	out := NewTensor(outShape...)

	for o := 0; o < outer; o++ {
		srcSliceBase := o * dim * inner
		dstSliceBase := o * outAxis * inner
		for k := 0; k < outAxis; k++ {
			srcIdx := start + k*step
			srcBase := srcSliceBase + srcIdx*inner
			dstBase := dstSliceBase + k*inner
			copy(out.Data[dstBase:dstBase+inner], in.Data[srcBase:srcBase+inner])
		}
	}
	return []*Tensor{out}
}

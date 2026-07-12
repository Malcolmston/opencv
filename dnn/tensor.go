package dnn

import (
	"fmt"
	"strings"
)

// Tensor is a dense, row-major n-dimensional array of float64 samples together
// with its integer shape. It is the value that flows between [Layer]s.
//
// Data is stored contiguously in row-major (C) order: for a shape
// [d0, d1, …, dk] the element at index (i0, i1, …, ik) lives at flat offset
//
//	i0*stride0 + i1*stride1 + … + ik*stridek
//
// where stride of the last axis is 1 and each earlier stride is the product of
// the sizes of all following axes. Image blobs use the NCHW layout — axis 0 is
// the batch, axis 1 the channel, then height and width — matching the output of
// [BlobFromImage].
//
// The zero value is not usable; build tensors with [NewTensor], [NewTensorFrom]
// or the transforms in this package. Although OpenCV's dnn works in single
// precision, this port stores samples as float64 for deterministic arithmetic;
// float32 storage is intentionally out of scope (see the package overview).
type Tensor struct {
	// Shape holds the size of each axis, outermost first. It has one entry per
	// dimension and every entry is positive.
	Shape []int
	// Data holds the samples in row-major order. Its length equals the product
	// of Shape.
	Data []float64
}

// NewTensor allocates a zero-filled Tensor with the given shape. It panics if
// no axis is given or any axis size is not positive.
func NewTensor(shape ...int) *Tensor {
	n := checkShape(shape)
	return &Tensor{
		Shape: append([]int(nil), shape...),
		Data:  make([]float64, n),
	}
}

// NewTensorFrom wraps data in a Tensor of the given shape. The data slice is
// used as-is (not copied), so later writes to it are visible through the
// Tensor. It panics if len(data) does not equal the product of shape.
func NewTensorFrom(shape []int, data []float64) *Tensor {
	n := checkShape(shape)
	if len(data) != n {
		panic(fmt.Sprintf("dnn: NewTensorFrom shape %v needs %d elements, got %d", shape, n, len(data)))
	}
	return &Tensor{
		Shape: append([]int(nil), shape...),
		Data:  data,
	}
}

// checkShape validates a shape and returns the number of elements it describes.
func checkShape(shape []int) int {
	if len(shape) == 0 {
		panic("dnn: tensor shape must have at least one axis")
	}
	n := 1
	for i, d := range shape {
		if d <= 0 {
			panic(fmt.Sprintf("dnn: tensor axis %d has non-positive size %d", i, d))
		}
		n *= d
	}
	return n
}

// Dims returns the number of axes (the rank) of the tensor.
func (t *Tensor) Dims() int { return len(t.Shape) }

// Len returns the total number of elements (the product of the shape).
func (t *Tensor) Len() int { return len(t.Data) }

// Dim returns the size of axis i. It panics if i is out of range.
func (t *Tensor) Dim(i int) int {
	if i < 0 || i >= len(t.Shape) {
		panic(fmt.Sprintf("dnn: Dim axis %d out of range for rank %d", i, len(t.Shape)))
	}
	return t.Shape[i]
}

// Offset converts a multi-index into the flat Data offset. It panics if the
// number of indices differs from the rank or any index is out of range.
func (t *Tensor) Offset(index ...int) int {
	if len(index) != len(t.Shape) {
		panic(fmt.Sprintf("dnn: Offset expects %d indices, got %d", len(t.Shape), len(index)))
	}
	off := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		idx := index[i]
		if idx < 0 || idx >= t.Shape[i] {
			panic(fmt.Sprintf("dnn: index %d out of range for axis %d size %d", idx, i, t.Shape[i]))
		}
		off += idx * stride
		stride *= t.Shape[i]
	}
	return off
}

// At returns the element at the given multi-index.
func (t *Tensor) At(index ...int) float64 {
	return t.Data[t.Offset(index...)]
}

// Set stores value at the given multi-index.
func (t *Tensor) Set(value float64, index ...int) {
	t.Data[t.Offset(index...)] = value
}

// Clone returns a deep copy with its own backing storage.
func (t *Tensor) Clone() *Tensor {
	out := &Tensor{
		Shape: append([]int(nil), t.Shape...),
		Data:  make([]float64, len(t.Data)),
	}
	copy(out.Data, t.Data)
	return out
}

// Reshape returns a view of the tensor with a new shape that shares the Data
// slice. Exactly one axis may be given as -1, in which case its size is
// inferred. It panics if the requested shape is incompatible with the element
// count.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("dnn: Reshape requires at least one axis")
	}
	out := append([]int(nil), shape...)
	infer := -1
	known := 1
	for i, d := range out {
		switch {
		case d == -1:
			if infer != -1 {
				panic("dnn: Reshape allows at most one inferred (-1) axis")
			}
			infer = i
		case d <= 0:
			panic(fmt.Sprintf("dnn: Reshape axis %d has invalid size %d", i, d))
		default:
			known *= d
		}
	}
	if infer != -1 {
		if known == 0 || len(t.Data)%known != 0 {
			panic(fmt.Sprintf("dnn: Reshape cannot infer axis for %d elements with %v", len(t.Data), shape))
		}
		out[infer] = len(t.Data) / known
		known *= out[infer]
	}
	if known != len(t.Data) {
		panic(fmt.Sprintf("dnn: Reshape %v needs %d elements, tensor has %d", shape, known, len(t.Data)))
	}
	return &Tensor{Shape: out, Data: t.Data}
}

// Equal reports whether t and other have the same shape and identical data.
func (t *Tensor) Equal(other *Tensor) bool {
	if other == nil || len(t.Shape) != len(other.Shape) {
		return false
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return false
		}
	}
	for i := range t.Data {
		if t.Data[i] != other.Data[i] {
			return false
		}
	}
	return true
}

// String returns a compact, human-readable description of the tensor's shape,
// useful in diagnostics. It does not print the (potentially large) data.
func (t *Tensor) String() string {
	parts := make([]string, len(t.Shape))
	for i, d := range t.Shape {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return "Tensor[" + strings.Join(parts, "x") + "]"
}

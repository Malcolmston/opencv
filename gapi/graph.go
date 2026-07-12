package gapi

import (
	cv "github.com/malcolmston/opencv"
)

// KernelContext carries the fully-resolved inputs to a single operation node at
// graph-execution time. A [KernelFunc] reads its concrete image inputs from
// Mats, its dynamic scalar inputs from Scalars, and the compile-time parameters
// that were captured when the graph was built from Ints, Floats and Strings.
//
// The slices are positional: Mats[0] is the first image argument passed to the
// op, Floats holds the numeric constants in the order the op recorded them, and
// so on. A KernelFunc must not retain or mutate the input Mats; it returns a
// freshly allocated result.
type KernelContext struct {
	// Mats holds the resolved image inputs in argument order.
	Mats []*cv.Mat
	// Scalars holds the resolved values of the op's GScalar inputs.
	Scalars []float64
	// Floats holds compile-time float parameters captured when building the op.
	Floats []float64
	// Ints holds compile-time int parameters captured when building the op.
	Ints []int
	// Strings holds compile-time string parameters captured when building the op.
	Strings []string
}

// KernelFunc computes the output of one operation node from its resolved
// context. It is the unit of work executed by [GCompiled] and the value a
// custom [GKernel] supplies to override a built-in operation.
type KernelFunc func(ctx KernelContext) *cv.Mat

// matNode is one vertex of the deferred computation graph that yields an image.
// A node is either a protocol input (isInput) whose value is bound at run time,
// or an operation with image predecessors (matInputs), scalar predecessors
// (scalarInputs), captured parameters and a default evaluator.
type matNode struct {
	op           string
	matInputs    []*matNode
	scalarInputs []*scalarNode
	floats       []float64
	ints         []int
	strings      []string
	eval         KernelFunc

	isInput bool
}

// scalarNode is a graph vertex that yields a scalar. It is either a protocol
// input bound at run time or a compile-time constant.
type scalarNode struct {
	isInput bool
	value   float64
}

// GMat is a symbolic handle to an image node in a [GComputation]'s deferred
// graph, mirroring OpenCV G-API's cv::GMat. It carries no pixels: building a
// pipeline out of the package's operations only records how each result is to be
// computed. The actual image is produced later, when a [GCompiled] runs the
// graph on concrete [cv.Mat] inputs.
//
// The zero GMat is invalid; obtain graph inputs from [NewMat] and derived nodes
// from the operations in this package.
type GMat struct {
	n *matNode
}

// Valid reports whether the GMat refers to a graph node.
func (g GMat) Valid() bool { return g.n != nil }

// GScalar is a symbolic handle to a scalar node in a deferred graph, mirroring
// cv::GScalar. Like [GMat] it holds no value until the graph is run. Obtain a
// runtime-bound scalar input from [NewScalar] or a constant from [ConstScalar].
type GScalar struct {
	n *scalarNode
}

// Valid reports whether the GScalar refers to a graph node.
func (g GScalar) Valid() bool { return g.n != nil }

// NewMat creates a protocol image input: a symbolic [GMat] whose pixels are
// supplied at run time. Declare one input per image the pipeline consumes, wire
// them through operations to build the graph, then list them (in the same order
// the concrete Mats will be provided) when constructing a [GComputation].
func NewMat() GMat {
	return GMat{n: &matNode{isInput: true}}
}

// NewScalar creates a protocol scalar input whose value is bound at run time.
func NewScalar() GScalar {
	return GScalar{n: &scalarNode{isInput: true}}
}

// ConstScalar creates a compile-time constant scalar node.
func ConstScalar(v float64) GScalar {
	return GScalar{n: &scalarNode{value: v}}
}

// newOp constructs an operation [GMat] node from its image inputs, scalar
// inputs, captured parameters and default evaluator. It is the single builder
// every operation in this package uses, which keeps the graph shape uniform.
func newOp(op string, mats []GMat, scalars []GScalar, ints []int, floats []float64, strs []string, eval KernelFunc) GMat {
	mn := make([]*matNode, len(mats))
	for i, m := range mats {
		if m.n == nil {
			panic("gapi: " + op + " given an invalid GMat input")
		}
		mn[i] = m.n
	}
	sn := make([]*scalarNode, len(scalars))
	for i, s := range scalars {
		if s.n == nil {
			panic("gapi: " + op + " given an invalid GScalar input")
		}
		sn[i] = s.n
	}
	return GMat{n: &matNode{
		op:           op,
		matInputs:    mn,
		scalarInputs: sn,
		ints:         ints,
		floats:       floats,
		strings:      strs,
		eval:         eval,
	}}
}

// resolveScalar returns the run-time value of a scalar node, reading bound
// inputs from the supplied binding map and constants from the node itself.
func resolveScalar(n *scalarNode, bound map[*scalarNode]float64) float64 {
	if n.isInput {
		return bound[n]
	}
	return n.value
}

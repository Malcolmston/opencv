package gapi

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Inputs bundles the concrete values bound to a computation's protocol inputs
// when the graph is run. Mats are matched, by position, to the image inputs the
// [GComputation] was declared with; Scalars are matched to its scalar inputs.
type Inputs struct {
	// Mats holds one image per declared [GMat] input, in declaration order.
	Mats []*cv.Mat
	// Scalars holds one value per declared [GScalar] input, in declaration order.
	Scalars []float64
}

// GComputation is a captured, compilable image-processing graph, mirroring
// cv::GComputation. It records a set of protocol inputs and outputs; everything
// reachable from the outputs forms the DAG that a [GCompiled] later evaluates.
// A computation is immutable once built and may be compiled more than once.
type GComputation struct {
	matInputs    []*matNode
	scalarInputs []*scalarNode
	outputs      []*matNode
}

// NewComputation builds the common single-input, single-output computation from
// an input [GMat] to an output GMat.
func NewComputation(in, out GMat) *GComputation {
	return NewComputationMulti([]GMat{in}, []GMat{out})
}

// NewComputationMulti builds a computation with several image inputs and
// outputs. The order of ins fixes the order concrete Mats must be supplied in;
// the order of outs fixes the order results are returned in.
func NewComputationMulti(ins, outs []GMat) *GComputation {
	return NewComputationIO(ins, nil, outs)
}

// NewComputationIO builds a computation with image inputs, scalar inputs and
// image outputs, giving full control over the protocol. It panics if any input
// is not a protocol placeholder created by [NewMat] or [NewScalar], or if an
// output is invalid.
func NewComputationIO(matIns []GMat, scalarIns []GScalar, outs []GMat) *GComputation {
	c := &GComputation{}
	for i, m := range matIns {
		if m.n == nil || !m.n.isInput {
			panic(fmt.Sprintf("gapi: computation image input %d is not a NewMat placeholder", i))
		}
		c.matInputs = append(c.matInputs, m.n)
	}
	for i, s := range scalarIns {
		if s.n == nil || !s.n.isInput {
			panic(fmt.Sprintf("gapi: computation scalar input %d is not a NewScalar placeholder", i))
		}
		c.scalarInputs = append(c.scalarInputs, s.n)
	}
	if len(outs) == 0 {
		panic("gapi: computation requires at least one output")
	}
	for i, o := range outs {
		if o.n == nil {
			panic(fmt.Sprintf("gapi: computation output %d is invalid", i))
		}
		c.outputs = append(c.outputs, o.n)
	}
	return c
}

// Compile prepares the graph for execution with the built-in kernels, returning
// a reusable [GCompiled].
func (c *GComputation) Compile() *GCompiled {
	return c.CompileWith(nil)
}

// CompileWith prepares the graph for execution, allowing a [GKernelPackage] to
// override the default implementation of any operation by name. A nil package
// uses the built-in kernels throughout.
func (c *GComputation) CompileWith(pkg *GKernelPackage) *GCompiled {
	order := topoOrder(c.outputs)
	return &GCompiled{comp: c, order: order, pkg: pkg}
}

// Apply is a convenience that compiles the graph and runs it once on the given
// scalar-free image inputs, returning the outputs. For repeated execution
// compile once and reuse the [GCompiled].
func (c *GComputation) Apply(mats ...*cv.Mat) []*cv.Mat {
	return c.Compile().Apply(mats...)
}

// GCompiled is an executable form of a [GComputation]: the graph with a fixed
// topological evaluation order and a chosen kernel package, mirroring
// cv::GCompiled. It is safe to run many times on different inputs.
type GCompiled struct {
	comp  *GComputation
	order []*matNode
	pkg   *GKernelPackage
}

// Apply runs the compiled graph on image inputs only (no scalar inputs are
// bound) and returns every declared output in order. It panics if the input
// count is wrong or an input is nil; use [GCompiled.Run] for error handling and
// scalar inputs.
func (c *GCompiled) Apply(mats ...*cv.Mat) []*cv.Mat {
	outs, err := c.Run(Inputs{Mats: mats})
	if err != nil {
		panic(err)
	}
	return outs
}

// Run executes the compiled graph on the bound inputs and returns the outputs.
// Each operation node is evaluated exactly once, in topological order, and its
// result is cached so shared sub-expressions are never recomputed. It returns an
// error if the supplied inputs do not match the declared protocol.
func (c *GCompiled) Run(in Inputs) ([]*cv.Mat, error) {
	if len(in.Mats) != len(c.comp.matInputs) {
		return nil, fmt.Errorf("gapi: expected %d image input(s), got %d", len(c.comp.matInputs), len(in.Mats))
	}
	if len(in.Scalars) != len(c.comp.scalarInputs) {
		return nil, fmt.Errorf("gapi: expected %d scalar input(s), got %d", len(c.comp.scalarInputs), len(in.Scalars))
	}

	matCache := make(map[*matNode]*cv.Mat, len(c.order))
	scalarBind := make(map[*scalarNode]float64, len(c.comp.scalarInputs))

	for i, n := range c.comp.matInputs {
		if in.Mats[i] == nil {
			return nil, fmt.Errorf("gapi: image input %d is nil", i)
		}
		matCache[n] = in.Mats[i]
	}
	for i, n := range c.comp.scalarInputs {
		scalarBind[n] = in.Scalars[i]
	}

	for _, n := range c.order {
		if n.isInput {
			if _, ok := matCache[n]; !ok {
				return nil, fmt.Errorf("gapi: graph references an unbound image input")
			}
			continue
		}
		ctx := KernelContext{
			Mats:    make([]*cv.Mat, len(n.matInputs)),
			Scalars: make([]float64, len(n.scalarInputs)),
			Floats:  n.floats,
			Ints:    n.ints,
			Strings: n.strings,
		}
		for i, m := range n.matInputs {
			ctx.Mats[i] = matCache[m]
		}
		for i, s := range n.scalarInputs {
			ctx.Scalars[i] = resolveScalar(s, scalarBind)
		}
		fn := n.eval
		if c.pkg != nil {
			if k, ok := c.pkg.lookup(n.op); ok {
				fn = k.Eval
			}
		}
		matCache[n] = fn(ctx)
	}

	outs := make([]*cv.Mat, len(c.comp.outputs))
	for i, o := range c.comp.outputs {
		outs[i] = matCache[o]
	}
	return outs, nil
}

// topoOrder returns the operation nodes reachable from outputs in a
// dependencies-first topological order via a depth-first post-order walk. It
// panics if the graph contains a cycle, which the acyclic builders never create.
func topoOrder(outputs []*matNode) []*matNode {
	const (
		unvisited = 0
		visiting  = 1
		done      = 2
	)
	state := map[*matNode]int{}
	order := make([]*matNode, 0)
	var visit func(n *matNode)
	visit = func(n *matNode) {
		switch state[n] {
		case done:
			return
		case visiting:
			panic("gapi: computation graph contains a cycle")
		}
		state[n] = visiting
		for _, m := range n.matInputs {
			visit(m)
		}
		state[n] = done
		order = append(order, n)
	}
	for _, o := range outputs {
		visit(o)
	}
	return order
}

// ComputationT is a typed, single-image-in single-image-out convenience wrapper
// mirroring G-API's cv::GComputationT. It captures a build function that turns
// one input [GMat] into one output GMat, so a whole pipeline can be defined and
// applied without manually declaring inputs and constructing a [GComputation].
type ComputationT struct {
	compiled *GCompiled
}

// NewComputationT captures build, which must construct the output GMat from the
// single input GMat it is given, and compiles the resulting graph once.
func NewComputationT(build func(in GMat) GMat) *ComputationT {
	in := NewMat()
	out := build(in)
	return &ComputationT{compiled: NewComputation(in, out).Compile()}
}

// Apply runs the captured pipeline on one image and returns the single result.
func (t *ComputationT) Apply(in *cv.Mat) *cv.Mat {
	return t.compiled.Apply(in)[0]
}

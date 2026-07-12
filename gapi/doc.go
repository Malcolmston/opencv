// Package gapi is a pure-Go port of OpenCV's G-API (graph API) module built on
// the standard-library-only OpenCV port github.com/malcolmston/opencv (imported
// here as cv). It offers a lazy, graph-based way to express image processing:
// operations do not run when you call them but instead record how a result is to
// be computed, and the whole pipeline executes later, once, over concrete
// images.
//
// # Why a graph
//
// The root cv package is eager: cv.GaussianBlur(img, 5, 1.4) allocates and fills
// a Mat immediately. G-API instead separates the description of a computation
// from its execution. You build a graph out of symbolic nodes, compile it, and
// then feed it data. The same compiled graph can be run many times on different
// inputs, and because the structure is known ahead of time it can be validated,
// reused and (in principle) optimised as a whole.
//
// # Symbolic nodes
//
// [GMat] is a symbolic image and [GScalar] a symbolic scalar. Neither holds any
// pixels. [NewMat] and [NewScalar] create protocol inputs — placeholders bound
// to real values at run time. Every operation in this package, for example
// [RGB2Gray], [GaussianBlur], [Canny] or [Add], takes GMats and returns a new
// GMat that records the operation and its arguments. Chaining them composes a
// directed acyclic graph:
//
//	in := gapi.NewMat()
//	edges := gapi.Canny(gapi.GaussianBlur(gapi.RGB2Gray(in), 5, 1.4), 50, 100)
//
// At this point nothing has been computed; edges is just the root of a graph.
//
// # Compilation and execution
//
// [NewComputation] (and its multi-input/output siblings) captures a graph by its
// protocol inputs and outputs into a [GComputation]. Compiling it yields a
// reusable [GCompiled]:
//
//	cc := gapi.NewComputation(in, edges).Compile()
//	out := cc.Apply(img) // out[0] is the edge map for img
//
// Execution performs a topological sort of the nodes reachable from the outputs
// and evaluates each exactly once, caching every intermediate result. Shared
// sub-expressions are therefore computed a single time, and graphs with several
// inputs or several outputs (see [NewComputationMulti] and [Split3]) are handled
// uniformly. Each node delegates to the corresponding eager routine in the root
// cv package at execution time, so a compiled graph produces bit-for-bit the
// same result as the equivalent hand-written eager pipeline.
//
// # Scalars
//
// Dynamic scalars flow through the graph as [GScalar]. A scalar input created
// with [NewScalar] is bound per run via [Inputs].Scalars, while [ConstScalar]
// captures a constant. Operations such as [AddC] and [MulC] consume them.
//
// # Custom kernels
//
// A [GKernel] supplies an alternative implementation for a named operation, and
// a [GKernelPackage] groups several. Passing a package to
// [GComputation.CompileWith] overrides the built-in evaluators for those
// operations, mirroring G-API's pluggable kernel backends. Operation names are
// the exported Op* constants, for example [OpGaussianBlur].
//
// # Typed helper
//
// [ComputationT] wraps the common single-image-in, single-image-out case: it
// captures a build function and compiles it once, so a pipeline can be defined
// and applied without manually declaring inputs.
package gapi

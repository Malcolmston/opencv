package gapi

// GKernel is a custom implementation for a named operation, mirroring the role
// of a G-API kernel. When a [GKernelPackage] containing a kernel for an
// operation is passed to [GComputation.CompileWith], the kernel's Eval replaces
// the built-in evaluator for every node of that operation. Eval receives the
// same [KernelContext] the built-in would, so it can reimplement the op fully
// from the resolved inputs and captured parameters.
//
// Op names are the identifiers this package assigns to its operations, for
// example "add", "gaussianBlur" or "canny" — the value passed as the first
// argument to the internal node builder. Look them up with the OpName* helpers.
type GKernel struct {
	// Op is the operation name this kernel implements.
	Op string
	// Eval computes the operation's output from the resolved context.
	Eval KernelFunc
}

// GKernelPackage is an ordered set of [GKernel] implementations keyed by
// operation name, mirroring cv::gapi::GKernelPackage. It is the unit passed to
// [GComputation.CompileWith] to customise how selected operations execute.
type GKernelPackage struct {
	kernels map[string]GKernel
}

// Kernels builds a package from the given kernels. If two kernels target the
// same operation the later one wins, matching G-API's "last include" rule.
func Kernels(ks ...GKernel) *GKernelPackage {
	p := &GKernelPackage{kernels: make(map[string]GKernel, len(ks))}
	for _, k := range ks {
		p.kernels[k.Op] = k
	}
	return p
}

// Include adds or replaces the kernel for an operation and returns the package
// for chaining.
func (p *GKernelPackage) Include(k GKernel) *GKernelPackage {
	if p.kernels == nil {
		p.kernels = make(map[string]GKernel)
	}
	p.kernels[k.Op] = k
	return p
}

// Combine returns a new package containing every kernel from p overlaid with
// every kernel from other, so entries in other take precedence on conflict.
// Neither receiver nor argument is modified. A nil package is treated as empty.
func (p *GKernelPackage) Combine(other *GKernelPackage) *GKernelPackage {
	out := &GKernelPackage{kernels: make(map[string]GKernel)}
	if p != nil {
		for op, k := range p.kernels {
			out.kernels[op] = k
		}
	}
	if other != nil {
		for op, k := range other.kernels {
			out.kernels[op] = k
		}
	}
	return out
}

// Size reports how many operations the package overrides.
func (p *GKernelPackage) Size() int {
	if p == nil {
		return 0
	}
	return len(p.kernels)
}

// lookup returns the kernel registered for op, if any.
func (p *GKernelPackage) lookup(op string) (GKernel, bool) {
	if p == nil {
		return GKernel{}, false
	}
	k, ok := p.kernels[op]
	return k, ok
}

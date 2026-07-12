package fuzzy

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// BasisFunction selects the shape of the one-dimensional fuzzy basis (membership)
// function used to build a fuzzy partition of the image domain. Both choices
// satisfy the Ruspini condition (a partition of unity) when the partition nodes
// are spaced radius pixels apart, which is what the F-transform routines in this
// package assume.
type BasisFunction int

const (
	// LinearBasis is a symmetric triangular membership function. Over the support
	// [-radius, radius] its value at offset d is 1 - |d|/radius, so it is 1 at the
	// node centre and falls linearly to 0 at the neighbouring nodes.
	LinearBasis BasisFunction = iota
	// SinusBasis is a raised-cosine ("sinusoidal") membership function. Over the
	// support [-radius, radius] its value at offset d is 0.5*(cos(pi*d/radius)+1),
	// so it is 1 at the node centre and falls smoothly to 0 at the neighbouring
	// nodes. It is smoother than [LinearBasis] and reconstructs gently curved
	// signals with less faceting.
	SinusBasis
)

// String returns the name of the basis function.
func (f BasisFunction) String() string {
	switch f {
	case LinearBasis:
		return "LinearBasis"
	case SinusBasis:
		return "SinusBasis"
	default:
		return fmt.Sprintf("BasisFunction(%d)", int(f))
	}
}

// basisVector returns the sampled 1-D membership function of length 2*radius+1.
// Index i corresponds to offset d = i - radius, so element radius is the peak
// (value 1) and elements 0 and 2*radius are the zero-valued endpoints.
func basisVector(function BasisFunction, radius int) []float64 {
	if radius < 1 {
		panic(fmt.Sprintf("fuzzy: radius must be >= 1, got %d", radius))
	}
	n := 2*radius + 1
	v := make([]float64, n)
	r := float64(radius)
	for i := 0; i < n; i++ {
		d := float64(i - radius)
		switch function {
		case LinearBasis:
			v[i] = 1 - math.Abs(d)/r
		case SinusBasis:
			v[i] = 0.5 * (math.Cos(math.Pi*d/r) + 1)
		default:
			panic(fmt.Sprintf("fuzzy: unknown basis function %d", int(function)))
		}
		if v[i] < 0 {
			v[i] = 0
		}
	}
	return v
}

// CreateKernel builds the two-dimensional fuzzy-partition kernel for the given
// basis function and radius, mirroring OpenCV's ft::createKernel. The kernel is
// the outer product of the 1-D membership vector with itself, so it has size
// (2*radius+1) x (2*radius+1), peaks at 1 in its centre and tapers to 0 at its
// border. It is returned as a single-channel [cv.FloatMat]; the same kernel is
// applied independently to every channel of an image.
//
// Because the underlying 1-D basis is a partition of unity at node spacing
// radius, four neighbouring copies of this kernel (two per axis) sum to 1 at
// every interior pixel, which is what makes the inverse F-transform reconstruct
// a smooth approximation of the input.
func CreateKernel(function BasisFunction, radius int) *cv.FloatMat {
	if radius < 1 {
		panic(fmt.Sprintf("fuzzy: CreateKernel radius must be >= 1, got %d", radius))
	}
	v := basisVector(function, radius)
	n := len(v)
	k := cv.NewFloatMat(n, n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			k.Data[y*n+x] = v[y] * v[x]
		}
	}
	return k
}

// kernelRadius recovers the radius encoded in a square kernel produced by
// [CreateKernel]. It panics if the kernel is not square with odd side length.
func kernelRadius(kernel *cv.FloatMat) int {
	if kernel == nil || kernel.Rows < 3 || kernel.Rows != kernel.Cols || kernel.Rows%2 == 0 {
		panic("fuzzy: kernel must be a square FloatMat with odd side length >= 3")
	}
	return (kernel.Cols - 1) / 2
}

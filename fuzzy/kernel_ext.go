package fuzzy

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// CreateKernel1D returns the sampled one-dimensional fuzzy basis (membership)
// vector for the given basis function and radius, mirroring the 1-D half of
// OpenCV's ft::createKernel1. The vector has length 2*radius+1; index i holds the
// membership value at offset d = i-radius, so element radius is the peak (1) and
// the two endpoints are zero. It is the building block from which the separable
// two-dimensional kernels are formed, and can be passed to [CreateKernelVec] or
// [CreateKernelAB].
func CreateKernel1D(function BasisFunction, radius int) []float64 {
	// basisVector already validates radius and function and panics otherwise.
	return basisVector(function, radius)
}

// CreateKernelVec builds a square two-dimensional fuzzy-partition kernel as the
// outer product of a one-dimensional membership vector with itself, mirroring the
// single-vector form of OpenCV's ft::createKernel1. This is the symmetric kernel
// used throughout the module; [CreateKernel] is exactly
// CreateKernelVec(CreateKernel1D(function, radius)).
//
// v must have odd length >= 3 (a valid 2*radius+1 membership vector). The result
// is an len(v) x len(v) single-channel [cv.FloatMat].
func CreateKernelVec(v []float64) *cv.FloatMat {
	return CreateKernelAB(v, v)
}

// CreateKernelAB builds a two-dimensional fuzzy-partition kernel as the outer
// product of two one-dimensional membership vectors, mirroring the two-vector
// form of OpenCV's ft::createKernel (createKernel(A, B)). Element (y, x) of the
// result is b[y]*a[x], so a is the horizontal (column) profile and b the vertical
// (row) profile. Passing different a and b yields an anisotropic kernel — for
// example a wider horizontal than vertical support — which the FT02D and FT12D
// routines apply unchanged, since they only require a square, odd-sided kernel
// when a and b have equal length.
//
// Both vectors must have odd length >= 3. When len(a) == len(b) the kernel is
// square and directly usable by every transform in this package; unequal lengths
// produce a rectangular kernel suitable for [CreateKernelAB]-aware callers.
func CreateKernelAB(a, b []float64) *cv.FloatMat {
	if len(a) < 3 || len(a)%2 == 0 {
		panic(fmt.Sprintf("fuzzy: CreateKernelAB vector a must have odd length >= 3, got %d", len(a)))
	}
	if len(b) < 3 || len(b)%2 == 0 {
		panic(fmt.Sprintf("fuzzy: CreateKernelAB vector b must have odd length >= 3, got %d", len(b)))
	}
	w, h := len(a), len(b)
	k := cv.NewFloatMat(h, w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			k.Data[y*w+x] = b[y] * a[x]
		}
	}
	return k
}

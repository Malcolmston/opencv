package imghash2

import "math"

// invSqrt2 is 1/√2, the orthonormal Haar normalisation factor.
const invSqrt2 = 0.70710678118654752440

// DCT1D returns the orthonormal type-II discrete cosine transform of vec:
//
//	C(u) = a(u) · Σ_{x=0}^{N-1} f(x) · cos[(2x+1)uπ/(2N)]
//
// with a(0)=√(1/N) and a(u)=√(2/N) for u>0. This is the transform used by JPEG
// and by OpenCV's cv::dct. It is implemented directly in O(N²); the block sizes
// used for perceptual hashing are small, so that is ample. [IDCT1D] is its
// exact inverse.
func DCT1D(vec []float64) []float64 {
	n := len(vec)
	out := make([]float64, n)
	a0 := math.Sqrt(1.0 / float64(n))
	ak := math.Sqrt(2.0 / float64(n))
	for u := 0; u < n; u++ {
		var sum float64
		for x := 0; x < n; x++ {
			sum += vec[x] * math.Cos((2*float64(x)+1)*float64(u)*math.Pi/(2*float64(n)))
		}
		if u == 0 {
			out[u] = a0 * sum
		} else {
			out[u] = ak * sum
		}
	}
	return out
}

// IDCT1D returns the inverse of [DCT1D] (the orthonormal type-III DCT),
// reconstructing the samples f(x) = Σ_u a(u) C(u) cos[(2x+1)uπ/(2N)]. Applying
// DCT1D then IDCT1D recovers the input up to floating-point rounding.
func IDCT1D(coeffs []float64) []float64 {
	n := len(coeffs)
	out := make([]float64, n)
	a0 := math.Sqrt(1.0 / float64(n))
	ak := math.Sqrt(2.0 / float64(n))
	for x := 0; x < n; x++ {
		var sum float64
		for u := 0; u < n; u++ {
			a := ak
			if u == 0 {
				a = a0
			}
			sum += a * coeffs[u] * math.Cos((2*float64(x)+1)*float64(u)*math.Pi/(2*float64(n)))
		}
		out[x] = sum
	}
	return out
}

// DCT2D returns the 2-D orthonormal type-II DCT of an n×n block stored
// row-major in a flat slice of length n·n. The transform is separable: it is
// applied to every row and then to every column. For a constant input every
// coefficient is zero except the DC term at index 0. [IDCT2D] is its exact
// inverse.
func DCT2D(in []float64, n int) []float64 {
	return separable2D(in, n, DCT1D)
}

// IDCT2D returns the inverse of [DCT2D], reconstructing an n×n block from its
// coefficients. Columns are inverted first and then rows, undoing the forward
// order.
func IDCT2D(in []float64, n int) []float64 {
	return separable2DColsFirst(in, n, IDCT1D)
}

// HaarDWT1D applies one level of the orthonormal Haar discrete wavelet
// transform to vec and returns a fresh slice. The n/2 scaling (approximation)
// coefficients (x0+x1)/√2 occupy the low half of the output and the n/2 wavelet
// (detail) coefficients (x0−x1)/√2 the high half. The length of vec must be
// even. [HaarIDWT1D] is its exact inverse.
func HaarDWT1D(vec []float64) []float64 {
	n := len(vec)
	if n%2 != 0 {
		panic("imghash2: HaarDWT1D requires an even length")
	}
	out := make([]float64, n)
	half := n / 2
	for i := 0; i < half; i++ {
		a := vec[2*i]
		b := vec[2*i+1]
		out[i] = (a + b) * invSqrt2
		out[half+i] = (a - b) * invSqrt2
	}
	return out
}

// HaarIDWT1D is the exact inverse of [HaarDWT1D], reconstructing the interleaved
// samples from the deinterleaved approximation and detail halves.
func HaarIDWT1D(vec []float64) []float64 {
	n := len(vec)
	if n%2 != 0 {
		panic("imghash2: HaarIDWT1D requires an even length")
	}
	out := make([]float64, n)
	half := n / 2
	for i := 0; i < half; i++ {
		a := vec[i]
		d := vec[half+i]
		out[2*i] = (a + d) * invSqrt2
		out[2*i+1] = (a - d) * invSqrt2
	}
	return out
}

// HaarDWT2D returns one level of the separable 2-D orthonormal Haar discrete
// wavelet transform of an n×n block stored row-major in a flat slice of length
// n·n. It is applied to every row and then every column, so the result splits
// into four n/2×n/2 subbands: LL (approximation, top-left), LH, HL and HH
// (horizontal, vertical and diagonal detail). n must be even. [HaarIDWT2D] is
// its exact inverse.
func HaarDWT2D(in []float64, n int) []float64 {
	if n%2 != 0 {
		panic("imghash2: HaarDWT2D requires an even side length")
	}
	return separable2D(in, n, HaarDWT1D)
}

// HaarIDWT2D is the exact inverse of [HaarDWT2D]: given one level of 2-D Haar
// coefficients for an n×n block it reconstructs the original samples, inverting
// columns first and then rows.
func HaarIDWT2D(in []float64, n int) []float64 {
	if n%2 != 0 {
		panic("imghash2: HaarIDWT2D requires an even side length")
	}
	return separable2DColsFirst(in, n, HaarIDWT1D)
}

// separable2D applies a 1-D transform to every row and then every column of an
// n×n block, the forward order shared by the DCT and Haar transforms.
func separable2D(in []float64, n int, t func([]float64) []float64) []float64 {
	if len(in) != n*n {
		panic("imghash2: separable transform input length must be n*n")
	}
	tmp := make([]float64, n*n)
	row := make([]float64, n)
	for y := 0; y < n; y++ {
		copy(row, in[y*n:y*n+n])
		r := t(row)
		copy(tmp[y*n:y*n+n], r)
	}
	out := make([]float64, n*n)
	col := make([]float64, n)
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			col[y] = tmp[y*n+x]
		}
		c := t(col)
		for y := 0; y < n; y++ {
			out[y*n+x] = c[y]
		}
	}
	return out
}

// separable2DColsFirst applies a 1-D transform to every column and then every
// row, the reverse order used by the inverse transforms.
func separable2DColsFirst(in []float64, n int, t func([]float64) []float64) []float64 {
	if len(in) != n*n {
		panic("imghash2: separable transform input length must be n*n")
	}
	tmp := make([]float64, n*n)
	copy(tmp, in)
	col := make([]float64, n)
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			col[y] = tmp[y*n+x]
		}
		c := t(col)
		for y := 0; y < n; y++ {
			tmp[y*n+x] = c[y]
		}
	}
	out := make([]float64, n*n)
	row := make([]float64, n)
	for y := 0; y < n; y++ {
		copy(row, tmp[y*n:y*n+n])
		r := t(row)
		copy(out[y*n:y*n+n], r)
	}
	return out
}

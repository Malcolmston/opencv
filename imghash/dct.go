package imghash

import "math"

// dct1D returns the orthonormal type-II discrete cosine transform of vec:
//
//	C(u) = a(u) * sum_{x=0}^{N-1} f(x) cos[(2x+1)uπ/(2N)]
//
// with a(0)=sqrt(1/N) and a(u)=sqrt(2/N) for u>0. The transform is the same one
// used by JPEG and by OpenCV's cv::dct, and it is its own inverse up to the
// normalisation, so applying it separably to rows and columns yields the 2-D
// DCT. It is implemented directly (O(N^2) per vector) rather than via an FFT;
// the block sizes used for perceptual hashing are small, so this is ample.
func dct1D(vec []float64) []float64 {
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

// dct2D returns the 2-D orthonormal type-II DCT of an n×n block stored
// row-major in a flat slice of length n*n. The transform is separable: it is
// applied to every row and then to every column. For a constant input every
// coefficient is zero except the DC term (0,0), a property the tests rely on.
func dct2D(in []float64, n int) []float64 {
	// Transform rows.
	rows := make([]float64, n*n)
	row := make([]float64, n)
	for y := 0; y < n; y++ {
		copy(row, in[y*n:y*n+n])
		r := dct1D(row)
		copy(rows[y*n:y*n+n], r)
	}
	// Transform columns.
	out := make([]float64, n*n)
	col := make([]float64, n)
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			col[y] = rows[y*n+x]
		}
		c := dct1D(col)
		for y := 0; y < n; y++ {
			out[y*n+x] = c[y]
		}
	}
	return out
}

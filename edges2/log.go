package edges2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// LaplacianOfGaussian convolves src with a Laplacian-of-Gaussian (LoG, also
// known as the Mexican-hat) kernel of the given standard deviation and returns
// the signed response as a [FloatGrid]. The kernel size is chosen to cover
// three standard deviations and the kernel is mean-corrected so a flat image
// yields a zero response. It panics on multi-channel input or non-positive
// sigma.
func LaplacianOfGaussian(src *cv.Mat, sigma float64) *FloatGrid {
	edges2RequireGray(src, "LaplacianOfGaussian")
	if sigma <= 0 {
		panic("edges2: LaplacianOfGaussian requires sigma > 0")
	}
	return edges2Convolve(src, edges2LoGKernel(sigma))
}

// edges2LoGKernel builds a mean-corrected Laplacian-of-Gaussian kernel.
func edges2LoGKernel(sigma float64) [][]float64 {
	r := int(math.Ceil(3 * sigma))
	if r < 1 {
		r = 1
	}
	size := 2*r + 1
	k := make([][]float64, size)
	s2 := sigma * sigma
	s4 := s2 * s2
	var sum float64
	for j := 0; j < size; j++ {
		k[j] = make([]float64, size)
		for i := 0; i < size; i++ {
			dy := float64(j - r)
			dx := float64(i - r)
			d2 := dx*dx + dy*dy
			v := (d2 - 2*s2) / s4 * math.Exp(-d2/(2*s2))
			k[j][i] = v
			sum += v
		}
	}
	// Remove the DC component so a constant image maps to zero.
	mean := sum / float64(size*size)
	for j := range k {
		for i := range k[j] {
			k[j][i] -= mean
		}
	}
	return k
}

// ZeroCrossings detects sign changes in a signed response grid (such as a LoG
// output) and returns a binary edge map (255 at a crossing, 0 otherwise). A
// pixel is marked when it and one of its right, bottom or diagonal neighbours
// have opposite signs and the magnitude of their difference exceeds thresh,
// which suppresses crossings caused by noise in near-flat regions.
func ZeroCrossings(g *FloatGrid, thresh float64) *cv.Mat {
	rows, cols := g.Rows, g.Cols
	dst := cv.NewMat(rows, cols, 1)
	neigh := [][2]int{{0, 1}, {1, 0}, {1, 1}, {1, -1}}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := g.Data[y*cols+x]
			for _, d := range neigh {
				ny := y + d[0]
				nx := x + d[1]
				if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
					continue
				}
				w := g.Data[ny*cols+nx]
				if (v > 0) != (w > 0) && math.Abs(v-w) > thresh {
					dst.Data[y*cols+x] = 255
					break
				}
			}
		}
	}
	return dst
}

// MarrHildreth runs the Marr-Hildreth edge detector: it computes the
// Laplacian of Gaussian of src at the given sigma and returns the zero
// crossings of that response whose local slope exceeds thresh as a binary edge
// map (edges 255, background 0). It panics on multi-channel input or
// non-positive sigma.
func MarrHildreth(src *cv.Mat, sigma, thresh float64) *cv.Mat {
	return ZeroCrossings(LaplacianOfGaussian(src, sigma), thresh)
}

// DifferenceOfGaussians returns the difference of two Gaussian-smoothed copies
// of src (blur at sigma1 minus blur at sigma2) as a signed [FloatGrid]. With
// sigma1 < sigma2 this band-pass response approximates the Laplacian of
// Gaussian and highlights edges and blobs. It panics on multi-channel input.
func DifferenceOfGaussians(src *cv.Mat, sigma1, sigma2 float64) *FloatGrid {
	edges2RequireGray(src, "DifferenceOfGaussians")
	b1 := edges2Blur(src, sigma1)
	b2 := edges2Blur(src, sigma2)
	out := NewFloatGrid(src.Rows, src.Cols)
	for i := range out.Data {
		out.Data[i] = float64(b1.Data[i]) - float64(b2.Data[i])
	}
	return out
}

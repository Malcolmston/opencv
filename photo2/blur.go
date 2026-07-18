package photo2

import (
	cv "github.com/malcolmston/opencv"
)

// GaussianBlurFloat convolves a single-channel float matrix with a separable
// Gaussian of the given standard deviation, using reflected borders. A
// non-positive sigma returns a clone.
func GaussianBlurFloat(f *cv.FloatMat, sigma float64) *cv.FloatMat {
	photo2RequireFloat(f, "GaussianBlurFloat")
	if sigma <= 0 {
		return photo2CloneFloat(f)
	}
	k := photo2GaussianKernel(sigma)
	return photo2SepConv(f, k)
}

// photo2SepConv applies the symmetric 1-D kernel k horizontally then vertically
// with reflected borders.
func photo2SepConv(f *cv.FloatMat, k []float64) *cv.FloatMat {
	rows, cols := f.Rows, f.Cols
	radius := len(k) / 2
	tmp := cv.NewFloatMat(rows, cols)
	// Horizontal pass.
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -radius; t <= radius; t++ {
				xx := photo2Reflect(x+t, cols)
				acc += k[t+radius] * f.Data[base+xx]
			}
			tmp.Data[base+x] = acc
		}
	}
	// Vertical pass.
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -radius; t <= radius; t++ {
				yy := photo2Reflect(y+t, rows)
				acc += k[t+radius] * tmp.Data[yy*cols+x]
			}
			out.Data[y*cols+x] = acc
		}
	}
	return out
}

// GaussianBlur convolves an 8-bit image with a separable Gaussian of the given
// standard deviation, independently per channel, using reflected borders. A
// non-positive sigma returns a clone.
func GaussianBlur(img *cv.Mat, sigma float64) *cv.Mat {
	photo2RequireImage(img, "GaussianBlur")
	if sigma <= 0 {
		return img.Clone()
	}
	planes := ToFloat(img)
	for c := range planes {
		planes[c] = GaussianBlurFloat(planes[c], sigma)
	}
	return FromFloat(planes)
}

// BoxBlur applies an averaging filter over a (2*radius+1) square window with
// reflected borders, independently per channel. A non-positive radius returns a
// clone.
func BoxBlur(img *cv.Mat, radius int) *cv.Mat {
	photo2RequireImage(img, "BoxBlur")
	if radius <= 0 {
		return img.Clone()
	}
	planes := ToFloat(img)
	for c := range planes {
		planes[c] = photo2BoxBlurFloat(planes[c], radius)
	}
	return FromFloat(planes)
}

// photo2BoxBlurFloat runs a separable box filter of the given radius on a float
// plane with reflected borders.
func photo2BoxBlurFloat(f *cv.FloatMat, radius int) *cv.FloatMat {
	rows, cols := f.Rows, f.Cols
	win := float64(2*radius + 1)
	tmp := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -radius; t <= radius; t++ {
				acc += f.Data[base+photo2Reflect(x+t, cols)]
			}
			tmp.Data[base+x] = acc / win
		}
	}
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			for t := -radius; t <= radius; t++ {
				acc += tmp.Data[photo2Reflect(y+t, rows)*cols+x]
			}
			out.Data[y*cols+x] = acc / win
		}
	}
	return out
}

// PyrDown blurs f with a Gaussian and downsamples it by a factor of two,
// returning a matrix of size ceil(rows/2) x ceil(cols/2). It is the reduction
// step of a Gaussian pyramid.
func PyrDown(f *cv.FloatMat) *cv.FloatMat {
	photo2RequireFloat(f, "PyrDown")
	blurred := photo2SepConv(f, photo2Pyr5Tap())
	nr := (f.Rows + 1) / 2
	nc := (f.Cols + 1) / 2
	out := cv.NewFloatMat(nr, nc)
	for y := 0; y < nr; y++ {
		for x := 0; x < nc; x++ {
			out.Data[y*nc+x] = blurred.Data[(2*y)*f.Cols+(2*x)]
		}
	}
	return out
}

// PyrUp upsamples f to the given target dimensions by pixel injection followed
// by a Gaussian smoothing (scaled by four to preserve brightness). rows and
// cols must be the size of the finer pyramid level, i.e. either 2*f.Rows or
// 2*f.Rows-1 and likewise for columns.
func PyrUp(f *cv.FloatMat, rows, cols int) *cv.FloatMat {
	photo2RequireFloat(f, "PyrUp")
	up := cv.NewFloatMat(rows, cols)
	for y := 0; y < f.Rows; y++ {
		yy := 2 * y
		if yy >= rows {
			continue
		}
		for x := 0; x < f.Cols; x++ {
			xx := 2 * x
			if xx >= cols {
				continue
			}
			up.Data[yy*cols+xx] = f.Data[y*f.Cols+x]
		}
	}
	k := photo2Pyr5Tap()
	for i := range k {
		k[i] *= 2 // per axis; 4x total across the separable pair
	}
	return photo2SepConv(up, k)
}

// photo2Pyr5Tap returns the normalised binomial 5-tap kernel [1 4 6 4 1]/16
// used by the pyramid reduce and expand operators.
func photo2Pyr5Tap() []float64 {
	return []float64{1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16}
}

// GaussianPyramid builds a Gaussian pyramid with the given number of levels.
// Level 0 is a copy of f; each subsequent level is [PyrDown] of the previous.
// levels must be at least one. Reduction stops early if a dimension reaches one.
func GaussianPyramid(f *cv.FloatMat, levels int) []*cv.FloatMat {
	photo2RequireFloat(f, "GaussianPyramid")
	if levels < 1 {
		levels = 1
	}
	pyr := make([]*cv.FloatMat, 0, levels)
	pyr = append(pyr, photo2CloneFloat(f))
	for l := 1; l < levels; l++ {
		prev := pyr[l-1]
		if prev.Rows < 2 || prev.Cols < 2 {
			break
		}
		pyr = append(pyr, PyrDown(prev))
	}
	return pyr
}

// LaplacianPyramid builds a Laplacian pyramid with the given number of levels.
// Each level holds the detail lost by [PyrDown]/[PyrUp] at that scale; the final
// level is the residual Gaussian (low-pass) image. Reconstruct the original with
// [ReconstructLaplacianPyramid].
func LaplacianPyramid(f *cv.FloatMat, levels int) []*cv.FloatMat {
	g := GaussianPyramid(f, levels)
	n := len(g)
	lap := make([]*cv.FloatMat, n)
	for l := 0; l < n-1; l++ {
		up := PyrUp(g[l+1], g[l].Rows, g[l].Cols)
		d := cv.NewFloatMat(g[l].Rows, g[l].Cols)
		for i := range d.Data {
			d.Data[i] = g[l].Data[i] - up.Data[i]
		}
		lap[l] = d
	}
	lap[n-1] = g[n-1]
	return lap
}

// ReconstructLaplacianPyramid collapses a Laplacian pyramid back into a single
// matrix, the inverse of [LaplacianPyramid]. The pyramid must be non-empty.
func ReconstructLaplacianPyramid(pyr []*cv.FloatMat) *cv.FloatMat {
	if len(pyr) == 0 {
		panic("photo2: ReconstructLaplacianPyramid given an empty pyramid")
	}
	cur := photo2CloneFloat(pyr[len(pyr)-1])
	for l := len(pyr) - 2; l >= 0; l-- {
		up := PyrUp(cur, pyr[l].Rows, pyr[l].Cols)
		out := cv.NewFloatMat(pyr[l].Rows, pyr[l].Cols)
		for i := range out.Data {
			out.Data[i] = up.Data[i] + pyr[l].Data[i]
		}
		cur = out
	}
	return cur
}

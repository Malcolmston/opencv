package pyramids

import (
	cv "github.com/malcolmston/opencv"
)

// PyrDownFloat blurs f with the separable binomial kernel and drops every other
// row and column, halving each dimension (rounding up). It is one reduction
// step of a Gaussian pyramid. A grid smaller than 1×1 cannot be produced; a
// 1×N or N×1 grid reduces along the larger axis only.
func PyrDownFloat(f *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(f, "PyrDownFloat")
	k := BinomialKernel()
	blurred := ConvolveSeparable(f, k, k)
	dh := (f.Rows + 1) / 2
	dw := (f.Cols + 1) / 2
	if dh < 1 {
		dh = 1
	}
	if dw < 1 {
		dw = 1
	}
	out := cv.NewFloatMat(dh, dw)
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			out.Data[y*dw+x] = blurred.Data[(y*2)*f.Cols+(x*2)]
		}
	}
	return out
}

// PyrUpFloat upsamples f to the given destination size by injecting its samples
// at even coordinates of a zero grid and smoothing with the binomial kernel
// scaled by four (to compensate for the injected zeros, preserving average
// brightness). The destination size must be the size of the finer level that f
// was reduced from — that is, dstRows in {2*f.Rows-1, 2*f.Rows} and likewise
// for dstCols — so that build and reconstruct stay consistent. It panics on an
// incompatible destination size.
func PyrUpFloat(f *cv.FloatMat, dstRows, dstCols int) *cv.FloatMat {
	pyramidsRequire(f, "PyrUpFloat")
	if dstRows != 2*f.Rows && dstRows != 2*f.Rows-1 {
		panic("pyramids: PyrUpFloat: incompatible destination rows")
	}
	if dstCols != 2*f.Cols && dstCols != 2*f.Cols-1 {
		panic("pyramids: PyrUpFloat: incompatible destination cols")
	}
	up := cv.NewFloatMat(dstRows, dstCols)
	for y := 0; y < f.Rows; y++ {
		dy := y * 2
		if dy >= dstRows {
			continue
		}
		for x := 0; x < f.Cols; x++ {
			dx := x * 2
			if dx >= dstCols {
				continue
			}
			up.Data[dy*dstCols+dx] = f.Data[y*f.Cols+x]
		}
	}
	k := BinomialKernel()
	sm := ConvolveSeparable(up, k, k)
	for i := range sm.Data {
		sm.Data[i] *= 4
	}
	return sm
}

// GaussianPyramid is a sequence of increasingly coarse, blurred and
// down-sampled versions of an image. Levels[0] is the original resolution and
// each subsequent level halves both dimensions (rounding up).
type GaussianPyramid struct {
	// Levels holds the pyramid images from finest (index 0) to coarsest.
	Levels []*cv.FloatMat
}

// BuildGaussianPyramid constructs a Gaussian pyramid of the requested number of
// levels from f. levels counts the base, so levels==1 yields just a copy of the
// input. Reduction stops early if a further level would be smaller than 1×1. It
// panics if levels is not positive.
func BuildGaussianPyramid(f *cv.FloatMat, levels int) *GaussianPyramid {
	pyramidsRequire(f, "BuildGaussianPyramid")
	if levels <= 0 {
		panic("pyramids: BuildGaussianPyramid: levels must be positive")
	}
	out := &GaussianPyramid{Levels: make([]*cv.FloatMat, 0, levels)}
	cur := CloneFloat(f)
	out.Levels = append(out.Levels, cur)
	for i := 1; i < levels; i++ {
		if cur.Rows <= 1 && cur.Cols <= 1 {
			break
		}
		cur = PyrDownFloat(cur)
		out.Levels = append(out.Levels, cur)
	}
	return out
}

// NumLevels returns the number of levels in the pyramid.
func (p *GaussianPyramid) NumLevels() int { return len(p.Levels) }

// Level returns the pyramid image at index i (0 is finest). It panics if i is
// out of range.
func (p *GaussianPyramid) Level(i int) *cv.FloatMat {
	if i < 0 || i >= len(p.Levels) {
		panic("pyramids: GaussianPyramid.Level: index out of range")
	}
	return p.Levels[i]
}

// Base returns the finest (full-resolution) level of the pyramid.
func (p *GaussianPyramid) Base() *cv.FloatMat { return p.Levels[0] }

// Coarsest returns the smallest (last) level of the pyramid.
func (p *GaussianPyramid) Coarsest() *cv.FloatMat { return p.Levels[len(p.Levels)-1] }

// LaplacianPyramid is a band-pass decomposition of an image. Each band holds
// the detail lost between two adjacent Gaussian levels, and Base holds the
// coarsest low-pass residual. Summing the expanded bands back onto Base
// reconstructs the original image exactly (in the float domain).
type LaplacianPyramid struct {
	// Bands holds the band-pass detail images from finest (index 0) to
	// coarsest; band i has the size of Gaussian level i.
	Bands []*cv.FloatMat
	// Base is the coarsest low-pass residual (the smallest Gaussian level).
	Base *cv.FloatMat
}

// BuildLaplacianPyramid constructs a Laplacian pyramid with the requested
// number of band levels plus a residual base. It computes the Gaussian pyramid
// and, for each adjacent pair, stores band_i = G_i - expand(G_{i+1}). The
// number of bands is one less than the number of Gaussian levels actually
// produced. It panics if levels is not positive.
func BuildLaplacianPyramid(f *cv.FloatMat, levels int) *LaplacianPyramid {
	g := BuildGaussianPyramid(f, levels)
	lp := &LaplacianPyramid{Bands: make([]*cv.FloatMat, 0, len(g.Levels)-1)}
	for i := 0; i+1 < len(g.Levels); i++ {
		fine := g.Levels[i]
		coarse := g.Levels[i+1]
		expanded := PyrUpFloat(coarse, fine.Rows, fine.Cols)
		lp.Bands = append(lp.Bands, SubtractFloat(fine, expanded))
	}
	lp.Base = g.Levels[len(g.Levels)-1]
	return lp
}

// NumBands returns the number of band-pass levels (excluding the base).
func (lp *LaplacianPyramid) NumBands() int { return len(lp.Bands) }

// Band returns the band-pass image at index i (0 is finest). It panics if i is
// out of range.
func (lp *LaplacianPyramid) Band(i int) *cv.FloatMat {
	if i < 0 || i >= len(lp.Bands) {
		panic("pyramids: LaplacianPyramid.Band: index out of range")
	}
	return lp.Bands[i]
}

// Reconstruct rebuilds the original image from the Laplacian pyramid by
// repeatedly expanding the running coarse image and adding the next finer band.
// For an unmodified pyramid the result equals the source image to within
// floating-point rounding.
func (lp *LaplacianPyramid) Reconstruct() *cv.FloatMat {
	cur := CloneFloat(lp.Base)
	for i := len(lp.Bands) - 1; i >= 0; i-- {
		band := lp.Bands[i]
		expanded := PyrUpFloat(cur, band.Rows, band.Cols)
		cur = AddFloat(expanded, band)
	}
	return cur
}

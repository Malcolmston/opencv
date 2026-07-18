package pyramids

import (
	cv "github.com/malcolmston/opencv"
)

// HaarBands holds one level of a 2-D orthonormal Haar wavelet transform: the
// LL approximation (low-low) and the three detail sub-bands LH (horizontal
// detail), HL (vertical detail) and HH (diagonal detail). Every band has half
// the width and half the height of the analysed image. The transform is
// orthonormal (scaled by 1/2 per 2×2 block), so [HaarInverse] recovers the
// input exactly.
type HaarBands struct {
	// LL is the coarse approximation (average) sub-band.
	LL *cv.FloatMat
	// LH is the horizontal-detail sub-band (responds to vertical edges).
	LH *cv.FloatMat
	// HL is the vertical-detail sub-band (responds to horizontal edges).
	HL *cv.FloatMat
	// HH is the diagonal-detail sub-band.
	HH *cv.FloatMat
}

// pyramidsEvenPad returns f padded by edge replication so both dimensions are
// even, together with the padded size. If f is already even it is returned
// unchanged.
func pyramidsEvenPad(f *cv.FloatMat) *cv.FloatMat {
	nr, nc := f.Rows, f.Cols
	if nr%2 != 0 {
		nr++
	}
	if nc%2 != 0 {
		nc++
	}
	if nr == f.Rows && nc == f.Cols {
		return f
	}
	out := cv.NewFloatMat(nr, nc)
	for y := 0; y < nr; y++ {
		for x := 0; x < nc; x++ {
			out.Data[y*nc+x] = pyramidsAt(f, y, x)
		}
	}
	return out
}

// HaarForward applies one level of the 2-D orthonormal Haar wavelet transform
// to f, returning the four half-size sub-bands. For each 2×2 block with samples
// a (top-left), b (top-right), c (bottom-left), d (bottom-right) the outputs
// are LL=(a+b+c+d)/2, LH=(a-b+c-d)/2, HL=(a+b-c-d)/2 and HH=(a-b-c+d)/2. Odd
// dimensions are edge-padded to even before analysis.
func HaarForward(f *cv.FloatMat) *HaarBands {
	pyramidsRequire(f, "HaarForward")
	p := pyramidsEvenPad(f)
	hr, hc := p.Rows/2, p.Cols/2
	h := &HaarBands{
		LL: cv.NewFloatMat(hr, hc),
		LH: cv.NewFloatMat(hr, hc),
		HL: cv.NewFloatMat(hr, hc),
		HH: cv.NewFloatMat(hr, hc),
	}
	for y := 0; y < hr; y++ {
		for x := 0; x < hc; x++ {
			a := p.Data[(2*y)*p.Cols+(2*x)]
			b := p.Data[(2*y)*p.Cols+(2*x+1)]
			c := p.Data[(2*y+1)*p.Cols+(2*x)]
			d := p.Data[(2*y+1)*p.Cols+(2*x+1)]
			i := y*hc + x
			h.LL.Data[i] = (a + b + c + d) / 2
			h.LH.Data[i] = (a - b + c - d) / 2
			h.HL.Data[i] = (a + b - c - d) / 2
			h.HH.Data[i] = (a - b - c + d) / 2
		}
	}
	return h
}

// HaarInverse reconstructs the image from one level of Haar sub-bands, the
// exact inverse of [HaarForward]. The four bands must share dimensions; the
// result is twice as large in each dimension. If the forward transform padded
// an odd input, the extra reconstructed row or column is the replicated edge
// and can be cropped by the caller.
func HaarInverse(h *HaarBands) *cv.FloatMat {
	pyramidsRequire(h.LL, "HaarInverse")
	pyramidsRequire(h.LH, "HaarInverse")
	pyramidsRequire(h.HL, "HaarInverse")
	pyramidsRequire(h.HH, "HaarInverse")
	pyramidsSameSize(h.LL, h.LH, "HaarInverse")
	pyramidsSameSize(h.LL, h.HL, "HaarInverse")
	pyramidsSameSize(h.LL, h.HH, "HaarInverse")
	hr, hc := h.LL.Rows, h.LL.Cols
	out := cv.NewFloatMat(hr*2, hc*2)
	for y := 0; y < hr; y++ {
		for x := 0; x < hc; x++ {
			i := y*hc + x
			ll, lh, hl, hh := h.LL.Data[i], h.LH.Data[i], h.HL.Data[i], h.HH.Data[i]
			a := (ll + lh + hl + hh) / 2
			b := (ll - lh + hl - hh) / 2
			c := (ll + lh - hl - hh) / 2
			d := (ll - lh - hl + hh) / 2
			out.Data[(2*y)*out.Cols+(2*x)] = a
			out.Data[(2*y)*out.Cols+(2*x+1)] = b
			out.Data[(2*y+1)*out.Cols+(2*x)] = c
			out.Data[(2*y+1)*out.Cols+(2*x+1)] = d
		}
	}
	return out
}

// WaveletPyramid is a multi-level Haar wavelet decomposition: the detail
// sub-bands at each level plus the coarsest approximation. Level 0 is the
// finest. Because each level records the original (possibly odd) size of its
// input, [WaveletPyramid.Reconstruct] recovers the source exactly.
type WaveletPyramid struct {
	// Levels holds the detail sub-bands from finest (index 0) to coarsest.
	Levels []*HaarBands
	// Base is the coarsest LL approximation.
	Base *cv.FloatMat
	// sizes records the (rows, cols) of the input to each forward level so the
	// inverse can crop replicated padding.
	sizes [][2]int
}

// BuildWaveletPyramid applies the Haar transform recursively, feeding each
// level's LL approximation into the next, for levels iterations (or until the
// approximation reaches 1×1). It panics if levels is not positive.
func BuildWaveletPyramid(f *cv.FloatMat, levels int) *WaveletPyramid {
	pyramidsRequire(f, "BuildWaveletPyramid")
	if levels <= 0 {
		panic("pyramids: BuildWaveletPyramid: levels must be positive")
	}
	wp := &WaveletPyramid{}
	cur := CloneFloat(f)
	for i := 0; i < levels; i++ {
		if cur.Rows < 2 && cur.Cols < 2 {
			break
		}
		wp.sizes = append(wp.sizes, [2]int{cur.Rows, cur.Cols})
		h := HaarForward(cur)
		wp.Levels = append(wp.Levels, h)
		cur = h.LL
	}
	wp.Base = cur
	return wp
}

// NumLevels returns the number of wavelet decomposition levels.
func (wp *WaveletPyramid) NumLevels() int { return len(wp.Levels) }

// Reconstruct rebuilds the original image by inverting each Haar level from
// coarsest to finest, cropping any edge-replication padding introduced for odd
// sizes. For an unmodified pyramid the result equals the source image to within
// floating-point rounding.
func (wp *WaveletPyramid) Reconstruct() *cv.FloatMat {
	cur := CloneFloat(wp.Base)
	for i := len(wp.Levels) - 1; i >= 0; i-- {
		h := wp.Levels[i]
		// Swap in the running approximation for the stored LL, invert, then
		// crop back to the original level size.
		rebuilt := HaarInverse(&HaarBands{LL: cur, LH: h.LH, HL: h.HL, HH: h.HH})
		r, c := wp.sizes[i][0], wp.sizes[i][1]
		if rebuilt.Rows == r && rebuilt.Cols == c {
			cur = rebuilt
			continue
		}
		cropped := cv.NewFloatMat(r, c)
		for y := 0; y < r; y++ {
			for x := 0; x < c; x++ {
				cropped.Data[y*c+x] = rebuilt.Data[y*rebuilt.Cols+x]
			}
		}
		cur = cropped
	}
	return cur
}

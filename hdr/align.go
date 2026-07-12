package hdr

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Bitmap is a single-channel binary image whose samples are 0 or 1. It is the
// working representation used by the median-threshold-bitmap aligner: a
// threshold bitmap marks pixels brighter than the image median, and an
// exclusion bitmap marks pixels far enough from the median to carry reliable
// alignment information.
type Bitmap struct {
	// Rows is the bitmap height.
	Rows int
	// Cols is the bitmap width.
	Cols int
	// Data holds Rows*Cols samples in row-major order, each 0 or 1.
	Data []uint8
}

// Count returns the number of set (non-zero) samples in the bitmap.
func (b *Bitmap) Count() int {
	n := 0
	for _, v := range b.Data {
		if v != 0 {
			n++
		}
	}
	return n
}

// AlignMTB aligns a bracket of hand-held exposures using Ward's (2003)
// median-threshold-bitmap (MTB) algorithm, the same method OpenCV exposes as
// createAlignMTB. Each image is reduced to a threshold bitmap (pixels above the
// median) that is invariant to exposure changes, and the translation that best
// registers two bitmaps is found by a coarse-to-fine search over an image
// pyramid, testing the nine integer shifts around the current estimate at each
// level. Only whole-pixel translations are considered; rotation and scale are
// not corrected.
type AlignMTB struct {
	// MaxBits is the number of pyramid levels; the largest detectable shift is
	// about 2^MaxBits pixels. Non-positive selects a default of 6.
	MaxBits int
	// ExclusionRange is the half-width, in 8-bit levels, of the band around the
	// median whose pixels are excluded from the error count (they carry little
	// information and are sensitive to noise). Negative selects a default of 4.
	ExclusionRange int
	// Cut enables the exclusion bitmap. When false every pixel contributes to
	// the alignment error.
	Cut bool
}

// NewAlignMTB returns an aligner with the given pyramid depth and OpenCV-style
// defaults (exclusion range 4, exclusion enabled). Pass a non-positive maxBits
// for the default depth of 6.
func NewAlignMTB(maxBits int) *AlignMTB {
	if maxBits <= 0 {
		maxBits = 6
	}
	return &AlignMTB{MaxBits: maxBits, ExclusionRange: 4, Cut: true}
}

// grayMTB converts a Mat to a single-channel luminance plane in [0,255]. A
// single-channel Mat is copied; a multi-channel Mat is reduced with the Rec.709
// luma weights on its first three channels.
func grayMTB(m *cv.Mat) *plane {
	p := newPlane(m.Rows, m.Cols)
	total := m.Rows * m.Cols
	if m.Channels == 1 {
		for i := 0; i < total; i++ {
			p.data[i] = float64(m.Data[i])
		}
		return p
	}
	for i := 0; i < total; i++ {
		base := i * m.Channels
		r := float64(m.Data[base+0])
		g := float64(m.Data[base+1])
		b := float64(m.Data[base+2])
		p.data[i] = 0.2126*r + 0.7152*g + 0.0722*b
	}
	return p
}

// medianPlane returns the median sample value of a plane.
func medianPlane(p *plane) float64 {
	cp := make([]float64, len(p.data))
	copy(cp, p.data)
	sort.Float64s(cp)
	n := len(cp)
	if n == 0 {
		return 0
	}
	return cp[n/2]
}

// bitmapsFromGray builds the threshold and exclusion bitmaps of a gray plane.
func (a *AlignMTB) bitmapsFromGray(g *plane) (tb, eb *Bitmap) {
	med := medianPlane(g)
	tb = &Bitmap{Rows: g.rows, Cols: g.cols, Data: make([]uint8, len(g.data))}
	eb = &Bitmap{Rows: g.rows, Cols: g.cols, Data: make([]uint8, len(g.data))}
	ex := float64(a.ExclusionRange)
	if ex < 0 {
		ex = 4
	}
	for i, v := range g.data {
		if v > med {
			tb.Data[i] = 1
		}
		if math.Abs(v-med) > ex {
			eb.Data[i] = 1
		}
	}
	return tb, eb
}

// ComputeBitmaps returns the threshold and exclusion bitmaps of a full
// resolution image, exactly as used internally by the alignment search. The
// threshold bitmap has a 1 wherever the pixel luminance exceeds the image
// median; the exclusion bitmap has a 1 wherever the luminance lies outside the
// [median-ExclusionRange, median+ExclusionRange] band.
func (a *AlignMTB) ComputeBitmaps(m *cv.Mat) (tb, eb *Bitmap) {
	return a.bitmapsFromGray(grayMTB(m))
}

// downsampleGray halves a gray plane by averaging each 2x2 block.
func downsampleGray(p *plane) *plane {
	nr := (p.rows + 1) / 2
	nc := (p.cols + 1) / 2
	out := newPlane(nr, nc)
	for y := 0; y < nr; y++ {
		for x := 0; x < nc; x++ {
			var s float64
			var n int
			for yy := 0; yy < 2; yy++ {
				for xx := 0; xx < 2; xx++ {
					sy, sx := 2*y+yy, 2*x+xx
					if sy < p.rows && sx < p.cols {
						s += p.at(sy, sx)
						n++
					}
				}
			}
			if n > 0 {
				out.set(y, x, s/float64(n))
			}
		}
	}
	return out
}

// effectiveLevels caps the pyramid depth so the coarsest level stays above a
// few pixels across, regardless of MaxBits.
func (a *AlignMTB) effectiveLevels(rows, cols int) int {
	maxBits := a.MaxBits
	if maxBits <= 0 {
		maxBits = 6
	}
	levels := 1
	m := minInt(rows, cols)
	for levels < maxBits && m > 8 {
		m = (m + 1) / 2
		levels++
	}
	return levels
}

// grayPyramidMTB returns the gray plane and its successive 2x downsamples.
func grayPyramidMTB(p *plane, levels int) []*plane {
	pyr := make([]*plane, levels)
	pyr[0] = p
	for l := 1; l < levels; l++ {
		pyr[l] = downsampleGray(pyr[l-1])
	}
	return pyr
}

// bitmapError counts the disagreements between tb0 and tb1 after shifting tb1
// (and its exclusion bitmap) by (xs, ys), masked by both exclusion bitmaps when
// cut is set. A shift of (xs, ys) means shifted[y][x] = tb1[y-ys][x-xs].
func bitmapError(tb0, eb0, tb1, eb1 *Bitmap, xs, ys int, cut bool) int {
	rows, cols := tb0.Rows, tb0.Cols
	err := 0
	for y := 0; y < rows; y++ {
		sy := y - ys
		for x := 0; x < cols; x++ {
			sx := x - xs
			var t1, e1 uint8
			if sy >= 0 && sy < rows && sx >= 0 && sx < cols {
				idx := sy*cols + sx
				t1 = tb1.Data[idx]
				e1 = eb1.Data[idx]
			}
			i := y*cols + x
			d := tb0.Data[i] ^ t1
			if cut {
				d &= eb0.Data[i] & e1
			}
			err += int(d)
		}
	}
	return err
}

// CalculateShift returns the integer (dx, dy) translation that best registers
// src onto ref: applying Shift(src, dx, dy) aligns it with ref. The search is
// coarse-to-fine over a median-threshold-bitmap pyramid.
func (a *AlignMTB) CalculateShift(ref, src *cv.Mat) (dx, dy int) {
	levels := a.effectiveLevels(ref.Rows, ref.Cols)
	pyr0 := grayPyramidMTB(grayMTB(ref), levels)
	pyr1 := grayPyramidMTB(grayMTB(src), levels)

	shiftX, shiftY := 0, 0
	for l := levels - 1; l >= 0; l-- {
		shiftX *= 2
		shiftY *= 2
		tb0, eb0 := a.bitmapsFromGray(pyr0[l])
		tb1, eb1 := a.bitmapsFromGray(pyr1[l])
		bestErr := math.MaxInt
		bx, by := shiftX, shiftY
		// Test the zero offset first so ties favour the (unbiased) current
		// estimate rather than an arbitrary neighbour.
		offsets := [9][2]int{{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}}
		for _, off := range offsets {
			xs, ys := shiftX+off[0], shiftY+off[1]
			e := bitmapError(tb0, eb0, tb1, eb1, xs, ys, a.Cut)
			if e < bestErr {
				bestErr = e
				bx, by = xs, ys
			}
		}
		shiftX, shiftY = bx, by
	}
	return shiftX, shiftY
}

// Shift returns a copy of m translated by (dx, dy): the output pixel (x, y)
// takes the value of the input pixel (x-dx, y-dy), with pixels shifted in from
// outside the image left black. Positive dx moves content right; positive dy
// moves it down.
func (a *AlignMTB) Shift(m *cv.Mat, dx, dy int) *cv.Mat {
	out := cv.NewMat(m.Rows, m.Cols, m.Channels)
	ch := m.Channels
	for y := 0; y < m.Rows; y++ {
		sy := y - dy
		if sy < 0 || sy >= m.Rows {
			continue
		}
		for x := 0; x < m.Cols; x++ {
			sx := x - dx
			if sx < 0 || sx >= m.Cols {
				continue
			}
			si := (sy*m.Cols + sx) * ch
			di := (y*m.Cols + x) * ch
			copy(out.Data[di:di+ch], m.Data[si:si+ch])
		}
	}
	return out
}

// Process aligns a bracket of exposures to the middle frame and returns the
// registered copies in the original order. The reference frame is returned as a
// clone; every other frame is shifted by its recovered translation. Inputs must
// share dimensions and channel count.
func (a *AlignMTB) Process(images []*cv.Mat) []*cv.Mat {
	n := len(images)
	out := make([]*cv.Mat, n)
	if n == 0 {
		return out
	}
	ref := n / 2
	for i := 0; i < n; i++ {
		if i == ref {
			out[i] = images[i].Clone()
			continue
		}
		dx, dy := a.CalculateShift(images[ref], images[i])
		out[i] = a.Shift(images[i], dx, dy)
	}
	return out
}

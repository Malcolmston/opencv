package fuzzy

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Components1 holds the degree-1 (linear) F-transform components of an image
// over a fuzzy partition, mirroring the FT12D family of OpenCV's ft module.
//
// Where the degree-0 [Components] stores a single value per node (the weighted
// average of the pixels beneath a basis function), the degree-1 transform fits a
// local linear polynomial c00 + c10*(x-x0) + c01*(y-y0) under each basis, where
// (x0, y0) is the node centre. The three coefficients are found by a fuzzy-
// weighted least-squares fit, so the components capture not just the local
// brightness (c00) but also its horizontal (c10) and vertical (c01) gradient.
// The inverse transform blends those local planes back through the partition of
// unity, reproducing gradients far more faithfully than the degree-0 transform.
//
// The partition has An nodes horizontally and Bn nodes vertically, spaced Radius
// pixels apart. C00, C10 and C01 each store one coefficient per node in row-major
// node order, interleaved by channel: the coefficient for node row i, node column
// o and channel c lives at index (i*An+o)*Channels+c. Rows and Cols record the
// original image size so [Components1.Reconstruct] can crop its reconstruction.
type Components1 struct {
	// An is the number of partition nodes along the x (column) axis.
	An int
	// Bn is the number of partition nodes along the y (row) axis.
	Bn int
	// Radius is the partition node spacing and kernel radius in pixels.
	Radius int
	// Channels is the number of image channels each node carries components for.
	Channels int
	// Rows and Cols are the dimensions of the original (un-padded) image.
	Rows, Cols int
	// Function is the basis function the kernel was built from.
	Function BasisFunction
	// C00 holds the constant (degree-0) coefficient of each node's local plane.
	C00 []float64
	// C10 holds the horizontal-gradient coefficient of each node's local plane.
	C10 []float64
	// C01 holds the vertical-gradient coefficient of each node's local plane.
	C01 []float64
	// valid[i*An+o] reports whether node (i, o) had enough unmasked pixels under
	// it to carry a meaningful component; invalid nodes are skipped by the inverse.
	valid []bool
	// kernel is the fuzzy-partition kernel used to compute these components; it is
	// reused by the inverse transform.
	kernel *cv.FloatMat
}

// At returns the constant (degree-0) coefficient c00 for node (nodeY, nodeX) and
// channel ch. It is the direct analogue of [Components.At] and equals the local
// weighted average, so it can be compared against the degree-0 components.
func (c *Components1) At(nodeY, nodeX, ch int) float64 {
	return c.C00[(nodeY*c.An+nodeX)*c.Channels+ch]
}

// Polynomial returns the three coefficients (c00, c10, c01) of the local linear
// polynomial fitted at node (nodeY, nodeX) for channel ch. The reconstructed
// value at an offset (dx, dy) pixels from the node centre is
// c00 + c10*dx + c01*dy.
func (c *Components1) Polynomial(nodeY, nodeX, ch int) (c00, c10, c01 float64) {
	idx := (nodeY*c.An+nodeX)*c.Channels + ch
	return c.C00[idx], c.C10[idx], c.C01[idx]
}

// FT12DComponents computes the degree-1 F-transform components of img over the
// fuzzy partition induced by kernel, mirroring OpenCV's ft::FT12D_components.
//
// For every node it solves the 3x3 fuzzy-weighted least-squares system for the
// best local linear plane through the pixels under the basis function. Solving
// the full system (rather than assuming the analytic symmetric-window result)
// keeps the fit correct at image borders and under an arbitrary validity mask,
// where the window is only partially populated. When the system is singular —
// too few valid pixels to determine a gradient — the node degrades gracefully to
// its degree-0 average with zero gradient.
//
// mask, when non-nil, is a per-pixel validity mask the same size as img: a pixel
// whose first mask channel is zero is excluded from the fit. kernel must be a
// square, odd-sided [cv.FloatMat] as produced by [CreateKernel]. img may have any
// number of channels; each is fitted independently.
func FT12DComponents(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat) *Components1 {
	if img == nil || img.Empty() {
		panic("fuzzy: FT12DComponents given an empty image")
	}
	radius := kernelRadius(kernel)
	if mask != nil && (mask.Rows != img.Rows || mask.Cols != img.Cols) {
		panic(fmt.Sprintf("fuzzy: mask %dx%d does not match image %dx%d", mask.Rows, mask.Cols, img.Rows, img.Cols))
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	bn, an := partitionCounts(rows, cols, radius)

	c := &Components1{
		An: an, Bn: bn, Radius: radius, Channels: ch,
		Rows: rows, Cols: cols, Function: LinearBasis,
		C00: make([]float64, bn*an*ch), C10: make([]float64, bn*an*ch),
		C01: make([]float64, bn*an*ch), valid: make([]bool, bn*an), kernel: kernel,
	}

	kside := 2*radius + 1
	for i := 0; i < bn; i++ {
		for o := 0; o < an; o++ {
			originY := i*radius - radius
			originX := o*radius - radius
			for cc := 0; cc < ch; cc++ {
				// Accumulate the weighted normal-equation moments. dx and dy are
				// pixel offsets from the node centre, which lies at kernel index
				// (radius, radius), i.e. dx = kx-radius, dy = ky-radius.
				var s0, sx, sy, sxx, sxy, syy float64
				var b0, bx, by float64
				for ky := 0; ky < kside; ky++ {
					iy := originY + ky
					if iy < 0 || iy >= rows {
						continue
					}
					dy := float64(ky - radius)
					for kx := 0; kx < kside; kx++ {
						ix := originX + kx
						if ix < 0 || ix >= cols {
							continue
						}
						if mask != nil && mask.Data[(iy*cols+ix)*mask.Channels] == 0 {
							continue
						}
						w := kernel.Data[ky*kside+kx]
						if w == 0 {
							continue
						}
						dx := float64(kx - radius)
						f := float64(img.Data[(iy*cols+ix)*ch+cc])
						s0 += w
						sx += w * dx
						sy += w * dy
						sxx += w * dx * dx
						sxy += w * dx * dy
						syy += w * dy * dy
						b0 += w * f
						bx += w * dx * f
						by += w * dy * f
					}
				}
				if s0 <= 0 {
					continue // no valid pixel: node stays invalid.
				}
				idx := (i*an+o)*ch + cc
				c00, c10, c01, ok := solve3(s0, sx, sy, sxx, sxy, syy, b0, bx, by)
				if !ok {
					// Singular system: fall back to the degree-0 average.
					c00, c10, c01 = b0/s0, 0, 0
				}
				c.C00[idx] = c00
				c.C10[idx] = c10
				c.C01[idx] = c01
				c.valid[i*an+o] = true
			}
		}
	}
	return c
}

// solve3 solves the symmetric 3x3 linear system arising from the degree-1
// weighted least-squares fit using Cramer's rule. The matrix is
//
//	[ s0  sx  sy  ] [c00]   [b0]
//	[ sx  sxx sxy ] [c10] = [bx]
//	[ sy  sxy syy ] [c01]   [by]
//
// It returns ok == false when the matrix is numerically singular (a degenerate
// window that cannot determine a gradient), so the caller can fall back.
func solve3(s0, sx, sy, sxx, sxy, syy, b0, bx, by float64) (c00, c10, c01 float64, ok bool) {
	// Cofactors of the symmetric matrix.
	m00 := sxx*syy - sxy*sxy
	m01 := sx*syy - sy*sxy
	m02 := sx*sxy - sy*sxx
	det := s0*m00 - sx*m01 + sy*m02
	// Scale the singularity threshold by the moment magnitude so it is invariant
	// to the number of pixels and the kernel weight range.
	scale := s0*sxx*syy + 1
	if math.Abs(det) < 1e-9*scale {
		return 0, 0, 0, false
	}
	inv := 1 / det
	// Inverse of the symmetric matrix (only the entries we need).
	i00 := m00 * inv
	i01 := -m01 * inv
	i02 := m02 * inv
	i11 := (s0*syy - sy*sy) * inv
	i12 := -(s0*sxy - sx*sy) * inv
	i22 := (s0*sxx - sx*sx) * inv
	c00 = i00*b0 + i01*bx + i02*by
	c10 = i01*b0 + i11*bx + i12*by
	c01 = i02*b0 + i12*bx + i22*by
	return c00, c10, c01, true
}

// FT12DInverse reconstructs an image from degree-1 F-transform components,
// mirroring OpenCV's ft::FT12D_inverse. Each node paints its local linear plane
// c00 + c10*dx + c01*dy (with dx, dy the pixel offset from the node centre) back
// through the basis; contributions are normalised by the total basis weight
// reaching each pixel, so interior pixels — covered by a partition of unity — are
// reproduced from a blend of neighbouring planes, and border pixels are handled
// without darkening. Because each node reconstructs a sloped plane rather than a
// constant, linear image content is recovered essentially exactly.
//
// The returned [cv.Mat] has the original image's dimensions and channel count,
// with values rounded and clamped to the 8-bit range.
func FT12DInverse(c *Components1) *cv.Mat {
	out, _ := c.inverseWithCoverage()
	return out
}

// inverseWithCoverage performs the degree-1 inverse and additionally reports,
// for every pixel, whether any valid node's basis reached it. Uncovered pixels
// are left at zero.
func (c *Components1) inverseWithCoverage() (*cv.Mat, []bool) {
	radius := c.Radius
	rows, cols, ch := c.Rows, c.Cols, c.Channels
	kside := 2*radius + 1
	kernel := c.kernel

	acc := make([]float64, rows*cols*ch)
	wsum := make([]float64, rows*cols)

	for i := 0; i < c.Bn; i++ {
		for o := 0; o < c.An; o++ {
			if !c.valid[i*c.An+o] {
				continue
			}
			originY := i*radius - radius
			originX := o*radius - radius
			for ky := 0; ky < kside; ky++ {
				iy := originY + ky
				if iy < 0 || iy >= rows {
					continue
				}
				dy := float64(ky - radius)
				for kx := 0; kx < kside; kx++ {
					ix := originX + kx
					if ix < 0 || ix >= cols {
						continue
					}
					w := kernel.Data[ky*kside+kx]
					if w == 0 {
						continue
					}
					dx := float64(kx - radius)
					p := iy*cols + ix
					wsum[p] += w
					base := (i*c.An + o) * ch
					for cc := 0; cc < ch; cc++ {
						plane := c.C00[base+cc] + c.C10[base+cc]*dx + c.C01[base+cc]*dy
						acc[p*ch+cc] += w * plane
					}
				}
			}
		}
	}

	out := cv.NewMat(rows, cols, ch)
	covered := make([]bool, rows*cols)
	for p := 0; p < rows*cols; p++ {
		if wsum[p] <= 0 {
			continue
		}
		covered[p] = true
		for cc := 0; cc < ch; cc++ {
			out.Data[p*ch+cc] = clampByte(acc[p*ch+cc] / wsum[p])
		}
	}
	return out, covered
}

// Reconstruct is a convenience method equivalent to [FT12DInverse](c); it paints
// the degree-1 components back into a full-size image.
func (c *Components1) Reconstruct() *cv.Mat {
	return FT12DInverse(c)
}

// FT12DProcess computes the degree-1 F-transform components of img with the given
// kernel and immediately reconstructs them, mirroring OpenCV's ft::FT12D_process.
// With a nil mask this is a gradient-preserving smoother; with a validity mask it
// also fills the masked-out pixels from the surrounding linear fits.
func FT12DProcess(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat) *cv.Mat {
	return FT12DInverse(FT12DComponents(img, kernel, mask))
}

// FT12DPolynomial computes the degree-1 F-transform components of img and returns
// the three polynomial-coefficient planes for the requested channel, mirroring
// OpenCV's ft::FT12D_polynomial. Each returned [cv.FloatMat] is Bn x An in node
// resolution: c00 is the local constant (average), c10 the horizontal gradient
// and c01 the vertical gradient of the fitted plane at each node. The full
// [Components1] is returned as well so the caller can invert or inspect other
// channels. channel must be a valid channel index of img.
func FT12DPolynomial(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat, channel int) (comps *Components1, c00, c10, c01 *cv.FloatMat) {
	comps = FT12DComponents(img, kernel, mask)
	if channel < 0 || channel >= comps.Channels {
		panic(fmt.Sprintf("fuzzy: FT12DPolynomial channel %d out of range [0,%d)", channel, comps.Channels))
	}
	c00 = comps.CoeffPlane(0, channel)
	c10 = comps.CoeffPlane(1, channel)
	c01 = comps.CoeffPlane(2, channel)
	return comps, c00, c10, c01
}

// CoeffPlane extracts one polynomial-coefficient plane at node resolution for a
// single channel as a Bn x An [cv.FloatMat]. which selects the coefficient:
// 0 for c00 (constant), 1 for c10 (x-gradient) and 2 for c01 (y-gradient). It
// panics for any other selector or an out-of-range channel.
func (c *Components1) CoeffPlane(which, channel int) *cv.FloatMat {
	if channel < 0 || channel >= c.Channels {
		panic(fmt.Sprintf("fuzzy: CoeffPlane channel %d out of range [0,%d)", channel, c.Channels))
	}
	var src []float64
	switch which {
	case 0:
		src = c.C00
	case 1:
		src = c.C10
	case 2:
		src = c.C01
	default:
		panic(fmt.Sprintf("fuzzy: CoeffPlane which must be 0, 1 or 2, got %d", which))
	}
	out := cv.NewFloatMat(c.Bn, c.An)
	for n := 0; n < c.Bn*c.An; n++ {
		out.Data[n] = src[n*c.Channels+channel]
	}
	return out
}

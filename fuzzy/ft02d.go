package fuzzy

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Components holds the degree-0 F-transform components of an image over a fuzzy
// partition. Each component is the weighted average of the input samples that
// fall under one basis function of the partition; together with the kernel they
// are all that is needed to reconstruct a smoothed image via [FT02DInverse].
//
// The partition has An nodes horizontally and Bn nodes vertically, spaced Radius
// pixels apart. Data stores the components in row-major node order, interleaved
// by channel: the component for node row i, node column o and channel c lives at
// Data[(i*An+o)*Channels+c]. Rows and Cols record the size of the original image
// so the inverse transform can crop its padded reconstruction back to it.
type Components struct {
	// An is the number of partition nodes along the x (column) axis.
	An int
	// Bn is the number of partition nodes along the y (row) axis.
	Bn int
	// Radius is the partition node spacing and kernel radius in pixels.
	Radius int
	// Channels is the number of image channels each node carries a component for.
	Channels int
	// Rows and Cols are the dimensions of the original (un-padded) image.
	Rows, Cols int
	// Function is the basis function the kernel was built from.
	Function BasisFunction
	// Data holds Bn*An*Channels component values (see the type doc for layout).
	Data []float64
	// valid[i*An+o] reports whether node (i, o) had any unmasked pixel under it
	// and therefore carries a meaningful component; fully-masked nodes are false
	// and are skipped by the inverse transform.
	valid []bool
	// kernel is the fuzzy-partition kernel used to compute these components; it
	// is reused by [FT02DInverse].
	kernel *cv.FloatMat
}

// At returns the component value for node (nodeY, nodeX) and channel c.
func (c *Components) At(nodeY, nodeX, ch int) float64 {
	return c.Data[(nodeY*c.An+nodeX)*c.Channels+ch]
}

// partitionCounts returns the number of partition nodes for an image of the
// given size at the given radius, following OpenCV's ft module: one node every
// radius pixels plus one, guaranteeing the padded partition fully covers the
// image with overlapping basis functions.
func partitionCounts(rows, cols, radius int) (bn, an int) {
	return rows/radius + 1, cols/radius + 1
}

// FT02DComponents computes the degree-0 F-transform components of img over the
// fuzzy partition induced by kernel, mirroring OpenCV's ft::FT02D_components.
//
// mask, when non-nil, is a per-pixel validity mask the same size as img: a pixel
// whose first mask channel is zero is excluded from every weighted average (its
// value is treated as unknown). This is what lets the components — and therefore
// the reconstruction — ignore corrupted pixels, and is the basis of [Inpaint].
// Pass a nil mask to use every pixel.
//
// The kernel must be a square, odd-sided [cv.FloatMat] as produced by
// [CreateKernel]. img may have any number of channels.
func FT02DComponents(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat) *Components {
	if img == nil || img.Empty() {
		panic("fuzzy: FT02DComponents given an empty image")
	}
	radius := kernelRadius(kernel)
	if mask != nil && (mask.Rows != img.Rows || mask.Cols != img.Cols) {
		panic(fmt.Sprintf("fuzzy: mask %dx%d does not match image %dx%d", mask.Rows, mask.Cols, img.Rows, img.Cols))
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	bn, an := partitionCounts(rows, cols, radius)

	c := &Components{
		An: an, Bn: bn, Radius: radius, Channels: ch,
		Rows: rows, Cols: cols, Function: LinearBasis,
		Data: make([]float64, bn*an*ch), valid: make([]bool, bn*an), kernel: kernel,
	}

	// Node centres live in a coordinate frame padded by radius on the top and
	// left; centre of node (i, o) is at padded pixel ((o+1)*radius, (i+1)*radius).
	// The corresponding un-padded image pixel is centre - radius.
	kside := 2*radius + 1
	for i := 0; i < bn; i++ {
		for o := 0; o < an; o++ {
			originY := i*radius - radius // top-left un-padded y of the kernel window
			originX := o*radius - radius
			for cc := 0; cc < ch; cc++ {
				var num, den float64
				for ky := 0; ky < kside; ky++ {
					iy := originY + ky
					if iy < 0 || iy >= rows {
						continue
					}
					for kx := 0; kx < kside; kx++ {
						ix := originX + kx
						if ix < 0 || ix >= cols {
							continue
						}
						if mask != nil && mask.Data[(iy*cols+ix)*mask.Channels] == 0 {
							continue
						}
						w := kernel.Data[ky*kside+kx]
						num += w * float64(img.Data[(iy*cols+ix)*ch+cc])
						den += w
					}
				}
				if den > 0 {
					c.Data[(i*an+o)*ch+cc] = num / den
					c.valid[i*an+o] = true
				}
				// den == 0 leaves the component at 0; a node with no valid pixel
				// in any channel stays invalid and is skipped by the inverse.
			}
		}
	}
	return c
}

// FT02DInverse reconstructs a smoothed image from F-transform components,
// mirroring OpenCV's ft::FT02D_inverse. Each node contributes its component
// scaled by the kernel to the surrounding pixels; contributions are normalised
// by the total basis weight reaching each pixel, so partition-of-unity interior
// pixels are reproduced exactly while border pixels — covered by fewer nodes —
// are still handled cleanly rather than darkened. Fully-masked nodes (component
// left at zero with no weight) do not contribute, letting reconstructed values
// bleed in from neighbouring valid nodes.
//
// The returned [cv.Mat] has the original image's dimensions and channel count,
// with values rounded and clamped to the 8-bit range.
func FT02DInverse(c *Components) *cv.Mat {
	out, _ := inverseWithCoverage(c)
	return out
}

// inverseWithCoverage performs the inverse transform and additionally reports,
// for every pixel, whether any valid node's basis reached it (covered == true).
// Pixels with covered == false lie outside the support of every valid node and
// are left at zero; the iterative inpainter uses this to know which unknown
// pixels a pass was actually able to fill.
func inverseWithCoverage(c *Components) (*cv.Mat, []bool) {
	radius := c.Radius
	rows, cols, ch := c.Rows, c.Cols, c.Channels
	kside := 2*radius + 1
	kernel := c.kernel

	acc := make([]float64, rows*cols*ch)
	wsum := make([]float64, rows*cols) // basis weight is channel-independent

	for i := 0; i < c.Bn; i++ {
		for o := 0; o < c.An; o++ {
			if !c.valid[i*c.An+o] {
				continue // fully-masked node: no meaningful component to spread.
			}
			originY := i*radius - radius
			originX := o*radius - radius
			for ky := 0; ky < kside; ky++ {
				iy := originY + ky
				if iy < 0 || iy >= rows {
					continue
				}
				for kx := 0; kx < kside; kx++ {
					ix := originX + kx
					if ix < 0 || ix >= cols {
						continue
					}
					w := kernel.Data[ky*kside+kx]
					if w == 0 {
						continue
					}
					p := iy*cols + ix
					wsum[p] += w
					for cc := 0; cc < ch; cc++ {
						acc[p*ch+cc] += w * c.Data[(i*c.An+o)*ch+cc]
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

// FT02DProcess computes the F-transform components of img with the given kernel
// and immediately reconstructs them, mirroring OpenCV's ft::FT02D_process. With
// a nil mask this smooths the image; with a validity mask it also fills the
// masked-out (zero) pixels from their surroundings — the ONE_STEP building block
// used by [Inpaint].
func FT02DProcess(img *cv.Mat, kernel *cv.FloatMat, mask *cv.Mat) *cv.Mat {
	return FT02DInverse(FT02DComponents(img, kernel, mask))
}

// Filter smooths img with the degree-0 F-transform over a fuzzy partition of the
// given basis function and radius. Larger radii remove more high-frequency
// detail. It is a convenience wrapper around [CreateKernel] and [FT02DProcess]
// and works on grayscale or multi-channel images. The input is not modified.
func Filter(img *cv.Mat, function BasisFunction, radius int) *cv.Mat {
	if img == nil || img.Empty() {
		panic("fuzzy: Filter given an empty image")
	}
	kernel := CreateKernel(function, radius)
	c := FT02DComponents(img, kernel, nil)
	c.Function = function
	return FT02DInverse(c)
}

// clampByte rounds v to the nearest integer and clamps it into [0, 255].
func clampByte(v float64) uint8 {
	v += 0.5
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

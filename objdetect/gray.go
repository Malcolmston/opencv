package objdetect

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// grayImage is an internal single-channel image of float64 luma samples in
// [0,255], stored row-major. It decouples the detectors from the root Mat's
// uint8 storage so intermediate computations (gradients, resampling) keep full
// precision.
type grayImage struct {
	w, h int
	pix  []float64
}

// matToGray converts a root cv.Mat into a grayImage. A single-channel Mat is
// copied verbatim; a Mat with three or more channels is reduced to BT.601 luma
// from its first three (RGB) channels; a two-channel Mat uses its first
// channel. It panics on an empty image.
func matToGray(img *cv.Mat) *grayImage {
	if img == nil || img.Empty() {
		panic("objdetect: nil or empty image")
	}
	w, h, ch := img.Cols, img.Rows, img.Channels
	g := &grayImage{w: w, h: h, pix: make([]float64, w*h)}
	n := w * h
	switch {
	case ch == 1:
		for i := 0; i < n; i++ {
			g.pix[i] = float64(img.Data[i])
		}
	case ch >= 3:
		for i := 0; i < n; i++ {
			base := i * ch
			r := float64(img.Data[base])
			gg := float64(img.Data[base+1])
			b := float64(img.Data[base+2])
			g.pix[i] = 0.299*r + 0.587*gg + 0.114*b
		}
	default: // ch == 2
		for i := 0; i < n; i++ {
			g.pix[i] = float64(img.Data[i*ch])
		}
	}
	return g
}

// at returns the sample at (x, y) with edge replication for out-of-range
// coordinates.
func (g *grayImage) at(x, y int) float64 {
	if x < 0 {
		x = 0
	} else if x >= g.w {
		x = g.w - 1
	}
	if y < 0 {
		y = 0
	} else if y >= g.h {
		y = g.h - 1
	}
	return g.pix[y*g.w+x]
}

// resize returns a bilinearly resampled copy of the image at the requested
// dimensions. It panics on non-positive dimensions.
func (g *grayImage) resize(nw, nh int) *grayImage {
	if nw <= 0 || nh <= 0 {
		panic("objdetect: resize requires positive dimensions")
	}
	out := &grayImage{w: nw, h: nh, pix: make([]float64, nw*nh)}
	sx := float64(g.w) / float64(nw)
	sy := float64(g.h) / float64(nh)
	for y := 0; y < nh; y++ {
		fy := (float64(y)+0.5)*sy - 0.5
		y0 := int(math.Floor(fy))
		dy := fy - float64(y0)
		for x := 0; x < nw; x++ {
			fx := (float64(x)+0.5)*sx - 0.5
			x0 := int(math.Floor(fx))
			dx := fx - float64(x0)
			v00 := g.at(x0, y0)
			v10 := g.at(x0+1, y0)
			v01 := g.at(x0, y0+1)
			v11 := g.at(x0+1, y0+1)
			top := v00 + (v10-v00)*dx
			bot := v01 + (v11-v01)*dx
			out.pix[y*nw+x] = top + (bot-top)*dy
		}
	}
	return out
}

// gradients returns per-pixel gradient magnitude and unsigned orientation (in
// degrees, folded into [0,180)) using a centred [-1,0,1] finite difference with
// edge replication. The returned slices are row-major of length w*h.
func (g *grayImage) gradients() (mag, ori []float64) {
	n := g.w * g.h
	mag = make([]float64, n)
	ori = make([]float64, n)
	for y := 0; y < g.h; y++ {
		for x := 0; x < g.w; x++ {
			gx := g.at(x+1, y) - g.at(x-1, y)
			gy := g.at(x, y+1) - g.at(x, y-1)
			i := y*g.w + x
			mag[i] = math.Hypot(gx, gy)
			a := math.Atan2(gy, gx) * 180 / math.Pi
			if a < 0 {
				a += 180
			}
			if a >= 180 {
				a -= 180
			}
			ori[i] = a
		}
	}
	return mag, ori
}

// IntegralImage is a summed-area table of an image's luma and squared luma. It
// answers the total (or squared total) of any axis-aligned rectangle in
// constant time, which is what makes Haar cascade evaluation fast.
//
// The table has (H+1)×(W+1) entries so that the sum over the half-open
// rectangle [x, x+w) × [y, y+h) is a four-corner difference with no special
// casing at the borders.
type IntegralImage struct {
	// W and H are the source image dimensions in pixels.
	W, H int
	sum  []float64 // (H+1)*(W+1)
	sq   []float64 // (H+1)*(W+1)
}

// NewIntegralImage builds the summed-area tables for img (reduced to luma).
func NewIntegralImage(img *cv.Mat) *IntegralImage {
	g := matToGray(img)
	return newIntegralFromGray(g)
}

func newIntegralFromGray(g *grayImage) *IntegralImage {
	w, h := g.w, g.h
	iw := w + 1
	ii := &IntegralImage{W: w, H: h, sum: make([]float64, iw*(h+1)), sq: make([]float64, iw*(h+1))}
	for y := 0; y < h; y++ {
		var rowSum, rowSq float64
		for x := 0; x < w; x++ {
			v := g.pix[y*w+x]
			rowSum += v
			rowSq += v * v
			ii.sum[(y+1)*iw+(x+1)] = ii.sum[y*iw+(x+1)] + rowSum
			ii.sq[(y+1)*iw+(x+1)] = ii.sq[y*iw+(x+1)] + rowSq
		}
	}
	return ii
}

// clampRect clips a rectangle to the image and reports whether anything
// remains.
func (ii *IntegralImage) clampRect(x, y, w, h int) (int, int, int, int, bool) {
	if x < 0 {
		w += x
		x = 0
	}
	if y < 0 {
		h += y
		y = 0
	}
	if x+w > ii.W {
		w = ii.W - x
	}
	if y+h > ii.H {
		h = ii.H - y
	}
	if w <= 0 || h <= 0 {
		return 0, 0, 0, 0, false
	}
	return x, y, w, h, true
}

// Sum returns the total luma over the rectangle [x, x+w) × [y, y+h). Portions
// of the rectangle outside the image contribute nothing.
func (ii *IntegralImage) Sum(x, y, w, h int) float64 {
	x, y, w, h, ok := ii.clampRect(x, y, w, h)
	if !ok {
		return 0
	}
	iw := ii.W + 1
	a := ii.sum[y*iw+x]
	b := ii.sum[y*iw+(x+w)]
	c := ii.sum[(y+h)*iw+x]
	d := ii.sum[(y+h)*iw+(x+w)]
	return d - b - c + a
}

// SqSum returns the total squared luma over the rectangle [x, x+w) × [y, y+h).
// Portions outside the image contribute nothing.
func (ii *IntegralImage) SqSum(x, y, w, h int) float64 {
	x, y, w, h, ok := ii.clampRect(x, y, w, h)
	if !ok {
		return 0
	}
	iw := ii.W + 1
	a := ii.sq[y*iw+x]
	b := ii.sq[y*iw+(x+w)]
	c := ii.sq[(y+h)*iw+x]
	d := ii.sq[(y+h)*iw+(x+w)]
	return d - b - c + a
}

package saliency

import (
	"math"
	"math/cmplx"

	cv "github.com/malcolmston/opencv"
)

// plane is a single-channel, row-major matrix of float64 samples used for all
// intermediate saliency computation. Working in float avoids the rounding and
// clamping that cv.Mat's 8-bit storage would otherwise impose between steps
// (log spectra, inverse transforms, multi-scale accumulation).
type plane struct {
	rows, cols int
	data       []float64
}

// newPlane allocates a zero-filled plane of the given size.
func newPlane(rows, cols int) *plane {
	return &plane{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

// at returns element (y, x).
func (p *plane) at(y, x int) float64 { return p.data[y*p.cols+x] }

// atReplicate returns element (y, x), clamping out-of-range coordinates to the
// nearest edge (BORDER_REPLICATE), matching the border policy of the root
// package's convolutions.
func (p *plane) atReplicate(y, x int) float64 {
	if y < 0 {
		y = 0
	} else if y >= p.rows {
		y = p.rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= p.cols {
		x = p.cols - 1
	}
	return p.data[y*p.cols+x]
}

// grayPlane reduces a cv.Mat to a single-channel float plane whose samples lie
// in [0,255]. Three-channel input is converted with the BT.601 luma weights via
// [cv.CvtColor]; single-channel input is copied; any other channel count uses
// the first channel. It panics if img is nil or empty.
func grayPlane(img *cv.Mat) *plane {
	if img == nil || img.Empty() {
		panic("saliency: input image is empty")
	}
	var g *cv.Mat
	switch img.Channels {
	case 1:
		g = img
	case 3:
		g = cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		g = cv.NewMat(img.Rows, img.Cols, 1)
		for p := 0; p < img.Total(); p++ {
			g.Data[p] = img.Data[p*img.Channels]
		}
	}
	out := newPlane(g.Rows, g.Cols)
	for i, v := range g.Data {
		out.data[i] = float64(v)
	}
	return out
}

// normalizedMat min-max normalizes the plane into a fresh single-channel
// [cv.Mat] spanning the full 8-bit range. A flat plane (max <= min) yields an
// all-zero Mat.
func (p *plane) normalizedMat() *cv.Mat {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range p.data {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	out := cv.NewMat(p.rows, p.cols, 1)
	if !(mx > mn) {
		return out
	}
	scale := 255.0 / (mx - mn)
	for i, v := range p.data {
		s := math.Round((v - mn) * scale)
		if s < 0 {
			s = 0
		} else if s > 255 {
			s = 255
		}
		out.Data[i] = uint8(s)
	}
	return out
}

// normalizeUnit rescales the plane in place so its samples span [0,1]. A flat
// plane is left untouched (all zeros).
func (p *plane) normalizeUnit() {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range p.data {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if !(mx > mn) {
		for i := range p.data {
			p.data[i] = 0
		}
		return
	}
	inv := 1.0 / (mx - mn)
	for i, v := range p.data {
		p.data[i] = (v - mn) * inv
	}
}

// integral returns the summed-area table of the plane with an extra zero row
// and column, so the returned slice has (rows+1)*(cols+1) elements and element
// (y+1, x+1) holds the sum of all samples in the rectangle [0,y]×[0,x].
func (p *plane) integral() []float64 {
	w := p.cols + 1
	sat := make([]float64, (p.rows+1)*w)
	for y := 0; y < p.rows; y++ {
		var rowSum float64
		for x := 0; x < p.cols; x++ {
			rowSum += p.at(y, x)
			sat[(y+1)*w+(x+1)] = sat[y*w+(x+1)] + rowSum
		}
	}
	return sat
}

// rectSum returns the sum of samples over the inclusive rectangle
// [y0,y1]×[x0,x1] using the summed-area table sat of an image with the given
// number of columns. Callers must pass in-bounds coordinates.
func rectSum(sat []float64, cols, y0, x0, y1, x1 int) float64 {
	w := cols + 1
	return sat[(y1+1)*w+(x1+1)] - sat[y0*w+(x1+1)] - sat[(y1+1)*w+x0] + sat[y0*w+x0]
}

// boxMean returns the mean of the samples in the window of radius r centred on
// (y, x), clamped to the image, using the summed-area table sat.
func boxMean(sat []float64, rows, cols, y, x, r int) float64 {
	y0, x0 := y-r, x-r
	y1, x1 := y+r, x+r
	if y0 < 0 {
		y0 = 0
	}
	if x0 < 0 {
		x0 = 0
	}
	if y1 > rows-1 {
		y1 = rows - 1
	}
	if x1 > cols-1 {
		x1 = cols - 1
	}
	area := float64((y1 - y0 + 1) * (x1 - x0 + 1))
	return rectSum(sat, cols, y0, x0, y1, x1) / area
}

// meanBlur returns a copy of p smoothed by a (2r+1)×(2r+1) box (mean) filter
// with replicated borders.
func meanBlur(p *plane, r int) *plane {
	sat := p.integral()
	out := newPlane(p.rows, p.cols)
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			out.data[y*p.cols+x] = boxMean(sat, p.rows, p.cols, y, x, r)
		}
	}
	return out
}

// gaussianBlurPlane convolves p with a separable Gaussian of size ksize and
// standard deviation sigma (auto-derived from ksize when sigma <= 0, matching
// [cv.GaussianBlur]). Borders are replicated. The kernel weights come from the
// root package's [cv.GaussianKernel1D].
func gaussianBlurPlane(p *plane, ksize int, sigma float64) *plane {
	k := cv.GaussianKernel1D(ksize, sigma)
	a := len(k) / 2
	tmp := newPlane(p.rows, p.cols)
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			var sum float64
			for i, w := range k {
				sum += w * p.atReplicate(y, x+i-a)
			}
			tmp.data[y*p.cols+x] = sum
		}
	}
	out := newPlane(p.rows, p.cols)
	for y := 0; y < p.rows; y++ {
		for x := 0; x < p.cols; x++ {
			var sum float64
			for i, w := range k {
				sum += w * tmp.atReplicate(y+i-a, x)
			}
			out.data[y*p.cols+x] = sum
		}
	}
	return out
}

// resizePlane returns p resampled to rows×cols with bilinear interpolation. It
// is an identity copy when the size is unchanged.
func resizePlane(p *plane, rows, cols int) *plane {
	out := newPlane(rows, cols)
	if p.rows == rows && p.cols == cols {
		copy(out.data, p.data)
		return out
	}
	sy := float64(p.rows) / float64(rows)
	sx := float64(p.cols) / float64(cols)
	for y := 0; y < rows; y++ {
		fy := (float64(y)+0.5)*sy - 0.5
		y0 := int(math.Floor(fy))
		wy := fy - float64(y0)
		for x := 0; x < cols; x++ {
			fx := (float64(x)+0.5)*sx - 0.5
			x0 := int(math.Floor(fx))
			wx := fx - float64(x0)
			v00 := p.atReplicate(y0, x0)
			v01 := p.atReplicate(y0, x0+1)
			v10 := p.atReplicate(y0+1, x0)
			v11 := p.atReplicate(y0+1, x0+1)
			top := v00 + (v01-v00)*wx
			bot := v10 + (v11-v10)*wx
			out.data[y*cols+x] = top + (bot-top)*wy
		}
	}
	return out
}

// nextPow2 returns the smallest power of two that is >= n (and at least 1).
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// fft1D computes the in-place radix-2 Cooley–Tukey transform of a. len(a) must
// be a power of two. When inverse is true it computes the inverse transform,
// including the 1/N normalisation.
func fft1D(a []complex128, inverse bool) {
	n := len(a)
	// Bit-reversal permutation.
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := 2 * math.Pi / float64(length)
		if !inverse {
			ang = -ang
		}
		wl := cmplx.Rect(1, ang)
		half := length / 2
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			for k := 0; k < half; k++ {
				u := a[i+k]
				v := a[i+k+half] * w
				a[i+k] = u + v
				a[i+k+half] = u - v
				w *= wl
			}
		}
	}
	if inverse {
		nc := complex(float64(n), 0)
		for i := range a {
			a[i] /= nc
		}
	}
}

// fft2D computes the in-place row–column 2-D transform of a row-major complex
// field of the given dimensions. Both rows and cols must be powers of two.
func fft2D(data []complex128, rows, cols int, inverse bool) {
	for y := 0; y < rows; y++ {
		fft1D(data[y*cols:(y+1)*cols], inverse)
	}
	col := make([]complex128, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			col[y] = data[y*cols+x]
		}
		fft1D(col, inverse)
		for y := 0; y < rows; y++ {
			data[y*cols+x] = col[y]
		}
	}
}

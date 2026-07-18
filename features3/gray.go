package features3

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// features3gray holds a single-channel float64 image extracted from a cv.Mat.
// It is the internal working representation shared by the operators in this
// package; pixel (x, y) is stored at index y*Cols+x.
type features3gray struct {
	Rows int
	Cols int
	Data []float64
}

// features3ToGray converts any accepted cv.Mat (1- or 3-channel) into a float64
// grayscale buffer using the BT.601 luma weights for colour input. It panics for
// an empty Mat or an unsupported channel count.
func features3ToGray(img *cv.Mat) *features3gray {
	if img == nil || img.Empty() {
		panic("features3: empty image")
	}
	g := &features3gray{Rows: img.Rows, Cols: img.Cols, Data: make([]float64, img.Rows*img.Cols)}
	switch img.Channels {
	case 1:
		for i := 0; i < len(g.Data); i++ {
			g.Data[i] = float64(img.Data[i])
		}
	case 3:
		for i := 0; i < g.Rows*g.Cols; i++ {
			r := float64(img.Data[i*3])
			gg := float64(img.Data[i*3+1])
			b := float64(img.Data[i*3+2])
			g.Data[i] = 0.299*r + 0.587*gg + 0.114*b
		}
	default:
		panic("features3: expected a 1- or 3-channel image")
	}
	return g
}

// at returns the pixel value at (x, y) without bounds checking.
func (g *features3gray) at(x, y int) float64 {
	return g.Data[y*g.Cols+x]
}

// atClamped returns the pixel value at (x, y), clamping out-of-range coordinates
// to the nearest edge (replicated border).
func (g *features3gray) atClamped(x, y int) float64 {
	if x < 0 {
		x = 0
	} else if x >= g.Cols {
		x = g.Cols - 1
	}
	if y < 0 {
		y = 0
	} else if y >= g.Rows {
		y = g.Rows - 1
	}
	return g.Data[y*g.Cols+x]
}

// inBounds reports whether (x, y) lies inside the image.
func (g *features3gray) inBounds(x, y int) bool {
	return x >= 0 && x < g.Cols && y >= 0 && y < g.Rows
}

// features3sobel computes the horizontal and vertical Sobel derivatives of a
// grayscale buffer using the standard 3×3 kernels with a replicated border.
func features3sobel(g *features3gray) (gx, gy []float64) {
	n := g.Rows * g.Cols
	gx = make([]float64, n)
	gy = make([]float64, n)
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			p00 := g.atClamped(x-1, y-1)
			p10 := g.atClamped(x, y-1)
			p20 := g.atClamped(x+1, y-1)
			p01 := g.atClamped(x-1, y)
			p21 := g.atClamped(x+1, y)
			p02 := g.atClamped(x-1, y+1)
			p12 := g.atClamped(x, y+1)
			p22 := g.atClamped(x+1, y+1)
			i := y*g.Cols + x
			gx[i] = (p20 + 2*p21 + p22) - (p00 + 2*p01 + p02)
			gy[i] = (p02 + 2*p12 + p22) - (p00 + 2*p10 + p20)
		}
	}
	return gx, gy
}

// features3gaussianKernel1D returns a normalised 1D Gaussian kernel with the
// given standard deviation. The radius is ceil(3*sigma), giving a kernel of
// length 2*radius+1.
func features3gaussianKernel1D(sigma float64) []float64 {
	if sigma <= 0 {
		return []float64{1}
	}
	radius := int(math.Ceil(3 * sigma))
	if radius < 1 {
		radius = 1
	}
	k := make([]float64, 2*radius+1)
	var sum float64
	for i := -radius; i <= radius; i++ {
		v := math.Exp(-float64(i*i) / (2 * sigma * sigma))
		k[i+radius] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// features3gaussianBlur applies a separable Gaussian blur with the given sigma
// to a grayscale buffer using a replicated border, returning a new buffer.
func features3gaussianBlur(g *features3gray, sigma float64) *features3gray {
	k := features3gaussianKernel1D(sigma)
	radius := len(k) / 2
	tmp := make([]float64, g.Rows*g.Cols)
	out := &features3gray{Rows: g.Rows, Cols: g.Cols, Data: make([]float64, g.Rows*g.Cols)}
	// Horizontal pass.
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			var s float64
			for t := -radius; t <= radius; t++ {
				s += k[t+radius] * g.atClamped(x+t, y)
			}
			tmp[y*g.Cols+x] = s
		}
	}
	// Vertical pass.
	clampY := func(y int) int {
		if y < 0 {
			return 0
		}
		if y >= g.Rows {
			return g.Rows - 1
		}
		return y
	}
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			var s float64
			for t := -radius; t <= radius; t++ {
				s += k[t+radius] * tmp[clampY(y+t)*g.Cols+x]
			}
			out.Data[y*g.Cols+x] = s
		}
	}
	return out
}

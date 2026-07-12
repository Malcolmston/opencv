package features2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// fimage is an internal single-channel floating-point image used by the
// scale-space detectors (SIFT, KAZE, AKAZE). Working in float rather than in
// 8-bit avoids the precision loss of repeated blurring and lets gradients and
// Hessians be computed accurately. It is deliberately unexported: it is an
// implementation detail, not part of the package API.
type fimage struct {
	rows, cols int
	data       []float64 // row-major, length rows*cols
}

// newFImage allocates a zeroed float image.
func newFImage(rows, cols int) *fimage {
	return &fimage{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

// fimageFromMat builds a float image from a single- or three-channel cv.Mat,
// converting to grayscale when necessary. Samples keep the 0..255 range.
func fimageFromMat(img *cv.Mat) *fimage {
	gray := toGray(img)
	f := newFImage(gray.Rows, gray.Cols)
	for i := range f.data {
		f.data[i] = float64(gray.Data[i])
	}
	return f
}

// at returns the sample at (x, y), clamping to the nearest edge for
// out-of-range coordinates (reflect-free replicate border).
func (f *fimage) at(y, x int) float64 {
	if y < 0 {
		y = 0
	} else if y >= f.rows {
		y = f.rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= f.cols {
		x = f.cols - 1
	}
	return f.data[y*f.cols+x]
}

// clone returns an independent copy.
func (f *fimage) clone() *fimage {
	out := newFImage(f.rows, f.cols)
	copy(out.data, f.data)
	return out
}

// gaussianKernel returns a normalised 1-D Gaussian of the given sigma with
// radius ceil(3*sigma).
func gaussianKernel(sigma float64) []float64 {
	if sigma < 1e-6 {
		return []float64{1}
	}
	radius := int(math.Ceil(3 * sigma))
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

// gaussianBlur returns a separably Gaussian-blurred copy using a replicate
// border. sigma is the absolute standard deviation of the blur to apply.
func (f *fimage) gaussianBlur(sigma float64) *fimage {
	k := gaussianKernel(sigma)
	radius := len(k) / 2
	tmp := newFImage(f.rows, f.cols)
	// Horizontal pass.
	for y := 0; y < f.rows; y++ {
		base := y * f.cols
		for x := 0; x < f.cols; x++ {
			var s float64
			for t := -radius; t <= radius; t++ {
				xx := x + t
				if xx < 0 {
					xx = 0
				} else if xx >= f.cols {
					xx = f.cols - 1
				}
				s += k[t+radius] * f.data[base+xx]
			}
			tmp.data[base+x] = s
		}
	}
	// Vertical pass.
	out := newFImage(f.rows, f.cols)
	for y := 0; y < f.rows; y++ {
		for x := 0; x < f.cols; x++ {
			var s float64
			for t := -radius; t <= radius; t++ {
				yy := y + t
				if yy < 0 {
					yy = 0
				} else if yy >= f.rows {
					yy = f.rows - 1
				}
				s += k[t+radius] * tmp.data[yy*f.cols+x]
			}
			out.data[y*f.cols+x] = s
		}
	}
	return out
}

// downsampleHalf returns a half-resolution image by taking every second pixel
// (nearest-neighbour), matching the octave step of a scale-space pyramid.
func (f *fimage) downsampleHalf() *fimage {
	nr, nc := f.rows/2, f.cols/2
	if nr < 1 {
		nr = 1
	}
	if nc < 1 {
		nc = 1
	}
	out := newFImage(nr, nc)
	for y := 0; y < nr; y++ {
		for x := 0; x < nc; x++ {
			out.data[y*nc+x] = f.data[(y*2)*f.cols+(x*2)]
		}
	}
	return out
}

// subtractF returns a-b (same dimensions assumed).
func subtractF(a, b *fimage) *fimage {
	out := newFImage(a.rows, a.cols)
	for i := range out.data {
		out.data[i] = a.data[i] - b.data[i]
	}
	return out
}

// gradXY returns the central-difference gradient (dx, dy) at (x, y).
func (f *fimage) gradXY(y, x int) (dx, dy float64) {
	dx = 0.5 * (f.at(y, x+1) - f.at(y, x-1))
	dy = 0.5 * (f.at(y+1, x) - f.at(y-1, x))
	return
}

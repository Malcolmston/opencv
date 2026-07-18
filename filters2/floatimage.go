package filters2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FloatImage is a dense, single-channel raster of float64 samples in row-major
// order. It is the exported carrier for the signed, unbounded responses
// produced by the linear analysis filters in this package (Laplacian of
// Gaussian, difference of Gaussians, Gabor and steerable filters) and converts
// to and from [cv.Mat] rather than duplicating that type.
//
// The value at row y, column x lives at Data[y*Cols+x]. The zero value is not
// usable; build instances with [NewFloatImage], [MatToFloatImage] or the
// filter functions that return one.
type FloatImage struct {
	// Rows is the image height.
	Rows int
	// Cols is the image width.
	Cols int
	// Data holds Rows*Cols samples in row-major order.
	Data []float64
}

// NewFloatImage allocates a zero-filled FloatImage with the given dimensions.
// It panics if either dimension is not positive.
func NewFloatImage(rows, cols int) *FloatImage {
	if rows <= 0 || cols <= 0 {
		panic("filters2: NewFloatImage requires positive dimensions")
	}
	return &FloatImage{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// At returns the sample at row y, column x. It panics if the coordinates are
// out of range.
func (f *FloatImage) At(y, x int) float64 {
	if y < 0 || y >= f.Rows || x < 0 || x >= f.Cols {
		panic("filters2: FloatImage.At out of range")
	}
	return f.Data[y*f.Cols+x]
}

// Set stores value at row y, column x. It panics if the coordinates are out of
// range.
func (f *FloatImage) Set(y, x int, value float64) {
	if y < 0 || y >= f.Rows || x < 0 || x >= f.Cols {
		panic("filters2: FloatImage.Set out of range")
	}
	f.Data[y*f.Cols+x] = value
}

// atReplicate returns the sample at (y, x) with out-of-range coordinates
// clamped to the nearest edge.
func (f *FloatImage) atReplicate(y, x int) float64 {
	y = clampIdx(y, f.Rows)
	x = clampIdx(x, f.Cols)
	return f.Data[y*f.Cols+x]
}

// Clone returns an independent deep copy of the image.
func (f *FloatImage) Clone() *FloatImage {
	out := NewFloatImage(f.Rows, f.Cols)
	copy(out.Data, f.Data)
	return out
}

// MinMax returns the smallest and largest sample values. It panics on an empty
// image.
func (f *FloatImage) MinMax() (min, max float64) {
	if len(f.Data) == 0 {
		panic("filters2: FloatImage.MinMax on empty image")
	}
	min, max = f.Data[0], f.Data[0]
	for _, v := range f.Data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}

// Mean returns the arithmetic mean of all samples. It panics on an empty image.
func (f *FloatImage) Mean() float64 {
	if len(f.Data) == 0 {
		panic("filters2: FloatImage.Mean on empty image")
	}
	var s float64
	for _, v := range f.Data {
		s += v
	}
	return s / float64(len(f.Data))
}

// Abs returns a new FloatImage holding the absolute value of every sample.
func (f *FloatImage) Abs() *FloatImage {
	out := NewFloatImage(f.Rows, f.Cols)
	for i, v := range f.Data {
		out.Data[i] = math.Abs(v)
	}
	return out
}

// Scale returns a new FloatImage with every sample multiplied by s.
func (f *FloatImage) Scale(s float64) *FloatImage {
	out := NewFloatImage(f.Rows, f.Cols)
	for i, v := range f.Data {
		out.Data[i] = v * s
	}
	return out
}

// Add returns the pointwise sum f+other. It panics on a size mismatch.
func (f *FloatImage) Add(other *FloatImage) *FloatImage {
	if f.Rows != other.Rows || f.Cols != other.Cols {
		panic("filters2: FloatImage.Add size mismatch")
	}
	out := NewFloatImage(f.Rows, f.Cols)
	for i := range f.Data {
		out.Data[i] = f.Data[i] + other.Data[i]
	}
	return out
}

// Sub returns the pointwise difference f-other. It panics on a size mismatch.
func (f *FloatImage) Sub(other *FloatImage) *FloatImage {
	if f.Rows != other.Rows || f.Cols != other.Cols {
		panic("filters2: FloatImage.Sub size mismatch")
	}
	out := NewFloatImage(f.Rows, f.Cols)
	for i := range f.Data {
		out.Data[i] = f.Data[i] - other.Data[i]
	}
	return out
}

// ToMat converts the image to an 8-bit single-channel [cv.Mat] by rounding and
// clamping each sample into the [0,255] range, without rescaling. Use this when
// the samples already occupy the intensity range; use [FloatImage.Normalize]
// to rescale an arbitrary response for display.
func (f *FloatImage) ToMat() *cv.Mat {
	out := cv.NewMat(f.Rows, f.Cols, 1)
	for i, v := range f.Data {
		out.Data[i] = clampU8(v)
	}
	return out
}

// Normalize linearly rescales the sample range [min,max] onto [0,255] and
// returns the result as an 8-bit single-channel [cv.Mat]. A constant image maps
// to all zeros. This is the standard way to visualise a signed filter response.
func (f *FloatImage) Normalize() *cv.Mat {
	out := cv.NewMat(f.Rows, f.Cols, 1)
	min, max := f.MinMax()
	span := max - min
	if span == 0 {
		return out
	}
	scale := 255.0 / span
	for i, v := range f.Data {
		out.Data[i] = clampU8((v - min) * scale)
	}
	return out
}

// Magnitude returns the pointwise Euclidean magnitude sqrt(a^2+b^2) of two
// equally sized responses, for example the real and imaginary parts of a
// complex Gabor filter. It panics on a size mismatch.
func Magnitude(a, b *FloatImage) *FloatImage {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("filters2: Magnitude size mismatch")
	}
	out := NewFloatImage(a.Rows, a.Cols)
	for i := range a.Data {
		out.Data[i] = math.Hypot(a.Data[i], b.Data[i])
	}
	return out
}

// MatToFloatImage converts a single-channel [cv.Mat] to a FloatImage, copying
// each sample as a float64. It panics on multi-channel input.
func MatToFloatImage(src *cv.Mat) *FloatImage {
	requireGray(src, "MatToFloatImage")
	out := NewFloatImage(src.Rows, src.Cols)
	for i, v := range src.Data {
		out.Data[i] = float64(v)
	}
	return out
}

// Convolve correlates src with a row-major kernel anchored at its centre and
// returns a new FloatImage of the same size, using edge replication at the
// borders. The kernel must be non-empty and rectangular. For symmetric kernels
// correlation and true convolution coincide.
func Convolve(src *FloatImage, kernel [][]float64) *FloatImage {
	kr, kc := kernelDims(kernel)
	ay, ax := kr/2, kc/2
	out := NewFloatImage(src.Rows, src.Cols)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			var sum float64
			for ky := 0; ky < kr; ky++ {
				sy := y + ky - ay
				row := kernel[ky]
				for kx := 0; kx < kc; kx++ {
					sum += row[kx] * src.atReplicate(sy, x+kx-ax)
				}
			}
			out.Data[y*src.Cols+x] = sum
		}
	}
	return out
}

// ConvolveMat correlates a single-channel [cv.Mat] with a row-major kernel
// anchored at its centre and returns the unclamped, signed result as a
// FloatImage, using edge replication at the borders. It panics on multi-channel
// input.
func ConvolveMat(src *cv.Mat, kernel [][]float64) *FloatImage {
	requireGray(src, "ConvolveMat")
	return Convolve(MatToFloatImage(src), kernel)
}

// kernelDims validates a rectangular kernel and returns its dimensions.
func kernelDims(kernel [][]float64) (rows, cols int) {
	rows = len(kernel)
	if rows == 0 || len(kernel[0]) == 0 {
		panic("filters2: empty kernel")
	}
	cols = len(kernel[0])
	for _, r := range kernel {
		if len(r) != cols {
			panic("filters2: non-rectangular kernel")
		}
	}
	return rows, cols
}

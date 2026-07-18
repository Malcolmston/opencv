package inpaint

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// FloatImage is a dense row-major matrix of float64 samples with the same
// channel-interleaved layout as [cv.Mat]. It holds intermediate gradient,
// divergence and Laplacian fields for gradient-domain editing, where the [0,255]
// integer range of [cv.Mat] would lose precision or clip. The zero value is not
// usable — construct with [NewFloatImage] or the field operators.
type FloatImage struct {
	// Rows is the image height.
	Rows int
	// Cols is the image width.
	Cols int
	// Channels is the number of samples per pixel.
	Channels int
	// Data holds Rows*Cols*Channels samples, channel-interleaved: the value at
	// row y, column x, channel c is Data[(y*Cols+x)*Channels+c].
	Data []float64
}

// NewFloatImage allocates a zero-filled FloatImage. It panics if any dimension
// is not positive.
func NewFloatImage(rows, cols, channels int) *FloatImage {
	if rows <= 0 || cols <= 0 || channels <= 0 {
		panic(fmt.Sprintf("inpaint: NewFloatImage requires positive dimensions, got rows=%d cols=%d channels=%d", rows, cols, channels))
	}
	return &FloatImage{Rows: rows, Cols: cols, Channels: channels, Data: make([]float64, rows*cols*channels)}
}

// FloatImageFromMat converts a [cv.Mat] into a FloatImage with the same shape,
// promoting each 8-bit sample to float64.
func FloatImageFromMat(m *cv.Mat) *FloatImage {
	inpaintRequireImage(m, "FloatImageFromMat")
	out := NewFloatImage(m.Rows, m.Cols, m.Channels)
	for i, v := range m.Data {
		out.Data[i] = float64(v)
	}
	return out
}

// index returns the flat offset of the first sample of pixel (x, y).
func (f *FloatImage) index(y, x int) int {
	return (y*f.Cols + x) * f.Channels
}

// At returns the sample at row y, column x, channel c. It panics if out of
// range.
func (f *FloatImage) At(y, x, c int) float64 {
	if y < 0 || y >= f.Rows || x < 0 || x >= f.Cols || c < 0 || c >= f.Channels {
		panic(fmt.Sprintf("inpaint: FloatImage.At out of range y=%d x=%d c=%d for %dx%dx%d", y, x, c, f.Rows, f.Cols, f.Channels))
	}
	return f.Data[f.index(y, x)+c]
}

// Set stores value at row y, column x, channel c. It panics if out of range.
func (f *FloatImage) Set(y, x, c int, value float64) {
	if y < 0 || y >= f.Rows || x < 0 || x >= f.Cols || c < 0 || c >= f.Channels {
		panic(fmt.Sprintf("inpaint: FloatImage.Set out of range y=%d x=%d c=%d for %dx%dx%d", y, x, c, f.Rows, f.Cols, f.Channels))
	}
	f.Data[f.index(y, x)+c] = value
}

// atRep returns the sample at (y, x, c) with BORDER_REPLICATE clamping.
func (f *FloatImage) atRep(y, x, c int) float64 {
	if y < 0 {
		y = 0
	} else if y >= f.Rows {
		y = f.Rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= f.Cols {
		x = f.Cols - 1
	}
	return f.Data[f.index(y, x)+c]
}

// Clone returns an independent deep copy.
func (f *FloatImage) Clone() *FloatImage {
	out := NewFloatImage(f.Rows, f.Cols, f.Channels)
	copy(out.Data, f.Data)
	return out
}

// ToMat converts the field back to a [cv.Mat], rounding and clamping every
// sample into [0,255].
func (f *FloatImage) ToMat() *cv.Mat {
	out := cv.NewMat(f.Rows, f.Cols, f.Channels)
	for i, v := range f.Data {
		out.Data[i] = inpaintClampU8(v)
	}
	return out
}

// GradientX returns the forward horizontal derivative field I(x+1)-I(x) of img
// (BORDER_REPLICATE at the right edge). The result has the same shape as img.
func GradientX(img *cv.Mat) *FloatImage {
	inpaintRequireImage(img, "GradientX")
	out := NewFloatImage(img.Rows, img.Cols, img.Channels)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			for c := 0; c < img.Channels; c++ {
				out.Set(y, x, c, float64(inpaintAtRep(img, y, x+1, c))-float64(img.At(y, x, c)))
			}
		}
	}
	return out
}

// GradientY returns the forward vertical derivative field I(y+1)-I(y) of img
// (BORDER_REPLICATE at the bottom edge). The result has the same shape as img.
func GradientY(img *cv.Mat) *FloatImage {
	inpaintRequireImage(img, "GradientY")
	out := NewFloatImage(img.Rows, img.Cols, img.Channels)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			for c := 0; c < img.Channels; c++ {
				out.Set(y, x, c, float64(inpaintAtRep(img, y+1, x, c))-float64(img.At(y, x, c)))
			}
		}
	}
	return out
}

// Divergence returns the divergence of the vector field (gx, gy) using backward
// differences: div(x,y) = (gx(x)-gx(x-1)) + (gy(y)-gy(y-1)), the discrete
// adjoint of the forward gradient. gx and gy must share shape; the Laplacian of
// an image equals Divergence(GradientX(img), GradientY(img)).
func Divergence(gx, gy *FloatImage) *FloatImage {
	if gx.Rows != gy.Rows || gx.Cols != gy.Cols || gx.Channels != gy.Channels {
		panic("inpaint: Divergence requires gx and gy of equal shape")
	}
	out := NewFloatImage(gx.Rows, gx.Cols, gx.Channels)
	for y := 0; y < gx.Rows; y++ {
		for x := 0; x < gx.Cols; x++ {
			for c := 0; c < gx.Channels; c++ {
				dxx := gx.At(y, x, c) - gx.atRep(y, x-1, c)
				dyy := gy.At(y, x, c) - gy.atRep(y-1, x, c)
				out.Set(y, x, c, dxx+dyy)
			}
		}
	}
	return out
}

// Laplacian returns the 5-point discrete Laplacian of img:
// L = I(x-1)+I(x+1)+I(y-1)+I(y+1)-4*I (BORDER_REPLICATE). For a linear ramp the
// Laplacian is zero everywhere.
func Laplacian(img *cv.Mat) *FloatImage {
	inpaintRequireImage(img, "Laplacian")
	out := NewFloatImage(img.Rows, img.Cols, img.Channels)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			for c := 0; c < img.Channels; c++ {
				v := float64(inpaintAtRep(img, y-1, x, c)) + float64(inpaintAtRep(img, y+1, x, c)) +
					float64(inpaintAtRep(img, y, x-1, c)) + float64(inpaintAtRep(img, y, x+1, c)) -
					4*float64(img.At(y, x, c))
				out.Set(y, x, c, v)
			}
		}
	}
	return out
}

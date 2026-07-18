package edges2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GradientField holds the signed horizontal (Gx) and vertical (Gy) first
// derivatives of an image, as produced by [Sobel], [Scharr], [Prewitt] and
// [Roberts]. Gx responds to vertical edges (intensity change along x) and Gy to
// horizontal edges (intensity change along y). The magnitude and orientation of
// the gradient are derived from the two components.
type GradientField struct {
	// Rows is the image height.
	Rows int
	// Cols is the image width.
	Cols int
	// Gx is the signed horizontal derivative (d/dx).
	Gx *FloatGrid
	// Gy is the signed vertical derivative (d/dy).
	Gy *FloatGrid
}

// At returns the gradient components (gx, gy) at row y, column x.
func (f *GradientField) At(y, x int) (gx, gy float64) {
	i := y*f.Cols + x
	return f.Gx.Data[i], f.Gy.Data[i]
}

// Magnitude returns the Euclidean gradient magnitude sqrt(Gx^2+Gy^2) as a
// [FloatGrid].
func (f *GradientField) Magnitude() *FloatGrid {
	out := NewFloatGrid(f.Rows, f.Cols)
	for i := range out.Data {
		out.Data[i] = math.Hypot(f.Gx.Data[i], f.Gy.Data[i])
	}
	return out
}

// Orientation returns the gradient direction atan2(Gy,Gx) in radians in the
// range (-pi, pi] as a [FloatGrid].
func (f *GradientField) Orientation() *FloatGrid {
	out := NewFloatGrid(f.Rows, f.Cols)
	for i := range out.Data {
		out.Data[i] = math.Atan2(f.Gy.Data[i], f.Gx.Data[i])
	}
	return out
}

// OrientationDegrees returns the gradient direction in degrees in the range
// [0,360) as a [FloatGrid].
func (f *GradientField) OrientationDegrees() *FloatGrid {
	out := NewFloatGrid(f.Rows, f.Cols)
	for i := range out.Data {
		a := math.Atan2(f.Gy.Data[i], f.Gx.Data[i]) * 180 / math.Pi
		if a < 0 {
			a += 360
		}
		out.Data[i] = a
	}
	return out
}

// MagnitudeMat returns the gradient magnitude as an 8-bit [cv.Mat], rescaling
// the response range onto [0,255] for display.
func (f *GradientField) MagnitudeMat() *cv.Mat {
	return f.Magnitude().ToMatNormalized()
}

// Sobel computes the image gradient with the 3×3 Sobel operator and returns it
// as a [GradientField]. It panics on multi-channel input.
func Sobel(src *cv.Mat) *GradientField {
	edges2RequireGray(src, "Sobel")
	kx := [][]float64{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}
	ky := [][]float64{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}
	return &GradientField{Rows: src.Rows, Cols: src.Cols,
		Gx: edges2Convolve(src, kx), Gy: edges2Convolve(src, ky)}
}

// Scharr computes the image gradient with the 3×3 Scharr operator, a variant of
// Sobel with better rotational symmetry, and returns it as a [GradientField].
// It panics on multi-channel input.
func Scharr(src *cv.Mat) *GradientField {
	edges2RequireGray(src, "Scharr")
	kx := [][]float64{{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}}
	ky := [][]float64{{-3, -10, -3}, {0, 0, 0}, {3, 10, 3}}
	return &GradientField{Rows: src.Rows, Cols: src.Cols,
		Gx: edges2Convolve(src, kx), Gy: edges2Convolve(src, ky)}
}

// Prewitt computes the image gradient with the 3×3 Prewitt operator (uniform
// smoothing weights) and returns it as a [GradientField]. It panics on
// multi-channel input.
func Prewitt(src *cv.Mat) *GradientField {
	edges2RequireGray(src, "Prewitt")
	kx := [][]float64{{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}}
	ky := [][]float64{{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}
	return &GradientField{Rows: src.Rows, Cols: src.Cols,
		Gx: edges2Convolve(src, kx), Gy: edges2Convolve(src, ky)}
}

// Roberts computes the image gradient with the 2×2 Roberts cross operator and
// returns it as a [GradientField]. The diagonal responses are mapped onto the
// Gx (main diagonal) and Gy (anti-diagonal) fields. It panics on multi-channel
// input.
func Roberts(src *cv.Mat) *GradientField {
	edges2RequireGray(src, "Roberts")
	// The Roberts cross uses 2×2 kernels; centring them in a 3×3 grid with the
	// anchor at the middle keeps the response aligned with the other operators.
	kx := [][]float64{{0, 0, 0}, {0, 1, 0}, {0, 0, -1}}
	ky := [][]float64{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}
	return &GradientField{Rows: src.Rows, Cols: src.Cols,
		Gx: edges2Convolve(src, kx), Gy: edges2Convolve(src, ky)}
}

// Laplacian computes the discrete Laplacian (second derivative) of the image
// with the 4-neighbour 3×3 kernel and returns the signed response as a
// [FloatGrid]. It panics on multi-channel input.
func Laplacian(src *cv.Mat) *FloatGrid {
	edges2RequireGray(src, "Laplacian")
	k := [][]float64{{0, 1, 0}, {1, -4, 1}, {0, 1, 0}}
	return edges2Convolve(src, k)
}

// GradientMagnitude is a convenience wrapper that computes the Sobel gradient
// of src and returns its magnitude as a display-ready 8-bit [cv.Mat]. It panics
// on multi-channel input.
func GradientMagnitude(src *cv.Mat) *cv.Mat {
	return Sobel(src).MagnitudeMat()
}

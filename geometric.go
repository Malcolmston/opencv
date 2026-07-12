package cv

import (
	"fmt"
	"math"
)

// InterpolationFlag selects the resampling method used by [Resize] and
// [WarpAffine].
type InterpolationFlag int

const (
	// InterNearest picks the nearest source sample (fast, blocky).
	InterNearest InterpolationFlag = iota
	// InterLinear uses bilinear interpolation of the four nearest samples.
	InterLinear
)

// Resize scales src to the given width and height using the chosen
// interpolation. Both dimensions must be positive.
func Resize(src *Mat, width, height int, interp InterpolationFlag) *Mat {
	if width <= 0 || height <= 0 {
		panic("cv: Resize requires positive width and height")
	}
	dst := NewMat(height, width, src.Channels)
	scaleX := float64(src.Cols) / float64(width)
	scaleY := float64(src.Rows) / float64(height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			switch interp {
			case InterNearest:
				sx := int(math.Floor((float64(x) + 0.5) * scaleX))
				sy := int(math.Floor((float64(y) + 0.5) * scaleY))
				if sx >= src.Cols {
					sx = src.Cols - 1
				}
				if sy >= src.Rows {
					sy = src.Rows - 1
				}
				for c := 0; c < src.Channels; c++ {
					dst.Data[dst.index(y, x)+c] = src.Data[src.index(sy, sx)+c]
				}
			case InterLinear:
				// Map to source coordinates using pixel-centre alignment.
				fx := (float64(x)+0.5)*scaleX - 0.5
				fy := (float64(y)+0.5)*scaleY - 0.5
				for c := 0; c < src.Channels; c++ {
					dst.Data[dst.index(y, x)+c] = clampToUint8(bilinearSample(src, fx, fy, c) + 0.5)
				}
			default:
				panic(fmt.Sprintf("cv: Resize unknown interpolation %d", interp))
			}
		}
	}
	return dst
}

// bilinearSample interpolates channel c of src at fractional coordinates
// (fx, fy), replicating the border for out-of-range neighbours.
func bilinearSample(src *Mat, fx, fy float64, c int) float64 {
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	dx := fx - float64(x0)
	dy := fy - float64(y0)
	v00 := float64(src.atReplicate(y0, x0, c))
	v01 := float64(src.atReplicate(y0, x0+1, c))
	v10 := float64(src.atReplicate(y0+1, x0, c))
	v11 := float64(src.atReplicate(y0+1, x0+1, c))
	top := v00*(1-dx) + v01*dx
	bot := v10*(1-dx) + v11*dx
	return top*(1-dy) + bot*dy
}

// FlipCode selects the axis for [Flip].
type FlipCode int

const (
	// FlipVertical mirrors around the horizontal axis (top-bottom).
	FlipVertical FlipCode = iota
	// FlipHorizontal mirrors around the vertical axis (left-right).
	FlipHorizontal
	// FlipBoth mirrors around both axes (equivalent to a 180° rotation).
	FlipBoth
)

// Flip mirrors src along the axis chosen by code and returns a new Mat.
func Flip(src *Mat, code FlipCode) *Mat {
	dst := NewMat(src.Rows, src.Cols, src.Channels)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			sy, sx := y, x
			switch code {
			case FlipVertical:
				sy = src.Rows - 1 - y
			case FlipHorizontal:
				sx = src.Cols - 1 - x
			case FlipBoth:
				sy = src.Rows - 1 - y
				sx = src.Cols - 1 - x
			default:
				panic(fmt.Sprintf("cv: Flip unknown code %d", code))
			}
			si := src.index(sy, sx)
			di := dst.index(y, x)
			copy(dst.Data[di:di+src.Channels], src.Data[si:si+src.Channels])
		}
	}
	return dst
}

// RotateCode selects one of the three lossless right-angle rotations for
// [Rotate].
type RotateCode int

const (
	// Rotate90CW rotates 90° clockwise.
	Rotate90CW RotateCode = iota
	// Rotate180 rotates 180°.
	Rotate180
	// Rotate90CCW rotates 90° counter-clockwise (270° clockwise).
	Rotate90CCW
)

// Rotate performs a lossless 90/180/270-degree rotation and returns a new Mat.
// For arbitrary angles use [GetRotationMatrix2D] with [WarpAffine].
func Rotate(src *Mat, code RotateCode) *Mat {
	switch code {
	case Rotate180:
		return Flip(src, FlipBoth)
	case Rotate90CW:
		dst := NewMat(src.Cols, src.Rows, src.Channels)
		for y := 0; y < src.Rows; y++ {
			for x := 0; x < src.Cols; x++ {
				ny, nx := x, src.Rows-1-y
				si := src.index(y, x)
				di := dst.index(ny, nx)
				copy(dst.Data[di:di+src.Channels], src.Data[si:si+src.Channels])
			}
		}
		return dst
	case Rotate90CCW:
		dst := NewMat(src.Cols, src.Rows, src.Channels)
		for y := 0; y < src.Rows; y++ {
			for x := 0; x < src.Cols; x++ {
				ny, nx := src.Cols-1-x, y
				si := src.index(y, x)
				di := dst.index(ny, nx)
				copy(dst.Data[di:di+src.Channels], src.Data[si:si+src.Channels])
			}
		}
		return dst
	default:
		panic(fmt.Sprintf("cv: Rotate unknown code %d", code))
	}
}

// Transpose swaps rows and columns, returning a new Mat of shape Cols×Rows.
func Transpose(src *Mat) *Mat {
	dst := NewMat(src.Cols, src.Rows, src.Channels)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			si := src.index(y, x)
			di := dst.index(x, y)
			copy(dst.Data[di:di+src.Channels], src.Data[si:si+src.Channels])
		}
	}
	return dst
}

// AffineMatrix is a 2×3 affine transform stored row-major:
//
//	[ m0 m1 m2 ]
//	[ m3 m4 m5 ]
//
// It maps a source point (x, y) to (m0*x + m1*y + m2, m3*x + m4*y + m5).
type AffineMatrix [6]float64

// GetRotationMatrix2D builds the 2×3 affine matrix that rotates around the
// point (cx, cy) by angle degrees (counter-clockwise, matching OpenCV) and
// uniformly scales by scale.
func GetRotationMatrix2D(cx, cy, angle, scale float64) AffineMatrix {
	rad := angle * math.Pi / 180
	a := scale * math.Cos(rad)
	b := scale * math.Sin(rad)
	return AffineMatrix{
		a, b, (1-a)*cx - b*cy,
		-b, a, b*cx + (1-a)*cy,
	}
}

// WarpAffine applies the inverse-mapped affine transform m to src, producing an
// output of the given width and height. Each destination pixel is sampled from
// src with the chosen interpolation; coordinates outside the source are filled
// with zero. width and height must be positive.
func WarpAffine(src *Mat, m AffineMatrix, width, height int, interp InterpolationFlag) *Mat {
	if width <= 0 || height <= 0 {
		panic("cv: WarpAffine requires positive width and height")
	}
	inv, ok := invertAffine(m)
	if !ok {
		panic("cv: WarpAffine transform is not invertible")
	}
	dst := NewMat(height, width, src.Channels)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sx := inv[0]*float64(x) + inv[1]*float64(y) + inv[2]
			sy := inv[3]*float64(x) + inv[4]*float64(y) + inv[5]
			switch interp {
			case InterNearest:
				rx := int(math.Round(sx))
				ry := int(math.Round(sy))
				if rx < 0 || rx >= src.Cols || ry < 0 || ry >= src.Rows {
					continue
				}
				si := src.index(ry, rx)
				di := dst.index(y, x)
				copy(dst.Data[di:di+src.Channels], src.Data[si:si+src.Channels])
			case InterLinear:
				if sx < -0.5 || sx > float64(src.Cols)-0.5 || sy < -0.5 || sy > float64(src.Rows)-0.5 {
					continue
				}
				for c := 0; c < src.Channels; c++ {
					dst.Data[dst.index(y, x)+c] = clampToUint8(bilinearSample(src, sx, sy, c) + 0.5)
				}
			default:
				panic(fmt.Sprintf("cv: WarpAffine unknown interpolation %d", interp))
			}
		}
	}
	return dst
}

// invertAffine returns the inverse of a 2×3 affine transform, reporting whether
// the 2×2 linear part is invertible.
func invertAffine(m AffineMatrix) (AffineMatrix, bool) {
	det := m[0]*m[4] - m[1]*m[3]
	if det == 0 {
		return AffineMatrix{}, false
	}
	id := 1 / det
	i0 := m[4] * id
	i1 := -m[1] * id
	i3 := -m[3] * id
	i4 := m[0] * id
	i2 := -(i0*m[2] + i1*m[5])
	i5 := -(i3*m[2] + i4*m[5])
	return AffineMatrix{i0, i1, i2, i3, i4, i5}, true
}

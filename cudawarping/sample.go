package cudawarping

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// clampU8 rounds v to the nearest integer and clamps it to the 8-bit range.
func clampU8(v float64) uint8 {
	r := math.Round(v)
	switch {
	case r <= 0:
		return 0
	case r >= 255:
		return 255
	default:
		return uint8(r)
	}
}

// matSample returns channel c of pixel (y, x) of src without bounds checking; it
// is the hot-path accessor used after coordinates have been resolved by
// borderIndex.
func matSample(src *cv.Mat, y, x, c int) float64 {
	return float64(src.Data[(y*src.Cols+x)*src.Channels+c])
}

// borderIndex maps a possibly out-of-range coordinate p along an axis of the
// given length to a valid index according to mode, mirroring
// cv::borderInterpolate. It returns -1 when mode is [BorderConstant] and p is
// outside [0, length), signalling that the caller should substitute the
// constant border value.
func borderIndex(p, length int, mode BorderMode) int {
	if p >= 0 && p < length {
		return p
	}
	switch mode {
	case BorderReplicate:
		if p < 0 {
			return 0
		}
		return length - 1
	case BorderReflect101:
		if length == 1 {
			return 0
		}
		for p < 0 || p >= length {
			if p < 0 {
				p = -p
			} else {
				p = 2*(length-1) - p
			}
		}
		return p
	case BorderReflect:
		if length == 1 {
			return 0
		}
		for p < 0 || p >= length {
			if p < 0 {
				p = -p - 1
			} else {
				p = 2*length - 1 - p
			}
		}
		return p
	case BorderWrap:
		p %= length
		if p < 0 {
			p += length
		}
		return p
	default: // BorderConstant
		return -1
	}
}

// cubicWeights returns the four bicubic tap weights for a fractional offset t in
// [0,1), using OpenCV's a = -0.75 kernel. The taps correspond to source offsets
// -1, 0, +1, +2 relative to floor(coordinate).
func cubicWeights(t float64) [4]float64 {
	const a = -0.75
	var w [4]float64
	w[0] = ((a*(t+1)-5*a)*(t+1)+8*a)*(t+1) - 4*a
	w[1] = ((a+2)*t-(a+3))*t*t + 1
	w[2] = ((a+2)*(1-t)-(a+3))*(1-t)*(1-t) + 1
	w[3] = 1 - w[0] - w[1] - w[2]
	return w
}

// sampleBorder samples channel c of src at the fractional coordinate (fx, fy)
// with the given interpolation and border handling, returning a value already
// in the 8-bit range. Coordinates resolved to the constant border contribute
// borderValue.
func sampleBorder(src *cv.Mat, fx, fy float64, c int, interp Interpolation, mode BorderMode, borderValue float64) float64 {
	switch interp {
	case InterNearest:
		rx := int(math.Round(fx))
		ry := int(math.Round(fy))
		ix := borderIndex(rx, src.Cols, mode)
		iy := borderIndex(ry, src.Rows, mode)
		if ix < 0 || iy < 0 {
			return borderValue
		}
		return matSample(src, iy, ix, c)
	case InterCubic:
		x0 := int(math.Floor(fx))
		y0 := int(math.Floor(fy))
		wx := cubicWeights(fx - float64(x0))
		wy := cubicWeights(fy - float64(y0))
		var acc float64
		for j := 0; j < 4; j++ {
			iy := borderIndex(y0-1+j, src.Rows, mode)
			var row float64
			for i := 0; i < 4; i++ {
				ix := borderIndex(x0-1+i, src.Cols, mode)
				var v float64
				if ix < 0 || iy < 0 {
					v = borderValue
				} else {
					v = matSample(src, iy, ix, c)
				}
				row += wx[i] * v
			}
			acc += wy[j] * row
		}
		return acc
	default: // InterLinear and InterArea (area falls back to bilinear when sampling)
		x0 := int(math.Floor(fx))
		y0 := int(math.Floor(fy))
		dx := fx - float64(x0)
		dy := fy - float64(y0)
		v00 := borderSample(src, y0, x0, c, mode, borderValue)
		v01 := borderSample(src, y0, x0+1, c, mode, borderValue)
		v10 := borderSample(src, y0+1, x0, c, mode, borderValue)
		v11 := borderSample(src, y0+1, x0+1, c, mode, borderValue)
		top := v00*(1-dx) + v01*dx
		bot := v10*(1-dx) + v11*dx
		return top*(1-dy) + bot*dy
	}
}

// borderSample resolves the integer sample at (y, x, c) through the border mode,
// returning borderValue when the coordinate maps to the constant border.
func borderSample(src *cv.Mat, y, x, c int, mode BorderMode, borderValue float64) float64 {
	ix := borderIndex(x, src.Cols, mode)
	iy := borderIndex(y, src.Rows, mode)
	if ix < 0 || iy < 0 {
		return borderValue
	}
	return matSample(src, iy, ix, c)
}

// remapWithBorder resamples src at the per-pixel source coordinates in xmap and
// ymap (which must share dimensions) using the given interpolation and border
// handling. It is the shared engine behind the non-default paths of the warp
// operations. The result has the map's dimensions and src's channel count.
func remapWithBorder(src *cv.Mat, xmap, ymap *cv.FloatMat, interp Interpolation, mode BorderMode, borderValue float64) *cv.Mat {
	if xmap.Rows != ymap.Rows || xmap.Cols != ymap.Cols {
		panic("cudawarping: remap map dimensions differ")
	}
	dst := cv.NewMat(xmap.Rows, xmap.Cols, src.Channels)
	for y := 0; y < xmap.Rows; y++ {
		for x := 0; x < xmap.Cols; x++ {
			sx := xmap.Data[y*xmap.Cols+x]
			sy := ymap.Data[y*ymap.Cols+x]
			di := (y*dst.Cols + x) * dst.Channels
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = clampU8(sampleBorder(src, sx, sy, c, interp, mode, borderValue))
			}
		}
	}
	return dst
}

// isDefaultBorder reports whether the (interp, mode, borderValue) triple matches
// the behaviour of the root cv package's warp helpers — bilinear/nearest
// interpolation with a zero constant border — so those helpers can be delegated
// to directly for an exact, well-tested result.
func isDefaultBorder(interp Interpolation, mode BorderMode, borderValue float64) bool {
	return mode == BorderConstant && borderValue == 0 &&
		(interp == InterNearest || interp == InterLinear)
}

// cvInterp maps a cudawarping [Interpolation] to the root package's flag for the
// two methods the root package implements.
func cvInterp(interp Interpolation) cv.InterpolationFlag {
	if interp == InterNearest {
		return cv.InterNearest
	}
	return cv.InterLinear
}

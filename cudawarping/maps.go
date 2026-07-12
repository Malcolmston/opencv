package cudawarping

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// invertAffine returns the inverse of a 2×3 affine transform, reporting whether
// the 2×2 linear part is invertible. It reimplements the root package's private
// inverse because that helper is unexported.
func invertAffine(m cv.AffineMatrix) (cv.AffineMatrix, bool) {
	det := m[0]*m[4] - m[1]*m[3]
	if det == 0 {
		return cv.AffineMatrix{}, false
	}
	id := 1 / det
	i0 := m[4] * id
	i1 := -m[1] * id
	i3 := -m[3] * id
	i4 := m[0] * id
	i2 := -(i0*m[2] + i1*m[5])
	i5 := -(i3*m[2] + i4*m[5])
	return cv.AffineMatrix{i0, i1, i2, i3, i4, i5}, true
}

// invertPerspective returns the inverse of a 3×3 projective transform, reporting
// whether it is invertible.
func invertPerspective(m cv.PerspectiveMatrix) (cv.PerspectiveMatrix, bool) {
	a, b, c := m[0], m[1], m[2]
	d, e, f := m[3], m[4], m[5]
	g, h, i := m[6], m[7], m[8]
	det := a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
	if math.Abs(det) < 1e-15 {
		return cv.PerspectiveMatrix{}, false
	}
	id := 1 / det
	var inv cv.PerspectiveMatrix
	inv[0] = (e*i - f*h) * id
	inv[1] = (c*h - b*i) * id
	inv[2] = (b*f - c*e) * id
	inv[3] = (f*g - d*i) * id
	inv[4] = (a*i - c*g) * id
	inv[5] = (c*d - a*f) * id
	inv[6] = (d*h - e*g) * id
	inv[7] = (b*g - a*h) * id
	inv[8] = (a*e - b*d) * id
	return inv, true
}

// checkSize panics unless dsize has positive width and height.
func checkSize(dsize image.Point, op string) {
	if dsize.X <= 0 || dsize.Y <= 0 {
		panic("cudawarping: " + op + " requires positive dsize (width and height)")
	}
}

// BuildWarpAffineMaps builds the pair of coordinate maps that [GpuMat.Remap]
// (or OpenCV's cuda::remap) needs to realise the affine transform m over an
// output of size dsize (width dsize.X, height dsize.Y). Element (x, y) of the
// returned maps holds the source coordinate that destination pixel (x, y) is
// sampled from.
//
// When inverse is true, m is taken to already map destination pixels to source
// pixels (as if [WarpInverseMap] were set) and is used directly; when false, m
// maps source to destination and is inverted first. This mirrors
// cv::cuda::buildWarpAffineMaps(M, inverse, dsize, xmap, ymap). The stream
// argument is accepted for API compatibility and ignored. It panics if dsize is
// not positive or m is not invertible when inversion is required.
func BuildWarpAffineMaps(m cv.AffineMatrix, inverse bool, dsize image.Point, stream *Stream) (xmap, ymap *cv.FloatMat) {
	checkSize(dsize, "BuildWarpAffineMaps")
	inv := m
	if !inverse {
		var ok bool
		inv, ok = invertAffine(m)
		if !ok {
			panic("cudawarping: BuildWarpAffineMaps transform is not invertible")
		}
	}
	xmap = cv.NewFloatMat(dsize.Y, dsize.X)
	ymap = cv.NewFloatMat(dsize.Y, dsize.X)
	for y := 0; y < dsize.Y; y++ {
		fy := float64(y)
		for x := 0; x < dsize.X; x++ {
			fx := float64(x)
			idx := y*dsize.X + x
			xmap.Data[idx] = inv[0]*fx + inv[1]*fy + inv[2]
			ymap.Data[idx] = inv[3]*fx + inv[4]*fy + inv[5]
		}
	}
	return xmap, ymap
}

// BuildWarpPerspectiveMaps builds the coordinate maps that realise the
// projective transform m over an output of size dsize, the perspective analogue
// of [BuildWarpAffineMaps]. When inverse is true m is used directly as the
// destination→source map; otherwise it is inverted first. It mirrors
// cv::cuda::buildWarpPerspectiveMaps. The stream argument is ignored. It panics
// if dsize is not positive or m is not invertible when inversion is required.
func BuildWarpPerspectiveMaps(m cv.PerspectiveMatrix, inverse bool, dsize image.Point, stream *Stream) (xmap, ymap *cv.FloatMat) {
	checkSize(dsize, "BuildWarpPerspectiveMaps")
	inv := m
	if !inverse {
		var ok bool
		inv, ok = invertPerspective(m)
		if !ok {
			panic("cudawarping: BuildWarpPerspectiveMaps transform is not invertible")
		}
	}
	xmap = cv.NewFloatMat(dsize.Y, dsize.X)
	ymap = cv.NewFloatMat(dsize.Y, dsize.X)
	for y := 0; y < dsize.Y; y++ {
		fy := float64(y)
		for x := 0; x < dsize.X; x++ {
			fx := float64(x)
			w := inv[6]*fx + inv[7]*fy + inv[8]
			idx := y*dsize.X + x
			if w == 0 {
				// Point at infinity: send it far outside the source so any
				// border mode treats it as an outlier.
				xmap.Data[idx] = math.Inf(1)
				ymap.Data[idx] = math.Inf(1)
				continue
			}
			xmap.Data[idx] = (inv[0]*fx + inv[1]*fy + inv[2]) / w
			ymap.Data[idx] = (inv[3]*fx + inv[4]*fy + inv[5]) / w
		}
	}
	return xmap, ymap
}

// rotationMatrix returns the 2×3 forward (source→destination) affine matrix used
// by OpenCV's cuda::rotate: a rotation by angle degrees about the origin plus a
// translation of (xShift, yShift).
func rotationMatrix(angle, xShift, yShift float64) cv.AffineMatrix {
	rad := angle * math.Pi / 180
	c := math.Cos(rad)
	s := math.Sin(rad)
	return cv.AffineMatrix{c, s, xShift, -s, c, yShift}
}

// BuildRotationMaps builds the coordinate maps that realise the cuda::rotate
// transform (rotation by angle degrees about the origin followed by a shift of
// (xShift, yShift)) over an output of size dsize. It is a convenience wrapper
// around [BuildWarpAffineMaps] for the rotation matrix and is handy for
// pre-computing the maps once and reusing them with [GpuMat.Remap]. The stream
// argument is ignored.
func BuildRotationMaps(angle, xShift, yShift float64, dsize image.Point, stream *Stream) (xmap, ymap *cv.FloatMat) {
	return BuildWarpAffineMaps(rotationMatrix(angle, xShift, yShift), false, dsize, stream)
}

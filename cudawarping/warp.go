package cudawarping

import (
	"image"

	cv "github.com/malcolmston/opencv"
)

// WarpAffine applies the 2×3 affine transform m to the GpuMat, producing an
// output of size dsize (width dsize.X, height dsize.Y). It mirrors
// cv::cuda::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue).
//
// flags is an [Interpolation] optionally OR-ed with [WarpInverseMap]: without
// that bit m maps source pixels to destination pixels and is inverted
// internally; with it m is taken to already map destination pixels to source
// pixels. borderMode selects how out-of-image samples are produced and
// borderValue is the fill used by [BorderConstant].
//
// The default case (a zero constant border with nearest or bilinear
// interpolation) delegates to [cv.WarpAffine] for an exact, well-tested result;
// other border modes, a non-zero border value or bicubic interpolation are
// resampled locally via the coordinate maps from [BuildWarpAffineMaps]. The
// stream argument is ignored. It panics on an empty GpuMat, a non-positive
// dsize, or a non-invertible m.
func (g *GpuMat) WarpAffine(m cv.AffineMatrix, dsize image.Point, flags int, borderMode BorderMode, borderValue float64, stream *Stream) *GpuMat {
	src := g.host("WarpAffine")
	checkSize(dsize, "WarpAffine")
	interp := Interpolation(flags & interMask)
	inverse := flags&WarpInverseMap != 0

	if isDefaultBorder(interp, borderMode, borderValue) {
		// cv.WarpAffine expects a forward (source→destination) matrix and
		// inverts it internally. When the caller supplied a destination→source
		// matrix, invert it first so the round-trip cancels.
		fwd := m
		if inverse {
			var ok bool
			fwd, ok = invertAffine(m)
			if !ok {
				panic("cudawarping: WarpAffine transform is not invertible")
			}
		}
		return &GpuMat{mat: cv.WarpAffine(src, fwd, dsize.X, dsize.Y, cvInterp(interp))}
	}

	xmap, ymap := BuildWarpAffineMaps(m, inverse, dsize, stream)
	return &GpuMat{mat: remapWithBorder(src, xmap, ymap, interp, borderMode, borderValue)}
}

// WarpPerspective applies the 3×3 projective transform m to the GpuMat,
// producing an output of size dsize, the perspective analogue of
// [GpuMat.WarpAffine]. It mirrors cv::cuda::warpPerspective with the same flag,
// border mode and border value semantics. The stream argument is ignored. It
// panics on an empty GpuMat, a non-positive dsize, or a non-invertible m.
func (g *GpuMat) WarpPerspective(m cv.PerspectiveMatrix, dsize image.Point, flags int, borderMode BorderMode, borderValue float64, stream *Stream) *GpuMat {
	src := g.host("WarpPerspective")
	checkSize(dsize, "WarpPerspective")
	interp := Interpolation(flags & interMask)
	inverse := flags&WarpInverseMap != 0

	if isDefaultBorder(interp, borderMode, borderValue) {
		fwd := m
		if inverse {
			var ok bool
			fwd, ok = invertPerspective(m)
			if !ok {
				panic("cudawarping: WarpPerspective transform is not invertible")
			}
		}
		return &GpuMat{mat: cv.WarpPerspective(src, fwd, dsize.X, dsize.Y, cvInterp(interp))}
	}

	xmap, ymap := BuildWarpPerspectiveMaps(m, inverse, dsize, stream)
	return &GpuMat{mat: remapWithBorder(src, xmap, ymap, interp, borderMode, borderValue)}
}

// Rotate rotates the GpuMat by angle degrees about the origin, then shifts the
// result by (xShift, yShift), producing an output of size dsize. It mirrors
// cv::cuda::rotate(src, dst, dsize, angle, xShift, yShift, interpolation): the
// forward transform is
//
//	[ cos(angle)  sin(angle)  xShift ]
//	[-sin(angle)  cos(angle)  yShift ]
//
// applied to source coordinates (with the origin at the top-left and y growing
// downwards, a positive angle rotates the image clockwise). Samples outside the
// source are filled with zero. The stream argument is ignored. It panics on an
// empty GpuMat or a non-positive dsize.
func (g *GpuMat) Rotate(dsize image.Point, angle, xShift, yShift float64, interp Interpolation, stream *Stream) *GpuMat {
	g.host("Rotate")
	checkSize(dsize, "Rotate")
	m := rotationMatrix(angle, xShift, yShift)
	return g.WarpAffine(m, dsize, int(interp), BorderConstant, 0, stream)
}

// Rotate90 performs a lossless 90/180/270-degree rotation of the GpuMat,
// delegating to [cv.Rotate]. Unlike [GpuMat.Rotate] it never resamples, so it is
// exact and preserves every sample; use it for right-angle rotations. The
// stream argument is ignored. It panics on an empty GpuMat.
func (g *GpuMat) Rotate90(code RotateCode, stream *Stream) *GpuMat {
	src := g.host("Rotate90")
	return &GpuMat{mat: cv.Rotate(src, code)}
}

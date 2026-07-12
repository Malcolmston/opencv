package cudawarping

import (
	cv "github.com/malcolmston/opencv"
)

// Remap resamples the GpuMat at the per-pixel source coordinates given by xmap
// and ymap: destination pixel (x, y) is taken from the source at
// (xmap[y,x], ymap[y,x]) using the chosen interpolation and border handling. It
// mirrors cv::cuda::remap(src, dst, xmap, ymap, interpolation, borderMode,
// borderValue). The output takes the maps' dimensions.
//
// The default case (a zero constant border with nearest or bilinear
// interpolation) delegates to [cv.Remap]; other border modes, a non-zero border
// value or bicubic interpolation are resampled locally. The stream argument is
// ignored. It panics on an empty GpuMat or if the maps differ in size.
func (g *GpuMat) Remap(xmap, ymap *cv.FloatMat, interp Interpolation, borderMode BorderMode, borderValue float64, stream *Stream) *GpuMat {
	src := g.host("Remap")
	if xmap.Rows != ymap.Rows || xmap.Cols != ymap.Cols {
		panic("cudawarping: Remap map dimensions differ")
	}
	if isDefaultBorder(interp, borderMode, borderValue) {
		return &GpuMat{mat: cv.Remap(src, xmap, ymap, cvInterp(interp))}
	}
	return &GpuMat{mat: remapWithBorder(src, xmap, ymap, interp, borderMode, borderValue)}
}

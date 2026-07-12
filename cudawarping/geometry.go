package cudawarping

import (
	cv "github.com/malcolmston/opencv"
)

// PyrDown blurs the GpuMat with the 5-tap binomial Gaussian kernel and drops
// every other row and column, halving each dimension (rounding up). It is one
// level of a Gaussian pyramid and delegates to [cv.PyrDown]. The stream argument
// is ignored. It panics on an empty GpuMat.
func (g *GpuMat) PyrDown(stream *Stream) *GpuMat {
	src := g.host("PyrDown")
	return &GpuMat{mat: cv.PyrDown(src)}
}

// PyrUp doubles each dimension of the GpuMat by inserting zero rows and columns
// and smoothing with the 5-tap Gaussian kernel, the up-sampling step of a
// Gaussian pyramid. It delegates to [cv.PyrUp]. The stream argument is ignored.
// It panics on an empty GpuMat.
func (g *GpuMat) PyrUp(stream *Stream) *GpuMat {
	src := g.host("PyrUp")
	return &GpuMat{mat: cv.PyrUp(src)}
}

// Transpose swaps the rows and columns of the GpuMat, returning a result of
// shape cols×rows. It delegates to [cv.Transpose] and mirrors
// cv::cuda::transpose. The stream argument is ignored. It panics on an empty
// GpuMat.
func (g *GpuMat) Transpose(stream *Stream) *GpuMat {
	src := g.host("Transpose")
	return &GpuMat{mat: cv.Transpose(src)}
}

// Flip mirrors the GpuMat about one or both axes and mirrors cv::cuda::flip.
// Following OpenCV's integer convention, flipCode == 0 flips about the x-axis
// (vertical flip, top↔bottom), flipCode > 0 flips about the y-axis (horizontal
// flip, left↔right) and flipCode < 0 flips about both axes. It delegates to
// [cv.Flip]. The stream argument is ignored. It panics on an empty GpuMat.
func (g *GpuMat) Flip(flipCode int, stream *Stream) *GpuMat {
	src := g.host("Flip")
	var code cv.FlipCode
	switch {
	case flipCode == 0:
		code = cv.FlipVertical
	case flipCode > 0:
		code = cv.FlipHorizontal
	default:
		code = cv.FlipBoth
	}
	return &GpuMat{mat: cv.Flip(src, code)}
}

// CopyMakeBorder returns a copy of the GpuMat enlarged by top, bottom, left and
// right rows/columns of border, filled according to borderMode (with
// borderValue used by [BorderConstant]). It mirrors cv::cuda::copyMakeBorder.
// The stream argument is ignored. It panics on an empty GpuMat or a negative
// border width.
func (g *GpuMat) CopyMakeBorder(top, bottom, left, right int, borderMode BorderMode, borderValue float64, stream *Stream) *GpuMat {
	src := g.host("CopyMakeBorder")
	if top < 0 || bottom < 0 || left < 0 || right < 0 {
		panic("cudawarping: CopyMakeBorder requires non-negative border widths")
	}
	dst := cv.NewMat(src.Rows+top+bottom, src.Cols+left+right, src.Channels)
	for y := 0; y < dst.Rows; y++ {
		sy := borderIndex(y-top, src.Rows, borderMode)
		for x := 0; x < dst.Cols; x++ {
			sx := borderIndex(x-left, src.Cols, borderMode)
			di := (y*dst.Cols + x) * dst.Channels
			if sx < 0 || sy < 0 {
				fill := clampU8(borderValue)
				for c := 0; c < dst.Channels; c++ {
					dst.Data[di+c] = fill
				}
				continue
			}
			si := (sy*src.Cols + sx) * src.Channels
			copy(dst.Data[di:di+dst.Channels], src.Data[si:si+src.Channels])
		}
	}
	return &GpuMat{mat: dst}
}

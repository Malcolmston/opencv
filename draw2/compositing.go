package draw2

import (
	cv "github.com/malcolmston/opencv"
)

// draw2sameShape reports whether a and b have identical dimensions.
func draw2sameShape(a, b *cv.Mat) bool {
	return a.Rows == b.Rows && a.Cols == b.Cols && a.Channels == b.Channels
}

// AlphaBlend composites src over dst in place with a single global opacity:
// dst = (1-alpha)*dst + alpha*src for every sample. alpha is clamped to [0,1].
// It panics if the two Mats do not have the same shape.
func AlphaBlend(dst, src *cv.Mat, alpha float64) {
	if !draw2sameShape(dst, src) {
		panic("draw2: AlphaBlend requires matching Mat shapes")
	}
	if alpha <= 0 {
		return
	}
	if alpha > 1 {
		alpha = 1
	}
	inv := 1 - alpha
	for i := range dst.Data {
		dst.Data[i] = draw2clamp8(float64(dst.Data[i])*inv + float64(src.Data[i])*alpha)
	}
}

// Blend returns a new Mat holding the weighted sum wa*a + wb*b + gamma of two
// equally-shaped Mats, mirroring OpenCV's addWeighted. Results are clamped to
// the 8-bit range. It panics if the shapes differ.
func Blend(a, b *cv.Mat, wa, wb, gamma float64) *cv.Mat {
	if !draw2sameShape(a, b) {
		panic("draw2: Blend requires matching Mat shapes")
	}
	out := cv.NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		out.Data[i] = draw2clamp8(float64(a.Data[i])*wa + float64(b.Data[i])*wb + gamma)
	}
	return out
}

// AlphaCompositeMask composites src over dst in place using a single-channel
// mask as a per-pixel opacity map: where the mask is 255 src replaces dst,
// where it is 0 dst is untouched, and intermediate values blend. dst and src
// must share the same shape and mask must have the same rows and cols with a
// single channel. It panics otherwise.
func AlphaCompositeMask(dst, src, mask *cv.Mat) {
	if !draw2sameShape(dst, src) {
		panic("draw2: AlphaCompositeMask requires matching dst/src shapes")
	}
	if mask.Rows != dst.Rows || mask.Cols != dst.Cols || mask.Channels != 1 {
		panic("draw2: AlphaCompositeMask requires a single-channel mask of matching size")
	}
	ch := dst.Channels
	for p := 0; p < dst.Rows*dst.Cols; p++ {
		a := float64(mask.Data[p]) / 255
		if a <= 0 {
			continue
		}
		inv := 1 - a
		base := p * ch
		for c := 0; c < ch; c++ {
			dst.Data[base+c] = draw2clamp8(float64(dst.Data[base+c])*inv + float64(src.Data[base+c])*a)
		}
	}
}

// FillRectangleAlpha fills the axis-aligned rectangle spanning pt1 and pt2 with
// color at the given opacity, alpha-compositing over the existing pixels. This
// is the building block for translucent overlays such as bounding-box
// highlights. alpha is clamped to [0,1].
func FillRectangleAlpha(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, alpha float64) {
	x0 := draw2maxInt(0, draw2minInt(pt1.X, pt2.X))
	x1 := draw2minInt(m.Cols-1, draw2maxInt(pt1.X, pt2.X))
	y0 := draw2maxInt(0, draw2minInt(pt1.Y, pt2.Y))
	y1 := draw2minInt(m.Rows-1, draw2maxInt(pt1.Y, pt2.Y))
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			draw2blend(m, x, y, color, alpha)
		}
	}
}

// OverlayColor fills the whole image with color at the given opacity,
// alpha-compositing over the existing pixels (a full-frame tint). alpha is
// clamped to [0,1].
func OverlayColor(m *cv.Mat, color cv.Scalar, alpha float64) {
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			draw2blend(m, x, y, color, alpha)
		}
	}
}

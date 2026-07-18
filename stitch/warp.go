package stitch

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ProjectCylindrical maps an image point (x, y) onto a cylinder of radius focal
// whose axis passes through the projection centre (cx, cy). It returns the
// cylindrical coordinates (u, v) relative to that centre: u is the arc length
// around the cylinder and v the height along its axis. The point at the centre
// maps to (0, 0).
func ProjectCylindrical(x, y, focal, cx, cy float64) (u, v float64) {
	xc := x - cx
	yc := y - cy
	u = focal * math.Atan2(xc, focal)
	v = focal * yc / math.Hypot(xc, focal)
	return u, v
}

// ProjectSpherical maps an image point (x, y) onto a unit sphere of radius focal
// centred on (cx, cy). It returns the spherical coordinates (u, v) relative to
// that centre, where u is longitude times focal and v latitude times focal. The
// point at the centre maps to (0, 0).
func ProjectSpherical(x, y, focal, cx, cy float64) (u, v float64) {
	xc := x - cx
	yc := y - cy
	u = focal * math.Atan2(xc, focal)
	v = focal * math.Atan2(yc, math.Hypot(xc, focal))
	return u, v
}

// warpProjected is the shared implementation of cylindrical and spherical
// warping. spherical selects the projection; the return convention matches the
// public wrappers.
func warpProjected(img *cv.Mat, focal float64, spherical bool) (*cv.Mat, int, int) {
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	cx := float64(cols-1) / 2
	cy := float64(rows-1) / 2
	project := ProjectCylindrical
	if spherical {
		project = ProjectSpherical
	}
	minU, minV := math.Inf(1), math.Inf(1)
	maxU, maxV := math.Inf(-1), math.Inf(-1)
	consider := func(x, y float64) {
		u, v := project(x, y, focal, cx, cy)
		minU = math.Min(minU, u)
		minV = math.Min(minV, v)
		maxU = math.Max(maxU, u)
		maxV = math.Max(maxV, v)
	}
	for x := 0; x < cols; x++ {
		consider(float64(x), 0)
		consider(float64(x), float64(rows-1))
	}
	for y := 0; y < rows; y++ {
		consider(0, float64(y))
		consider(float64(cols-1), float64(y))
	}
	offsetX := int(math.Floor(minU))
	offsetY := int(math.Floor(minV))
	outW := int(math.Ceil(maxU)) - offsetX + 1
	outH := int(math.Ceil(maxV)) - offsetY + 1
	if outW < 1 {
		outW = 1
	}
	if outH < 1 {
		outH = 1
	}
	out := cv.NewMat(outH, outW, ch)
	for oy := 0; oy < outH; oy++ {
		v := float64(oy + offsetY)
		for ox := 0; ox < outW; ox++ {
			u := float64(ox + offsetX)
			theta := u / focal
			xc := focal * math.Tan(theta)
			var yc float64
			if spherical {
				phi := v / focal
				yc = math.Tan(phi) * math.Hypot(xc, focal)
			} else {
				yc = v * math.Hypot(xc, focal) / focal
			}
			sx := xc + cx
			sy := yc + cy
			di := (oy*outW + ox) * ch
			for c := 0; c < ch; c++ {
				val, covered := sampleBilinear(img, sx, sy, c)
				if covered {
					out.Data[di+c] = clampByte(val)
				}
			}
		}
	}
	return out, offsetX, offsetY
}

// WarpCylindrical projects img onto a cylinder of radius focal (the camera focal
// length in pixels) and returns the rectified image together with the integer
// offset (offsetX, offsetY) of its top-left corner in the projected coordinate
// frame. Cylindrical warping straightens horizontal panning so overlapping
// frames align by a pure translation. Larger focal values curve the image less.
func WarpCylindrical(img *cv.Mat, focal float64) (out *cv.Mat, offsetX, offsetY int) {
	return warpProjected(img, focal, false)
}

// WarpSpherical projects img onto a sphere of radius focal (the camera focal
// length in pixels) and returns the rectified image together with the integer
// offset (offsetX, offsetY) of its top-left corner in the projected coordinate
// frame. Spherical warping suits panoramas that pan in both azimuth and
// elevation.
func WarpSpherical(img *cv.Mat, focal float64) (out *cv.Mat, offsetX, offsetY int) {
	return warpProjected(img, focal, true)
}

// WarpPerspectiveToCanvas warps img by the homography h onto the mosaic region
// described by canvas, producing a [Layer] sized to that region. h maps source
// pixels to global canvas coordinates; the returned layer's pixel (X, Y)
// corresponds to global coordinate (canvas.MinX+X, canvas.MinY+Y). Colours are
// bilinearly resampled and each covered pixel receives a feather weight (highest
// at the source image centre, tapering to the source border) so the layer can be
// blended directly. Pixels not covered by the warped image are left transparent
// (zero colour, zero weight). It returns a zero-value Layer if h is singular or
// canvas is empty.
func WarpPerspectiveToCanvas(img *cv.Mat, h Homography, canvas Bounds) Layer {
	hInv, ok := h.Inverse()
	if !ok || canvas.Empty() {
		return Layer{}
	}
	ch := img.Channels
	w := canvas.Width()
	hgt := canvas.Height()
	color := cv.NewMat(hgt, w, ch)
	weight := cv.NewFloatMat(hgt, w)
	srcWeight := FeatherWeightMap(img.Cols, img.Rows, 1.0)
	for oy := 0; oy < hgt; oy++ {
		gy := float64(oy + canvas.MinY)
		for ox := 0; ox < w; ox++ {
			gx := float64(ox + canvas.MinX)
			sxf, syf := hInv.ApplyXY(gx, gy)
			if sxf < 0 || syf < 0 || sxf > float64(img.Cols-1) || syf > float64(img.Rows-1) {
				continue
			}
			di := (oy*w + ox) * ch
			for c := 0; c < ch; c++ {
				val, covered := sampleBilinear(img, sxf, syf, c)
				if covered {
					color.Data[di+c] = clampByte(val)
				}
			}
			wv, _ := sampleBilinearFloat(srcWeight, sxf, syf)
			weight.Data[oy*w+ox] = wv
		}
	}
	return Layer{Image: color, Weight: weight}
}

// sampleBilinearFloat bilinearly samples a FloatMat at the continuous location
// (fx, fy) and reports whether the location lies within the matrix.
func sampleBilinearFloat(m *cv.FloatMat, fx, fy float64) (float64, bool) {
	if fx < 0 || fy < 0 || fx > float64(m.Cols-1) || fy > float64(m.Rows-1) {
		return 0, false
	}
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	x1 := x0 + 1
	y1 := y0 + 1
	ax := fx - float64(x0)
	ay := fy - float64(y0)
	if x1 > m.Cols-1 {
		x1 = m.Cols - 1
	}
	if y1 > m.Rows-1 {
		y1 = m.Rows - 1
	}
	v00 := m.Data[y0*m.Cols+x0]
	v01 := m.Data[y0*m.Cols+x1]
	v10 := m.Data[y1*m.Cols+x0]
	v11 := m.Data[y1*m.Cols+x1]
	top := v00*(1-ax) + v01*ax
	bot := v10*(1-ax) + v11*ax
	return top*(1-ay) + bot*ay, true
}

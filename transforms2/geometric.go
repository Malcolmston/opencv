package transforms2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Resize returns src resampled to the given width and height using the chosen
// interpolation. Border effects use edge replication. It panics if width or
// height is non-positive.
func Resize(src *cv.Mat, width, height int, interp Interpolation) *cv.Mat {
	if width <= 0 || height <= 0 {
		panic("transforms2: Resize requires positive width and height")
	}
	scaleX := float64(src.Cols) / float64(width)
	scaleY := float64(src.Rows) / float64(height)
	return transforms2warpInverse(src, width, height, interp, BorderReplicate, 0, func(x, y float64) (float64, float64) {
		return (x+0.5)*scaleX - 0.5, (y+0.5)*scaleY - 0.5
	})
}

// Scale returns src scaled by the factors fx and fy about the origin, with the
// output size derived by rounding src's dimensions. It panics if a factor is
// non-positive.
func Scale(src *cv.Mat, fx, fy float64, interp Interpolation) *cv.Mat {
	if fx <= 0 || fy <= 0 {
		panic("transforms2: Scale requires positive factors")
	}
	w := int(math.Round(float64(src.Cols) * fx))
	h := int(math.Round(float64(src.Rows) * fy))
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	return Resize(src, w, h, interp)
}

// Translate returns src shifted by (tx, ty) pixels while keeping the output the
// same size, filling exposed areas per the border mode.
func Translate(src *cv.Mat, tx, ty float64, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	return WarpAffine(src, AffineTranslation(tx, ty), src.Cols, src.Rows, interp, border, fill)
}

// Shear returns src sheared by shx (along x, proportional to y) and shy (along
// y, proportional to x) while keeping the output the same size.
func Shear(src *cv.Mat, shx, shy float64, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	return WarpAffine(src, AffineShear(shx, shy), src.Cols, src.Rows, interp, border, fill)
}

// Rotate returns src rotated about its centre by angleDeg degrees
// (counter-clockwise) and uniformly scaled by scale, keeping the output the
// same size. Corners rotated outside the frame are discarded.
func Rotate(src *cv.Mat, angleDeg, scale float64, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	cx := float64(src.Cols-1) / 2
	cy := float64(src.Rows-1) / 2
	m := AffineRotationAround(cx, cy, angleDeg, scale)
	return WarpAffine(src, m, src.Cols, src.Rows, interp, border, fill)
}

// RotateBound returns src rotated about its centre by angleDeg degrees
// (counter-clockwise), enlarging the output canvas so that no content is
// clipped. The whole rotated image is contained in the result.
func RotateBound(src *cv.Mat, angleDeg float64, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	cx := float64(src.Cols-1) / 2
	cy := float64(src.Rows-1) / 2
	rot := AffineRotationAround(cx, cy, angleDeg, 1)
	// Bounding box of the rotated corners.
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	corners := [4][2]float64{{0, 0}, {float64(src.Cols - 1), 0}, {0, float64(src.Rows - 1)}, {float64(src.Cols - 1), float64(src.Rows - 1)}}
	for _, c := range corners {
		px, py := ApplyAffine(rot, c[0], c[1])
		minX = math.Min(minX, px)
		minY = math.Min(minY, py)
		maxX = math.Max(maxX, px)
		maxY = math.Max(maxY, py)
	}
	w := int(math.Round(maxX-minX)) + 1
	h := int(math.Round(maxY-minY)) + 1
	// Translate so the bounding box starts at the origin.
	m := ComposeAffine(AffineTranslation(-minX, -minY), rot)
	return WarpAffine(src, m, w, h, interp, border, fill)
}

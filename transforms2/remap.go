package transforms2

import (
	cv "github.com/malcolmston/opencv"
)

// Remap resamples src at the per-pixel source coordinates given by mapX and
// mapY: destination pixel (x, y) is taken from src at (mapX[y,x], mapY[y,x])
// using the chosen interpolation and border handling. mapX and mapY must have
// identical, positive dimensions, which become the output size. It panics
// otherwise.
func Remap(src *cv.Mat, mapX, mapY *cv.FloatMat, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	if mapX.Rows != mapY.Rows || mapX.Cols != mapY.Cols {
		panic("transforms2: Remap map dimensions differ")
	}
	if mapX.Rows <= 0 || mapX.Cols <= 0 {
		panic("transforms2: Remap requires positive map dimensions")
	}
	cols := mapX.Cols
	return transforms2warpInverse(src, cols, mapX.Rows, interp, border, fill, func(x, y float64) (float64, float64) {
		idx := int(y)*cols + int(x)
		return mapX.Data[idx], mapY.Data[idx]
	})
}

// RemapFunc resamples src into a width x height output where the source
// coordinate for each destination pixel (x, y) is returned by fn. It is a
// convenience wrapper that avoids materialising coordinate maps. It panics if
// the size is non-positive.
func RemapFunc(src *cv.Mat, width, height int, fn func(x, y float64) (sx, sy float64), interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	return transforms2warpInverse(src, width, height, interp, border, fill, fn)
}

// MakeIdentityMaps returns coordinate maps of the given size for which
// destination pixel (x, y) maps to source (x, y). They are a useful starting
// point for building custom displacement fields. It panics if the size is
// non-positive.
func MakeIdentityMaps(width, height int) (mapX, mapY *cv.FloatMat) {
	if width <= 0 || height <= 0 {
		panic("transforms2: MakeIdentityMaps requires positive size")
	}
	mapX = cv.NewFloatMat(height, width)
	mapY = cv.NewFloatMat(height, width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			mapX.Data[y*width+x] = float64(x)
			mapY.Data[y*width+x] = float64(y)
		}
	}
	return mapX, mapY
}

// MakeAffineMaps returns coordinate maps of the given size that realise the
// inverse of the affine transform m, so that Remap with them reproduces
// WarpAffine(src, m, ...). It panics if m is not invertible or the size is
// non-positive.
func MakeAffineMaps(m cv.AffineMatrix, width, height int) (mapX, mapY *cv.FloatMat) {
	if width <= 0 || height <= 0 {
		panic("transforms2: MakeAffineMaps requires positive size")
	}
	inv, ok := InvertAffine(m)
	if !ok {
		panic("transforms2: MakeAffineMaps transform is not invertible")
	}
	mapX = cv.NewFloatMat(height, width)
	mapY = cv.NewFloatMat(height, width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sx, sy := ApplyAffine(inv, float64(x), float64(y))
			mapX.Data[y*width+x] = sx
			mapY.Data[y*width+x] = sy
		}
	}
	return mapX, mapY
}

// MakePerspectiveMaps returns coordinate maps of the given size that realise the
// inverse of the homography m, so that Remap with them reproduces
// WarpPerspective(src, m, ...). It panics if m is not invertible or the size is
// non-positive.
func MakePerspectiveMaps(m cv.PerspectiveMatrix, width, height int) (mapX, mapY *cv.FloatMat) {
	if width <= 0 || height <= 0 {
		panic("transforms2: MakePerspectiveMaps requires positive size")
	}
	inv, ok := InvertPerspective(m)
	if !ok {
		panic("transforms2: MakePerspectiveMaps transform is not invertible")
	}
	mapX = cv.NewFloatMat(height, width)
	mapY = cv.NewFloatMat(height, width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sx, sy := ApplyPerspective(inv, float64(x), float64(y))
			mapX.Data[y*width+x] = sx
			mapY.Data[y*width+x] = sy
		}
	}
	return mapX, mapY
}

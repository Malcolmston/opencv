package calib2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// pixelToNormalized converts a pixel coordinate to an ideal normalized image
// coordinate using the intrinsic matrix (including skew).
func pixelToNormalized(u, v float64, k CameraMatrix) (x, y float64) {
	y = (v - k.Cy) / k.Fy
	x = (u - k.Cx - k.Skew*y) / k.Fx
	return x, y
}

// normalizedToPixel converts a normalized image coordinate to a pixel
// coordinate using the intrinsic matrix (including skew).
func normalizedToPixel(x, y float64, k CameraMatrix) (u, v float64) {
	u = k.Fx*x + k.Skew*y + k.Cx
	v = k.Fy*y + k.Cy
	return u, v
}

// undistortNormalized inverts the Brown–Conrady distortion model, recovering
// the ideal normalized coordinate (x, y) from a distorted normalized coordinate
// (xd, yd) by fixed-point iteration. It converges rapidly for realistic
// distortion magnitudes.
func undistortNormalized(xd, yd float64, d DistortionCoeffs) (x, y float64) {
	x, y = xd, yd
	if d.IsZero() {
		return x, y
	}
	for i := 0; i < 20; i++ {
		r2 := x*x + y*y
		radial := 1 + d.K1*r2 + d.K2*r2*r2 + d.K3*r2*r2*r2
		dxT := 2*d.P1*x*y + d.P2*(r2+2*x*x)
		dyT := d.P1*(r2+2*y*y) + 2*d.P2*x*y
		nx := (xd - dxT) / radial
		ny := (yd - dyT) / radial
		if math.Abs(nx-x) < 1e-12 && math.Abs(ny-y) < 1e-12 {
			x, y = nx, ny
			break
		}
		x, y = nx, ny
	}
	return x, y
}

// DistortPoint maps an ideal (undistortion-corrected) image pixel to the pixel
// where the lens would actually image that ray, applying the Brown–Conrady
// distortion model. It is the inverse of [UndistortPoint].
func DistortPoint(pt cv.Point2f, k CameraMatrix, d DistortionCoeffs) cv.Point2f {
	x, y := pixelToNormalized(pt.X, pt.Y, k)
	xd, yd := distortNormalized(x, y, d)
	u, v := normalizedToPixel(xd, yd, k)
	return cv.Point2f{X: u, Y: v}
}

// UndistortPoint maps a distorted image pixel to the ideal pixel it would
// occupy under a distortion-free pinhole camera with the same intrinsics,
// inverting the lens model by iteration. It is the inverse of [DistortPoint].
func UndistortPoint(pt cv.Point2f, k CameraMatrix, d DistortionCoeffs) cv.Point2f {
	xd, yd := pixelToNormalized(pt.X, pt.Y, k)
	x, y := undistortNormalized(xd, yd, d)
	u, v := normalizedToPixel(x, y, k)
	return cv.Point2f{X: u, Y: v}
}

// DistortPoints applies [DistortPoint] to every point, returning a new slice in
// the same order.
func DistortPoints(pts []cv.Point2f, k CameraMatrix, d DistortionCoeffs) []cv.Point2f {
	out := make([]cv.Point2f, len(pts))
	for i, p := range pts {
		out[i] = DistortPoint(p, k, d)
	}
	return out
}

// UndistortPoints applies [UndistortPoint] to every point, returning a new
// slice in the same order.
func UndistortPoints(pts []cv.Point2f, k CameraMatrix, d DistortionCoeffs) []cv.Point2f {
	out := make([]cv.Point2f, len(pts))
	for i, p := range pts {
		out[i] = UndistortPoint(p, k, d)
	}
	return out
}

// InitUndistortRectifyMap builds the per-pixel resampling maps that convert a
// distorted source image into an undistorted, rectified image of size
// rows×cols. For each destination pixel it computes the corresponding
// floating-point source coordinate: mapX[v][u] and mapY[v][u] give the source
// column and row for destination pixel (u, v). The rectification rotation rect
// (pass [Mat3Identity] for plain undistortion) is applied in normalized space,
// newK is the intrinsic matrix of the destination image and k / d describe the
// source camera. The maps are suitable for [Remap].
func InitUndistortRectifyMap(k CameraMatrix, d DistortionCoeffs, rect [3][3]float64, newK CameraMatrix, rows, cols int) (mapX, mapY [][]float64) {
	rinv, ok := Mat3Inverse(rect)
	if !ok {
		rinv = Mat3Identity()
	}
	mapX = make([][]float64, rows)
	mapY = make([][]float64, rows)
	for v := 0; v < rows; v++ {
		mapX[v] = make([]float64, cols)
		mapY[v] = make([]float64, cols)
		for u := 0; u < cols; u++ {
			// Destination pixel -> normalized ray in destination frame.
			xn, yn := pixelToNormalized(float64(u), float64(v), newK)
			// Undo the rectification rotation to reach the source camera frame.
			ray := Mat3VecMul(rinv, [3]float64{xn, yn, 1})
			if ray[2] == 0 {
				ray[2] = 1e-12
			}
			x := ray[0] / ray[2]
			y := ray[1] / ray[2]
			// Apply source distortion and source intrinsics.
			xd, yd := distortNormalized(x, y, d)
			su, sv := normalizedToPixel(xd, yd, k)
			mapX[v][u] = su
			mapY[v][u] = sv
		}
	}
	return mapX, mapY
}

// Remap resamples src according to the coordinate maps produced by
// [InitUndistortRectifyMap], using bilinear interpolation. The output has the
// dimensions of the maps and the same channel count as src; destination pixels
// whose source coordinate falls outside src are left black. It panics if mapX
// and mapY differ in shape.
func Remap(src *cv.Mat, mapX, mapY [][]float64) *cv.Mat {
	rows := len(mapX)
	if rows == 0 || len(mapY) != rows {
		panic("calib2: Remap map shape mismatch")
	}
	cols := len(mapX[0])
	dst := cv.NewMat(rows, cols, src.Channels)
	for v := 0; v < rows; v++ {
		if len(mapY[v]) != cols || len(mapX[v]) != cols {
			panic("calib2: Remap ragged maps")
		}
		for u := 0; u < cols; u++ {
			sx := mapX[v][u]
			sy := mapY[v][u]
			if sx < 0 || sy < 0 || sx > float64(src.Cols-1) || sy > float64(src.Rows-1) {
				continue
			}
			x0 := int(math.Floor(sx))
			y0 := int(math.Floor(sy))
			x1 := x0 + 1
			y1 := y0 + 1
			if x1 > src.Cols-1 {
				x1 = src.Cols - 1
			}
			if y1 > src.Rows-1 {
				y1 = src.Rows - 1
			}
			fx := sx - float64(x0)
			fy := sy - float64(y0)
			for c := 0; c < src.Channels; c++ {
				p00 := float64(src.At(y0, x0, c))
				p01 := float64(src.At(y0, x1, c))
				p10 := float64(src.At(y1, x0, c))
				p11 := float64(src.At(y1, x1, c))
				top := p00*(1-fx) + p01*fx
				bot := p10*(1-fx) + p11*fx
				val := top*(1-fy) + bot*fy
				iv := int(val + 0.5)
				if iv < 0 {
					iv = 0
				} else if iv > 255 {
					iv = 255
				}
				dst.Set(v, u, c, uint8(iv))
			}
		}
	}
	return dst
}

// UndistortImage removes lens distortion from src, producing a new image of the
// same size in which straight lines in the scene appear straight. It uses the
// same intrinsics for the output, so the field of view is preserved; pixels
// that map outside the source are left black.
func UndistortImage(src *cv.Mat, k CameraMatrix, d DistortionCoeffs) *cv.Mat {
	mapX, mapY := InitUndistortRectifyMap(k, d, Mat3Identity(), k, src.Rows, src.Cols)
	return Remap(src, mapX, mapY)
}

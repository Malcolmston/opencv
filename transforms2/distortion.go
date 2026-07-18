package transforms2

import (
	cv "github.com/malcolmston/opencv"
)

// CameraMatrix holds the pinhole intrinsic parameters: focal lengths (Fx, Fy)
// in pixels and the principal point (Cx, Cy).
type CameraMatrix struct {
	// Fx is the focal length along x in pixels.
	Fx float64
	// Fy is the focal length along y in pixels.
	Fy float64
	// Cx is the x coordinate of the principal point.
	Cx float64
	// Cy is the y coordinate of the principal point.
	Cy float64
}

// DistortionCoeffs holds Brown-Conrady lens distortion coefficients: radial
// terms K1, K2, K3 and tangential terms P1, P2, matching OpenCV's ordering
// (k1, k2, p1, p2, k3).
type DistortionCoeffs struct {
	// K1, K2, K3 are the radial distortion coefficients.
	K1, K2, K3 float64
	// P1, P2 are the tangential distortion coefficients.
	P1, P2 float64
}

// transforms2distortNormalized applies the distortion model to an ideal
// normalized coordinate (x, y), returning the distorted normalized coordinate.
func transforms2distortNormalized(d DistortionCoeffs, x, y float64) (float64, float64) {
	r2 := x*x + y*y
	radial := 1 + d.K1*r2 + d.K2*r2*r2 + d.K3*r2*r2*r2
	xd := x*radial + 2*d.P1*x*y + d.P2*(r2+2*x*x)
	yd := y*radial + d.P1*(r2+2*y*y) + 2*d.P2*x*y
	return xd, yd
}

// DistortPoint maps an ideal (undistorted) pixel (x, y) to the pixel where the
// lens actually images it, given the camera intrinsics and distortion.
func DistortPoint(cam CameraMatrix, dist DistortionCoeffs, x, y float64) (float64, float64) {
	nx := (x - cam.Cx) / cam.Fx
	ny := (y - cam.Cy) / cam.Fy
	xd, yd := transforms2distortNormalized(dist, nx, ny)
	return xd*cam.Fx + cam.Cx, yd*cam.Fy + cam.Cy
}

// UndistortPoint maps a distorted pixel (u, v), as recorded by the lens, to the
// ideal pinhole pixel, given the camera intrinsics and distortion. The inverse
// distortion is solved iteratively.
func UndistortPoint(cam CameraMatrix, dist DistortionCoeffs, u, v float64) (float64, float64) {
	xd := (u - cam.Cx) / cam.Fx
	yd := (v - cam.Cy) / cam.Fy
	x, y := xd, yd
	for iter := 0; iter < 20; iter++ {
		r2 := x*x + y*y
		radial := 1 + dist.K1*r2 + dist.K2*r2*r2 + dist.K3*r2*r2*r2
		dxT := 2*dist.P1*x*y + dist.P2*(r2+2*x*x)
		dyT := dist.P1*(r2+2*y*y) + 2*dist.P2*x*y
		x = (xd - dxT) / radial
		y = (yd - dyT) / radial
	}
	return x*cam.Fx + cam.Cx, y*cam.Fy + cam.Cy
}

// DistortPoints applies [DistortPoint] to every point in pts, returning a fresh
// slice.
func DistortPoints(cam CameraMatrix, dist DistortionCoeffs, pts []cv.Point2f) []cv.Point2f {
	out := make([]cv.Point2f, len(pts))
	for i, p := range pts {
		x, y := DistortPoint(cam, dist, p.X, p.Y)
		out[i] = cv.Point2f{X: x, Y: y}
	}
	return out
}

// UndistortPoints applies [UndistortPoint] to every point in pts, returning a
// fresh slice.
func UndistortPoints(cam CameraMatrix, dist DistortionCoeffs, pts []cv.Point2f) []cv.Point2f {
	out := make([]cv.Point2f, len(pts))
	for i, p := range pts {
		x, y := UndistortPoint(cam, dist, p.X, p.Y)
		out[i] = cv.Point2f{X: x, Y: y}
	}
	return out
}

// InitUndistortRectifyMap builds coordinate maps of the given size such that
// Remap with them removes lens distortion: destination pixel (x, y) in the
// ideal image is sampled from the distorted source at the coordinate the lens
// imaged it to. It panics if the size is non-positive.
func InitUndistortRectifyMap(cam CameraMatrix, dist DistortionCoeffs, width, height int) (mapX, mapY *cv.FloatMat) {
	if width <= 0 || height <= 0 {
		panic("transforms2: InitUndistortRectifyMap requires positive size")
	}
	mapX = cv.NewFloatMat(height, width)
	mapY = cv.NewFloatMat(height, width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sx, sy := DistortPoint(cam, dist, float64(x), float64(y))
			mapX.Data[y*width+x] = sx
			mapY.Data[y*width+x] = sy
		}
	}
	return mapX, mapY
}

// Undistort removes lens distortion from src, returning an image of the same
// size in which straight lines are straight again. Pixels are resampled with
// the chosen interpolation and border handling.
func Undistort(src *cv.Mat, cam CameraMatrix, dist DistortionCoeffs, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	mapX, mapY := InitUndistortRectifyMap(cam, dist, src.Cols, src.Rows)
	return Remap(src, mapX, mapY, interp, border, fill)
}

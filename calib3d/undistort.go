package calib3d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Undistort removes lens distortion from an image, returning a new [cv.Mat] of
// the same size and channel count as img. K is the 3×3 intrinsic matrix and dist
// holds the Brown–Conrady coefficients in [K1, K2, P1, P2, K3] order (radial K1,
// K2, K3 and tangential P1, P2); a nil slice leaves the image unchanged up to
// resampling.
//
// For every pixel of the output (which lies on the ideal, undistorted image
// plane) the function maps back through the distortion model to the fractional
// source location and samples img there with bilinear interpolation, replicating
// the border for out-of-range neighbours. This inverse-map formulation makes the
// output free of holes.
func Undistort(img *cv.Mat, K [3][3]float64, dist []float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("calib3d: Undistort requires a non-empty image")
	}
	k1, k2, p1, p2, k3 := distParams(dist)
	fx, fy, cx, cy := K[0][0], K[1][1], K[0][2], K[1][2]
	if fx == 0 || fy == 0 {
		panic("calib3d: Undistort requires non-zero focal lengths")
	}
	dst := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for v := 0; v < img.Rows; v++ {
		for u := 0; u < img.Cols; u++ {
			// Normalize the ideal pixel, distort it, and re-project to find the
			// matching location in the distorted source image.
			x := (float64(u) - cx) / fx
			y := (float64(v) - cy) / fy
			xd, yd := distortNormalized(x, y, k1, k2, p1, p2, k3)
			su := fx*xd + cx
			sv := fy*yd + cy
			for c := 0; c < img.Channels; c++ {
				val := bilinearReplicate(img, su, sv, c)
				dst.Set(v, u, c, clampToUint8(val+0.5))
			}
		}
	}
	return dst
}

// bilinearReplicate samples channel c of img at the fractional location (fx, fy)
// using bilinear interpolation, replicating the nearest edge sample for
// out-of-range neighbours (BORDER_REPLICATE). It reimplements the root package's
// unexported sampler so calib3d depends only on the public cv API.
func bilinearReplicate(img *cv.Mat, fx, fy float64, c int) float64 {
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	dx := fx - float64(x0)
	dy := fy - float64(y0)
	v00 := float64(atReplicate(img, y0, x0, c))
	v01 := float64(atReplicate(img, y0, x0+1, c))
	v10 := float64(atReplicate(img, y0+1, x0, c))
	v11 := float64(atReplicate(img, y0+1, x0+1, c))
	top := v00*(1-dx) + v01*dx
	bot := v10*(1-dx) + v11*dx
	return top*(1-dy) + bot*dy
}

// atReplicate returns the sample at (y, x, c) with coordinates clamped to the
// image bounds (BORDER_REPLICATE).
func atReplicate(img *cv.Mat, y, x, c int) uint8 {
	if y < 0 {
		y = 0
	} else if y >= img.Rows {
		y = img.Rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= img.Cols {
		x = img.Cols - 1
	}
	return img.At(y, x, c)
}

// clampToUint8 rounds toward zero (after the caller adds any bias) and clamps
// into [0,255]. It reimplements the root package's unexported helper.
func clampToUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

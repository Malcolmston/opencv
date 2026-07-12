package imgprocx

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// LinearPolar remaps src from Cartesian coordinates into a polar (radius, angle)
// grid about center, mirroring cv2.linearPolar's forward transform. The output
// is width×height (columns index radius, rows index angle): destination pixel
// (col, row) is sampled from src at
//
//	ρ = maxRadius · col / width
//	φ = 2π · row / height
//	(x, y) = (center.X + ρ·cosφ, center.Y + ρ·sinφ)
//
// using bilinear interpolation, with samples outside src treated as zero. width
// and height must be positive and maxRadius must be positive. Because rotation
// and scaling of src about center become vertical and horizontal translations in
// the output, this is a common registration pre-processing step.
func LinearPolar(src *cv.Mat, center Point2f, maxRadius float64, width, height int) *cv.Mat {
	if width <= 0 || height <= 0 {
		panic("imgprocx: LinearPolar requires positive width and height")
	}
	if maxRadius <= 0 {
		panic("imgprocx: LinearPolar requires maxRadius > 0")
	}
	return polarRemap(src, center, width, height, func(col int) float64 {
		return maxRadius * float64(col) / float64(width)
	})
}

// LogPolar remaps src from Cartesian coordinates into a log-polar
// (log-radius, angle) grid about center, mirroring cv2.logPolar's forward
// transform. It behaves like [LinearPolar] except that the radial axis is
// logarithmic: destination column col maps to radius
//
//	ρ = exp(col / M),   M = width / ln(maxRadius)
//
// so the innermost column samples radius 1 and the outermost samples maxRadius.
// The angular axis is unchanged. width and height must be positive and maxRadius
// must be greater than one. In the log-polar domain both rotation and uniform
// scaling about center become translations, which is why it underpins
// scale/rotation-invariant matching.
func LogPolar(src *cv.Mat, center Point2f, maxRadius float64, width, height int) *cv.Mat {
	if width <= 0 || height <= 0 {
		panic("imgprocx: LogPolar requires positive width and height")
	}
	if maxRadius <= 1 {
		panic("imgprocx: LogPolar requires maxRadius > 1")
	}
	m := float64(width) / math.Log(maxRadius)
	return polarRemap(src, center, width, height, func(col int) float64 {
		return math.Exp(float64(col) / m)
	})
}

// polarRemap builds a width×height image whose column col maps to the radius
// radiusFn(col) and whose row maps to an angle spanning [0,2π); each output
// pixel is bilinearly sampled from src about center.
func polarRemap(src *cv.Mat, center Point2f, width, height int, radiusFn func(col int) float64) *cv.Mat {
	dst := cv.NewMat(height, width, src.Channels)
	for row := 0; row < height; row++ {
		angle := 2 * math.Pi * float64(row) / float64(height)
		ca := math.Cos(angle)
		sa := math.Sin(angle)
		for col := 0; col < width; col++ {
			rho := radiusFn(col)
			fx := center.X + rho*ca
			fy := center.Y + rho*sa
			di := (row*width + col) * src.Channels
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = clampUint8(bilinearMat(src, fx, fy, c))
			}
		}
	}
	return dst
}

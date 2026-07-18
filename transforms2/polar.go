package transforms2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PolarMode selects the radial mapping used by [WarpPolar] and
// [WarpPolarInverse].
type PolarMode int

const (
	// PolarLinear maps radius linearly: column x corresponds to radius
	// maxRadius * x / width.
	PolarLinear PolarMode = iota
	// PolarLog maps radius logarithmically (log-polar): column x corresponds
	// to radius maxRadius^(x/width), emphasising the region near the centre.
	PolarLog
)

// transforms2polarRadius converts a radius column index (in [0, width)) to a
// source radius for the given mode.
func transforms2polarRadius(col float64, width int, maxRadius float64, mode PolarMode) float64 {
	f := col / float64(width)
	if mode == PolarLog {
		return math.Pow(maxRadius, f)
	}
	return maxRadius * f
}

// WarpPolar remaps the Cartesian image src into a polar representation of size
// width (radius axis) by height (angle axis) about the centre (centerX,
// centerY). Output column x encodes radius (per mode) up to maxRadius and output
// row y encodes angle 2*pi*y/height. It panics if the size is non-positive or
// maxRadius is non-positive.
func WarpPolar(src *cv.Mat, centerX, centerY, maxRadius float64, width, height int, mode PolarMode, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	if width <= 0 || height <= 0 {
		panic("transforms2: WarpPolar requires positive size")
	}
	if maxRadius <= 0 {
		panic("transforms2: WarpPolar requires positive maxRadius")
	}
	return transforms2warpInverse(src, width, height, interp, border, fill, func(x, y float64) (float64, float64) {
		angle := 2 * math.Pi * y / float64(height)
		r := transforms2polarRadius(x, width, maxRadius, mode)
		return centerX + r*math.Cos(angle), centerY + r*math.Sin(angle)
	})
}

// LinearPolar is a convenience wrapper for [WarpPolar] with [PolarLinear].
func LinearPolar(src *cv.Mat, centerX, centerY, maxRadius float64, width, height int, interp Interpolation) *cv.Mat {
	return WarpPolar(src, centerX, centerY, maxRadius, width, height, PolarLinear, interp, BorderConstant, 0)
}

// LogPolar is a convenience wrapper for [WarpPolar] with [PolarLog].
func LogPolar(src *cv.Mat, centerX, centerY, maxRadius float64, width, height int, interp Interpolation) *cv.Mat {
	return WarpPolar(src, centerX, centerY, maxRadius, width, height, PolarLog, interp, BorderConstant, 0)
}

// WarpPolarInverse reconstructs a Cartesian image of the given width and height
// from the polar image polar (as produced by [WarpPolar] with the same centre,
// maxRadius and mode). It panics if the size is non-positive or maxRadius is
// non-positive.
func WarpPolarInverse(polar *cv.Mat, centerX, centerY, maxRadius float64, width, height int, mode PolarMode, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	if width <= 0 || height <= 0 {
		panic("transforms2: WarpPolarInverse requires positive size")
	}
	if maxRadius <= 0 {
		panic("transforms2: WarpPolarInverse requires positive maxRadius")
	}
	pw := float64(polar.Cols)
	ph := float64(polar.Rows)
	return transforms2warpInverse(polar, width, height, interp, border, fill, func(x, y float64) (float64, float64) {
		dx := x - centerX
		dy := y - centerY
		r := math.Hypot(dx, dy)
		angle := math.Atan2(dy, dx)
		if angle < 0 {
			angle += 2 * math.Pi
		}
		py := angle / (2 * math.Pi) * ph
		var px float64
		if mode == PolarLog {
			if r <= 0 {
				px = 0
			} else {
				px = pw * math.Log(r) / math.Log(maxRadius)
			}
		} else {
			px = r / maxRadius * pw
		}
		return px, py
	})
}

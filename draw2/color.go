package draw2

import (
	cv "github.com/malcolmston/opencv"
)

// RGB builds an opaque colour Scalar from 8-bit red, green and blue
// components. The alpha component is set to 255.
func RGB(r, g, b uint8) cv.Scalar {
	return cv.Scalar{float64(r), float64(g), float64(b), 255}
}

// RGBA builds a colour Scalar from 8-bit red, green, blue and alpha
// components. The alpha component is carried in the fourth slot and used by
// the compositing routines in this package.
func RGBA(r, g, b, a uint8) cv.Scalar {
	return cv.Scalar{float64(r), float64(g), float64(b), float64(a)}
}

// Gray builds a colour Scalar whose three colour components all equal v, i.e.
// a neutral grey. The alpha component is set to 255.
func Gray(v uint8) cv.Scalar {
	f := float64(v)
	return cv.Scalar{f, f, f, 255}
}

// LerpColor linearly interpolates between colours a and b. t is clamped to
// [0,1]; t==0 yields a and t==1 yields b. All four components are blended.
func LerpColor(a, b cv.Scalar, t float64) cv.Scalar {
	if t <= 0 {
		return a
	}
	if t >= 1 {
		return b
	}
	var out cv.Scalar
	for i := 0; i < 4; i++ {
		out[i] = a[i] + (b[i]-a[i])*t
	}
	return out
}

// Alpha returns the alpha (fourth) component of a colour normalised to [0,1].
func Alpha(c cv.Scalar) float64 {
	a := c[3] / 255
	if a < 0 {
		return 0
	}
	if a > 1 {
		return 1
	}
	return a
}

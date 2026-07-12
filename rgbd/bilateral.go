package rgbd

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BilateralDepthFilter applies an edge-preserving bilateral filter to a depth
// map. Each valid pixel is replaced by a weighted average of the valid depths
// in a (2·radius+1)² window, where a neighbour's weight is the product of a
// spatial Gaussian (controlled by sigmaSpace, in pixels) and a range Gaussian
// on the depth difference (controlled by sigmaDepth, in depth units). This
// smooths sensor noise on flat regions while keeping depth discontinuities
// crisp, unlike a plain box or Gaussian blur.
//
// Invalid pixels (depth ≤ 0) are neither read nor written: they stay invalid in
// the output. The filter is separable-free and fully deterministic. It panics if
// depth is nil/empty, radius is negative, or either sigma is not positive.
func BilateralDepthFilter(depth *cv.FloatMat, radius int, sigmaSpace, sigmaDepth float64) *cv.FloatMat {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: BilateralDepthFilter given an empty depth map")
	}
	if radius < 0 {
		panic("rgbd: BilateralDepthFilter requires a non-negative radius")
	}
	if sigmaSpace <= 0 || sigmaDepth <= 0 {
		panic("rgbd: BilateralDepthFilter requires positive sigmas")
	}
	out := cv.NewFloatMat(depth.Rows, depth.Cols)
	invSpace := 1.0 / (2 * sigmaSpace * sigmaSpace)
	invDepth := 1.0 / (2 * sigmaDepth * sigmaDepth)
	for v := 0; v < depth.Rows; v++ {
		for u := 0; u < depth.Cols; u++ {
			z0 := depth.At(v, u)
			if z0 <= 0 {
				continue
			}
			var wsum, zsum float64
			for dv := -radius; dv <= radius; dv++ {
				vv := v + dv
				if vv < 0 || vv >= depth.Rows {
					continue
				}
				for du := -radius; du <= radius; du++ {
					uu := u + du
					if uu < 0 || uu >= depth.Cols {
						continue
					}
					z := depth.At(vv, uu)
					if z <= 0 {
						continue
					}
					spatial := float64(du*du+dv*dv) * invSpace
					rng := (z - z0) * (z - z0) * invDepth
					w := math.Exp(-spatial - rng)
					wsum += w
					zsum += w * z
				}
			}
			if wsum > 0 {
				out.Data[v*depth.Cols+u] = zsum / wsum
			}
		}
	}
	return out
}

// RescaleDepth returns a copy of depth with every valid measurement multiplied
// by scale, the standard way to convert integer depth units (for example
// millimetres) into metres, or otherwise change the depth unit. Invalid pixels
// (depth ≤ 0) are left as zero. It panics if depth is nil/empty or scale is not
// positive.
func RescaleDepth(depth *cv.FloatMat, scale float64) *cv.FloatMat {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: RescaleDepth given an empty depth map")
	}
	if scale <= 0 {
		panic("rgbd: RescaleDepth requires a positive scale")
	}
	out := cv.NewFloatMat(depth.Rows, depth.Cols)
	for i, z := range depth.Data {
		if z > 0 {
			out.Data[i] = z * scale
		}
	}
	return out
}

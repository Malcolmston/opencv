package rgbd

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// RegisterDepth warps a depth map captured by one camera into the image frame
// of a second (colour) camera. Each valid pixel of depth is back-projected to a
// 3-D point using the depth intrinsics kd, transformed into the colour camera
// frame by the rigid pose (R, t) as p' = R·p + t, and projected into the colour
// image with the colour intrinsics kc. The transformed depth p'_z is written to
// the output pixel it lands on; when several source points project to the same
// output pixel the nearest (smallest Z) is kept, resolving occlusion.
//
// The output is an outRows×outCols [github.com/malcolmston/opencv.FloatMat]
// sized for the colour image, with zero marking pixels that received no depth.
// It panics if depth is nil/empty, either intrinsic matrix has a zero focal
// length, or the output size is not positive.
func RegisterDepth(depth *cv.FloatMat, kd, kc [3][3]float64, r [3][3]float64, t [3]float64, outRows, outCols int) *cv.FloatMat {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: RegisterDepth given an empty depth map")
	}
	if outRows <= 0 || outCols <= 0 {
		panic("rgbd: RegisterDepth requires a positive output size")
	}
	validK(kd)
	validK(kc)
	fx, fy := kc[0][0], kc[1][1]
	cx, cy := kc[0][2], kc[1][2]

	out := cv.NewFloatMat(outRows, outCols)
	for v := 0; v < depth.Rows; v++ {
		for u := 0; u < depth.Cols; u++ {
			z := depth.At(v, u)
			if z <= 0 {
				continue
			}
			p := backProject(u, v, z, kd)
			// Transform into the colour camera frame.
			p = add3(matVec3(r, p), t)
			if p[2] <= 0 {
				continue // behind the colour camera
			}
			// Project into the colour image.
			ou := int(math.Round(fx*p[0]/p[2] + cx))
			ov := int(math.Round(fy*p[1]/p[2] + cy))
			if ou < 0 || ou >= outCols || ov < 0 || ov >= outRows {
				continue
			}
			cur := out.Data[ov*outCols+ou]
			if cur == 0 || p[2] < cur {
				out.Data[ov*outCols+ou] = p[2]
			}
		}
	}
	return out
}

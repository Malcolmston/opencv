package rgbd

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// distortNormalized applies the Brown–Conrady radial-tangential distortion model
// to a normalized image coordinate (x, y) = (X/Z, Y/Z), returning the distorted
// normalized coordinate. dist is (k1, k2, p1, p2, k3): k* are radial and p* are
// tangential coefficients, matching OpenCV's ordering.
func distortNormalized(x, y float64, dist [5]float64) (float64, float64) {
	k1, k2, p1, p2, k3 := dist[0], dist[1], dist[2], dist[3], dist[4]
	r2 := x*x + y*y
	radial := 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
	xd := x*radial + 2*p1*x*y + p2*(r2+2*x*x)
	yd := y*radial + p1*(r2+2*y*y) + 2*p2*x*y
	return xd, yd
}

// RegisterDepthDistorted warps a depth map from a depth camera into the image
// frame of a colour camera, exactly like [RegisterDepth], but additionally
// models the colour camera's lens distortion. After each back-projected point is
// transformed into the colour frame by (R, t), its normalized coordinates are
// passed through the Brown–Conrady model with coefficients dist = (k1, k2, p1,
// p2, k3) before projection with the colour intrinsics kc. The transformed depth
// is written to the pixel it lands on, keeping the nearest (smallest Z) on
// collision to resolve occlusion.
//
// Passing an all-zero dist reproduces [RegisterDepth] exactly. The output is an
// outRows×outCols [github.com/malcolmston/opencv.FloatMat] with zero marking
// pixels that received no depth. It panics if depth is nil/empty, either
// intrinsic matrix has a zero focal length, or the output size is not positive.
func RegisterDepthDistorted(depth *cv.FloatMat, kd, kc [3][3]float64, dist [5]float64, r [3][3]float64, t [3]float64, outRows, outCols int) *cv.FloatMat {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: RegisterDepthDistorted given an empty depth map")
	}
	if outRows <= 0 || outCols <= 0 {
		panic("rgbd: RegisterDepthDistorted requires a positive output size")
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
			p = add3(matVec3(r, p), t)
			if p[2] <= 0 {
				continue
			}
			xn, yn := distortNormalized(p[0]/p[2], p[1]/p[2], dist)
			ou := int(math.Round(fx*xn + cx))
			ov := int(math.Round(fy*yn + cy))
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

package rgbd

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Camera holds the intrinsic parameters of a pinhole camera: the focal lengths
// Fx, Fy (in pixels) and the principal point (Cx, Cy). It is a convenience for
// building the 3×3 intrinsic matrix K that the package functions accept.
type Camera struct {
	Fx float64
	Fy float64
	Cx float64
	Cy float64
}

// K returns the intrinsic parameters as the 3×3 row-major matrix
//
//	[ Fx  0  Cx ]
//	[  0 Fy  Cy ]
//	[  0  0   1 ]
func (c Camera) K() [3][3]float64 {
	return [3][3]float64{
		{c.Fx, 0, c.Cx},
		{0, c.Fy, c.Cy},
		{0, 0, 1},
	}
}

// backProject returns the camera-frame 3-D point for pixel (u, v) at depth z,
// using intrinsics K. It performs no validity checking.
func backProject(u, v int, z float64, k [3][3]float64) [3]float64 {
	fx, fy := k[0][0], k[1][1]
	cx, cy := k[0][2], k[1][2]
	return [3]float64{
		(float64(u) - cx) * z / fx,
		(float64(v) - cy) * z / fy,
		z,
	}
}

// validK reports whether the intrinsic matrix has usable (non-zero) focal
// lengths, panicking with a descriptive message otherwise.
func validK(k [3][3]float64) {
	if k[0][0] == 0 || k[1][1] == 0 {
		panic(fmt.Sprintf("rgbd: intrinsic matrix has zero focal length fx=%g fy=%g", k[0][0], k[1][1]))
	}
}

// DepthTo3D back-projects every pixel of a depth map into a 3-D point using the
// pinhole intrinsics K. The result is a dense, row-major slice of length
// depth.Rows*depth.Cols: the point for pixel (row v, column u) is at index
// v*depth.Cols + u, so the grid structure is preserved for downstream routines
// such as [Compute3DNormals].
//
// A depth of zero or less marks a missing measurement; the corresponding entry
// is the origin {0, 0, 0}. Use [ValidPoints] to drop those if a bare point
// cloud is wanted. It panics if depth is nil/empty or K has a zero focal
// length.
func DepthTo3D(depth *cv.FloatMat, k [3][3]float64) [][3]float64 {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: DepthTo3D given an empty depth map")
	}
	validK(k)
	out := make([][3]float64, depth.Rows*depth.Cols)
	for v := 0; v < depth.Rows; v++ {
		for u := 0; u < depth.Cols; u++ {
			z := depth.At(v, u)
			if z > 0 {
				out[v*depth.Cols+u] = backProject(u, v, z, k)
			}
		}
	}
	return out
}

// ValidPoints returns the subset of points whose depth (Z component) is
// strictly positive, in their original order. It is the usual way to turn the
// dense grid from [DepthTo3D] into a bare point cloud.
func ValidPoints(points [][3]float64) [][3]float64 {
	out := make([][3]float64, 0, len(points))
	for _, p := range points {
		if p[2] > 0 {
			out = append(out, p)
		}
	}
	return out
}

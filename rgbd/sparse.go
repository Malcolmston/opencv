package rgbd

import cv "github.com/malcolmston/opencv"

// DepthTo3dSparse back-projects a chosen set of pixels of a depth map into 3-D
// points, the sparse counterpart of [DepthTo3D]. Each element of pixels is a
// {column u, row v} coordinate; the returned slice holds the camera-frame point
// for each, in the same order.
//
// Pixels that fall outside the image or carry an invalid (non-positive) depth
// yield the origin {0,0,0} and are flagged as false in the returned valid mask,
// so callers can filter them without losing the index correspondence. It panics
// if depth is nil/empty or K has a zero focal length.
func DepthTo3dSparse(depth *cv.FloatMat, k [3][3]float64, pixels [][2]int) (points [][3]float64, valid []bool) {
	if depth == nil || len(depth.Data) == 0 {
		panic("rgbd: DepthTo3dSparse given an empty depth map")
	}
	validK(k)
	points = make([][3]float64, len(pixels))
	valid = make([]bool, len(pixels))
	for i, px := range pixels {
		u, v := px[0], px[1]
		p, ok := pointAt(depth, k, u, v)
		if ok {
			points[i] = p
			valid[i] = true
		}
	}
	return points, valid
}

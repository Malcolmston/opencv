package stereo

import cv "github.com/malcolmston/opencv"

// ReprojectImageTo3D maps a disparity map to 3-D coordinates through the 4×4
// reprojection matrix Q (as produced by stereo rectification). For each pixel at
// column x, row y with disparity d it forms the homogeneous vector [x, y, d, 1],
// multiplies by Q, and divides by the resulting w component:
//
//	[X Y Z W]ᵀ = Q · [x y d 1]ᵀ,   point = (X/W, Y/W, Z/W)
//
// The result is a slice of length Rows*Cols in row-major order; entry
// (y*Cols + x) holds the (X, Y, Z) coordinates of that pixel. A typical Q has
// the form
//
//	[ 1  0   0   -cx      ]
//	[ 0  1   0   -cy      ]
//	[ 0  0   0    f       ]
//	[ 0  0 -1/Tx (cx-cx')/Tx ]
//
// for which a constant disparity yields a constant depth Z regardless of pixel
// position, since the third and fourth output rows depend only on d.
//
// Pixels holding [InvalidDisparity] are reprojected using their raw value like
// any other; callers who need to exclude no-match pixels should mask them using
// the disparity map. When W is zero the point is returned as (0, 0, 0). It
// panics if disparity is nil, empty, or not single-channel.
func ReprojectImageTo3D(disparity *cv.Mat, Q [4][4]float64) [][3]float64 {
	if disparity == nil || disparity.Empty() {
		panic("stereo: ReprojectImageTo3D given a nil or empty disparity map")
	}
	if disparity.Channels != 1 {
		panic("stereo: ReprojectImageTo3D requires a single-channel disparity map")
	}
	rows, cols := disparity.Rows, disparity.Cols
	out := make([][3]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			d := float64(disparity.Data[y*cols+x])
			fx, fy, fz := float64(x), float64(y), d
			X := Q[0][0]*fx + Q[0][1]*fy + Q[0][2]*fz + Q[0][3]
			Y := Q[1][0]*fx + Q[1][1]*fy + Q[1][2]*fz + Q[1][3]
			Z := Q[2][0]*fx + Q[2][1]*fy + Q[2][2]*fz + Q[2][3]
			W := Q[3][0]*fx + Q[3][1]*fy + Q[3][2]*fz + Q[3][3]
			if W == 0 {
				out[y*cols+x] = [3]float64{0, 0, 0}
				continue
			}
			out[y*cols+x] = [3]float64{X / W, Y / W, Z / W}
		}
	}
	return out
}

// Rectify is a passthrough stub for stereo rectification. Full rectification
// warps each image so that corresponding scene points share an image row, which
// requires the camera intrinsics, the relative pose and the resulting
// rectification homographies. That calibration-driven machinery is deferred (see
// the package overview), so this helper simply returns independent clones of its
// inputs and assumes the pair is already rectified — the same assumption the
// matchers make. It panics if either input is nil or empty.
func Rectify(left, right *cv.Mat) (rectLeft, rectRight *cv.Mat) {
	if left == nil || left.Empty() || right == nil || right.Empty() {
		panic("stereo: Rectify given a nil or empty image")
	}
	return left.Clone(), right.Clone()
}

package calib3d_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/calib3d"
)

// ExampleFindHomography estimates the projective transform mapping four source
// corners onto four destination corners and verifies it maps the first corner.
func ExampleFindHomography() {
	src := []cv.Point{{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 100, Y: 100}, {X: 0, Y: 100}}
	dst := []cv.Point{{X: 10, Y: 12}, {X: 190, Y: 30}, {X: 210, Y: 220}, {X: 5, Y: 205}}

	h, inliers := calib3d.FindHomography(src, dst, calib3d.MethodDirect, 0)
	fmt.Printf("inliers=%d h22=%.0f\n", len(inliers), h[2][2])
	// Output: inliers=4 h22=1
}

// ExampleRodriguesToMatrix converts a 90° rotation about Z into a rotation
// matrix.
func ExampleRodriguesToMatrix() {
	const halfPi = 1.5707963267948966
	r := calib3d.RodriguesToMatrix([3]float64{0, 0, halfPi})
	fmt.Printf("%.0f %.0f\n%.0f %.0f\n", r[0][0], r[0][1], r[1][0], r[1][1])
	// Output:
	// 0 -1
	// 1 0
}

// ExampleProjectPoints projects the world origin through a camera translated 5
// units along Z; it lands on the principal point.
func ExampleProjectPoints() {
	K := calib3d.CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	pts := calib3d.ProjectPoints(
		[][3]float64{{0, 0, 0}},
		[3]float64{0, 0, 0},
		[3]float64{0, 0, 5},
		K, nil,
	)
	fmt.Println(pts[0].X, pts[0].Y)
	// Output: 320 240
}

// ExampleUndistort resamples an image with no distortion, which reproduces the
// input exactly.
func ExampleUndistort() {
	img := cv.NewMat(4, 4, 1)
	img.Set(1, 2, 0, 200)
	K := calib3d.CameraMatrix{Fx: 100, Fy: 100, Cx: 2, Cy: 2}.Matrix()
	out := calib3d.Undistort(img, K, nil)
	fmt.Println(out.At(1, 2, 0))
	// Output: 200
}

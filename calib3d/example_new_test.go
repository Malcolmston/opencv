package calib3d_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/calib3d"
)

// ExampleConvertPointsToHomogeneous appends the homogeneous coordinate to a set
// of 2D points.
func ExampleConvertPointsToHomogeneous() {
	h := calib3d.ConvertPointsToHomogeneous([][2]float64{{3, 4}})
	fmt.Println(h[0])
	// Output: [3 4 1]
}

// ExampleConvertPointsFromHomogeneous divides by the homogeneous weight to
// recover the inhomogeneous point.
func ExampleConvertPointsFromHomogeneous() {
	p := calib3d.ConvertPointsFromHomogeneous([][3]float64{{6, 8, 2}})
	fmt.Println(p[0])
	// Output: [3 4]
}

// ExampleSolvePnP recovers a camera pose from six 3D–2D correspondences and
// reprojects the object, landing back on the observations.
func ExampleSolvePnP() {
	K := calib3d.CameraMatrix{Fx: 500, Fy: 500, Cx: 320, Cy: 240}.Matrix()
	obj := [][3]float64{
		{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1},
	}
	// Synthesize observations for a known pose to feed the solver.
	rvec := [3]float64{0.1, -0.05, 0.02}
	tvec := [3]float64{0.2, 0.1, 8}
	pix := calib3d.ProjectPoints(obj, rvec, tvec, K, nil)
	img := make([][2]float64, len(pix))
	for i, p := range pix {
		img[i] = [2]float64{float64(p.X), float64(p.Y)}
	}
	_, t, ok := calib3d.SolvePnP(obj, img, K, nil)
	fmt.Printf("ok=%v depth≈%.0f\n", ok, t[2])
	// Output: ok=true depth≈8
}

// ExampleFindChessboardCorners detects the inner corners of a small synthetic
// chessboard.
func ExampleFindChessboardCorners() {
	// Build a 4x4-square board (3x3 inner corners) with a white quiet zone.
	const sq, margin = 20, 20
	w := 4*sq + 2*margin
	img := cv.NewMat(w, w, 1)
	img.SetTo(255)
	for sy := 0; sy < 4; sy++ {
		for sx := 0; sx < 4; sx++ {
			if (sx+sy)%2 == 0 {
				continue
			}
			for y := 0; y < sq; y++ {
				for x := 0; x < sq; x++ {
					img.Set(margin+sy*sq+y, margin+sx*sq+x, 0, 0)
				}
			}
		}
	}
	corners, found := calib3d.FindChessboardCorners(img, [2]int{3, 3})
	fmt.Printf("found=%v n=%d\n", found, len(corners))
	// Output: found=true n=9
}

// ExampleStereoRectify computes rectification transforms for a horizontal stereo
// pair and reports the rectified baseline sign.
func ExampleStereoRectify() {
	K := calib3d.CameraMatrix{Fx: 600, Fy: 600, Cx: 320, Cy: 240}.Matrix()
	R := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	T := [3]float64{-1.5, 0, 0}
	_, _, _, P2, _ := calib3d.StereoRectify(K, nil, K, nil, [2]int{640, 480}, R, T)
	fmt.Printf("P2 baseline term negative: %v\n", P2[0][3] < 0)
	// Output: P2 baseline term negative: true
}

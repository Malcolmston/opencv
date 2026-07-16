package ccalib_test

import (
	"fmt"

	"github.com/malcolmston/opencv/ccalib"
)

// Example sets up an omnidirectional (fisheye) camera with the unified sphere
// model, projects a known 3D point through it, then undistorts that pixel back
// to a rectilinear view — printing both, deterministically.
func Example() {
	// A wide field-of-view camera: pinhole intrinsics plus mirror parameter Xi
	// and Brown–Conrady distortion terms.
	cam := ccalib.OmniModel{
		Fx: 280, Fy: 285, Cx: 320, Cy: 240,
		Xi: 1.05, K1: -0.02, K2: 0.01, P1: 0.001, P2: -0.0008,
	}

	// Project one 3D point (in the camera frame) to a pixel. Zero rotation and
	// translation keep the geometry easy to follow.
	obj := [][3]float64{{0.5, 0.3, 4}}
	pix := ccalib.Omnidir.ProjectPoints(obj, [3]float64{0, 0, 0}, [3]float64{0, 0, 0}, cam.K(), cam.Xi, cam.Dist())

	// Rectify that distorted pixel to a pinhole camera to recover a clean ray.
	Knew := [3][3]float64{{250, 0, 320}, {0, 250, 240}, {0, 0, 1}}
	R := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	rect := ccalib.Omnidir.Undistort(pix, cam.K(), cam.Xi, cam.Dist(), Knew, R)

	fmt.Printf("projected: (%.2f, %.2f)\n", pix[0][0], pix[0][1])
	fmt.Printf("rectified: (%.2f, %.2f)\n", rect[0][0], rect[0][1])
	// Output:
	// projected: (336.98, 250.37)
	// rectified: (351.25, 258.75)
}

// ExampleOmniModel_K shows that an OmniModel exposes its intrinsics as a 3×3
// matrix and its distortion as the [K1, K2, P1, P2] slice.
func ExampleOmniModel_K() {
	cam := ccalib.OmniModel{Fx: 280, Fy: 285, Cx: 320, Cy: 240, K1: -0.02, P2: -0.0008}
	K := cam.K()
	fmt.Printf("fx=%.0f fy=%.0f cx=%.0f cy=%.0f\n", K[0][0], K[1][1], K[0][2], K[1][2])
	fmt.Printf("dist=%v\n", cam.Dist())
	// Output:
	// fx=280 fy=285 cx=320 cy=240
	// dist=[-0.02 0 0 -0.0008]
}

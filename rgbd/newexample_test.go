package rgbd_test

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/rgbd"
)

// ExampleRodrigues rotates a vector 90° about the Z axis.
func ExampleRodrigues() {
	r := rgbd.Rodrigues([3]float64{0, 0, math.Pi / 2})
	p := rgbd.PoseFromRt(r, [3]float64{0, 0, 0}).Apply([3]float64{1, 0, 0})
	snap := func(x float64) float64 {
		if x > -1e-9 && x < 1e-9 {
			return 0
		}
		return x
	}
	fmt.Printf("(%.1f, %.1f, %.1f)\n", snap(p[0]), snap(p[1]), snap(p[2]))
	// Output:
	// (0.0, 1.0, 0.0)
}

// ExampleICPPointToPlane recovers a translation along the surface normal.
func ExampleICPPointToPlane() {
	var src, normals [][3]float64
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			src = append(src, [3]float64{float64(i) * 0.1, float64(j) * 0.1, 0})
			normals = append(normals, [3]float64{0, 0, 1})
		}
	}
	// Shift the destination 0.1 along the (shared) normal.
	dst := make([][3]float64, len(src))
	for i, p := range src {
		dst[i] = [3]float64{p[0], p[1], p[2] + 0.1}
	}
	pose, residual := rgbd.ICPPointToPlane(src, dst, normals, 10)
	fmt.Printf("tz=%.2f residual=%.3f\n", pose.T[2], residual)
	// Output:
	// tz=0.10 residual=0.000
}

// ExampleTSDFVolume integrates a fronto-parallel plane and raycasts it back.
func ExampleTSDFVolume() {
	k := rgbd.Camera{Fx: 80, Fy: 80, Cx: 24, Cy: 24}.K()
	depth := cv.NewFloatMat(48, 48)
	for i := range depth.Data {
		depth.Data[i] = 1.0 // a plane one metre away
	}
	vol := rgbd.NewTSDFVolume([3]int{60, 60, 60}, 0.02, [3]float64{-0.6, -0.6, 0.5}, 0.1)
	vol.Integrate(depth, k, rgbd.IdentityPose())
	out := vol.Raycast(k, rgbd.IdentityPose(), 48, 48)
	fmt.Printf("center depth=%.1f\n", out.At(24, 24))
	// Output:
	// center depth=1.0
}

// ExampleDepthCleaner fills a single-pixel hole from its neighbours.
func ExampleDepthCleaner() {
	depth := cv.NewFloatMat(5, 5)
	for i := range depth.Data {
		depth.Data[i] = 2.0
	}
	depth.Data[2*5+2] = 0 // a hole in the centre
	out := rgbd.NewDepthCleaner(1).Clean(depth)
	fmt.Printf("filled=%.1f\n", out.At(2, 2))
	// Output:
	// filled=2.0
}

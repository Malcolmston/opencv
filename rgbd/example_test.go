package rgbd_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/rgbd"
)

// ExampleDepthTo3D back-projects a tiny constant-depth map into a point cloud.
func ExampleDepthTo3D() {
	cam := rgbd.Camera{Fx: 50, Fy: 50, Cx: 1, Cy: 1}
	depth := cv.NewFloatMat(3, 3)
	for i := range depth.Data {
		depth.Data[i] = 2 // 2 metres everywhere
	}
	pts := rgbd.DepthTo3D(depth, cam.K())
	// The centre pixel (1,1) lies on the optical axis.
	c := pts[1*3+1]
	fmt.Printf("center=(%.1f, %.1f, %.1f)\n", c[0], c[1], c[2])
	// Output:
	// center=(0.0, 0.0, 2.0)
}

// ExampleICP recovers a known rigid transform between two clouds.
func ExampleICP() {
	src := [][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}}
	// Translate the source by a small (overlapping) offset so nearest-neighbour
	// correspondences are correct from the identity initialisation.
	dst := make([][3]float64, len(src))
	for i, p := range src {
		dst[i] = [3]float64{p[0] + 0.1, p[1], p[2]}
	}
	_, t, err := rgbd.ICP(src, dst, 20)
	// snap folds values within 1e-9 of zero to +0 for stable printing.
	snap := func(x float64) float64 {
		if x > -1e-9 && x < 1e-9 {
			return 0
		}
		return x
	}
	fmt.Printf("t=(%.1f, %.1f, %.1f) err=%.3f\n", snap(t[0]), snap(t[1]), snap(t[2]), snap(err))
	// Output:
	// t=(0.1, 0.0, 0.0) err=0.000
}

// ExampleVoxelDownsample collapses a small cluster into a single centroid.
func ExampleVoxelDownsample() {
	pts := [][3]float64{
		{0.01, 0.01, 0.01}, {0.02, 0.02, 0.02}, {0.03, 0.01, 0.02},
		{5.0, 5.0, 5.0},
	}
	out := rgbd.VoxelDownsample(pts, 1.0)
	fmt.Printf("%d points\n", len(out))
	// Output:
	// 2 points
}

// ExamplePlaneSegmentation finds the dominant plane in a point cloud.
func ExamplePlaneSegmentation() {
	var pts [][3]float64
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			pts = append(pts, [3]float64{float64(i) * 0.1, float64(j) * 0.1, 0})
		}
	}
	opts := rgbd.DefaultPlaneOptions()
	opts.DistanceThreshold = 0.01
	opts.MaxPlanes = 1
	planes, _ := rgbd.PlaneSegmentation(pts, opts)
	fmt.Printf("planes=%d inliers=%d\n", len(planes), planes[0].Inliers)
	// Output:
	// planes=1 inliers=100
}

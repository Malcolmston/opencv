package cudafeatures2d_test

import (
	"fmt"

	"github.com/malcolmston/opencv/cudafeatures2d"
)

// Example uploads a synthetic scene to a device matrix, detects ORB keypoints
// and computes their binary descriptors, then reports that the detector fired
// and produced one fixed-length descriptor row per keypoint.
func Example() {
	img := cudafeatures2d.NewGpuMat(scene(128))
	orb := cudafeatures2d.CreateORB(50)

	kps, desc := orb.DetectAndCompute(img, nil)
	rows, cols := desc.Size()

	fmt.Printf("keypoints>0: %v, one row per keypoint: %v, descriptor bytes: %d\n",
		len(kps) > 0, rows == len(kps), cols)
	// Output: keypoints>0: true, one row per keypoint: true, descriptor bytes: 32
}

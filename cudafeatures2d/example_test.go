package cudafeatures2d_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudafeatures2d"
)

// scene builds a light image with dark squares whose corners are the strong
// interest points the examples detect.
func scene(size int) *cv.Mat {
	m := cv.NewMat(size, size, 1)
	m.SetTo(200)
	for _, r := range [][2]int{{20, 20}, {70, 30}, {40, 75}, {90, 90}} {
		for y := r[1]; y < r[1]+16 && y < size; y++ {
			for x := r[0]; x < r[0]+16 && x < size; x++ {
				m.Data[y*size+x] = 20
			}
		}
	}
	return m
}

// ExampleORB detects ORB features on a device image and matches the descriptors
// against themselves, which yields an exact self-match for every keypoint.
func ExampleORB() {
	img := cudafeatures2d.NewGpuMat(scene(128))
	orb := cudafeatures2d.CreateORB(50)

	kps, desc := orb.DetectAndCompute(img, nil)

	matcher := cudafeatures2d.CreateBFMatcher(orb.DefaultNorm())
	matches := matcher.Match(desc, desc)

	allExact := true
	for _, m := range matches {
		if m.Distance != 0 {
			allExact = false
		}
	}
	fmt.Printf("keypoints>0: %v, self-matches exact: %v\n", len(kps) > 0, allExact)
	// Output: keypoints>0: true, self-matches exact: true
}

// ExampleCornersDetector finds Shi-Tomasi corners on a device image.
func ExampleCornersDetector() {
	img := cudafeatures2d.NewGpuMat(scene(128))
	det := cudafeatures2d.CreateGoodFeaturesToTrackDetector(30, 0.01, 5, 3)
	corners := det.Detect(img, nil)
	fmt.Println(len(corners) > 0)
	// Output: true
}

// ExampleDescriptorMatcher_KnnMatch ranks train descriptors by Hamming distance.
func ExampleDescriptorMatcher_KnnMatch() {
	query := cudafeatures2d.DescriptorsToGpuMat([][]byte{{0x00}})
	train := cudafeatures2d.DescriptorsToGpuMat([][]byte{{0xFF}, {0x01}, {0x0F}})

	matcher := cudafeatures2d.CreateBFMatcher(cudafeatures2d.NormHamming)
	knn := matcher.KnnMatch(query, train, 2)

	fmt.Printf("nearest train %d at distance %v\n", knn[0][0].TrainIdx, knn[0][0].Distance)
	// Output: nearest train 1 at distance 1
}

package cudaobjdetect_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudaobjdetect"
)

// Example builds the default 64x128 people-detector HOG, reports its descriptor
// length, then runs it over a single flat (featureless) window uploaded to a
// device matrix and confirms the detector does not fire.
func Example() {
	h := cudaobjdetect.NewDefaultHOG()
	h.SetSVMDetector(h.GetDefaultPeopleDetector())

	window := cv.NewMat(128, 64, 1)
	window.SetTo(128)
	locs, _ := h.Detect(cudaobjdetect.NewGpuMatFromMat(window), nil)

	fmt.Printf("descriptor size: %d, detections on flat window: %d\n",
		h.GetDescriptorSize(), len(locs))
	// Output: descriptor size: 3780, detections on flat window: 0
}

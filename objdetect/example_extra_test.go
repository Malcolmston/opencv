package objdetect_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/objdetect"
)

// ExampleNMSBoxes suppresses overlapping detections, keeping the strongest.
func ExampleNMSBoxes() {
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 10, Height: 10},
		{X: 1, Y: 1, Width: 10, Height: 10}, // overlaps the first
		{X: 50, Y: 50, Width: 10, Height: 10},
	}
	scores := []float64{0.9, 0.8, 0.7}
	kept := objdetect.NMSBoxes(boxes, scores, 0.5, 0.5)
	fmt.Println(kept)
	// Output: [0 2]
}

// ExampleGroupRectangles merges nearby detections into one averaged rectangle.
func ExampleGroupRectangles() {
	rects := []cv.Rect{
		{X: 10, Y: 10, Width: 20, Height: 20},
		{X: 12, Y: 11, Width: 20, Height: 20},
		{X: 11, Y: 12, Width: 20, Height: 20},
	}
	grouped := objdetect.GroupRectangles(rects, 2, 0.2)
	fmt.Println(len(grouped))
	// Output: 1
}

// ExampleComputeLBP shows that a uniform image maps to the all-ones LBP code.
func ExampleComputeLBP() {
	img := cv.NewMat(4, 4, 1)
	img.SetTo(100)
	lbp := objdetect.ComputeLBP(img)
	fmt.Println(lbp.At(2, 2, 0))
	// Output: 255
}

// ExampleHOGDescriptor_DefaultPeopleDetector reports the approximate people
// detector's weight-vector length for the canonical geometry.
func ExampleHOGDescriptor_DefaultPeopleDetector() {
	h := objdetect.NewHOGDescriptor()
	w := h.DefaultPeopleDetector()
	fmt.Println(len(w)) // DescriptorSize()+1 = 3780+1
	// Output: 3781
}

// ExampleDetectionBasedTracker follows one object across two frames, keeping a
// stable identity.
func ExampleDetectionBasedTracker() {
	var clf objdetect.SoftCascade // any Detector works; use a scripted stand-in below
	_ = clf

	det := &stepDetector{rects: [][]cv.Rect{
		{{X: 10, Y: 10, Width: 20, Height: 20}},
		{{X: 12, Y: 10, Width: 20, Height: 20}},
	}}
	tr := &objdetect.DetectionBasedTracker{Detector: det}
	img := cv.NewMat(1, 1, 1)
	tr.Process(img)
	tr.Process(img)
	objs := tr.Objects()
	fmt.Printf("%d track, id=%d\n", len(objs), objs[0].ID)
	// Output: 1 track, id=0
}

// stepDetector is a tiny Detector that replays scripted detections.
type stepDetector struct {
	rects [][]cv.Rect
	i     int
}

func (s *stepDetector) DetectMultiScale(_ *cv.Mat) []cv.Rect {
	if s.i >= len(s.rects) {
		return nil
	}
	r := s.rects[s.i]
	s.i++
	return r
}

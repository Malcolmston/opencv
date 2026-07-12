package cudaobjdetect

import (
	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/objdetect"
)

// GroupRectangles clusters overlapping detection rectangles and returns one
// averaged rectangle per cluster whose membership is at least groupThreshold,
// following OpenCV's cv::groupRectangles semantics. It is a thin pass-through to
// [objdetect.GroupRectangles]; eps <= 0 uses the OpenCV default of 0.2.
func GroupRectangles(rects []cv.Rect, groupThreshold int, eps float64) []cv.Rect {
	return objdetect.GroupRectangles(rects, groupThreshold, eps)
}

// GroupRectanglesWeights clusters rectangles like [GroupRectangles] but also
// carries a per-rectangle score; each surviving cluster reports the maximum
// score among its members. It delegates to [objdetect.GroupRectanglesWeights]
// and panics if rects and weights differ in length.
func GroupRectanglesWeights(rects []cv.Rect, weights []float64, groupThreshold int, eps float64) ([]cv.Rect, []float64) {
	return objdetect.GroupRectanglesWeights(rects, weights, groupThreshold, eps)
}

// NMSBoxes performs greedy non-maximum suppression on scored detection boxes and
// returns the kept indices in descending-score order, delegating to
// [objdetect.NMSBoxes]. It panics if boxes and scores differ in length.
func NMSBoxes(boxes []cv.Rect, scores []float64, scoreThreshold, nmsThreshold float64) []int {
	return objdetect.NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold)
}

// SoftNMSBoxes performs Gaussian Soft-NMS on scored detection boxes, returning
// the kept indices and their decayed scores, delegating to
// [objdetect.SoftNMSBoxes]. It panics if boxes and scores differ in length.
func SoftNMSBoxes(boxes []cv.Rect, scores []float64, scoreThreshold, sigma float64) (indices []int, keptScores []float64) {
	return objdetect.SoftNMSBoxes(boxes, scores, scoreThreshold, sigma)
}

// RectIoU returns the intersection-over-union overlap of two rectangles, a value
// in [0,1], delegating to [objdetect.RectIoU]. It is the overlap metric used by
// the non-maximum-suppression helpers.
func RectIoU(a, b cv.Rect) float64 {
	return objdetect.RectIoU(a, b)
}

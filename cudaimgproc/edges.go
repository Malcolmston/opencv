package cudaimgproc

import cv "github.com/malcolmston/opencv"

// CannyEdgeDetector is a CPU-backed Canny edge detector, mirroring
// cv::cuda::CannyEdgeDetector. Create one with [CreateCannyEdgeDetector] and run
// it with [CannyEdgeDetector.Detect]. Thresholds can be adjusted between
// detections.
type CannyEdgeDetector struct {
	lowThresh  float64
	highThresh float64
}

// CreateCannyEdgeDetector returns a [CannyEdgeDetector] with the given
// hysteresis thresholds, mirroring cuda::createCannyEdgeDetector. lowThresh and
// highThresh are gradient-magnitude thresholds; they are ordered automatically
// if supplied the wrong way round.
func CreateCannyEdgeDetector(lowThresh, highThresh float64) *CannyEdgeDetector {
	return &CannyEdgeDetector{lowThresh: lowThresh, highThresh: highThresh}
}

// SetLowThreshold updates the low hysteresis threshold.
func (d *CannyEdgeDetector) SetLowThreshold(v float64) { d.lowThresh = v }

// SetHighThreshold updates the high hysteresis threshold.
func (d *CannyEdgeDetector) SetHighThreshold(v float64) { d.highThresh = v }

// Detect runs the Canny pipeline on a single-channel GpuMat and returns a
// binary edge map (edges are 255, background 0). The trailing Stream argument is
// accepted and ignored. It panics unless src is single-channel.
func (d *CannyEdgeDetector) Detect(src GpuMat, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("CannyEdgeDetector.Detect")
	return wrap(cv.Canny(m, d.lowThresh, d.highThresh))
}

// HoughLinesDetector is a CPU-backed standard Hough line detector, mirroring
// cv::cuda::HoughLinesDetector. Create one with [CreateHoughLinesDetector] and
// run it with [HoughLinesDetector.Detect].
type HoughLinesDetector struct {
	rho       float64
	theta     float64
	threshold int
}

// CreateHoughLinesDetector returns a [HoughLinesDetector] with the given
// accumulator resolution and vote threshold, mirroring
// cuda::createHoughLinesDetector. rho is the distance resolution in pixels and
// theta the angle resolution in radians.
func CreateHoughLinesDetector(rho, theta float64, threshold int) *HoughLinesDetector {
	return &HoughLinesDetector{rho: rho, theta: theta, threshold: threshold}
}

// Detect finds lines in a binary edge image and returns them in Hesse normal
// form (rho, theta), sorted by descending vote count. The trailing Stream
// argument is accepted and ignored. It panics unless src is single-channel.
func (d *HoughLinesDetector) Detect(src GpuMat, streams ...Stream) []cv.PolarLine {
	_ = firstStream(streams)
	m := src.requireHost("HoughLinesDetector.Detect")
	return cv.HoughLines(m, d.rho, d.theta, d.threshold)
}

// HoughSegmentDetector is a CPU-backed probabilistic Hough line-segment
// detector, mirroring cv::cuda::HoughSegmentDetector. Create one with
// [CreateHoughSegmentDetector] and run it with [HoughSegmentDetector.Detect].
type HoughSegmentDetector struct {
	rho           float64
	theta         float64
	threshold     int
	minLineLength int
	maxLineGap    int
}

// CreateHoughSegmentDetector returns a [HoughSegmentDetector] with the given
// accumulator resolution, vote threshold and segment constraints, mirroring
// cuda::createHoughSegmentDetector. rho is in pixels, theta in radians,
// minLineLength and maxLineGap in pixels. Unlike OpenCV's randomised segment
// growth, the traversal here is deterministic.
func CreateHoughSegmentDetector(rho, theta float64, threshold, minLineLength, maxLineGap int) *HoughSegmentDetector {
	return &HoughSegmentDetector{
		rho:           rho,
		theta:         theta,
		threshold:     threshold,
		minLineLength: minLineLength,
		maxLineGap:    maxLineGap,
	}
}

// Detect finds line segments in a binary edge image. The trailing Stream
// argument is accepted and ignored. It panics unless src is single-channel.
func (d *HoughSegmentDetector) Detect(src GpuMat, streams ...Stream) []cv.LineSegment {
	_ = firstStream(streams)
	m := src.requireHost("HoughSegmentDetector.Detect")
	return cv.HoughLinesP(m, d.rho, d.theta, d.threshold, d.minLineLength, d.maxLineGap)
}

// HoughCirclesDetector is a CPU-backed Hough-gradient circle detector,
// mirroring cv::cuda::HoughCirclesDetector. Create one with
// [CreateHoughCirclesDetector] and run it with [HoughCirclesDetector.Detect].
type HoughCirclesDetector struct {
	minDist   float64
	cannyHigh float64
	votes     float64
	minRadius int
	maxRadius int
}

// CreateHoughCirclesDetector returns a [HoughCirclesDetector], mirroring
// cuda::createHoughCirclesDetector. minDist is the minimum distance between
// detected centres, cannyThreshold the high Canny threshold used for edge
// extraction, votesThreshold the accumulator threshold, and [minRadius,
// maxRadius] the radius search range.
func CreateHoughCirclesDetector(minDist, cannyThreshold, votesThreshold float64, minRadius, maxRadius int) *HoughCirclesDetector {
	return &HoughCirclesDetector{
		minDist:   minDist,
		cannyHigh: cannyThreshold,
		votes:     votesThreshold,
		minRadius: minRadius,
		maxRadius: maxRadius,
	}
}

// Detect finds circles in a single-channel image. The trailing Stream argument
// is accepted and ignored. It panics unless src is single-channel or the radius
// range is invalid.
func (d *HoughCirclesDetector) Detect(src GpuMat, streams ...Stream) []cv.HoughCircle {
	_ = firstStream(streams)
	m := src.requireHost("HoughCirclesDetector.Detect")
	return cv.HoughCircles(m, d.minDist, d.cannyHigh, d.votes, d.minRadius, d.maxRadius)
}

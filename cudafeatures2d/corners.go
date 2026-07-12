package cudafeatures2d

import (
	cv "github.com/malcolmston/opencv"
)

// CornersDetector is a CPU-backed mirror of cv::cuda::CornersDetector. This
// package provides the Shi–Tomasi ("good features to track") variant, delegating
// to cv.GoodFeaturesToTrack.
//
// Construct one with [CreateGoodFeaturesToTrackDetector].
type CornersDetector struct {
	maxCorners   int
	qualityLevel float64
	minDistance  float64
	blockSize    int
}

// CreateGoodFeaturesToTrackDetector returns a Shi–Tomasi corner detector. It
// mirrors cv::cuda::createGoodFeaturesToTrackDetector. maxCorners <= 0 keeps all
// corners; qualityLevel is the minimum accepted corner strength as a fraction of
// the strongest corner; minDistance is the minimum spacing between corners in
// pixels; blockSize is the structure-tensor window size (a value <= 0 defaults
// to 3).
func CreateGoodFeaturesToTrackDetector(maxCorners int, qualityLevel, minDistance float64, blockSize int) *CornersDetector {
	if blockSize <= 0 {
		blockSize = 3
	}
	if qualityLevel <= 0 {
		qualityLevel = 0.01
	}
	if minDistance <= 0 {
		minDistance = 1
	}
	return &CornersDetector{
		maxCorners:   maxCorners,
		qualityLevel: qualityLevel,
		minDistance:  minDistance,
		blockSize:    blockSize,
	}
}

// Detect returns the strong Shi–Tomasi corners of the device image as host
// points, strongest first, optionally filtered by mask (pass a nil or empty
// GpuMat for no mask). It mirrors cv::cuda::CornersDetector::detect. It panics if
// image is empty or not single-channel.
func (d *CornersDetector) Detect(image, mask *GpuMat) []cv.Point {
	if image.Empty() {
		panic("cudafeatures2d: CornersDetector.Detect on empty image")
	}
	pts := cv.GoodFeaturesToTrack(image.host(), d.maxCorners, d.qualityLevel, d.minDistance, d.blockSize)
	if mask.Empty() {
		return pts
	}
	m := mask.host()
	out := pts[:0:0]
	for _, p := range pts {
		if p.X < 0 || p.Y < 0 || p.X >= m.Cols || p.Y >= m.Rows {
			continue
		}
		if m.Data[(p.Y*m.Cols+p.X)*m.Channels] == 0 {
			continue
		}
		out = append(out, p)
	}
	return out
}

// DetectAsync is the streamed form of [CornersDetector.Detect]. The stream is
// inert in this synchronous port; a nil stream is accepted.
func (d *CornersDetector) DetectAsync(image, mask *GpuMat, _ *Stream) []cv.Point {
	return d.Detect(image, mask)
}

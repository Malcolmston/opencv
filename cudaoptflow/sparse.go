package cudaoptflow

import (
	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/video"
)

// SparsePyrLKOpticalFlow is the CPU-backed mirror of
// cv::cuda::SparsePyrLKOpticalFlow: pyramidal Lucas-Kanade tracking of a sparse
// set of points from one frame to the next. Construct it with
// [NewSparsePyrLKOpticalFlow] and call [SparsePyrLKOpticalFlow.Calc] per frame
// pair.
type SparsePyrLKOpticalFlow struct {
	// WinSize is the side length of the square integration window (odd; rounded
	// up if even). Larger windows tolerate more motion but blur across boundaries.
	WinSize int
	// MaxLevel is the highest 0-based pyramid level (0 disables the pyramid).
	MaxLevel int
	// Iters is retained for API compatibility with OpenCV's iteration-count
	// termination criterion; the underlying solver uses its own fixed iteration
	// schedule, so this field is advisory.
	Iters int
}

// NewSparsePyrLKOpticalFlow creates a sparse pyramidal Lucas-Kanade tracker,
// mirroring cv::cuda::SparsePyrLKOpticalFlow::create(winSize, maxLevel, iters).
// winSize must be >= 1 and maxLevel >= 0. OpenCV's defaults are winSize 21,
// maxLevel 3, iters 30.
func NewSparsePyrLKOpticalFlow(winSize, maxLevel, iters int) *SparsePyrLKOpticalFlow {
	if winSize < 1 {
		panic("cudaoptflow: NewSparsePyrLKOpticalFlow requires winSize >= 1")
	}
	if maxLevel < 0 {
		panic("cudaoptflow: NewSparsePyrLKOpticalFlow requires maxLevel >= 0")
	}
	return &SparsePyrLKOpticalFlow{WinSize: winSize, MaxLevel: maxLevel, Iters: iters}
}

// Calc tracks each point of prevPts from prev to next with pyramidal
// Lucas-Kanade, mirroring cv::cuda::SparsePyrLKOpticalFlow::calc. For every input
// point it returns the estimated new location, a status byte (1 = found, 0 =
// lost, matching OpenCV's CV_8U status) and a tracking error (the RMS intensity
// residual over the window). The three returned slices are parallel to prevPts.
//
// stream is accepted for API compatibility and ignored. prev and next must be
// non-empty and equally sized. Delegates to video.CalcOpticalFlowPyrLK.
func (o *SparsePyrLKOpticalFlow) Calc(prev, next *GpuMat, prevPts []cv.Point, stream *Stream) (nextPts []cv.Point, status []uint8, err []float32) {
	requireFramePair(prev, next, "SparsePyrLKOpticalFlow.Calc")
	_ = stream
	pts, ok, errs := video.CalcOpticalFlowPyrLK(prev.mat, next.mat, prevPts, o.WinSize, o.MaxLevel)
	status = make([]uint8, len(ok))
	err = make([]float32, len(errs))
	for i := range ok {
		if ok[i] {
			status[i] = 1
		}
		err[i] = float32(errs[i])
	}
	return pts, status, err
}

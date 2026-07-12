package cudastereo

import "github.com/malcolmston/opencv/stereo"

// StereoBM is the CPU-backed mirror of cv::cuda::StereoBM: local block matching
// over a rectified pair. It holds the same tuning knobs as the CUDA class and
// delegates the actual matching to the sibling stereo package's
// [github.com/malcolmston/opencv/stereo.StereoBM], operating on [GpuMat] inputs.
//
// Build one with [CreateStereoBM]; fields may then be adjusted before
// [StereoBM.Compute].
type StereoBM struct {
	// NumDisparities is the width of the disparity search range in pixels;
	// disparities d in [0, NumDisparities) are considered. Defaults to 64.
	NumDisparities int
	// BlockSize is the odd side length of the square SAD window. Defaults to 19,
	// the OpenCV CUDA default.
	BlockSize int
	// TextureThreshold rejects low-texture windows whose intensity range falls
	// below it. Defaults to 4 when non-positive.
	TextureThreshold int
	// UniquenessRatio is the percent margin by which the best match must beat the
	// second-best non-adjacent match. Defaults to 10 when non-positive.
	UniquenessRatio int
}

// CreateStereoBM constructs a StereoBM with the given search range and window
// size, mirroring cv::cuda::createStereoBM(numDisparities, blockSize).
// Non-positive arguments fall back to the OpenCV CUDA defaults (64 and 19).
func CreateStereoBM(numDisparities, blockSize int) *StereoBM {
	if numDisparities <= 0 {
		numDisparities = 64
	}
	if blockSize <= 0 {
		blockSize = 19
	}
	return &StereoBM{
		NumDisparities:   numDisparities,
		BlockSize:        blockSize,
		TextureThreshold: 4,
		UniquenessRatio:  10,
	}
}

// Compute matches left against right and returns a single-channel 8-bit
// disparity map as a [GpuMat], mirroring cv::cuda::StereoBM::compute. The stream
// argument is accepted for API compatibility and may be nil; work runs
// synchronously on the CPU regardless.
//
// It panics if either input is empty, the inputs differ in size, or an input has
// an unsupported channel count.
func (bm *StereoBM) Compute(left, right *GpuMat, stream *Stream) *GpuMat {
	_ = stream
	l := matOf(left, "left")
	r := matOf(right, "right")
	disp := stereo.StereoBM{
		NumDisparities:   bm.NumDisparities,
		BlockSize:        bm.BlockSize,
		TextureThreshold: bm.TextureThreshold,
		UniquenessRatio:  bm.UniquenessRatio,
	}.Compute(l, r)
	return &GpuMat{mat: disp}
}

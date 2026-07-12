package cudastereo

import "github.com/malcolmston/opencv/stereo"

// SGMMode selects the aggregation-path layout of [StereoSGM], mirroring the CUDA
// module's MODE_HH4 / MODE_HH flags.
type SGMMode int

const (
	// ModeHH4 aggregates along the four cardinal directions (left, right, up,
	// down). It is the lighter, default mode.
	ModeHH4 SGMMode = iota
	// ModeHH aggregates along all eight directions, adding the four diagonals for
	// higher accuracy at greater cost.
	ModeHH
)

// StereoSGM is the CPU-backed mirror of cv::cuda::StereoSGM: semi-global
// matching over a rectified pair. It delegates to the sibling stereo package's
// [github.com/malcolmston/opencv/stereo.StereoSGM].
//
// Build one with [CreateStereoSGM]; fields may then be adjusted before
// [StereoSGM.Compute].
type StereoSGM struct {
	// MinDisparity is the smallest disparity searched (usually 0).
	MinDisparity int
	// NumDisparities is the width of the search range; disparities d in
	// [MinDisparity, MinDisparity+NumDisparities) are considered. Defaults to 128.
	NumDisparities int
	// BlockSize is the odd side length of the matching window. Defaults to 5.
	BlockSize int
	// P1 penalises a disparity change of one between neighbours. Defaults to
	// 8*BlockSize when non-positive.
	P1 int
	// P2 penalises larger disparity jumps and must exceed P1. Defaults to
	// 32*BlockSize when non-positive.
	P2 int
	// UniquenessRatio is the percent margin by which the best aggregated cost must
	// beat the second-best non-adjacent cost. Defaults to 5.
	UniquenessRatio int
	// Mode selects the aggregation-path layout. The zero value is [ModeHH4].
	Mode SGMMode
}

// CreateStereoSGM constructs a StereoSGM, mirroring
// cv::cuda::createStereoSGM(minDisparity, numDisparities, P1, P2,
// uniquenessRatio, mode). Non-positive numDisparities falls back to 128.
func CreateStereoSGM(minDisparity, numDisparities, p1, p2, uniquenessRatio int, mode SGMMode) *StereoSGM {
	if numDisparities <= 0 {
		numDisparities = 128
	}
	if uniquenessRatio <= 0 {
		uniquenessRatio = 5
	}
	return &StereoSGM{
		MinDisparity:    minDisparity,
		NumDisparities:  numDisparities,
		BlockSize:       5,
		P1:              p1,
		P2:              p2,
		UniquenessRatio: uniquenessRatio,
		Mode:            mode,
	}
}

// Compute matches left against right and returns a single-channel 8-bit
// disparity map as a [GpuMat], mirroring cv::cuda::StereoSGM::compute. The stream
// argument is accepted for API compatibility and may be nil.
//
// It panics if either input is empty, the inputs differ in size, an input has an
// unsupported channel count, or BlockSize is not a positive odd integer.
func (s *StereoSGM) Compute(left, right *GpuMat, stream *Stream) *GpuMat {
	_ = stream
	l := matOf(left, "left")
	r := matOf(right, "right")
	mode := stereo.ModeSGBM
	if s.Mode == ModeHH {
		mode = stereo.ModeHH
	}
	disp := stereo.StereoSGM{
		MinDisparity:    s.MinDisparity,
		NumDisparities:  s.NumDisparities,
		BlockSize:       s.BlockSize,
		P1:              s.P1,
		P2:              s.P2,
		UniquenessRatio: s.UniquenessRatio,
		Disp12MaxDiff:   -1,
		Mode:            mode,
	}.Compute(l, r)
	return &GpuMat{mat: disp}
}

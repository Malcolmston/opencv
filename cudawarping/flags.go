package cudawarping

import cv "github.com/malcolmston/opencv"

// Interpolation selects the resampling method used by the warp operations. The
// numeric values match OpenCV's cv::InterpolationFlags so that constants copied
// from OpenCV code keep their meaning.
type Interpolation int

const (
	// InterNearest picks the nearest source sample (fast, blocky).
	InterNearest Interpolation = 0
	// InterLinear uses bilinear interpolation of the four nearest samples.
	InterLinear Interpolation = 1
	// InterCubic uses bicubic interpolation over a 4×4 neighbourhood
	// (Catmull-Rom style with OpenCV's a = -0.75).
	InterCubic Interpolation = 2
	// InterArea resamples using pixel-area relation; for shrinking it averages
	// the covered source pixels (moiré-free), and for enlarging it falls back to
	// nearest-neighbour, matching cv::INTER_AREA.
	InterArea Interpolation = 3
)

// interMask isolates the interpolation bits of a combined flags value, matching
// OpenCV's INTER_MAX-based masking (the low bits carry the interpolation, the
// high bits carry the WARP_* flags).
const interMask = 7

// Warp flags may be OR-ed with an [Interpolation] and passed to [GpuMat.WarpAffine]
// and [GpuMat.WarpPerspective], matching cv::WarpPolarMode / cv::InterpolationFlags.
const (
	// WarpFillOutliers is accepted for source compatibility; outliers are always
	// filled from the selected border mode, so the flag has no additional effect.
	WarpFillOutliers = 8
	// WarpInverseMap indicates that the supplied transform already maps
	// destination pixels to source pixels (dst→src), so it is used directly
	// instead of being inverted. This mirrors cv::WARP_INVERSE_MAP.
	WarpInverseMap = 16
)

// BorderMode selects how samples outside the source image are produced, matching
// OpenCV's cv::BorderTypes numeric values.
type BorderMode int

const (
	// BorderConstant pads with a fixed value (fedcba|abcdefgh|hgfedcb → 000000|abcdefgh|000000).
	BorderConstant BorderMode = 0
	// BorderReplicate repeats the edge sample (aaaaaa|abcdefgh|hhhhhhh).
	BorderReplicate BorderMode = 1
	// BorderReflect mirrors including the edge sample (fedcba|abcdefgh|hgfedcb).
	BorderReflect BorderMode = 2
	// BorderWrap tiles the image (cdefgh|abcdefgh|abcdefg).
	BorderWrap BorderMode = 3
	// BorderReflect101 mirrors excluding the edge sample (gfedcb|abcdefgh|gfedcba).
	BorderReflect101 BorderMode = 4
)

// Polar mode flags select the radial mapping for [GpuMat.WarpPolar] and may be
// OR-ed with [WarpInverseMap]. The values match cv::WarpPolarMode.
const (
	// WarpPolarLinear maps the radius linearly (semilog off).
	WarpPolarLinear = 0
	// WarpPolarLog maps the radius logarithmically (semilog on).
	WarpPolarLog = 256
)

// RotateCode selects one of the three lossless right-angle rotations for
// [GpuMat.Rotate90]. It aliases [cv.RotateCode] so the root package's constants
// can be used directly.
type RotateCode = cv.RotateCode

// Right-angle rotation codes, re-exported from the root package.
const (
	// Rotate90CW rotates 90° clockwise.
	Rotate90CW = cv.Rotate90CW
	// Rotate180 rotates 180°.
	Rotate180 = cv.Rotate180
	// Rotate90CCW rotates 90° counter-clockwise.
	Rotate90CCW = cv.Rotate90CCW
)

// Point2f is a sub-pixel image coordinate (x is the column, y is the row), used
// as the centre for the polar warps.
type Point2f struct {
	X float64
	Y float64
}

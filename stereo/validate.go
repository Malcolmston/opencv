package stereo

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// ValidateDisparity enforces the left-right consistency constraint between a
// left-referenced and a right-referenced disparity map, mirroring OpenCV's
// cv::validateDisparity. A left pixel x with disparity dL is kept only if its
// corresponding right pixel xr = x - dL carries a disparity dR that agrees to
// within disp12MaxDiff pixels; otherwise it is an occlusion or mismatch and is
// set to newVal. Pixels already holding [InvalidDisparity], and those whose
// match falls outside the image, are likewise set to newVal.
//
// dispLeft is not modified; the validated copy is returned. Both maps must be
// single-channel and the same size. A negative disp12MaxDiff disables the check
// and returns a clone of dispLeft. It panics on nil, empty, multi-channel or
// mismatched inputs.
func ValidateDisparity(dispLeft, dispRight *cv.Mat, disp12MaxDiff int, newVal uint8) *cv.Mat {
	if dispLeft == nil || dispLeft.Empty() || dispRight == nil || dispRight.Empty() {
		panic("stereo: ValidateDisparity given a nil or empty disparity map")
	}
	if dispLeft.Channels != 1 || dispRight.Channels != 1 {
		panic("stereo: ValidateDisparity requires single-channel disparity maps")
	}
	if dispLeft.Rows != dispRight.Rows || dispLeft.Cols != dispRight.Cols {
		panic(fmt.Sprintf("stereo: ValidateDisparity size mismatch %dx%d vs %dx%d",
			dispLeft.Rows, dispLeft.Cols, dispRight.Rows, dispRight.Cols))
	}
	out := dispLeft.Clone()
	if disp12MaxDiff < 0 {
		return out
	}
	rows, cols := out.Rows, out.Cols
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := y*cols + x
			dL := int(out.Data[p])
			if out.Data[p] == InvalidDisparity {
				out.Data[p] = newVal
				continue
			}
			xr := x - dL
			if xr < 0 {
				out.Data[p] = newVal
				continue
			}
			dR := int(dispRight.Data[y*cols+xr])
			if dispRight.Data[y*cols+xr] == InvalidDisparity || absInt(dL-dR) > disp12MaxDiff {
				out.Data[p] = newVal
			}
		}
	}
	return out
}

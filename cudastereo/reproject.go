package cudastereo

import (
	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/stereo"
)

// ReprojectImageTo3D maps a disparity [GpuMat] to 3-D coordinates through the 4×4
// reprojection matrix Q, mirroring cv::cuda::reprojectImageTo3D. It delegates to
// [github.com/malcolmston/opencv/stereo.ReprojectImageTo3D]: for each pixel at
// column x, row y with disparity d it forms [x, y, d, 1], multiplies by Q, and
// divides by the resulting w component. The result is a row-major slice of length
// Rows*Cols whose entry (y*Cols + x) holds the (X, Y, Z) coordinates.
//
// The stream argument is accepted for API compatibility and may be nil. It panics
// if disparity is empty or not single-channel.
func ReprojectImageTo3D(disparity *GpuMat, Q [4][4]float64, stream *Stream) [][3]float64 {
	_ = stream
	return stereo.ReprojectImageTo3D(matOf(disparity, "disparity"), Q)
}

// DrawColorDisp renders a disparity [GpuMat] as a three-channel RGB colour image
// for visualisation, mirroring cv::cuda::drawColorDisp. Each disparity is mapped
// to a hue (near surfaces warm, far surfaces cool) at full saturation and value;
// pixels holding [github.com/malcolmston/opencv/stereo.InvalidDisparity] (0) are
// drawn black. If ndisp is non-positive it is inferred from the maximum disparity
// present.
//
// The returned GpuMat is a 3-channel RGB image the same size as the input. The
// stream argument is accepted for API compatibility and may be nil. It panics if
// disparity is empty or not single-channel.
func DrawColorDisp(disparity *GpuMat, ndisp int, stream *Stream) *GpuMat {
	_ = stream
	disp := matOf(disparity, "disparity")
	if disp.Channels != 1 {
		panic("cudastereo: DrawColorDisp requires a single-channel disparity map")
	}
	rows, cols := disp.Rows, disp.Cols

	if ndisp <= 0 {
		maxD := 0
		for _, v := range disp.Data {
			if int(v) > maxD {
				maxD = int(v)
			}
		}
		ndisp = maxD + 1
	}

	hsv := cv.NewMat(rows, cols, 3)
	for i, v := range disp.Data {
		base := i * 3
		if v == 0 {
			// Black for no-match pixels.
			continue
		}
		// Map disparity to hue in [0, 120]: near (large disparity) -> red (0),
		// far (small disparity) -> green/cyan. OpenCV's HSV hue channel is [0,179].
		frac := float64(v) / float64(ndisp)
		if frac > 1 {
			frac = 1
		}
		hue := int((1 - frac) * 120)
		hsv.Data[base+0] = uint8(clampInt(hue, 0, 179))
		hsv.Data[base+1] = 255
		hsv.Data[base+2] = 255
	}
	rgb := cv.CvtColor(hsv, cv.ColorHSV2RGB)
	return &GpuMat{mat: rgb}
}

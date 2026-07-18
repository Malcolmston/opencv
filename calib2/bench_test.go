package calib2

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// BenchmarkCalibrateCamera exercises the heaviest routine in the package: a full
// Zhang calibration (per-view homographies, the intrinsic null-space solve,
// per-view extrinsics and distortion fit) over several synthetic views.
func BenchmarkCalibrateCamera(b *testing.B) {
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	obj, img, _ := synthCalibViews(k, DistortionCoeffs{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, _, err := CalibrateCamera(obj, img); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkUndistortImage measures the per-pixel undistortion resampling path.
func BenchmarkUndistortImage(b *testing.B) {
	k := CameraMatrix{Fx: 300, Fy: 300, Cx: 160, Cy: 120}
	d := DistortionCoeffs{K1: -0.2, K2: 0.05}
	src := cv.NewMat(240, 320, 3)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			src.Set(y, x, 0, uint8(x))
			src.Set(y, x, 1, uint8(y))
			src.Set(y, x, 2, uint8(x+y))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		UndistortImage(src, k, d)
	}
}

package imghash2

import cv "github.com/malcolmston/opencv"

// ToGray returns a single-channel version of img. A one-channel image is
// returned unchanged (no copy); a three-channel image is reduced with the
// BT.601 luma weights via [cv.CvtColor]; any other channel count is averaged
// across its channels. The result never aliases a multi-channel input.
func ToGray(img *cv.Mat) *cv.Mat {
	requireImage(img, "ToGray")
	switch img.Channels {
	case 1:
		return img
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		gray := cv.NewMat(img.Rows, img.Cols, 1)
		for p := 0; p < img.Total(); p++ {
			var sum int
			base := p * img.Channels
			for c := 0; c < img.Channels; c++ {
				sum += int(img.Data[base+c])
			}
			gray.Data[p] = uint8((sum + img.Channels/2) / img.Channels)
		}
		return gray
	}
}

// GrayResize converts img to grayscale with [ToGray] and rescales it to
// width×height using bilinear interpolation, the common front end of every
// hasher. It panics if img is nil or empty or if either dimension is not
// positive.
func GrayResize(img *cv.Mat, width, height int) *cv.Mat {
	requireImage(img, "GrayResize")
	return cv.Resize(ToGray(img), width, height, cv.InterLinear)
}

// grayFloats converts img to a width×height grayscale image and returns its
// samples as a fresh row-major float64 slice, the shared input to the transform
// hashers.
func grayFloats(img *cv.Mat, width, height int) []float64 {
	small := GrayResize(img, width, height)
	out := make([]float64, width*height)
	for i := range out {
		out[i] = float64(small.Data[i])
	}
	return out
}

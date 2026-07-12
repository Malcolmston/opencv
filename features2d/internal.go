package features2d

import cv "github.com/malcolmston/opencv"

// toGray returns a single-channel view of img. A 1-channel Mat is returned
// unchanged (cloned); a 3-channel Mat is converted with the BT.601 luma weights
// via cv.CvtColor. It panics for other channel counts.
func toGray(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return img.Clone()
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		panic("features2d: expected a 1- or 3-channel image")
	}
}

// toColor returns a 3-channel copy of img so drawing primitives can use colour.
func toColor(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 3:
		return img.Clone()
	case 1:
		return cv.CvtColor(img, cv.ColorGray2RGB)
	default:
		panic("features2d: expected a 1- or 3-channel image")
	}
}

// sampleClamped returns the sample of a single-channel Mat at (x, y), clamping
// out-of-range coordinates to the nearest edge. It lets descriptor sampling near
// the image border proceed without panicking.
func sampleClamped(m *cv.Mat, x, y int) int {
	if x < 0 {
		x = 0
	} else if x >= m.Cols {
		x = m.Cols - 1
	}
	if y < 0 {
		y = 0
	} else if y >= m.Rows {
		y = m.Rows - 1
	}
	return int(m.Data[y*m.Cols+x])
}

package freqdomain

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MatToFloat converts an 8-bit cv.Mat to a single-channel cv.FloatMat of
// samples in [0,255]. A single-channel matrix is copied directly; a
// three-channel matrix is converted to luma with the Rec. 601 weights
// 0.299R+0.587G+0.114B; any other channel count is averaged across channels. It
// panics on an empty matrix.
func MatToFloat(m *cv.Mat) *cv.FloatMat {
	if m == nil || m.Empty() {
		panic("freqdomain: MatToFloat on empty matrix")
	}
	out := cv.NewFloatMat(m.Rows, m.Cols)
	switch m.Channels {
	case 1:
		for i := range out.Data {
			out.Data[i] = float64(m.Data[i])
		}
	case 3:
		for p := 0; p < m.Rows*m.Cols; p++ {
			r := float64(m.Data[p*3+0])
			g := float64(m.Data[p*3+1])
			b := float64(m.Data[p*3+2])
			out.Data[p] = 0.299*r + 0.587*g + 0.114*b
		}
	default:
		ch := m.Channels
		for p := 0; p < m.Rows*m.Cols; p++ {
			var sum float64
			for c := 0; c < ch; c++ {
				sum += float64(m.Data[p*ch+c])
			}
			out.Data[p] = sum / float64(ch)
		}
	}
	return out
}

// FloatToMat converts a single-channel cv.FloatMat to an 8-bit cv.Mat by
// rounding and clamping each sample to [0,255]. Use [FloatToMatScaled] first if
// the data must be linearly rescaled into range.
func FloatToMat(f *cv.FloatMat) *cv.Mat {
	out := cv.NewMat(f.Rows, f.Cols, 1)
	for i, v := range f.Data {
		out.Data[i] = clampByte(v)
	}
	return out
}

// FloatToMatScaled linearly rescales a cv.FloatMat so its minimum maps to 0 and
// its maximum to 255, then rounds to an 8-bit cv.Mat. A constant image maps to
// all zeros. This is the standard way to display a filtered float result whose
// range is unknown.
func FloatToMatScaled(f *cv.FloatMat) *cv.Mat {
	return normalizeToMat(f)
}

// clampByte rounds v and clamps it to the 8-bit range [0,255].
func clampByte(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

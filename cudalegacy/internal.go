package cudalegacy

import (
	cv "github.com/malcolmston/opencv"
)

// grayPlane reduces a frame to a per-pixel float intensity plane. A
// single-channel frame is copied through; a three-channel frame is converted to
// BT.601 luma; any other channel count is averaged.
func grayPlane(m *cv.Mat) *cv.FloatMat {
	out := cv.NewFloatMat(m.Rows, m.Cols)
	n := m.Rows * m.Cols
	switch m.Channels {
	case 1:
		for p := 0; p < n; p++ {
			out.Data[p] = float64(m.Data[p])
		}
	case 3:
		for p := 0; p < n; p++ {
			b := p * 3
			out.Data[p] = 0.299*float64(m.Data[b]) + 0.587*float64(m.Data[b+1]) + 0.114*float64(m.Data[b+2])
		}
	default:
		ch := m.Channels
		for p := 0; p < n; p++ {
			b := p * ch
			sum := 0.0
			for c := 0; c < ch; c++ {
				sum += float64(m.Data[b+c])
			}
			out.Data[p] = sum / float64(ch)
		}
	}
	return out
}

// bilerpMat bilinearly samples channel c of a uint8 Mat at (fx, fy), clamping to
// the border, and returns a float in [0,255].
func bilerpMat(m *cv.Mat, fx, fy float64, c int) float64 {
	if fx < 0 {
		fx = 0
	} else if fx > float64(m.Cols-1) {
		fx = float64(m.Cols - 1)
	}
	if fy < 0 {
		fy = 0
	} else if fy > float64(m.Rows-1) {
		fy = float64(m.Rows - 1)
	}
	x0 := int(fx)
	y0 := int(fy)
	x1 := x0 + 1
	if x1 > m.Cols-1 {
		x1 = m.Cols - 1
	}
	y1 := y0 + 1
	if y1 > m.Rows-1 {
		y1 = m.Rows - 1
	}
	ax := fx - float64(x0)
	ay := fy - float64(y0)
	ch := m.Channels
	v00 := float64(m.Data[(y0*m.Cols+x0)*ch+c])
	v01 := float64(m.Data[(y0*m.Cols+x1)*ch+c])
	v10 := float64(m.Data[(y1*m.Cols+x0)*ch+c])
	v11 := float64(m.Data[(y1*m.Cols+x1)*ch+c])
	top := v00 + ax*(v01-v00)
	bot := v10 + ax*(v11-v10)
	return top + ay*(bot-top)
}

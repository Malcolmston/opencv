package videoproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// videoprocToGray returns a single-channel grayscale view of m. A frame that is
// already single-channel is returned unchanged (not copied); a multi-channel
// frame is converted with the Rec. 601 luma weights, treating the first three
// channels as R, G, B. Frames with two channels use channel 0.
func videoprocToGray(m *cv.Mat) *cv.Mat {
	if m == nil || m.Empty() {
		panic("videoproc: nil or empty frame")
	}
	if m.Channels == 1 {
		return m
	}
	out := cv.NewMat(m.Rows, m.Cols, 1)
	n := m.Total()
	if m.Channels < 3 {
		for i := 0; i < n; i++ {
			out.Data[i] = m.Data[i*m.Channels]
		}
		return out
	}
	for i := 0; i < n; i++ {
		base := i * m.Channels
		r := float64(m.Data[base])
		g := float64(m.Data[base+1])
		b := float64(m.Data[base+2])
		out.Data[i] = videoprocClampU8(0.299*r + 0.587*g + 0.114*b + 0.5)
	}
	return out
}

// videoprocClampU8 rounds and clamps v into the 0..255 uint8 range.
func videoprocClampU8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// videoprocSameSize reports whether a and b share Rows, Cols and Channels.
func videoprocSameSize(a, b *cv.Mat) bool {
	return a.Rows == b.Rows && a.Cols == b.Cols && a.Channels == b.Channels
}

// videoprocRequireSame panics unless a and b share dimensions and channels.
func videoprocRequireSame(fn string, a, b *cv.Mat) {
	if a == nil || b == nil || a.Empty() || b.Empty() {
		panic("videoproc: " + fn + " requires two non-empty frames")
	}
	if !videoprocSameSize(a, b) {
		panic("videoproc: " + fn + " frame size/channel mismatch")
	}
}

// videoprocGrayAtClamp returns the intensity of gray Mat g at integer (x, y)
// with edge clamping (BORDER_REPLICATE). g must be single-channel.
func videoprocGrayAtClamp(g *cv.Mat, x, y int) float64 {
	if x < 0 {
		x = 0
	} else if x >= g.Cols {
		x = g.Cols - 1
	}
	if y < 0 {
		y = 0
	} else if y >= g.Rows {
		y = g.Rows - 1
	}
	return float64(g.Data[y*g.Cols+x])
}

// videoprocSampleBilinear samples channel c of m at fractional (x, y) with
// bilinear interpolation and edge clamping.
func videoprocSampleBilinear(m *cv.Mat, x, y float64, c int) float64 {
	x0f := math.Floor(x)
	y0f := math.Floor(y)
	x0 := int(x0f)
	y0 := int(y0f)
	dx := x - x0f
	dy := y - y0f
	p00 := videoprocChanAtClamp(m, x0, y0, c)
	p10 := videoprocChanAtClamp(m, x0+1, y0, c)
	p01 := videoprocChanAtClamp(m, x0, y0+1, c)
	p11 := videoprocChanAtClamp(m, x0+1, y0+1, c)
	top := p00*(1-dx) + p10*dx
	bot := p01*(1-dx) + p11*dx
	return top*(1-dy) + bot*dy
}

// videoprocChanAtClamp returns channel c of m at integer (x, y) with clamping.
func videoprocChanAtClamp(m *cv.Mat, x, y, c int) float64 {
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
	return float64(m.Data[(y*m.Cols+x)*m.Channels+c])
}

// videoprocMedianU8 returns the median of a slice of bytes, sorting a scratch
// copy. It returns 0 for an empty slice.
func videoprocMedianU8(vals []uint8) uint8 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	scratch := make([]uint8, n)
	copy(scratch, vals)
	videoprocInsertionSortU8(scratch)
	return scratch[n/2]
}

// videoprocInsertionSortU8 sorts s ascending in place (small slices only).
func videoprocInsertionSortU8(s []uint8) {
	for i := 1; i < len(s); i++ {
		v := s[i]
		j := i - 1
		for j >= 0 && s[j] > v {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = v
	}
}

// videoprocMedianF returns the median of a float slice using a scratch copy.
func videoprocMedianF(vals []float64) float64 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	scratch := make([]float64, n)
	copy(scratch, vals)
	// simple insertion sort; slices are small in this package's use
	for i := 1; i < n; i++ {
		v := scratch[i]
		j := i - 1
		for j >= 0 && scratch[j] > v {
			scratch[j+1] = scratch[j]
			j--
		}
		scratch[j+1] = v
	}
	if n%2 == 1 {
		return scratch[n/2]
	}
	return 0.5 * (scratch[n/2-1] + scratch[n/2])
}

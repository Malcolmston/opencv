package pyramids

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// pyramidsAt returns the value of f at (y, x), replicating the nearest edge
// sample for out-of-range coordinates (BORDER_REPLICATE).
func pyramidsAt(f *cv.FloatMat, y, x int) float64 {
	if y < 0 {
		y = 0
	} else if y >= f.Rows {
		y = f.Rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= f.Cols {
		x = f.Cols - 1
	}
	return f.Data[y*f.Cols+x]
}

// pyramidsSet stores v at (y, x) of f without bounds checking.
func pyramidsSet(f *cv.FloatMat, y, x int, v float64) {
	f.Data[y*f.Cols+x] = v
}

// pyramidsRequire panics unless f is a usable (non-nil, non-empty) FloatMat.
func pyramidsRequire(f *cv.FloatMat, op string) {
	if f == nil || f.Rows <= 0 || f.Cols <= 0 || len(f.Data) != f.Rows*f.Cols {
		panic("pyramids: " + op + ": nil or malformed FloatMat")
	}
}

// pyramidsSameSize panics unless a and b share dimensions.
func pyramidsSameSize(a, b *cv.FloatMat, op string) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("pyramids: " + op + ": size mismatch")
	}
}

// GrayFloat converts an 8-bit [cv.Mat] to a single-channel float64 grid. A
// three-channel image is reduced with the Rec.601 luma weights
// (0.299, 0.587, 0.114); a single-channel image is copied directly; any other
// channel count is averaged. Values keep the 0..255 scale of the source.
func GrayFloat(m *cv.Mat) *cv.FloatMat {
	if m == nil || m.Empty() {
		panic("pyramids: GrayFloat: empty source matrix")
	}
	out := cv.NewFloatMat(m.Rows, m.Cols)
	switch m.Channels {
	case 1:
		for i, v := range m.Data {
			out.Data[i] = float64(v)
		}
	case 3:
		for p := 0; p < m.Rows*m.Cols; p++ {
			b := p * 3
			out.Data[p] = 0.299*float64(m.Data[b]) + 0.587*float64(m.Data[b+1]) + 0.114*float64(m.Data[b+2])
		}
	default:
		ch := m.Channels
		for p := 0; p < m.Rows*m.Cols; p++ {
			var s float64
			for c := 0; c < ch; c++ {
				s += float64(m.Data[p*ch+c])
			}
			out.Data[p] = s / float64(ch)
		}
	}
	return out
}

// ChannelFloat extracts channel c of an 8-bit [cv.Mat] as a float64 grid,
// keeping the 0..255 scale. It panics if c is out of range.
func ChannelFloat(m *cv.Mat, c int) *cv.FloatMat {
	if m == nil || m.Empty() {
		panic("pyramids: ChannelFloat: empty source matrix")
	}
	if c < 0 || c >= m.Channels {
		panic("pyramids: ChannelFloat: channel out of range")
	}
	out := cv.NewFloatMat(m.Rows, m.Cols)
	ch := m.Channels
	for p := 0; p < m.Rows*m.Cols; p++ {
		out.Data[p] = float64(m.Data[p*ch+c])
	}
	return out
}

// SplitFloat separates every channel of an 8-bit [cv.Mat] into its own float64
// grid, returning them in channel order. It is the inverse of [MergeFloat].
func SplitFloat(m *cv.Mat) []*cv.FloatMat {
	if m == nil || m.Empty() {
		panic("pyramids: SplitFloat: empty source matrix")
	}
	planes := make([]*cv.FloatMat, m.Channels)
	for c := range planes {
		planes[c] = ChannelFloat(m, c)
	}
	return planes
}

// MergeFloat combines equally sized float64 grids into an interleaved 8-bit
// [cv.Mat], one grid per channel, rounding and clamping each value to 0..255.
// It panics if planes is empty or the grids differ in size.
func MergeFloat(planes []*cv.FloatMat) *cv.Mat {
	if len(planes) == 0 {
		panic("pyramids: MergeFloat: no planes")
	}
	rows, cols := planes[0].Rows, planes[0].Cols
	for _, p := range planes {
		pyramidsRequire(p, "MergeFloat")
		if p.Rows != rows || p.Cols != cols {
			panic("pyramids: MergeFloat: size mismatch")
		}
	}
	ch := len(planes)
	out := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		src := planes[c].Data
		for p := 0; p < rows*cols; p++ {
			out.Data[p*ch+c] = pyramidsClampU8(src[p])
		}
	}
	return out
}

// pyramidsClampU8 rounds v to the nearest integer and clamps it to 0..255.
func pyramidsClampU8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// FloatToMat converts a single-channel float64 grid to an 8-bit grayscale
// [cv.Mat], rounding and clamping each value to the 0..255 range.
func FloatToMat(f *cv.FloatMat) *cv.Mat {
	pyramidsRequire(f, "FloatToMat")
	out := cv.NewMat(f.Rows, f.Cols, 1)
	for i, v := range f.Data {
		out.Data[i] = pyramidsClampU8(v)
	}
	return out
}

// FloatToMatNormalized converts a float64 grid to an 8-bit grayscale [cv.Mat]
// after linearly rescaling its actual minimum to 0 and maximum to 255. It is
// intended for visualising signed data such as Laplacian or wavelet detail
// bands. A constant grid maps to all zeros.
func FloatToMatNormalized(f *cv.FloatMat) *cv.Mat {
	pyramidsRequire(f, "FloatToMatNormalized")
	lo, hi := f.Data[0], f.Data[0]
	for _, v := range f.Data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	out := cv.NewMat(f.Rows, f.Cols, 1)
	if hi <= lo {
		return out
	}
	scale := 255.0 / (hi - lo)
	for i, v := range f.Data {
		out.Data[i] = pyramidsClampU8((v - lo) * scale)
	}
	return out
}

// CloneFloat returns an independent deep copy of a float64 grid.
func CloneFloat(f *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(f, "CloneFloat")
	out := cv.NewFloatMat(f.Rows, f.Cols)
	copy(out.Data, f.Data)
	return out
}

// AddFloat returns the element-wise sum a+b of two equally sized grids.
func AddFloat(a, b *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(a, "AddFloat")
	pyramidsRequire(b, "AddFloat")
	pyramidsSameSize(a, b, "AddFloat")
	out := cv.NewFloatMat(a.Rows, a.Cols)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

// SubtractFloat returns the element-wise difference a-b of two equally sized
// grids. The result may be negative, which is why pyramid bands are held in
// the float domain.
func SubtractFloat(a, b *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(a, "SubtractFloat")
	pyramidsRequire(b, "SubtractFloat")
	pyramidsSameSize(a, b, "SubtractFloat")
	out := cv.NewFloatMat(a.Rows, a.Cols)
	for i := range out.Data {
		out.Data[i] = a.Data[i] - b.Data[i]
	}
	return out
}

// ScaleFloat returns a new grid with every element of f multiplied by s.
func ScaleFloat(f *cv.FloatMat, s float64) *cv.FloatMat {
	pyramidsRequire(f, "ScaleFloat")
	out := cv.NewFloatMat(f.Rows, f.Cols)
	for i, v := range f.Data {
		out.Data[i] = v * s
	}
	return out
}

// AddScaledFloat returns a + s*b for two equally sized grids, a fused
// multiply-add convenient for blending and reconstruction.
func AddScaledFloat(a *cv.FloatMat, s float64, b *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(a, "AddScaledFloat")
	pyramidsRequire(b, "AddScaledFloat")
	pyramidsSameSize(a, b, "AddScaledFloat")
	out := cv.NewFloatMat(a.Rows, a.Cols)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + s*b.Data[i]
	}
	return out
}

// AbsFloat returns a new grid holding the absolute value of every element of f.
func AbsFloat(f *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(f, "AbsFloat")
	out := cv.NewFloatMat(f.Rows, f.Cols)
	for i, v := range f.Data {
		out.Data[i] = math.Abs(v)
	}
	return out
}

// NormalizeFloat linearly rescales f so its minimum maps to lo and its maximum
// to hi, returning a new grid. A constant input maps entirely to lo.
func NormalizeFloat(f *cv.FloatMat, lo, hi float64) *cv.FloatMat {
	pyramidsRequire(f, "NormalizeFloat")
	mn, mx := f.Data[0], f.Data[0]
	for _, v := range f.Data {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	out := cv.NewFloatMat(f.Rows, f.Cols)
	if mx <= mn {
		for i := range out.Data {
			out.Data[i] = lo
		}
		return out
	}
	scale := (hi - lo) / (mx - mn)
	for i, v := range f.Data {
		out.Data[i] = lo + (v-mn)*scale
	}
	return out
}

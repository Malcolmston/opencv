package texture

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// textureRequire panics unless m is a non-empty image.
func textureRequire(m *cv.Mat, name string) {
	if m.Empty() {
		panic(fmt.Sprintf("texture: %s given an empty image", name))
	}
}

// textureLuma reduces img to a single 8-bit brightness plane of length
// Rows*Cols. A one-channel image is copied as-is; a three-channel image is
// combined with the Rec. 601 weights; any other channel count uses channel 0.
func textureLuma(img *cv.Mat) []uint8 {
	n := img.Total()
	ch := img.Channels
	out := make([]uint8, n)
	switch ch {
	case 1:
		copy(out, img.Data[:n])
	case 3:
		for p := 0; p < n; p++ {
			base := p * 3
			r := float64(img.Data[base+0])
			g := float64(img.Data[base+1])
			b := float64(img.Data[base+2])
			out[p] = textureClampU8(0.299*r + 0.587*g + 0.114*b + 0.5)
		}
	default:
		for p := 0; p < n; p++ {
			out[p] = img.Data[p*ch]
		}
	}
	return out
}

// textureLumaFloat is like textureLuma but returns unclamped float luminance in
// [0,255], avoiding the rounding of the uint8 path where full precision helps
// (Tamura, Gabor, fractal).
func textureLumaFloat(img *cv.Mat) []float64 {
	n := img.Total()
	ch := img.Channels
	out := make([]float64, n)
	switch ch {
	case 1:
		for p := 0; p < n; p++ {
			out[p] = float64(img.Data[p])
		}
	case 3:
		for p := 0; p < n; p++ {
			base := p * 3
			r := float64(img.Data[base+0])
			g := float64(img.Data[base+1])
			b := float64(img.Data[base+2])
			out[p] = 0.299*r + 0.587*g + 0.114*b
		}
	default:
		for p := 0; p < n; p++ {
			out[p] = float64(img.Data[p*ch])
		}
	}
	return out
}

// textureQuantize maps an 8-bit luminance plane onto integer gray levels in
// [0, levels-1]. The map is uniform over the full [0,255] input range, so
// value v yields level min(levels-1, v*levels/256).
func textureQuantize(luma []uint8, levels int) []int {
	out := make([]int, len(luma))
	for i, v := range luma {
		q := int(v) * levels / 256
		if q >= levels {
			q = levels - 1
		}
		out[i] = q
	}
	return out
}

// textureClampU8 rounds toward zero after the caller's bias and clamps to
// [0,255], matching the root package's clampToUint8 semantics.
func textureClampU8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// textureField is a plain row-major grid of float64 samples, used internally as
// a filter response before it is reduced to scalar features. It is exported as
// nothing; helper reducers below operate on it.
type textureField struct {
	rows, cols int
	data       []float64
}

func textureNewField(rows, cols int) *textureField {
	return &textureField{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

func (f *textureField) at(y, x int) float64 { return f.data[y*f.cols+x] }

// textureMeanAbs returns the mean of |v| over a field, the standard Laws/Gabor
// energy reducer.
func textureMeanAbs(f *textureField) float64 {
	if len(f.data) == 0 {
		return 0
	}
	var s float64
	for _, v := range f.data {
		s += math.Abs(v)
	}
	return s / float64(len(f.data))
}

// textureStd returns the population standard deviation of a field.
func textureStd(f *textureField) float64 {
	n := float64(len(f.data))
	if n == 0 {
		return 0
	}
	var s, s2 float64
	for _, v := range f.data {
		s += v
		s2 += v * v
	}
	mean := s / n
	varr := s2/n - mean*mean
	if varr < 0 {
		varr = 0
	}
	return math.Sqrt(varr)
}

// textureReflect maps an index into [0, n) by mirror ("reflect-101") boundary
// handling, so that out-of-range convolution taps reuse interior samples.
func textureReflect(i, n int) int {
	if n == 1 {
		return 0
	}
	for i < 0 || i >= n {
		if i < 0 {
			i = -i
		}
		if i >= n {
			i = 2*(n-1) - i
		}
	}
	return i
}

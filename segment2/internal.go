package segment2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// segment2offset is a signed neighbour displacement in image coordinates.
type segment2offset struct{ dx, dy int }

// segment2neighbors4 is the 4-connected neighbourhood (edge neighbours).
var segment2neighbors4 = []segment2offset{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}

// segment2neighbors8 is the 8-connected neighbourhood (edge and corner
// neighbours).
var segment2neighbors8 = []segment2offset{
	{1, 0}, {-1, 0}, {0, 1}, {0, -1},
	{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
}

// segment2clampU8 rounds v to the nearest integer and clamps it into [0, 255].
func segment2clampU8(v float64) uint8 {
	v += 0.5
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// segment2colorAt returns the channel samples of pixel (x, y) of m as float64.
// The returned slice is freshly allocated and safe to retain.
func segment2colorAt(m *cv.Mat, x, y int) []float64 {
	base := (y*m.Cols + x) * m.Channels
	out := make([]float64, m.Channels)
	for c := 0; c < m.Channels; c++ {
		out[c] = float64(m.Data[base+c])
	}
	return out
}

// segment2colorInto writes the channel samples of pixel (x, y) into dst, which
// must have length m.Channels.
func segment2colorInto(m *cv.Mat, x, y int, dst []float64) {
	base := (y*m.Cols + x) * m.Channels
	for c := 0; c < m.Channels; c++ {
		dst[c] = float64(m.Data[base+c])
	}
}

// segment2colorDist2 returns the squared Euclidean distance between two colour
// vectors of equal length.
func segment2colorDist2(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// segment2colorDist returns the Euclidean distance between two colour vectors.
func segment2colorDist(a, b []float64) float64 {
	return math.Sqrt(segment2colorDist2(a, b))
}

// segment2gray converts m to a per-pixel intensity slice in row-major order.
// Three-channel images use a fixed 0.299/0.587/0.114 luminance weighting; other
// channel counts average their channels. Values lie in [0, 255].
func segment2gray(m *cv.Mat) []float64 {
	n := m.Rows * m.Cols
	out := make([]float64, n)
	ch := m.Channels
	if ch == 3 {
		for i := 0; i < n; i++ {
			b := i * 3
			out[i] = 0.299*float64(m.Data[b]) + 0.587*float64(m.Data[b+1]) + 0.114*float64(m.Data[b+2])
		}
		return out
	}
	for i := 0; i < n; i++ {
		var s float64
		b := i * ch
		for c := 0; c < ch; c++ {
			s += float64(m.Data[b+c])
		}
		out[i] = s / float64(ch)
	}
	return out
}

// segment2sobelMag computes the Sobel gradient magnitude of a grayscale field
// with replicate border handling. rows and cols describe gray's layout.
func segment2sobelMag(gray []float64, rows, cols int) []float64 {
	out := make([]float64, rows*cols)
	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return gray[y*cols+x]
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gx := (at(y-1, x+1) + 2*at(y, x+1) + at(y+1, x+1)) -
				(at(y-1, x-1) + 2*at(y, x-1) + at(y+1, x-1))
			gy := (at(y+1, x-1) + 2*at(y+1, x) + at(y+1, x+1)) -
				(at(y-1, x-1) + 2*at(y-1, x) + at(y-1, x+1))
			out[y*cols+x] = math.Sqrt(gx*gx + gy*gy)
		}
	}
	return out
}

// segment2requireNonEmpty panics with a package-qualified message if m is empty.
func segment2requireNonEmpty(m *cv.Mat, who string) {
	if m == nil || m.Empty() {
		panic("segment2: " + who + " on empty image")
	}
}

// segment2neighbors returns the neighbourhood offsets for the given
// connectivity.
func segment2neighbors(conn Connectivity) []segment2offset {
	if conn == Conn8 {
		return segment2neighbors8
	}
	return segment2neighbors4
}

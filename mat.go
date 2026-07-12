package cv

import (
	"fmt"
	"image"
	"image/color"
)

// Mat is a dense, row-major matrix of 8-bit unsigned samples and the central
// type of the package, analogous to OpenCV's cv::Mat for the CV_8U depth.
//
// Samples are stored interleaved by channel in the flat Data slice: the value
// for row y, column x and channel c is at index (y*Cols+x)*Channels + c. A
// grayscale image uses Channels == 1; an RGB image uses Channels == 3. The
// zero value of Mat is not usable — construct instances with [NewMat],
// [FromImage], [Mat.Clone] and the transform functions in this package.
type Mat struct {
	// Rows is the image height (number of rows).
	Rows int
	// Cols is the image width (number of columns).
	Cols int
	// Channels is the number of samples per pixel.
	Channels int
	// Data holds the samples, length Rows*Cols*Channels, in row-major,
	// channel-interleaved order.
	Data []uint8
}

// NewMat allocates a zero-filled Mat with the given dimensions. It panics if
// any dimension is not positive.
func NewMat(rows, cols, channels int) *Mat {
	if rows <= 0 || cols <= 0 || channels <= 0 {
		panic(fmt.Sprintf("cv: NewMat requires positive dimensions, got rows=%d cols=%d channels=%d", rows, cols, channels))
	}
	return &Mat{
		Rows:     rows,
		Cols:     cols,
		Channels: channels,
		Data:     make([]uint8, rows*cols*channels),
	}
}

// Empty reports whether the Mat has no samples.
func (m *Mat) Empty() bool {
	return m == nil || len(m.Data) == 0 || m.Rows == 0 || m.Cols == 0
}

// Size returns the matrix dimensions as (rows, cols).
func (m *Mat) Size() (rows, cols int) {
	return m.Rows, m.Cols
}

// Total returns the number of pixels (Rows*Cols), ignoring channels.
func (m *Mat) Total() int {
	return m.Rows * m.Cols
}

// index returns the flat Data offset of the first sample of pixel (x, y).
func (m *Mat) index(y, x int) int {
	return (y*m.Cols + x) * m.Channels
}

// inBounds reports whether pixel (x, y) lies inside the matrix.
func (m *Mat) inBounds(y, x int) bool {
	return y >= 0 && y < m.Rows && x >= 0 && x < m.Cols
}

// At returns the sample at row y, column x and channel c. It panics if the
// coordinates are out of range, mirroring a Go slice index.
func (m *Mat) At(y, x, c int) uint8 {
	if !m.inBounds(y, x) || c < 0 || c >= m.Channels {
		panic(fmt.Sprintf("cv: At out of range y=%d x=%d c=%d for %dx%dx%d", y, x, c, m.Rows, m.Cols, m.Channels))
	}
	return m.Data[m.index(y, x)+c]
}

// Set stores value at row y, column x and channel c. It panics if the
// coordinates are out of range.
func (m *Mat) Set(y, x, c int, value uint8) {
	if !m.inBounds(y, x) || c < 0 || c >= m.Channels {
		panic(fmt.Sprintf("cv: Set out of range y=%d x=%d c=%d for %dx%dx%d", y, x, c, m.Rows, m.Cols, m.Channels))
	}
	m.Data[m.index(y, x)+c] = value
}

// AtPixel returns all channel samples of pixel (x, y) as a fresh slice.
func (m *Mat) AtPixel(y, x int) []uint8 {
	if !m.inBounds(y, x) {
		panic(fmt.Sprintf("cv: AtPixel out of range y=%d x=%d for %dx%d", y, x, m.Rows, m.Cols))
	}
	i := m.index(y, x)
	out := make([]uint8, m.Channels)
	copy(out, m.Data[i:i+m.Channels])
	return out
}

// SetPixel stores every channel sample of pixel (x, y). It panics if values
// does not have exactly Channels elements or the coordinates are out of range.
func (m *Mat) SetPixel(y, x int, values []uint8) {
	if !m.inBounds(y, x) {
		panic(fmt.Sprintf("cv: SetPixel out of range y=%d x=%d for %dx%d", y, x, m.Rows, m.Cols))
	}
	if len(values) != m.Channels {
		panic(fmt.Sprintf("cv: SetPixel expects %d channels, got %d", m.Channels, len(values)))
	}
	i := m.index(y, x)
	copy(m.Data[i:i+m.Channels], values)
}

// atReplicate returns the sample at (y, x, c) clamping out-of-range
// coordinates to the nearest edge (BORDER_REPLICATE).
func (m *Mat) atReplicate(y, x, c int) uint8 {
	if y < 0 {
		y = 0
	} else if y >= m.Rows {
		y = m.Rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= m.Cols {
		x = m.Cols - 1
	}
	return m.Data[m.index(y, x)+c]
}

// Clone returns a deep copy of the Mat with its own backing storage.
func (m *Mat) Clone() *Mat {
	out := &Mat{
		Rows:     m.Rows,
		Cols:     m.Cols,
		Channels: m.Channels,
		Data:     make([]uint8, len(m.Data)),
	}
	copy(out.Data, m.Data)
	return out
}

// SetTo fills every sample of every channel with value.
func (m *Mat) SetTo(value uint8) {
	for i := range m.Data {
		m.Data[i] = value
	}
}

// Region returns a deep-copied sub-matrix (region of interest) covering the
// half-open rectangle [x, x+width) × [y, y+height). Unlike OpenCV's ROI, which
// shares memory with the parent, the returned Mat is independent. It panics if
// the rectangle is empty or extends outside the source.
func (m *Mat) Region(y, x, height, width int) *Mat {
	if width <= 0 || height <= 0 {
		panic(fmt.Sprintf("cv: Region requires positive size, got height=%d width=%d", height, width))
	}
	if x < 0 || y < 0 || x+width > m.Cols || y+height > m.Rows {
		panic(fmt.Sprintf("cv: Region [y=%d x=%d h=%d w=%d] out of bounds for %dx%d", y, x, height, width, m.Rows, m.Cols))
	}
	out := NewMat(height, width, m.Channels)
	for ry := 0; ry < height; ry++ {
		srcStart := m.index(y+ry, x)
		dstStart := out.index(ry, 0)
		copy(out.Data[dstStart:dstStart+width*m.Channels], m.Data[srcStart:srcStart+width*m.Channels])
	}
	return out
}

// CopyTo writes this Mat into dst so that its top-left sample lands at column
// x, row y of dst. Samples that fall outside dst are clipped. It panics if the
// channel counts differ.
func (m *Mat) CopyTo(dst *Mat, y, x int) {
	if m.Channels != dst.Channels {
		panic(fmt.Sprintf("cv: CopyTo channel mismatch %d vs %d", m.Channels, dst.Channels))
	}
	for ry := 0; ry < m.Rows; ry++ {
		dy := y + ry
		if dy < 0 || dy >= dst.Rows {
			continue
		}
		for rx := 0; rx < m.Cols; rx++ {
			dx := x + rx
			if dx < 0 || dx >= dst.Cols {
				continue
			}
			si := m.index(ry, rx)
			di := dst.index(dy, dx)
			copy(dst.Data[di:di+m.Channels], m.Data[si:si+m.Channels])
		}
	}
}

// Split separates a multi-channel Mat into a slice of single-channel Mats, one
// per channel, in channel order.
func (m *Mat) Split() []*Mat {
	out := make([]*Mat, m.Channels)
	for c := 0; c < m.Channels; c++ {
		plane := NewMat(m.Rows, m.Cols, 1)
		for p := 0; p < m.Total(); p++ {
			plane.Data[p] = m.Data[p*m.Channels+c]
		}
		out[c] = plane
	}
	return out
}

// Merge combines single-channel Mats into one interleaved multi-channel Mat.
// Every input must be single-channel and share the same dimensions. It panics
// otherwise or if no planes are given.
func Merge(planes []*Mat) *Mat {
	if len(planes) == 0 {
		panic("cv: Merge requires at least one plane")
	}
	rows, cols := planes[0].Rows, planes[0].Cols
	for i, p := range planes {
		if p.Channels != 1 {
			panic(fmt.Sprintf("cv: Merge plane %d has %d channels, want 1", i, p.Channels))
		}
		if p.Rows != rows || p.Cols != cols {
			panic(fmt.Sprintf("cv: Merge plane %d is %dx%d, want %dx%d", i, p.Rows, p.Cols, rows, cols))
		}
	}
	out := NewMat(rows, cols, len(planes))
	for c, p := range planes {
		for i := 0; i < out.Total(); i++ {
			out.Data[i*out.Channels+c] = p.Data[i]
		}
	}
	return out
}

// FromImage converts any image.Image into a Mat. If the source is grayscale
// (color.Gray or color.Gray16) the result is single-channel; otherwise it is
// three-channel RGB with the alpha channel dropped. Samples are scaled to the
// 8-bit range.
func FromImage(img image.Image) *Mat {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		panic("cv: FromImage given an empty image")
	}
	switch img.(type) {
	case *image.Gray:
		g := img.(*image.Gray)
		m := NewMat(h, w, 1)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				m.Data[y*w+x] = g.GrayAt(b.Min.X+x, b.Min.Y+y).Y
			}
		}
		return m
	default:
		m := NewMat(h, w, 3)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, gg, bb, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
				i := m.index(y, x)
				m.Data[i+0] = uint8(r >> 8)
				m.Data[i+1] = uint8(gg >> 8)
				m.Data[i+2] = uint8(bb >> 8)
			}
		}
		return m
	}
}

// ToImage converts the Mat into an image from the standard library. A
// single-channel Mat becomes *image.Gray; a three-channel Mat becomes
// *image.RGBA with full opacity. Other channel counts are rendered by using
// the first channel as gray. The Mat is treated as RGB, not BGR.
func (m *Mat) ToImage() image.Image {
	switch m.Channels {
	case 1:
		g := image.NewGray(image.Rect(0, 0, m.Cols, m.Rows))
		for y := 0; y < m.Rows; y++ {
			for x := 0; x < m.Cols; x++ {
				g.SetGray(x, y, color.Gray{Y: m.Data[m.index(y, x)]})
			}
		}
		return g
	case 3:
		rgba := image.NewRGBA(image.Rect(0, 0, m.Cols, m.Rows))
		for y := 0; y < m.Rows; y++ {
			for x := 0; x < m.Cols; x++ {
				i := m.index(y, x)
				rgba.SetRGBA(x, y, color.RGBA{R: m.Data[i], G: m.Data[i+1], B: m.Data[i+2], A: 255})
			}
		}
		return rgba
	default:
		g := image.NewGray(image.Rect(0, 0, m.Cols, m.Rows))
		for y := 0; y < m.Rows; y++ {
			for x := 0; x < m.Cols; x++ {
				g.SetGray(x, y, color.Gray{Y: m.Data[m.index(y, x)]})
			}
		}
		return g
	}
}

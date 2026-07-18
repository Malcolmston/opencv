package inpaint

import (
	"fmt"
	"image"

	cv "github.com/malcolmston/opencv"
)

// Mask is a dense binary grid marking the pixels an inpainting or editing
// routine should treat as the region of interest (typically the hole to fill).
// A true entry means "selected" / "unknown"; a false entry means "known".
// Mask interoperates with the root package's [cv.Mat] through [MaskFromMat] and
// [Mask.ToMat]. The zero value is not usable — construct with [NewMask].
type Mask struct {
	// Rows is the mask height.
	Rows int
	// Cols is the mask width.
	Cols int
	// Data holds Rows*Cols booleans in row-major order; Data[y*Cols+x] is the
	// value at pixel (x, y).
	Data []bool
}

// NewMask allocates an all-false mask with the given dimensions. It panics if
// either dimension is not positive.
func NewMask(rows, cols int) *Mask {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("inpaint: NewMask requires positive dimensions, got rows=%d cols=%d", rows, cols))
	}
	return &Mask{Rows: rows, Cols: cols, Data: make([]bool, rows*cols)}
}

// MaskFromMat builds a Mask from the first channel of m: a pixel is selected
// (true) where the sample is strictly greater than threshold. The common
// convention threshold=0 selects every non-zero pixel.
func MaskFromMat(m *cv.Mat, threshold uint8) *Mask {
	inpaintRequireImage(m, "MaskFromMat")
	out := NewMask(m.Rows, m.Cols)
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			out.Data[y*m.Cols+x] = m.At(y, x, 0) > threshold
		}
	}
	return out
}

// ToMat renders the mask as a single-channel [cv.Mat] with 255 where selected
// and 0 elsewhere, suitable for saving or passing to Mat-based routines.
func (m *Mask) ToMat() *cv.Mat {
	out := cv.NewMat(m.Rows, m.Cols, 1)
	for i, v := range m.Data {
		if v {
			out.Data[i] = 255
		}
	}
	return out
}

// At reports whether pixel (x, y) is selected. It panics if the coordinates lie
// outside the mask.
func (m *Mask) At(y, x int) bool {
	if y < 0 || y >= m.Rows || x < 0 || x >= m.Cols {
		panic(fmt.Sprintf("inpaint: Mask.At out of range y=%d x=%d for %dx%d", y, x, m.Rows, m.Cols))
	}
	return m.Data[y*m.Cols+x]
}

// Set marks pixel (x, y) selected (v true) or not. It panics if the coordinates
// lie outside the mask.
func (m *Mask) Set(y, x int, v bool) {
	if y < 0 || y >= m.Rows || x < 0 || x >= m.Cols {
		panic(fmt.Sprintf("inpaint: Mask.Set out of range y=%d x=%d for %dx%d", y, x, m.Rows, m.Cols))
	}
	m.Data[y*m.Cols+x] = v
}

// get reports selection without bounds checking; out-of-range is treated false.
func (m *Mask) get(y, x int) bool {
	if y < 0 || y >= m.Rows || x < 0 || x >= m.Cols {
		return false
	}
	return m.Data[y*m.Cols+x]
}

// Count returns the number of selected (true) pixels.
func (m *Mask) Count() int {
	n := 0
	for _, v := range m.Data {
		if v {
			n++
		}
	}
	return n
}

// Empty reports whether no pixel is selected.
func (m *Mask) Empty() bool {
	return m.Count() == 0
}

// Clone returns an independent deep copy of the mask.
func (m *Mask) Clone() *Mask {
	out := NewMask(m.Rows, m.Cols)
	copy(out.Data, m.Data)
	return out
}

// SetAll sets every pixel to v.
func (m *Mask) SetAll(v bool) {
	for i := range m.Data {
		m.Data[i] = v
	}
}

// FillRect selects (v true) or clears every pixel inside the rectangle r,
// intersected with the mask bounds. r follows image.Rectangle conventions
// (Max exclusive).
func (m *Mask) FillRect(r image.Rectangle, v bool) {
	r = r.Intersect(image.Rect(0, 0, m.Cols, m.Rows))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			m.Data[y*m.Cols+x] = v
		}
	}
}

// Invert returns a new mask with every selection flipped.
func (m *Mask) Invert() *Mask {
	out := NewMask(m.Rows, m.Cols)
	for i, v := range m.Data {
		out.Data[i] = !v
	}
	return out
}

// Union returns a new mask selecting pixels selected in either m or o. It
// panics if the masks differ in size.
func (m *Mask) Union(o *Mask) *Mask {
	m.requireSame(o, "Union")
	out := NewMask(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] || o.Data[i]
	}
	return out
}

// Intersect returns a new mask selecting pixels selected in both m and o. It
// panics if the masks differ in size.
func (m *Mask) Intersect(o *Mask) *Mask {
	m.requireSame(o, "Intersect")
	out := NewMask(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] && o.Data[i]
	}
	return out
}

// Subtract returns a new mask selecting pixels selected in m but not in o. It
// panics if the masks differ in size.
func (m *Mask) Subtract(o *Mask) *Mask {
	m.requireSame(o, "Subtract")
	out := NewMask(m.Rows, m.Cols)
	for i := range m.Data {
		out.Data[i] = m.Data[i] && !o.Data[i]
	}
	return out
}

// Dilate returns a new mask grown by radius pixels using a square (Chebyshev)
// structuring element: a pixel is selected if any pixel within the radius is
// selected. radius<=0 returns a clone.
func (m *Mask) Dilate(radius int) *Mask {
	if radius <= 0 {
		return m.Clone()
	}
	out := NewMask(m.Rows, m.Cols)
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			sel := false
			for dy := -radius; dy <= radius && !sel; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					if m.get(y+dy, x+dx) {
						sel = true
						break
					}
				}
			}
			out.Data[y*m.Cols+x] = sel
		}
	}
	return out
}

// Erode returns a new mask shrunk by radius pixels using a square structuring
// element: a pixel stays selected only if every pixel within the radius (and in
// bounds) is selected. radius<=0 returns a clone.
func (m *Mask) Erode(radius int) *Mask {
	if radius <= 0 {
		return m.Clone()
	}
	out := NewMask(m.Rows, m.Cols)
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			keep := true
			for dy := -radius; dy <= radius && keep; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					ny, nx := y+dy, x+dx
					if ny < 0 || ny >= m.Rows || nx < 0 || nx >= m.Cols || !m.Data[ny*m.Cols+nx] {
						keep = false
						break
					}
				}
			}
			out.Data[y*m.Cols+x] = keep
		}
	}
	return out
}

// BoundingBox returns the tight bounding rectangle of the selected pixels
// (Max exclusive) and whether any pixel is selected. When nothing is selected
// it returns the zero rectangle and false.
func (m *Mask) BoundingBox() (image.Rectangle, bool) {
	minX, minY := m.Cols, m.Rows
	maxX, maxY := -1, -1
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			if m.Data[y*m.Cols+x] {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < 0 {
		return image.Rectangle{}, false
	}
	return image.Rect(minX, minY, maxX+1, maxY+1), true
}

// Boundary returns the selected pixels that touch (4-connected) at least one
// unselected or out-of-bounds pixel — the inner rim of the region — in
// row-major order.
func (m *Mask) Boundary() []image.Point {
	var pts []image.Point
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			if !m.Data[y*m.Cols+x] {
				continue
			}
			for _, d := range neighbors4 {
				if !m.get(y+d[0], x+d[1]) {
					pts = append(pts, image.Pt(x, y))
					break
				}
			}
		}
	}
	return pts
}

// requireSame panics if o differs from m in size.
func (m *Mask) requireSame(o *Mask, name string) {
	if o == nil {
		panic(fmt.Sprintf("inpaint: Mask.%s given a nil mask", name))
	}
	if m.Rows != o.Rows || m.Cols != o.Cols {
		panic(fmt.Sprintf("inpaint: Mask.%s size mismatch %dx%d vs %dx%d",
			name, m.Rows, m.Cols, o.Rows, o.Cols))
	}
}

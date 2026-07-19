package morph2

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Shape selects the geometry of a structuring element built by [NewElement].
type Shape int

// ShapeRect, ShapeCross, ShapeEllipse, ShapeDiamond and ShapeDisk are the
// supported structuring-element shapes.
const (
	// ShapeRect is a solid rectangle: every cell is set.
	ShapeRect Shape = iota
	// ShapeCross is a plus sign: only the anchor row and anchor column are set.
	ShapeCross
	// ShapeEllipse is a filled ellipse inscribed in the element rectangle.
	ShapeEllipse
	// ShapeDiamond is a filled L1 ball (Manhattan disk) centred on the anchor.
	ShapeDiamond
	// ShapeDisk is a filled L2 ball (Euclidean disk) centred on the anchor;
	// for a square element it is identical to ShapeEllipse.
	ShapeDisk
)

// Element is a flat structuring element: a rectangular grid of boolean cells
// with an explicit anchor. A cell that is set participates in the min/max of a
// morphological operation; an unset cell is ignored. Offsets used by erosion
// and dilation are measured relative to the anchor, so the anchor need not be
// the geometric centre.
type Element struct {
	// Rows is the number of rows in the element grid.
	Rows int
	// Cols is the number of columns in the element grid.
	Cols int
	// AnchorY is the row of the anchor, in [0, Rows).
	AnchorY int
	// AnchorX is the column of the anchor, in [0, Cols).
	AnchorX int
	// cells holds the set/unset state in row-major order (length Rows*Cols).
	cells []bool
}

// NewElement builds a structuring element of the given shape and size. rows and
// cols must be positive; the anchor is placed at the centre (rows/2, cols/2).
// It panics on a non-positive size.
func NewElement(shape Shape, rows, cols int) *Element {
	if rows <= 0 || cols <= 0 {
		panic("morph2: NewElement requires positive size")
	}
	e := &Element{Rows: rows, Cols: cols, AnchorY: rows / 2, AnchorX: cols / 2, cells: make([]bool, rows*cols)}
	cy := float64(rows-1) / 2
	cx := float64(cols-1) / 2
	ry := float64(rows) / 2
	rx := float64(cols) / 2
	if ry == 0 {
		ry = 1
	}
	if rx == 0 {
		rx = 1
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var on bool
			switch shape {
			case ShapeRect:
				on = true
			case ShapeCross:
				on = x == e.AnchorX || y == e.AnchorY
			case ShapeEllipse, ShapeDisk:
				dy := (float64(y) - cy) / ry
				dx := (float64(x) - cx) / rx
				on = dx*dx+dy*dy <= 1.0
			case ShapeDiamond:
				on = abs(y-e.AnchorY)+abs(x-e.AnchorX) <= min(e.AnchorY, e.AnchorX)
			default:
				panic(fmt.Sprintf("morph2: unknown shape %d", shape))
			}
			e.cells[y*cols+x] = on
		}
	}
	return e
}

// RectElement returns a solid rectangular structuring element of the given size.
func RectElement(rows, cols int) *Element { return NewElement(ShapeRect, rows, cols) }

// CrossElement returns a cross-shaped structuring element of the given size.
func CrossElement(rows, cols int) *Element { return NewElement(ShapeCross, rows, cols) }

// EllipseElement returns an elliptical structuring element of the given size.
func EllipseElement(rows, cols int) *Element { return NewElement(ShapeEllipse, rows, cols) }

// DiskElement returns a Euclidean disk of the given radius; the element is a
// square of side 2*radius+1 anchored at its centre. It panics if radius < 0.
func DiskElement(radius int) *Element {
	if radius < 0 {
		panic("morph2: DiskElement requires radius >= 0")
	}
	return NewElement(ShapeDisk, 2*radius+1, 2*radius+1)
}

// DiamondElement returns a Manhattan (L1) disk of the given radius; the element
// is a square of side 2*radius+1 anchored at its centre. It panics if radius < 0.
func DiamondElement(radius int) *Element {
	if radius < 0 {
		panic("morph2: DiamondElement requires radius >= 0")
	}
	return NewElement(ShapeDiamond, 2*radius+1, 2*radius+1)
}

// ElementFromMat builds a structuring element from a single-channel [cv.Mat]
// kernel: a cell is set where the corresponding sample is non-zero. The anchor
// is placed at (anchorY, anchorX); pass -1 for either to default to the centre.
// It panics if the kernel is not single-channel or the anchor is out of range.
func ElementFromMat(kernel *cv.Mat, anchorY, anchorX int) *Element {
	if kernel.Channels != 1 {
		panic("morph2: ElementFromMat requires a single-channel kernel")
	}
	if anchorY < 0 {
		anchorY = kernel.Rows / 2
	}
	if anchorX < 0 {
		anchorX = kernel.Cols / 2
	}
	if anchorY >= kernel.Rows || anchorX >= kernel.Cols {
		panic("morph2: ElementFromMat anchor out of range")
	}
	e := &Element{Rows: kernel.Rows, Cols: kernel.Cols, AnchorY: anchorY, AnchorX: anchorX, cells: make([]bool, kernel.Rows*kernel.Cols)}
	for i := range e.cells {
		e.cells[i] = kernel.Data[i] != 0
	}
	return e
}

// At reports whether the cell at (y, x) is set. It panics on out-of-range
// coordinates.
func (e *Element) At(y, x int) bool {
	if y < 0 || y >= e.Rows || x < 0 || x >= e.Cols {
		panic("morph2: Element.At out of range")
	}
	return e.cells[y*e.Cols+x]
}

// Set marks the cell at (y, x) as set or unset. It panics on out-of-range
// coordinates.
func (e *Element) Set(y, x int, on bool) {
	if y < 0 || y >= e.Rows || x < 0 || x >= e.Cols {
		panic("morph2: Element.Set out of range")
	}
	e.cells[y*e.Cols+x] = on
}

// Count returns the number of set cells in the element.
func (e *Element) Count() int {
	n := 0
	for _, v := range e.cells {
		if v {
			n++
		}
	}
	return n
}

// Clone returns an independent deep copy of the element.
func (e *Element) Clone() *Element {
	c := &Element{Rows: e.Rows, Cols: e.Cols, AnchorY: e.AnchorY, AnchorX: e.AnchorX, cells: make([]bool, len(e.cells))}
	copy(c.cells, e.cells)
	return c
}

// Reflect returns the point reflection of the element through its anchor, i.e.
// the transposed element used to form the adjunct operation. Erosion by e and
// dilation by e.Reflect form an adjunction.
func (e *Element) Reflect() *Element {
	r := &Element{Rows: e.Rows, Cols: e.Cols, AnchorY: e.Rows - 1 - e.AnchorY, AnchorX: e.Cols - 1 - e.AnchorX, cells: make([]bool, len(e.cells))}
	for y := 0; y < e.Rows; y++ {
		for x := 0; x < e.Cols; x++ {
			if e.cells[y*e.Cols+x] {
				r.cells[(e.Rows-1-y)*e.Cols+(e.Cols-1-x)] = true
			}
		}
	}
	return r
}

// ToMat converts the element to a single-channel [cv.Mat] whose set cells hold
// 1 and unset cells hold 0, matching the root package's GetStructuringElement.
func (e *Element) ToMat() *cv.Mat {
	m := cv.NewMat(e.Rows, e.Cols, 1)
	for i, v := range e.cells {
		if v {
			m.Data[i] = 1
		}
	}
	return m
}

// offsets returns the (dy, dx) displacements of the set cells relative to the
// anchor.
func (e *Element) offsets() [][2]int {
	out := make([][2]int, 0, len(e.cells))
	for y := 0; y < e.Rows; y++ {
		for x := 0; x < e.Cols; x++ {
			if e.cells[y*e.Cols+x] {
				out = append(out, [2]int{y - e.AnchorY, x - e.AnchorX})
			}
		}
	}
	return out
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

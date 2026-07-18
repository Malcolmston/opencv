package barcode

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements detection of the Data Matrix "L" finder pattern: the two
// solid, adjacent borders that let a reader locate a Data Matrix symbol and
// establish its orientation. The opposite two borders carry the alternating
// "clock" (timing) track, which the detector uses together with the solid
// border thickness to estimate the symbol's module dimensions. The detector
// works on a clean, axis-aligned render (the form produced by this package's
// synthetic tests and by typical label printers); it is a deterministic,
// standard-library-only routine over the package's [cv.Mat] type.

// DataMatrixCorner identifies which corner of a Data Matrix symbol holds the
// solid "L" finder pattern.
type DataMatrixCorner int

const (
	// DataMatrixBottomLeft is the conventional Data Matrix orientation: the
	// solid L occupies the left column and the bottom row.
	DataMatrixBottomLeft DataMatrixCorner = iota
	// DataMatrixTopLeft has the solid L on the left column and top row.
	DataMatrixTopLeft
	// DataMatrixBottomRight has the solid L on the right column and bottom row.
	DataMatrixBottomRight
	// DataMatrixTopRight has the solid L on the right column and top row.
	DataMatrixTopRight
)

// String returns a human-readable name for the corner.
func (c DataMatrixCorner) String() string {
	switch c {
	case DataMatrixBottomLeft:
		return "bottom-left"
	case DataMatrixTopLeft:
		return "top-left"
	case DataMatrixBottomRight:
		return "bottom-right"
	case DataMatrixTopRight:
		return "top-right"
	default:
		return fmt.Sprintf("DataMatrixCorner(%d)", int(c))
	}
}

// DataMatrixFinder describes a detected Data Matrix finder pattern: the
// symbol's bounding box in pixel coordinates, which corner carries the solid L,
// and the estimated module dimensions.
type DataMatrixFinder struct {
	// MinX, MinY, MaxX, MaxY are the inclusive pixel bounds of the symbol
	// (excluding the quiet zone).
	MinX, MinY, MaxX, MaxY int
	// Corner is the corner occupied by the solid L finder pattern.
	Corner DataMatrixCorner
	// Rows and Cols are the estimated module dimensions of the symbol.
	Rows, Cols int
}

// Width returns the symbol's pixel width (MaxX-MinX+1).
func (f DataMatrixFinder) Width() int { return f.MaxX - f.MinX + 1 }

// Height returns the symbol's pixel height (MaxY-MinY+1).
func (f DataMatrixFinder) Height() int { return f.MaxY - f.MinY + 1 }

// borderFraction returns the fraction of dark cells along one edge of the
// bounding box of dark.
func dmBorderFraction(dark [][]bool, minX, minY, maxX, maxY int, edge int) float64 {
	dcount, total := 0, 0
	switch edge {
	case 0: // top row
		for x := minX; x <= maxX; x++ {
			total++
			if dark[minY][x] {
				dcount++
			}
		}
	case 1: // bottom row
		for x := minX; x <= maxX; x++ {
			total++
			if dark[maxY][x] {
				dcount++
			}
		}
	case 2: // left col
		for y := minY; y <= maxY; y++ {
			total++
			if dark[y][minX] {
				dcount++
			}
		}
	case 3: // right col
		for y := minY; y <= maxY; y++ {
			total++
			if dark[y][maxX] {
				dcount++
			}
		}
	}
	if total == 0 {
		return 0
	}
	return float64(dcount) / float64(total)
}

// dmRunFrom counts consecutive dark cells starting at (x,y) and stepping by
// (dx,dy) while inside [minX,maxX]x[minY,maxY].
func dmRunFrom(dark [][]bool, x, y, dx, dy, minX, minY, maxX, maxY int) int {
	n := 0
	for x >= minX && x <= maxX && y >= minY && y <= maxY && dark[y][x] {
		n++
		x += dx
		y += dy
	}
	return n
}

// FindDataMatrixFinder locates the solid "L" finder pattern of a Data Matrix
// symbol in img. It Otsu-binarises the image, takes the bounding box of the
// dark region, and identifies the pair of adjacent, fully-dark borders that
// form the L. From the thickness of those borders it estimates the module
// dimensions. It returns the finder and true on success, or the zero value and
// false when the image is empty, has no dark pixels, or does not present
// exactly one pair of adjacent solid borders.
func FindDataMatrixFinder(img *cv.Mat) (DataMatrixFinder, bool) {
	if img == nil || img.Empty() {
		return DataMatrixFinder{}, false
	}
	dark := toDarkGrid(img)
	h := len(dark)
	if h == 0 {
		return DataMatrixFinder{}, false
	}
	w := len(dark[0])
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if dark[y][x] {
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
	if maxX < minX || maxY < minY {
		return DataMatrixFinder{}, false
	}

	const solid = 0.9
	top := dmBorderFraction(dark, minX, minY, maxX, maxY, 0) >= solid
	bottom := dmBorderFraction(dark, minX, minY, maxX, maxY, 1) >= solid
	left := dmBorderFraction(dark, minX, minY, maxX, maxY, 2) >= solid
	right := dmBorderFraction(dark, minX, minY, maxX, maxY, 3) >= solid

	// Exactly one vertical border and one horizontal border must be solid.
	if left == right || top == bottom {
		return DataMatrixFinder{}, false
	}

	var corner DataMatrixCorner
	switch {
	case left && bottom:
		corner = DataMatrixBottomLeft
	case left && top:
		corner = DataMatrixTopLeft
	case right && bottom:
		corner = DataMatrixBottomRight
	default: // right && top
		corner = DataMatrixTopRight
	}

	// Estimate module pixel size from the solid border thickness. The solid
	// vertical border is one module wide; the solid horizontal border is one
	// module tall.
	midY := (minY + maxY) / 2
	midX := (minX + maxX) / 2
	var moduleW, moduleH int
	if left {
		moduleW = dmRunFrom(dark, minX, midY, 1, 0, minX, minY, maxX, maxY)
	} else {
		moduleW = dmRunFrom(dark, maxX, midY, -1, 0, minX, minY, maxX, maxY)
	}
	if bottom {
		moduleH = dmRunFrom(dark, midX, maxY, 0, -1, minX, minY, maxX, maxY)
	} else {
		moduleH = dmRunFrom(dark, midX, minY, 0, 1, minX, minY, maxX, maxY)
	}
	if moduleW <= 0 || moduleH <= 0 {
		return DataMatrixFinder{}, false
	}
	cols := int(float64(maxX-minX+1)/float64(moduleW) + 0.5)
	rows := int(float64(maxY-minY+1)/float64(moduleH) + 0.5)

	return DataMatrixFinder{
		MinX: minX, MinY: minY, MaxX: maxX, MaxY: maxY,
		Corner: corner,
		Rows:   rows, Cols: cols,
	}, true
}

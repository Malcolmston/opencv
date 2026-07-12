package text

import (
	cv "github.com/malcolmston/opencv"
)

// ProjectionProfile returns the ink projection of an image: the number of
// foreground samples (grayscale value strictly greater than threshold, i.e.
// bright ink on a dark background) in each column when vertical is true, or in
// each row when vertical is false. Vertical profiles have length img width;
// horizontal profiles have length img height. A colour image is reduced to
// grayscale first.
//
// Projection profiles are the classical basis for cutting a text image into
// lines (horizontal profile) and characters (vertical profile).
func ProjectionProfile(img *cv.Mat, threshold uint8, vertical bool) []int {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	if vertical {
		prof := make([]int, cols)
		for y := 0; y < rows; y++ {
			base := y * cols
			for x := 0; x < cols; x++ {
				if gray.Data[base+x] > threshold {
					prof[x]++
				}
			}
		}
		return prof
	}
	prof := make([]int, rows)
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			if gray.Data[base+x] > threshold {
				prof[y]++
			}
		}
	}
	return prof
}

// runsOf returns the [start, end) index ranges of maximal runs of positive values
// in prof (an ink profile), i.e. the bands separated by empty gaps.
func runsOf(prof []int) [][2]int {
	var runs [][2]int
	start := -1
	for i, v := range prof {
		if v > 0 {
			if start < 0 {
				start = i
			}
		} else if start >= 0 {
			runs = append(runs, [2]int{start, i})
			start = -1
		}
	}
	if start >= 0 {
		runs = append(runs, [2]int{start, len(prof)})
	}
	return runs
}

// SegmentChars splits a single line of bright-on-dark text into per-character
// bounding boxes by cutting the vertical ink projection at empty columns, then
// tightening each box to its inked rows. Boxes are returned left-to-right. It
// works best on cleanly separated glyphs (the built-in font rendered by
// [RenderText] is designed to segment this way); touching characters are not
// split.
func SegmentChars(img *cv.Mat, threshold uint8) []cv.Rect {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	colProf := make([]int, cols)
	// colInk[x] records, per column, the min/max inked row for vertical tightening.
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			if gray.Data[base+x] > threshold {
				colProf[x]++
			}
		}
	}

	var boxes []cv.Rect
	for _, run := range runsOf(colProf) {
		x0, x1 := run[0], run[1]
		minY, maxY := rows, -1
		for y := 0; y < rows; y++ {
			base := y * cols
			inked := false
			for x := x0; x < x1; x++ {
				if gray.Data[base+x] > threshold {
					inked = true
					break
				}
			}
			if inked {
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
		if maxY < 0 {
			continue
		}
		boxes = append(boxes, cv.Rect{
			X: x0, Y: minY, Width: x1 - x0, Height: maxY - minY + 1,
		})
	}
	return boxes
}

// SegmentLines splits a multi-line text image into per-line bounding boxes by
// cutting the horizontal ink projection at empty rows, tightening each line box
// to its inked columns. Boxes are returned top-to-bottom.
func SegmentLines(img *cv.Mat, threshold uint8) []cv.Rect {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	rowProf := make([]int, rows)
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			if gray.Data[base+x] > threshold {
				rowProf[y]++
			}
		}
	}

	var boxes []cv.Rect
	for _, run := range runsOf(rowProf) {
		y0, y1 := run[0], run[1]
		minX, maxX := cols, -1
		for y := y0; y < y1; y++ {
			base := y * cols
			for x := 0; x < cols; x++ {
				if gray.Data[base+x] > threshold {
					if x < minX {
						minX = x
					}
					if x > maxX {
						maxX = x
					}
				}
			}
		}
		if maxX < 0 {
			continue
		}
		boxes = append(boxes, cv.Rect{
			X: minX, Y: y0, Width: maxX - minX + 1, Height: y1 - y0,
		})
	}
	return boxes
}

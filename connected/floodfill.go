package connected

import cv "github.com/malcolmston/opencv"

// connectedScanFill performs a span-based (scanline) flood fill. inRegion
// reports whether the pixel at flat index i still belongs to the region and has
// not yet been consumed; visit marks a pixel as filled. The fill starts at
// (sx, sy) and honours conn. It returns the number of pixels visited.
//
// The algorithm fills a whole horizontal run at once, then enqueues the runs on
// the rows above and below, extended by one column at each end when conn is
// Conn8 so diagonal neighbours are reached. This visits every pixel at most
// once and needs no per-pixel recursion.
func connectedScanFill(w, h, sx, sy int, conn Connectivity, inRegion func(i int) bool, visit func(i int)) int {
	if sx < 0 || sy < 0 || sx >= w || sy >= h {
		return 0
	}
	if !inRegion(sy*w + sx) {
		return 0
	}
	diag := 0
	if conn == Conn8 {
		diag = 1
	}
	filled := 0
	stack := [][2]int{{sx, sy}}
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x, y := p[0], p[1]
		row := y * w
		if !inRegion(row + x) {
			continue
		}
		// Expand the horizontal run around x.
		lx := x
		for lx-1 >= 0 && inRegion(row+lx-1) {
			lx--
		}
		rx := x
		for rx+1 < w && inRegion(row+rx+1) {
			rx++
		}
		for i := lx; i <= rx; i++ {
			visit(row + i)
			filled++
		}
		// Enqueue seeds on the adjacent rows.
		for _, ny := range [2]int{y - 1, y + 1} {
			if ny < 0 || ny >= h {
				continue
			}
			nrow := ny * w
			lo := lx - diag
			if lo < 0 {
				lo = 0
			}
			hi := rx + diag
			if hi > w-1 {
				hi = w - 1
			}
			for i := lo; i <= hi; i++ {
				if inRegion(nrow + i) {
					stack = append(stack, [2]int{i, ny})
				}
			}
		}
	}
	return filled
}

// FloodFill fills the connected region of m that contains the seed pixel and
// shares its exact value, replacing it with newVal. Connectivity is set by
// conn. It returns the number of pixels changed. If the seed already holds
// newVal nothing is written and 0 is returned. m is modified in place. It
// panics on a non-binary matrix; the seed may be out of bounds (returns 0).
func FloodFill(m *cv.Mat, seedX, seedY int, newVal uint8, conn Connectivity) int {
	connectedRequireBinary(m, "FloodFill")
	connectedCheckConn(conn, "FloodFill")
	if seedX < 0 || seedY < 0 || seedX >= m.Cols || seedY >= m.Rows {
		return 0
	}
	target := m.Data[seedY*m.Cols+seedX]
	if target == newVal {
		return 0
	}
	inRegion := func(i int) bool { return m.Data[i] == target }
	visit := func(i int) { m.Data[i] = newVal }
	return connectedScanFill(m.Cols, m.Rows, seedX, seedY, conn, inRegion, visit)
}

// FloodFillMask flood-fills the connected region of m sharing the seed pixel's
// value and returns a binary mask marking that region (255 inside, 0 outside)
// together with its pixel count. m is not modified. It panics on a non-binary
// matrix; an out-of-bounds seed yields an all-zero mask and a count of 0.
func FloodFillMask(m *cv.Mat, seedX, seedY int, conn Connectivity) (*cv.Mat, int) {
	connectedRequireBinary(m, "FloodFillMask")
	connectedCheckConn(conn, "FloodFillMask")
	mask := cv.NewMat(m.Rows, m.Cols, 1)
	if seedX < 0 || seedY < 0 || seedX >= m.Cols || seedY >= m.Rows {
		return mask, 0
	}
	target := m.Data[seedY*m.Cols+seedX]
	inRegion := func(i int) bool { return m.Data[i] == target && mask.Data[i] == 0 }
	visit := func(i int) { mask.Data[i] = 255 }
	n := connectedScanFill(m.Cols, m.Rows, seedX, seedY, conn, inRegion, visit)
	return mask, n
}

// FloodFillTolerance fills the region reachable from the seed whose samples lie
// within tolerance of the seed's original value, replacing them with newVal.
// A pixel joins the region when the absolute difference between its value and
// the seed value is at most tolerance. It returns the number of pixels changed
// and modifies m in place. It panics on a non-binary matrix or negative
// tolerance; an out-of-bounds seed returns 0.
func FloodFillTolerance(m *cv.Mat, seedX, seedY int, newVal uint8, tolerance int, conn Connectivity) int {
	connectedRequireBinary(m, "FloodFillTolerance")
	connectedCheckConn(conn, "FloodFillTolerance")
	if tolerance < 0 {
		panic("connected: FloodFillTolerance: tolerance must be >= 0")
	}
	if seedX < 0 || seedY < 0 || seedX >= m.Cols || seedY >= m.Rows {
		return 0
	}
	seed := int(m.Data[seedY*m.Cols+seedX])
	// Track membership separately so a pixel repainted to newVal is not
	// re-tested against the seed value (which could re-admit or exclude it).
	done := make([]bool, len(m.Data))
	inRegion := func(i int) bool {
		if done[i] {
			return false
		}
		d := int(m.Data[i]) - seed
		if d < 0 {
			d = -d
		}
		return d <= tolerance
	}
	filled := 0
	visit := func(i int) {
		done[i] = true
		if m.Data[i] != newVal {
			filled++
		}
		m.Data[i] = newVal
	}
	connectedScanFill(m.Cols, m.Rows, seedX, seedY, conn, inRegion, visit)
	return filled
}

// RegionPoints returns the coordinates of every pixel in the connected region
// of m that contains the seed and shares its value, in raster order. m is not
// modified. It panics on a non-binary matrix; an out-of-bounds seed yields nil.
func RegionPoints(m *cv.Mat, seedX, seedY int, conn Connectivity) []cv.Point {
	mask, n := FloodFillMask(m, seedX, seedY, conn)
	if n == 0 {
		return nil
	}
	pts := make([]cv.Point, 0, n)
	for y := 0; y < mask.Rows; y++ {
		for x := 0; x < mask.Cols; x++ {
			if mask.Data[y*mask.Cols+x] != 0 {
				pts = append(pts, cv.Point{X: x, Y: y})
			}
		}
	}
	return pts
}

// ConnectedAt returns a binary mask of the single foreground component that
// contains pixel (x, y). If the seed is background or out of bounds the mask is
// all zero. It panics on a non-binary matrix.
func ConnectedAt(m *cv.Mat, x, y int, conn Connectivity) *cv.Mat {
	connectedRequireBinary(m, "ConnectedAt")
	connectedCheckConn(conn, "ConnectedAt")
	if x < 0 || y < 0 || x >= m.Cols || y >= m.Rows || m.Data[y*m.Cols+x] == 0 {
		return cv.NewMat(m.Rows, m.Cols, 1)
	}
	mask, _ := FloodFillMask(m, x, y, conn)
	return mask
}

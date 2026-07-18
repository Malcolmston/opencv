package connected

import cv "github.com/malcolmston/opencv"

// Boundary returns the inner boundary of the foreground: every foreground pixel
// that has at least one background neighbour under conn, or that lies on the
// image edge, is marked 255; all other pixels are 0. src is not modified.
func Boundary(src *cv.Mat, conn Connectivity) *cv.Mat {
	connectedRequireBinary(src, "Boundary")
	connectedCheckConn(conn, "Boundary")
	w, h := src.Cols, src.Rows
	off := connectedOffsets(conn)
	out := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if src.Data[y*w+x] == 0 {
				continue
			}
			edge := false
			for _, d := range off {
				ny, nx := y+d[0], x+d[1]
				if nx < 0 || ny < 0 || nx >= w || ny >= h || src.Data[ny*w+nx] == 0 {
					edge = true
					break
				}
			}
			if edge {
				out.Data[y*w+x] = 255
			}
		}
	}
	return out
}

// OuterBoundary returns the outer boundary of the foreground: every background
// pixel that has at least one foreground neighbour under conn is marked 255.
// This is the one-pixel-thick ring just outside each component. src is not
// modified.
func OuterBoundary(src *cv.Mat, conn Connectivity) *cv.Mat {
	connectedRequireBinary(src, "OuterBoundary")
	connectedCheckConn(conn, "OuterBoundary")
	w, h := src.Cols, src.Rows
	off := connectedOffsets(conn)
	out := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if src.Data[y*w+x] != 0 {
				continue
			}
			for _, d := range off {
				ny, nx := y+d[0], x+d[1]
				if nx >= 0 && ny >= 0 && nx < w && ny < h && src.Data[ny*w+nx] != 0 {
					out.Data[y*w+x] = 255
					break
				}
			}
		}
	}
	return out
}

// BoundaryPoints returns, in raster order, the coordinates of all inner
// boundary pixels of the foreground (the pixels [Boundary] marks).
func BoundaryPoints(src *cv.Mat, conn Connectivity) []cv.Point {
	b := Boundary(src, conn)
	var pts []cv.Point
	for y := 0; y < b.Rows; y++ {
		for x := 0; x < b.Cols; x++ {
			if b.Data[y*b.Cols+x] != 0 {
				pts = append(pts, cv.Point{X: x, Y: y})
			}
		}
	}
	return pts
}

// Perimeter returns the number of inner boundary pixels of the foreground, a
// simple pixel-count estimate of the total contour length under conn.
func Perimeter(src *cv.Mat, conn Connectivity) int {
	b := Boundary(src, conn)
	n := 0
	for _, v := range b.Data {
		if v != 0 {
			n++
		}
	}
	return n
}

// connectedMoore lists the eight neighbour displacements (dx, dy) in clockwise
// order, beginning at east. Because rows grow downward, this sequence
// E, SE, S, SW, W, NW, N, NE is genuinely clockwise in image coordinates.
var connectedMoore = [8][2]int{
	{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1},
}

// connectedMooreDir returns the index into connectedMoore of the displacement
// from c to its adjacent neighbour n, or -1 if n is not an 8-neighbour of c.
func connectedMooreDir(cx, cy, nx, ny int) int {
	dx, dy := nx-cx, ny-cy
	for i, d := range connectedMoore {
		if d[0] == dx && d[1] == dy {
			return i
		}
	}
	return -1
}

// TraceBoundary follows the outer contour of the connected component that
// contains the foreground pixel (startX, startY) using Moore-neighbour tracing
// with Jacob's stopping criterion, and returns its boundary pixels in clockwise
// order. Tracing is 8-connected, the standard for contour following; the first
// and last points coincide with the start pixel is not repeated. An isolated
// pixel yields a single-point slice. It panics on a non-binary matrix or when
// the start pixel is background or out of bounds.
func TraceBoundary(src *cv.Mat, startX, startY int) []cv.Point {
	connectedRequireBinary(src, "TraceBoundary")
	w, h := src.Cols, src.Rows
	if startX < 0 || startY < 0 || startX >= w || startY >= h || src.Data[startY*w+startX] == 0 {
		panic("connected: TraceBoundary: start pixel must be foreground and in bounds")
	}
	isFg := func(x, y int) bool {
		return x >= 0 && y >= 0 && x < w && y < h && src.Data[y*w+x] != 0
	}

	// Choose an initial backtrack (background) neighbour. The west neighbour is
	// background whenever the start is the top-left pixel of its component; for
	// an arbitrary start, scan clockwise for any background neighbour.
	bx, by := startX-1, startY
	if isFg(bx, by) {
		found := false
		for _, d := range connectedMoore {
			if !isFg(startX+d[0], startY+d[1]) {
				bx, by = startX+d[0], startY+d[1]
				found = true
				break
			}
		}
		if !found {
			// Fully surrounded: not a boundary pixel, nothing to trace.
			return []cv.Point{{X: startX, Y: startY}}
		}
	}

	// search returns the next foreground boundary pixel found by rotating
	// clockwise from the backtrack neighbour, along with the background pixel
	// examined just before it (the next backtrack).
	search := func(cx, cy, backX, backY int) (nx, ny, nbx, nby int, ok bool) {
		start := connectedMooreDir(cx, cy, backX, backY)
		if start < 0 {
			start = 0
		}
		px, py := backX, backY
		for k := 1; k <= 8; k++ {
			d := connectedMoore[(start+k)&7]
			cxx, cyy := cx+d[0], cy+d[1]
			if isFg(cxx, cyy) {
				return cxx, cyy, px, py, true
			}
			px, py = cxx, cyy
		}
		return 0, 0, 0, 0, false
	}

	cx, cy := startX, startY
	nx, ny, nbx, nby, ok := search(cx, cy, bx, by)
	if !ok {
		return []cv.Point{{X: startX, Y: startY}}
	}
	boundary := []cv.Point{{X: startX, Y: startY}}
	secondX, secondY := nx, ny
	limit := 4*(w*h) + 4
	for step := 0; step < limit; step++ {
		boundary = append(boundary, cv.Point{X: nx, Y: ny})
		cx, cy = nx, ny
		bx, by = nbx, nby
		nx, ny, nbx, nby, ok = search(cx, cy, bx, by)
		if !ok {
			break
		}
		// Jacob's stopping criterion: back at the start pixel, about to repeat
		// the very first step.
		if cx == startX && cy == startY && nx == secondX && ny == secondY {
			break
		}
	}
	// Drop the trailing repeat of the start pixel produced by the criterion.
	if len(boundary) > 1 {
		last := boundary[len(boundary)-1]
		if last.X == startX && last.Y == startY {
			boundary = boundary[:len(boundary)-1]
		}
	}
	return boundary
}

// ComponentBoundaries labels src and returns the traced outer contour of every
// foreground component, ordered by label. Each contour is an ordered,
// 8-connected clockwise list of boundary pixels (see [TraceBoundary]).
func ComponentBoundaries(src *cv.Mat, conn Connectivity) [][]cv.Point {
	lbl := Label(src, conn)
	if lbl.Count == 0 {
		return nil
	}
	// First pixel (raster order) of each label is its top-left, a valid start.
	starts := make([]cv.Point, lbl.Count+1)
	seen := make([]bool, lbl.Count+1)
	for y := 0; y < lbl.Height; y++ {
		for x := 0; x < lbl.Width; x++ {
			v := lbl.Data[y*lbl.Width+x]
			if v != 0 && !seen[v] {
				seen[v] = true
				starts[v] = cv.Point{X: x, Y: y}
			}
		}
	}
	out := make([][]cv.Point, lbl.Count)
	for k := 1; k <= lbl.Count; k++ {
		out[k-1] = TraceBoundary(src, starts[k].X, starts[k].Y)
	}
	return out
}

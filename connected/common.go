package connected

import cv "github.com/malcolmston/opencv"

// Connectivity selects the pixel neighbourhood used when deciding whether two
// foreground pixels belong to the same component.
type Connectivity int

const (
	// Conn4 is 4-connectivity: only the horizontal and vertical neighbours
	// (N, S, E, W) are considered adjacent.
	Conn4 Connectivity = 4
	// Conn8 is 8-connectivity: the four diagonal neighbours are adjacent in
	// addition to the four edge neighbours.
	Conn8 Connectivity = 8
)

// connectedRequireBinary panics unless m is a usable single-channel matrix.
func connectedRequireBinary(m *cv.Mat, who string) {
	if m == nil || m.Empty() {
		panic("connected: " + who + ": empty source matrix")
	}
	if m.Channels != 1 {
		panic("connected: " + who + ": operation requires a single-channel matrix")
	}
}

// connectedCheckConn panics unless conn is Conn4 or Conn8.
func connectedCheckConn(conn Connectivity, who string) {
	if conn != Conn4 && conn != Conn8 {
		panic("connected: " + who + ": connectivity must be Conn4 or Conn8")
	}
}

// connectedNewMask allocates a zeroed single-channel matrix matching the
// dimensions of m.
func connectedNewMask(m *cv.Mat) *cv.Mat { return cv.NewMat(m.Rows, m.Cols, 1) }

// connectedOffsets returns the neighbour displacements (dy, dx) for conn.
func connectedOffsets(conn Connectivity) [][2]int {
	if conn == Conn8 {
		return [][2]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}
	}
	return [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
}

// connectedForeground returns a boolean mask, in row-major order, that is true
// wherever src has a non-zero sample.
func connectedForeground(src *cv.Mat) []bool {
	m := make([]bool, src.Rows*src.Cols)
	for i, v := range src.Data {
		m[i] = v != 0
	}
	return m
}

// connectedBackground returns a boolean mask, in row-major order, that is true
// wherever src has a zero sample.
func connectedBackground(src *cv.Mat) []bool {
	m := make([]bool, src.Rows*src.Cols)
	for i, v := range src.Data {
		m[i] = v == 0
	}
	return m
}

// connectedLabelMask runs the two-pass union-find labeller over an arbitrary
// boolean region mask of the given width and height. It returns a per-pixel
// label slice (0 for pixels outside the mask, 1..count for regions) and the
// number of labelled regions. Labels are compact and gap-free, assigned in
// raster-scan order of each region's first pixel.
func connectedLabelMask(mask []bool, w, h int, conn Connectivity) (labels []int, count int) {
	prov := make([]int, w*h)
	uf := connectedNewUnionFind()
	diag := conn == Conn8

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if !mask[i] {
				continue
			}
			var neigh []int
			// West.
			if x > 0 && prov[i-1] != 0 {
				neigh = append(neigh, prov[i-1])
			}
			// North.
			if y > 0 && prov[i-w] != 0 {
				neigh = append(neigh, prov[i-w])
			}
			if diag {
				// North-west.
				if y > 0 && x > 0 && prov[i-w-1] != 0 {
					neigh = append(neigh, prov[i-w-1])
				}
				// North-east.
				if y > 0 && x < w-1 && prov[i-w+1] != 0 {
					neigh = append(neigh, prov[i-w+1])
				}
			}
			if len(neigh) == 0 {
				prov[i] = uf.makeSet()
				continue
			}
			min := neigh[0]
			for _, n := range neigh[1:] {
				if n < min {
					min = n
				}
			}
			prov[i] = min
			for _, n := range neigh {
				uf.union(min, n)
			}
		}
	}

	// Second pass: resolve each provisional label to its root, then compact
	// the roots to consecutive labels 1..count.
	remap := make(map[int]int)
	labels = make([]int, w*h)
	for i, p := range prov {
		if p == 0 {
			continue
		}
		r := uf.find(p)
		lbl, ok := remap[r]
		if !ok {
			count++
			lbl = count
			remap[r] = lbl
		}
		labels[i] = lbl
	}
	return labels, count
}

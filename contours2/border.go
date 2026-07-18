package contours2

import (
	cv "github.com/malcolmston/opencv"
)

// contours2dirs holds the eight neighbour offsets (row, col) in clockwise order
// starting at east, used by the Suzuki–Abe border tracer.
var contours2dirs = [8][2]int{
	{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
}

// contours2border records one traced border during Suzuki–Abe following.
type contours2border struct {
	seqNum  int
	isOuter bool
	parent  int // parent border sequence number, or 1 for the image frame
	points  []cv.Point
}

// FindContours extracts contours from a binary single-channel image using the
// Suzuki–Abe border-following algorithm. Any non-zero sample is treated as
// foreground. mode selects the retrieval strategy ([RetrExternal], [RetrList],
// [RetrCComp] or [RetrTree]) and approx selects point storage
// ([ChainApproxNone] or [ChainApproxSimple]).
//
// It returns the contours together with a parallel hierarchy slice (one
// [HierarchyNode] per contour, indexed identically). The source Mat is not
// modified. It panics if src is nil, empty, or not single-channel.
func FindContours(src *cv.Mat, mode RetrievalMode, approx ChainApproxMethod) ([]Contour, []HierarchyNode) {
	contours2requireGray(src, "FindContours")
	rows, cols := src.Rows, src.Cols
	f := make([]int, rows*cols)
	for i, v := range src.Data {
		if v != 0 {
			f[i] = 1
		}
	}
	get := func(r, c int) int {
		if r < 0 || r >= rows || c < 0 || c >= cols {
			return 0
		}
		return f[r*cols+c]
	}
	set := func(r, c, v int) {
		if r >= 0 && r < rows && c >= 0 && c < cols {
			f[r*cols+c] = v
		}
	}

	// borders[0] is the image frame (seqNum 1, a hole border, no parent).
	borders := []contours2border{{seqNum: 1, isOuter: false, parent: -1}}
	seqToIdx := map[int]int{1: 0}
	nbd := 1

	for i := 0; i < rows; i++ {
		lnbd := 1
		for j := 0; j < cols; j++ {
			fij := f[i*cols+j]
			if fij == 0 {
				continue
			}
			var i2, j2 int
			isOuter := false
			isStart := false
			if fij == 1 && get(i, j-1) == 0 {
				nbd++
				isOuter = true
				isStart = true
				i2, j2 = i, j-1
			} else if fij >= 1 && get(i, j+1) == 0 {
				nbd++
				isStart = true
				i2, j2 = i, j+1
				if fij > 1 {
					lnbd = fij
				}
			}
			if !isStart {
				if f[i*cols+j] != 1 {
					lnbd = contours2abs(f[i*cols+j])
				}
				continue
			}

			// Determine the parent border from LNBD.
			prev := borders[seqToIdx[lnbd]]
			var parentSeq int
			if isOuter {
				if prev.isOuter {
					parentSeq = prev.parent
				} else {
					parentSeq = prev.seqNum
				}
			} else {
				if prev.isOuter {
					parentSeq = prev.seqNum
				} else {
					parentSeq = prev.parent
				}
			}

			pts := contours2trace(i, j, i2, j2, nbd, get, set)
			borders = append(borders, contours2border{
				seqNum: nbd, isOuter: isOuter, parent: parentSeq, points: pts,
			})
			seqToIdx[nbd] = len(borders) - 1

			if f[i*cols+j] != 1 {
				lnbd = contours2abs(f[i*cols+j])
			}
		}
	}

	return contours2assemble(borders, seqToIdx, mode, approx)
}

// FindExternalContours is a convenience wrapper that returns only the outermost
// contours ([RetrExternal]) with the given chain approximation. The hierarchy
// is discarded since every external contour is a root.
func FindExternalContours(src *cv.Mat, approx ChainApproxMethod) []Contour {
	c, _ := FindContours(src, RetrExternal, approx)
	return c
}

// contours2dirOf returns the direction index of the neighbour offset (dr, dc),
// or -1 if it is not one of the eight neighbours.
func contours2dirOf(dr, dc int) int {
	for d := 0; d < 8; d++ {
		if contours2dirs[d][0] == dr && contours2dirs[d][1] == dc {
			return d
		}
	}
	return -1
}

// contours2trace performs Suzuki steps 3.1–3.5 starting at (i,j) with examined
// background neighbour (i2,j2), collecting the border points and relabelling.
func contours2trace(i, j, i2, j2, nbd int, get func(int, int) int, set func(int, int, int)) []cv.Point {
	d0 := contours2dirOf(i2-i, j2-j)
	i1, j1 := -1, -1
	for k := 0; k < 8; k++ {
		d := (d0 + k) % 8
		r, c := i+contours2dirs[d][0], j+contours2dirs[d][1]
		if get(r, c) != 0 {
			i1, j1 = r, c
			break
		}
	}
	if i1 == -1 {
		// Isolated pixel.
		set(i, j, -nbd)
		return []cv.Point{{X: j, Y: i}}
	}

	pts := make([]cv.Point, 0, 8)
	i2, j2 = i1, j1
	i3, j3 := i, j
	for {
		d2 := contours2dirOf(i2-i3, j2-j3)
		i4, j4 := i3, j3
		examinedEastZero := false
		for k := 1; k <= 8; k++ {
			d := (d2 + 8 - k) % 8
			r, c := i3+contours2dirs[d][0], j3+contours2dirs[d][1]
			if get(r, c) != 0 {
				i4, j4 = r, c
				break
			}
			if d == 0 {
				examinedEastZero = true
			}
		}

		pts = append(pts, cv.Point{X: j3, Y: i3})

		if examinedEastZero {
			set(i3, j3, -nbd)
		} else if get(i3, j3) == 1 {
			set(i3, j3, nbd)
		}

		if i4 == i && j4 == j && i3 == i1 && j3 == j1 {
			break
		}
		i2, j2 = i3, j3
		i3, j3 = i4, j4
	}
	return pts
}

// contours2assemble filters traced borders by retrieval mode, applies the chain
// approximation, and builds the hierarchy.
func contours2assemble(borders []contours2border, seqToIdx map[int]int, mode RetrievalMode, approx ChainApproxMethod) ([]Contour, []HierarchyNode) {
	var selected []int
	for idx := 1; idx < len(borders); idx++ {
		if mode == RetrExternal {
			if borders[idx].isOuter && borders[idx].parent == 1 {
				selected = append(selected, idx)
			}
		} else {
			selected = append(selected, idx)
		}
	}

	outIdxOf := make(map[int]int, len(selected))
	for out, idx := range selected {
		outIdxOf[idx] = out
	}

	contours := make([]Contour, len(selected))
	for out, idx := range selected {
		pts := borders[idx].points
		if approx == ChainApproxSimple {
			pts = contours2simplify(pts)
		}
		c := make(Contour, len(pts))
		copy(c, pts)
		contours[out] = c
	}

	hierarchy := make([]HierarchyNode, len(selected))
	for out := range hierarchy {
		hierarchy[out] = HierarchyNode{Next: -1, Prev: -1, FirstChild: -1, Parent: -1}
	}

	// Full tree parent (output index) for each selected contour.
	treeParent := make([]int, len(selected))
	for out, idx := range selected {
		pSeq := borders[idx].parent
		if pSeq <= 1 {
			treeParent[out] = -1
		} else if p, ok := outIdxOf[seqToIdx[pSeq]]; ok {
			treeParent[out] = p
		} else {
			treeParent[out] = -1
		}
	}

	var parentOut []int
	switch mode {
	case RetrTree:
		parentOut = treeParent
	case RetrCComp:
		parentOut = contours2ccomp(borders, selected, treeParent)
	default: // RetrExternal, RetrList
		parentOut = make([]int, len(selected))
		for out := range parentOut {
			parentOut[out] = -1
		}
	}

	for out := range hierarchy {
		hierarchy[out].Parent = parentOut[out]
	}
	contours2link(hierarchy, parentOut)
	return contours, hierarchy
}

// contours2ccomp collapses the full parent tree into the two-level connected
// component hierarchy used by [RetrCComp]: outer boundaries become top-level
// roots and hole boundaries become children of their enclosing outer boundary.
func contours2ccomp(borders []contours2border, selected, treeParent []int) []int {
	parentOut := make([]int, len(selected))
	for out, idx := range selected {
		if borders[idx].isOuter {
			// Outer boundary: a top-level component root.
			parentOut[out] = -1
		} else {
			// Hole boundary: child of its (outer) tree parent.
			parentOut[out] = treeParent[out]
		}
	}
	return parentOut
}

// contours2link fills Next/Prev/FirstChild links from a parent-index slice,
// preserving the order in which contours were discovered.
func contours2link(hierarchy []HierarchyNode, parentOut []int) {
	childrenOf := map[int][]int{}
	order := []int{}
	seen := map[int]bool{}
	for out, p := range parentOut {
		if !seen[p] {
			seen[p] = true
			order = append(order, p)
		}
		childrenOf[p] = append(childrenOf[p], out)
	}
	for _, p := range order {
		kids := childrenOf[p]
		for k := 0; k < len(kids); k++ {
			if k > 0 {
				hierarchy[kids[k]].Prev = kids[k-1]
			}
			if k < len(kids)-1 {
				hierarchy[kids[k]].Next = kids[k+1]
			}
		}
		if p >= 0 && len(kids) > 0 {
			hierarchy[p].FirstChild = kids[0]
		}
	}
}

// contours2simplify removes interior points of straight (horizontal, vertical
// or diagonal) runs, keeping only points where the chain direction changes.
// This is the CHAIN_APPROX_SIMPLE reduction.
func contours2simplify(pts []cv.Point) []cv.Point {
	n := len(pts)
	if n <= 2 {
		out := make([]cv.Point, n)
		copy(out, pts)
		return out
	}
	out := make([]cv.Point, 0, n)
	for i := 0; i < n; i++ {
		prev := pts[(i-1+n)%n]
		cur := pts[i]
		next := pts[(i+1)%n]
		d1x, d1y := contours2sign(cur.X-prev.X), contours2sign(cur.Y-prev.Y)
		d2x, d2y := contours2sign(next.X-cur.X), contours2sign(next.Y-cur.Y)
		if d1x != d2x || d1y != d2y {
			out = append(out, cur)
		}
	}
	if len(out) == 0 {
		out = append(out, pts[0])
	}
	return out
}

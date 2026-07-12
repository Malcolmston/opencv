package cv

import "math"

// Contour is an ordered list of boundary points describing a connected shape,
// as produced by [FindContours]. Points are in traversal order around the
// border.
type Contour []Point

// RetrievalMode selects which contours [FindContours] returns and how their
// hierarchy is built, mirroring OpenCV's RETR_* modes.
type RetrievalMode int

const (
	// RetrExternal returns only the outermost contours, discarding any holes
	// nested inside them. Every returned entry has no parent.
	RetrExternal RetrievalMode = iota
	// RetrList returns every contour (outer borders and holes) as a flat list
	// with no parent/child relationships recorded.
	RetrList
	// RetrTree returns every contour and reconstructs the full nesting tree in
	// the hierarchy (parent, child and sibling links).
	RetrTree
)

// ContourApproxMode selects how [FindContours] stores each border's points.
type ContourApproxMode int

const (
	// ChainApproxNone stores every point along the border.
	ChainApproxNone ContourApproxMode = iota
	// ChainApproxSimple collapses straight horizontal, vertical and diagonal
	// runs to their end points, so a rectangle keeps only its four corners.
	ChainApproxSimple
)

// HierarchyNode mirrors OpenCV's 4-element hierarchy entry. Each field is an
// index into the slice of contours returned by [FindContours], or -1 when
// absent. Next and Prev link siblings at the same nesting level; FirstChild is
// the first contour nested one level deeper; Parent is the enclosing contour.
type HierarchyNode struct {
	Next       int
	Prev       int
	FirstChild int
	Parent     int
}

// suzuki neighbour offsets in clockwise order (row, col) starting at east, used
// by the border-following tracer.
var contourDirs = [8][2]int{
	{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
}

type borderInfo struct {
	seqNum  int
	isOuter bool
	parent  int // parent seqNum, or -1 for the frame
	points  []Point
}

// FindContours extracts contours from a binary single-channel image using the
// Suzuki–Abe border-following algorithm. Any non-zero sample is treated as
// foreground. mode selects the retrieval strategy ([RetrExternal], [RetrList]
// or [RetrTree]) and approx selects point storage ([ChainApproxNone] or
// [ChainApproxSimple]).
//
// It returns the contours together with a parallel hierarchy slice (one
// [HierarchyNode] per contour, same indexing). For [RetrExternal] and
// [RetrList] every hierarchy entry's Parent and FirstChild are -1. The source
// is not modified. It panics if src is not single-channel.
func FindContours(src *Mat, mode RetrievalMode, approx ContourApproxMode) ([]Contour, []HierarchyNode) {
	requireChannels(src, 1, "FindContours")
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
	dirOf := func(dr, dc int) int {
		for d := 0; d < 8; d++ {
			if contourDirs[d][0] == dr && contourDirs[d][1] == dc {
				return d
			}
		}
		return -1
	}

	// borders[0] is the image frame (seqNum 1, a hole border, no parent).
	borders := []borderInfo{{seqNum: 1, isOuter: false, parent: -1}}
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
					lnbd = absInt(f[i*cols+j])
				}
				continue
			}

			// Step 2: determine the parent border from LNBD.
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

			// Step 3: follow the border, collecting points and labelling.
			pts := traceBorder(i, j, i2, j2, nbd, get, set, dirOf)
			borders = append(borders, borderInfo{
				seqNum: nbd, isOuter: isOuter, parent: parentSeq, points: pts,
			})
			seqToIdx[nbd] = len(borders) - 1

			// Step 4.
			if f[i*cols+j] != 1 {
				lnbd = absInt(f[i*cols+j])
			}
		}
	}

	return assembleContours(borders, seqToIdx, mode, approx)
}

// traceBorder performs Suzuki steps 3.1–3.5 starting at (i,j) with examined
// background neighbour (i2,j2), collecting the border points.
func traceBorder(i, j, i2, j2, nbd int, get func(int, int) int, set func(int, int, int), dirOf func(int, int) int) []Point {
	// Step 3.1: search clockwise from (i2,j2) for the first foreground pixel.
	d0 := dirOf(i2-i, j2-j)
	i1, j1 := -1, -1
	for k := 0; k < 8; k++ {
		d := (d0 + k) % 8
		r, c := i+contourDirs[d][0], j+contourDirs[d][1]
		if get(r, c) != 0 {
			i1, j1 = r, c
			break
		}
	}
	if i1 == -1 {
		// Isolated pixel.
		set(i, j, -nbd)
		return []Point{{X: j, Y: i}}
	}

	pts := make([]Point, 0, 8)
	i2, j2 = i1, j1
	i3, j3 := i, j
	for {
		// Step 3.3: search counter-clockwise from just after (i2,j2).
		d2 := dirOf(i2-i3, j2-j3)
		i4, j4 := i3, j3
		examinedEastZero := false
		for k := 1; k <= 8; k++ {
			d := (d2 + 8 - k) % 8
			r, c := i3+contourDirs[d][0], j3+contourDirs[d][1]
			if get(r, c) != 0 {
				i4, j4 = r, c
				break
			}
			if d == 0 {
				examinedEastZero = true
			}
		}

		pts = append(pts, Point{X: j3, Y: i3})

		// Step 3.4: relabel the current border pixel.
		if examinedEastZero {
			set(i3, j3, -nbd)
		} else if get(i3, j3) == 1 {
			set(i3, j3, nbd)
		}

		// Step 3.5: termination.
		if i4 == i && j4 == j && i3 == i1 && j3 == j1 {
			break
		}
		i2, j2 = i3, j3
		i3, j3 = i4, j4
	}
	return pts
}

// assembleContours filters the traced borders by retrieval mode, applies the
// chain approximation, and builds the hierarchy.
func assembleContours(borders []borderInfo, seqToIdx map[int]int, mode RetrievalMode, approx ContourApproxMode) ([]Contour, []HierarchyNode) {
	// Select which borders (indices into borders, skipping the frame) survive.
	var selected []int
	switch mode {
	case RetrExternal:
		for idx := 1; idx < len(borders); idx++ {
			if borders[idx].isOuter && borders[idx].parent == 1 {
				selected = append(selected, idx)
			}
		}
	case RetrList, RetrTree:
		for idx := 1; idx < len(borders); idx++ {
			selected = append(selected, idx)
		}
	default:
		panic("cv: FindContours unknown retrieval mode")
	}

	outIdxOf := make(map[int]int, len(selected))
	for out, idx := range selected {
		outIdxOf[idx] = out
	}

	contours := make([]Contour, len(selected))
	for out, idx := range selected {
		pts := borders[idx].points
		if approx == ChainApproxSimple {
			pts = simplifyChain(pts)
		}
		c := make(Contour, len(pts))
		copy(c, pts)
		contours[out] = c
	}

	hierarchy := make([]HierarchyNode, len(selected))
	for out := range hierarchy {
		hierarchy[out] = HierarchyNode{Next: -1, Prev: -1, FirstChild: -1, Parent: -1}
	}

	if mode == RetrTree {
		// Parent output index for each selected contour.
		parentOut := make([]int, len(selected))
		for out, idx := range selected {
			pSeq := borders[idx].parent
			if pSeq <= 1 {
				parentOut[out] = -1
			} else if p, ok := outIdxOf[seqToIdx[pSeq]]; ok {
				parentOut[out] = p
			} else {
				parentOut[out] = -1
			}
			hierarchy[out].Parent = parentOut[out]
		}
		linkSiblings(hierarchy, parentOut)
	} else {
		// Flat list: siblings at the root, no parents or children.
		parentOut := make([]int, len(selected))
		for out := range parentOut {
			parentOut[out] = -1
		}
		linkSiblings(hierarchy, parentOut)
	}

	return contours, hierarchy
}

// linkSiblings fills Next/Prev/FirstChild links from a parent-index slice.
func linkSiblings(hierarchy []HierarchyNode, parentOut []int) {
	childrenOf := map[int][]int{}
	for out, p := range parentOut {
		childrenOf[p] = append(childrenOf[p], out)
	}
	for p, kids := range childrenOf {
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

// simplifyChain removes interior points of straight (horizontal, vertical or
// diagonal) runs, keeping only the points where direction changes. This is the
// CHAIN_APPROX_SIMPLE reduction.
func simplifyChain(pts []Point) []Point {
	n := len(pts)
	if n <= 2 {
		out := make([]Point, n)
		copy(out, pts)
		return out
	}
	out := make([]Point, 0, n)
	for i := 0; i < n; i++ {
		prev := pts[(i-1+n)%n]
		cur := pts[i]
		next := pts[(i+1)%n]
		d1x, d1y := sign(cur.X-prev.X), sign(cur.Y-prev.Y)
		d2x, d2y := sign(next.X-cur.X), sign(next.Y-cur.Y)
		if d1x != d2x || d1y != d2y {
			out = append(out, cur)
		}
	}
	if len(out) == 0 {
		out = append(out, pts[0])
	}
	return out
}

// DrawContours renders contours onto m. When contourIdx is negative every
// contour is drawn; otherwise only that index. A positive thickness draws the
// closed outline; a negative thickness (or [Filled]) fills each contour.
func DrawContours(m *Mat, contours []Contour, contourIdx int, color Scalar, thickness int) {
	draw := func(c Contour) {
		if len(c) == 0 {
			return
		}
		poly := []Point(c)
		if thickness < 0 {
			FillPoly(m, [][]Point{poly}, color)
		} else {
			Polylines(m, [][]Point{poly}, true, color, thickness)
		}
	}
	if contourIdx < 0 {
		for _, c := range contours {
			draw(c)
		}
		return
	}
	if contourIdx < len(contours) {
		draw(contours[contourIdx])
	}
}

// ContourArea returns the area enclosed by a contour using the shoelace
// formula. The result is the polygon area through the contour's points and is
// always non-negative; a contour of fewer than three points has zero area.
//
// Because border points sit on pixel centres, the area of a solid W×H block is
// (W-1)*(H-1) rather than W*H.
func ContourArea(c Contour) float64 {
	n := len(c)
	if n < 3 {
		return 0
	}
	var a float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += float64(c[i].X)*float64(c[j].Y) - float64(c[j].X)*float64(c[i].Y)
	}
	return math.Abs(a) / 2
}

// ArcLength returns the perimeter (closed) or length (open) of a curve as the
// sum of Euclidean distances between consecutive points. When closed is true
// the segment from the last point back to the first is included.
func ArcLength(c []Point, closed bool) float64 {
	n := len(c)
	if n < 2 {
		return 0
	}
	var total float64
	for i := 0; i < n-1; i++ {
		total += math.Hypot(float64(c[i+1].X-c[i].X), float64(c[i+1].Y-c[i].Y))
	}
	if closed {
		total += math.Hypot(float64(c[0].X-c[n-1].X), float64(c[0].Y-c[n-1].Y))
	}
	return total
}

// Rect is an axis-aligned rectangle with an integer top-left corner (X, Y) and
// a Width and Height in pixels, matching OpenCV's cv::Rect.
type Rect struct {
	X      int
	Y      int
	Width  int
	Height int
}

// BoundingRect returns the smallest upright rectangle that contains every point.
// Width and Height count pixels inclusively, so a single point yields a 1×1
// rectangle. It panics on an empty point set.
func BoundingRect(pts []Point) Rect {
	if len(pts) == 0 {
		panic("cv: BoundingRect on empty point set")
	}
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := pts[0].X, pts[0].Y
	for _, p := range pts {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	return Rect{X: minX, Y: minY, Width: maxX - minX + 1, Height: maxY - minY + 1}
}

// ConvexHull computes the convex hull of a set of points with Andrew's monotone
// chain algorithm and returns its vertices in counter-clockwise order (in image
// coordinates, where y grows downward). Duplicate and collinear interior points
// are removed. Fewer than three input points are returned as-is (deduplicated).
func ConvexHull(pts []Point) []Point {
	n := len(pts)
	if n < 3 {
		out := make([]Point, n)
		copy(out, pts)
		return out
	}
	sorted := make([]Point, n)
	copy(sorted, pts)
	sortPoints(sorted)
	// Remove duplicates.
	uniq := sorted[:1]
	for _, p := range sorted[1:] {
		last := uniq[len(uniq)-1]
		if p.X != last.X || p.Y != last.Y {
			uniq = append(uniq, p)
		}
	}
	if len(uniq) < 3 {
		return uniq
	}
	cross := func(o, a, b Point) int {
		return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
	}
	m := len(uniq)
	hull := make([]Point, 0, 2*m)
	// Lower hull.
	for _, p := range uniq {
		for len(hull) >= 2 && cross(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	// Upper hull.
	lower := len(hull) + 1
	for i := m - 2; i >= 0; i-- {
		p := uniq[i]
		for len(hull) >= lower && cross(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	return hull[:len(hull)-1]
}

// sortPoints orders points by X then Y (ascending), used by ConvexHull.
func sortPoints(pts []Point) {
	for i := 1; i < len(pts); i++ {
		for j := i; j > 0; j-- {
			a, b := pts[j-1], pts[j]
			if a.X < b.X || (a.X == b.X && a.Y <= b.Y) {
				break
			}
			pts[j-1], pts[j] = pts[j], pts[j-1]
		}
	}
}

// RotatedRect is a rectangle that may be rotated about its centre, matching
// OpenCV's cv::RotatedRect. The centre is in fractional pixel coordinates,
// Width and Height are the side lengths, and Angle is the rotation in degrees.
type RotatedRect struct {
	CenterX float64
	CenterY float64
	Width   float64
	Height  float64
	Angle   float64
}

// Points returns the four corner points of the rotated rectangle in order,
// rounded to the nearest pixel.
func (r RotatedRect) Points() [4]Point {
	rad := r.Angle * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	hw, hh := r.Width/2, r.Height/2
	// Corners in the rectangle's own frame.
	local := [4][2]float64{{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}}
	var out [4]Point
	for i, p := range local {
		x := r.CenterX + p[0]*c - p[1]*s
		y := r.CenterY + p[0]*s + p[1]*c
		out[i] = Point{X: int(math.Round(x)), Y: int(math.Round(y))}
	}
	return out
}

// MinAreaRect returns the minimum-area rotated rectangle enclosing the points,
// found by rotating calipers over the convex hull: for each hull edge the
// axis-aligned bounding box in that edge's frame is measured and the smallest
// is kept. It panics on an empty point set.
func MinAreaRect(pts []Point) RotatedRect {
	if len(pts) == 0 {
		panic("cv: MinAreaRect on empty point set")
	}
	hull := ConvexHull(pts)
	if len(hull) == 1 {
		return RotatedRect{CenterX: float64(hull[0].X), CenterY: float64(hull[0].Y)}
	}
	if len(hull) == 2 {
		dx := float64(hull[1].X - hull[0].X)
		dy := float64(hull[1].Y - hull[0].Y)
		return RotatedRect{
			CenterX: float64(hull[0].X+hull[1].X) / 2,
			CenterY: float64(hull[0].Y+hull[1].Y) / 2,
			Width:   math.Hypot(dx, dy),
			Height:  0,
			Angle:   math.Atan2(dy, dx) * 180 / math.Pi,
		}
	}
	best := RotatedRect{}
	bestArea := math.Inf(1)
	n := len(hull)
	for i := 0; i < n; i++ {
		a := hull[i]
		b := hull[(i+1)%n]
		ex := float64(b.X - a.X)
		ey := float64(b.Y - a.Y)
		length := math.Hypot(ex, ey)
		if length == 0 {
			continue
		}
		ux, uy := ex/length, ey/length // edge direction (unit)
		// Project all hull points onto the edge axis (u) and its normal.
		minU, maxU := math.Inf(1), math.Inf(-1)
		minV, maxV := math.Inf(1), math.Inf(-1)
		for _, p := range hull {
			px, py := float64(p.X), float64(p.Y)
			pu := px*ux + py*uy
			pv := -px*uy + py*ux
			minU, maxU = math.Min(minU, pu), math.Max(maxU, pu)
			minV, maxV = math.Min(minV, pv), math.Max(maxV, pv)
		}
		w := maxU - minU
		h := maxV - minV
		area := w * h
		if area < bestArea {
			bestArea = area
			cu := (minU + maxU) / 2
			cv := (minV + maxV) / 2
			// Map centre back to image coordinates.
			cx := cu*ux - cv*uy
			cy := cu*uy + cv*ux
			best = RotatedRect{
				CenterX: cx,
				CenterY: cy,
				Width:   w,
				Height:  h,
				Angle:   math.Atan2(uy, ux) * 180 / math.Pi,
			}
		}
	}
	return best
}

// ApproxPolyDP simplifies a curve with the Douglas–Peucker algorithm, dropping
// points that lie within epsilon of the retained polyline. Larger epsilon
// yields a coarser approximation. When closed is true the curve is treated as a
// closed polygon.
func ApproxPolyDP(curve []Point, epsilon float64, closed bool) []Point {
	n := len(curve)
	if n < 3 {
		out := make([]Point, n)
		copy(out, curve)
		return out
	}
	if closed {
		// Split the closed curve at its two most distant points, approximate
		// each half, then join.
		i0, i1 := 0, 0
		maxD := -1.0
		for i := 0; i < n; i++ {
			d := math.Hypot(float64(curve[i].X-curve[0].X), float64(curve[i].Y-curve[0].Y))
			if d > maxD {
				maxD = d
				i1 = i
			}
		}
		first := dpSegment(curve, i0, i1, epsilon)
		second := dpSegment(curve, i1, n-1, epsilon)
		// dpSegment includes both endpoints; stitch avoiding duplicates and the
		// wrap-around closing point.
		result := first
		result = append(result, second[1:]...)
		if len(result) > 1 {
			result = result[:len(result)-1]
		}
		return result
	}
	return dpSegment(curve, 0, n-1, epsilon)
}

// dpSegment recursively simplifies curve[lo..hi] inclusive and returns the kept
// points from lo to hi.
func dpSegment(curve []Point, lo, hi int, epsilon float64) []Point {
	if hi <= lo+1 {
		return []Point{curve[lo], curve[hi]}
	}
	a, b := curve[lo], curve[hi]
	maxD := -1.0
	idx := lo
	for i := lo + 1; i < hi; i++ {
		d := perpDistance(curve[i], a, b)
		if d > maxD {
			maxD = d
			idx = i
		}
	}
	if maxD <= epsilon {
		return []Point{a, b}
	}
	left := dpSegment(curve, lo, idx, epsilon)
	right := dpSegment(curve, idx, hi, epsilon)
	return append(left[:len(left)-1], right...)
}

// perpDistance returns the perpendicular distance from p to the line through a
// and b (or the distance to a when a==b).
func perpDistance(p, a, b Point) float64 {
	dx := float64(b.X - a.X)
	dy := float64(b.Y - a.Y)
	if dx == 0 && dy == 0 {
		return math.Hypot(float64(p.X-a.X), float64(p.Y-a.Y))
	}
	num := math.Abs(dy*float64(p.X-a.X) - dx*float64(p.Y-a.Y))
	return num / math.Hypot(dx, dy)
}

// Moments holds the spatial, central and normalised central moments of an image
// up to third order, mirroring OpenCV's cv::Moments. Spatial moments are Mpq,
// central moments (translation invariant) are Mupq and normalised central
// moments (scale invariant) are Nupq.
type Moments struct {
	M00, M10, M01, M20, M11, M02, M30, M21, M12, M03 float64
	Mu20, Mu11, Mu02, Mu30, Mu21, Mu12, Mu03         float64
	Nu20, Nu11, Nu02, Nu30, Nu21, Nu12, Nu03         float64
}

// Centroid returns the intensity-weighted centre of mass (M10/M00, M01/M00). It
// returns (0, 0) for an image of zero total mass.
func (m Moments) Centroid() (x, y float64) {
	if m.M00 == 0 {
		return 0, 0
	}
	return m.M10 / m.M00, m.M01 / m.M00
}

// ImageMoments computes the moments of a single-channel image, weighting each
// pixel (x, y) by its sample value. For a binary mask this yields geometric
// moments of the white region. It panics if src is not single-channel.
func ImageMoments(src *Mat) Moments {
	requireChannels(src, 1, "ImageMoments")
	var m Moments
	for y := 0; y < src.Rows; y++ {
		fy := float64(y)
		for x := 0; x < src.Cols; x++ {
			v := float64(src.Data[y*src.Cols+x])
			if v == 0 {
				continue
			}
			fx := float64(x)
			m.M00 += v
			m.M10 += fx * v
			m.M01 += fy * v
			m.M20 += fx * fx * v
			m.M11 += fx * fy * v
			m.M02 += fy * fy * v
			m.M30 += fx * fx * fx * v
			m.M21 += fx * fx * fy * v
			m.M12 += fx * fy * fy * v
			m.M03 += fy * fy * fy * v
		}
	}
	if m.M00 != 0 {
		cx := m.M10 / m.M00
		cy := m.M01 / m.M00
		m.Mu20 = m.M20 - cx*m.M10
		m.Mu11 = m.M11 - cx*m.M01
		m.Mu02 = m.M02 - cy*m.M01
		m.Mu30 = m.M30 - 3*cx*m.M20 + 2*cx*cx*m.M10
		m.Mu21 = m.M21 - 2*cx*m.M11 - cy*m.M20 + 2*cx*cx*m.M01
		m.Mu12 = m.M12 - 2*cy*m.M11 - cx*m.M02 + 2*cy*cy*m.M10
		m.Mu03 = m.M03 - 3*cy*m.M02 + 2*cy*cy*m.M01
		// Normalised central moments: nu_pq = mu_pq / m00^(1+(p+q)/2).
		inv2 := 1 / (m.M00 * m.M00)
		s2 := inv2
		s25 := inv2 / math.Sqrt(m.M00)
		m.Nu20 = m.Mu20 * s2
		m.Nu11 = m.Mu11 * s2
		m.Nu02 = m.Mu02 * s2
		m.Nu30 = m.Mu30 * s25
		m.Nu21 = m.Mu21 * s25
		m.Nu12 = m.Mu12 * s25
		m.Nu03 = m.Mu03 * s25
	}
	return m
}

// absInt returns the absolute value of an int.
func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

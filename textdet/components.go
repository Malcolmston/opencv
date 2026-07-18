package textdet

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Connectivity selects the pixel neighbourhood used when grouping foreground
// pixels into connected components.
type Connectivity int

const (
	// Conn4 considers the four edge-adjacent neighbours (N, S, E, W).
	Conn4 Connectivity = 4
	// Conn8 additionally considers the four diagonal neighbours.
	Conn8 Connectivity = 8
)

// textdetUnionFind is a disjoint-set forest over provisional component labels.
type textdetUnionFind struct {
	parent []int
}

func (u *textdetUnionFind) makeSet() int {
	id := len(u.parent)
	u.parent = append(u.parent, id)
	return id
}

func (u *textdetUnionFind) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

func (u *textdetUnionFind) union(a, b int) {
	ra, rb := u.find(a), u.find(b)
	if ra == rb {
		return
	}
	if ra < rb {
		u.parent[rb] = ra
	} else {
		u.parent[ra] = rb
	}
}

// textdetLabelMask labels a boolean foreground mask with two-pass union-find
// and returns a dense label image (0 = background, 1..count are components) and
// the number of foreground components.
func textdetLabelMask(fg []bool, rows, cols int, conn Connectivity) (labels []int, count int) {
	prov := make([]int, rows*cols)
	uf := &textdetUnionFind{parent: []int{0}} // index 0 reserved for background
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			idx := y*cols + x
			if !fg[idx] {
				continue
			}
			var neigh []int
			if x > 0 && prov[idx-1] != 0 {
				neigh = append(neigh, prov[idx-1])
			}
			if y > 0 && prov[idx-cols] != 0 {
				neigh = append(neigh, prov[idx-cols])
			}
			if conn == Conn8 {
				if y > 0 && x > 0 && prov[idx-cols-1] != 0 {
					neigh = append(neigh, prov[idx-cols-1])
				}
				if y > 0 && x < cols-1 && prov[idx-cols+1] != 0 {
					neigh = append(neigh, prov[idx-cols+1])
				}
			}
			if len(neigh) == 0 {
				prov[idx] = uf.makeSet()
				continue
			}
			m := neigh[0]
			for _, n := range neigh[1:] {
				if n < m {
					m = n
				}
			}
			prov[idx] = m
			for _, n := range neigh {
				uf.union(m, n)
			}
		}
	}
	// Second pass: flatten to consecutive labels.
	remap := make(map[int]int)
	labels = make([]int, rows*cols)
	for i, p := range prov {
		if p == 0 {
			continue
		}
		root := uf.find(p)
		lab, ok := remap[root]
		if !ok {
			count++
			lab = count
			remap[root] = lab
		}
		labels[i] = lab
	}
	return labels, count
}

// Component describes one connected foreground region: its label, pixel area,
// axis-aligned bounding box and centroid.
type Component struct {
	// Label is the component's identifier within its [ComponentSet] (>= 1).
	Label int
	// Area is the number of foreground pixels in the component.
	Area int
	// Bounds is the smallest upright rectangle containing the component.
	Bounds cv.Rect
	// CentroidX is the mean column of the component's pixels.
	CentroidX float64
	// CentroidY is the mean row of the component's pixels.
	CentroidY float64
}

// AspectRatio returns the component's bounding-box width divided by its height.
// It returns 0 for a degenerate zero-height box.
func (c Component) AspectRatio() float64 {
	if c.Bounds.Height == 0 {
		return 0
	}
	return float64(c.Bounds.Width) / float64(c.Bounds.Height)
}

// FillRatio returns the fraction of the component's bounding box occupied by
// foreground pixels, a value in (0,1]. It returns 0 for an empty box.
func (c Component) FillRatio() float64 {
	area := c.Bounds.Width * c.Bounds.Height
	if area == 0 {
		return 0
	}
	return float64(c.Area) / float64(area)
}

// ComponentSet is the result of connected-component labelling: the dense label
// image together with per-component statistics.
type ComponentSet struct {
	// Labels is the row-major label image (0 = background).
	Labels []int
	// Rows is the image height.
	Rows int
	// Cols is the image width.
	Cols int
	// Components holds one entry per foreground component, ordered by label.
	Components []Component
}

// LabelComponents labels the foreground of a binary single-channel image (any
// non-zero sample is foreground) and returns the label image and per-component
// statistics. conn selects 4- or 8-connectivity. It returns [ErrEmpty] for an
// empty image and [ErrInvalidArgument] for an invalid connectivity.
func LabelComponents(binary *cv.Mat, conn Connectivity) (*ComponentSet, error) {
	if conn != Conn4 && conn != Conn8 {
		return nil, ErrInvalidArgument
	}
	fg, rows, cols, err := textdetForeground(binary)
	if err != nil {
		return nil, err
	}
	labels, count := textdetLabelMask(fg, rows, cols, conn)

	area := make([]int, count+1)
	minX := make([]int, count+1)
	minY := make([]int, count+1)
	maxX := make([]int, count+1)
	maxY := make([]int, count+1)
	sumX := make([]float64, count+1)
	sumY := make([]float64, count+1)
	for l := 1; l <= count; l++ {
		minX[l], minY[l] = cols, rows
		maxX[l], maxY[l] = -1, -1
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			l := labels[y*cols+x]
			if l == 0 {
				continue
			}
			area[l]++
			if x < minX[l] {
				minX[l] = x
			}
			if x > maxX[l] {
				maxX[l] = x
			}
			if y < minY[l] {
				minY[l] = y
			}
			if y > maxY[l] {
				maxY[l] = y
			}
			sumX[l] += float64(x)
			sumY[l] += float64(y)
		}
	}
	comps := make([]Component, 0, count)
	for l := 1; l <= count; l++ {
		comps = append(comps, Component{
			Label:     l,
			Area:      area[l],
			Bounds:    cv.Rect{X: minX[l], Y: minY[l], Width: maxX[l] - minX[l] + 1, Height: maxY[l] - minY[l] + 1},
			CentroidX: sumX[l] / float64(area[l]),
			CentroidY: sumY[l] / float64(area[l]),
		})
	}
	return &ComponentSet{Labels: labels, Rows: rows, Cols: cols, Components: comps}, nil
}

// Count returns the number of foreground components in the set.
func (s *ComponentSet) Count() int { return len(s.Components) }

// Mask renders a single component as a fresh single-channel 0/255 [cv.Mat] of
// the full image size. Pixels belonging to the given label are 255. It returns
// [ErrInvalidArgument] if label is out of range.
func (s *ComponentSet) Mask(label int) (*cv.Mat, error) {
	if label < 1 || label > len(s.Components) {
		return nil, ErrInvalidArgument
	}
	dst := cv.NewMat(s.Rows, s.Cols, 1)
	for i, l := range s.Labels {
		if l == label {
			dst.Data[i] = 255
		}
	}
	return dst, nil
}

// FilterBySize returns the components whose pixel area lies in the inclusive
// range [minArea, maxArea]. A maxArea <= 0 means no upper bound.
func FilterBySize(comps []Component, minArea, maxArea int) []Component {
	out := make([]Component, 0, len(comps))
	for _, c := range comps {
		if c.Area < minArea {
			continue
		}
		if maxArea > 0 && c.Area > maxArea {
			continue
		}
		out = append(out, c)
	}
	return out
}

// FilterByAspectRatio returns the components whose bounding-box aspect ratio
// (width/height) lies in the inclusive range [minRatio, maxRatio].
func FilterByAspectRatio(comps []Component, minRatio, maxRatio float64) []Component {
	out := make([]Component, 0, len(comps))
	for _, c := range comps {
		r := c.AspectRatio()
		if r >= minRatio && r <= maxRatio {
			out = append(out, c)
		}
	}
	return out
}

// FilterByFillRatio returns the components whose foreground fill ratio
// (area / bounding-box area) lies in the inclusive range [minFill, maxFill].
func FilterByFillRatio(comps []Component, minFill, maxFill float64) []Component {
	out := make([]Component, 0, len(comps))
	for _, c := range comps {
		f := c.FillRatio()
		if f >= minFill && f <= maxFill {
			out = append(out, c)
		}
	}
	return out
}

// TextComponentOptions bounds the geometric properties a connected component
// must satisfy to be considered a text-glyph candidate by [FilterTextComponents].
type TextComponentOptions struct {
	// MinArea and MaxArea bound the pixel area. MaxArea <= 0 disables the
	// upper bound.
	MinArea, MaxArea int
	// MinAspect and MaxAspect bound the bounding-box aspect ratio (width /
	// height). Typical glyphs are taller than wide or roughly square.
	MinAspect, MaxAspect float64
	// MinFill and MaxFill bound the foreground fill ratio, rejecting both
	// near-empty frames and solid blocks.
	MinFill, MaxFill float64
}

// DefaultTextComponentOptions returns loose defaults suitable for medium-sized
// glyphs: area in [10, 0], aspect in [0.05, 3], fill in [0.1, 0.95].
func DefaultTextComponentOptions() TextComponentOptions {
	return TextComponentOptions{
		MinArea:   10,
		MaxArea:   0,
		MinAspect: 0.05,
		MaxAspect: 3.0,
		MinFill:   0.10,
		MaxFill:   0.95,
	}
}

// FilterTextComponents keeps only the components that satisfy every bound in
// opts, returning the glyph-like candidates in input order.
func FilterTextComponents(comps []Component, opts TextComponentOptions) []Component {
	out := make([]Component, 0, len(comps))
	for _, c := range comps {
		if c.Area < opts.MinArea {
			continue
		}
		if opts.MaxArea > 0 && c.Area > opts.MaxArea {
			continue
		}
		r := c.AspectRatio()
		if r < opts.MinAspect || r > opts.MaxAspect {
			continue
		}
		f := c.FillRatio()
		if f < opts.MinFill || f > opts.MaxFill {
			continue
		}
		out = append(out, c)
	}
	return out
}

// TextLine is a horizontal grouping of glyph components that share a baseline.
type TextLine struct {
	// Bounds is the upright bounding box enclosing every member component.
	Bounds cv.Rect
	// Components holds the members ordered left-to-right by bounding-box X.
	Components []Component
}

// GroupTextLines groups glyph-like components into text lines. Two components
// join the same line when their vertical extents overlap by at least
// vertOverlap (a fraction of the smaller height, in [0,1]) and the horizontal
// gap between them is at most maxGap pixels. Components are first sorted
// left-to-right; the result is ordered top-to-bottom by line top edge. It
// returns [ErrInvalidArgument] if vertOverlap is outside [0,1] or maxGap < 0.
func GroupTextLines(comps []Component, vertOverlap float64, maxGap int) ([]TextLine, error) {
	if vertOverlap < 0 || vertOverlap > 1 || maxGap < 0 {
		return nil, ErrInvalidArgument
	}
	n := len(comps)
	ordered := make([]Component, n)
	copy(ordered, comps)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Bounds.X != ordered[j].Bounds.X {
			return ordered[i].Bounds.X < ordered[j].Bounds.X
		}
		return ordered[i].Bounds.Y < ordered[j].Bounds.Y
	})

	uf := &textdetUnionFind{parent: make([]int, n)}
	for i := range uf.parent {
		uf.parent[i] = i
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			a, b := ordered[i].Bounds, ordered[j].Bounds
			// Vertical overlap as a fraction of the shorter box.
			top := math.Max(float64(a.Y), float64(b.Y))
			bot := math.Min(float64(a.Y+a.Height), float64(b.Y+b.Height))
			ov := bot - top
			if ov <= 0 {
				continue
			}
			shorter := math.Min(float64(a.Height), float64(b.Height))
			if shorter <= 0 || ov/shorter < vertOverlap {
				continue
			}
			// Horizontal gap (0 if boxes overlap or touch).
			gap := b.X - (a.X + a.Width)
			if a.X > b.X {
				gap = a.X - (b.X + b.Width)
			}
			if gap <= maxGap {
				uf.union(i, j)
			}
		}
	}

	groups := make(map[int][]Component)
	for i := 0; i < n; i++ {
		r := uf.find(i)
		groups[r] = append(groups[r], ordered[i])
	}
	lines := make([]TextLine, 0, len(groups))
	for _, members := range groups {
		bx0, by0 := members[0].Bounds.X, members[0].Bounds.Y
		bx1, by1 := members[0].Bounds.X+members[0].Bounds.Width, members[0].Bounds.Y+members[0].Bounds.Height
		for _, m := range members[1:] {
			if m.Bounds.X < bx0 {
				bx0 = m.Bounds.X
			}
			if m.Bounds.Y < by0 {
				by0 = m.Bounds.Y
			}
			if m.Bounds.X+m.Bounds.Width > bx1 {
				bx1 = m.Bounds.X + m.Bounds.Width
			}
			if m.Bounds.Y+m.Bounds.Height > by1 {
				by1 = m.Bounds.Y + m.Bounds.Height
			}
		}
		lines = append(lines, TextLine{
			Bounds:     cv.Rect{X: bx0, Y: by0, Width: bx1 - bx0, Height: by1 - by0},
			Components: members,
		})
	}
	sort.SliceStable(lines, func(i, j int) bool {
		if lines[i].Bounds.Y != lines[j].Bounds.Y {
			return lines[i].Bounds.Y < lines[j].Bounds.Y
		}
		return lines[i].Bounds.X < lines[j].Bounds.X
	})
	return lines, nil
}

// EdgeDensityMap measures local edge density: it computes the gradient
// magnitude of src, marks strong edges (magnitude above magThresh), and for
// every pixel returns the fraction of edge pixels within a (2*radius+1)-square
// window. The result is a [cv.FloatMat] of the same size with values in [0,1].
// High values concentrate where text produces dense, high-contrast strokes. It
// returns [ErrEmpty] for an empty image and [ErrInvalidArgument] if radius < 1.
func EdgeDensityMap(src *cv.Mat, radius int, magThresh float64) (*cv.FloatMat, error) {
	if radius < 1 {
		return nil, ErrInvalidArgument
	}
	gray, rows, cols, err := textdetGray(src)
	if err != nil {
		return nil, err
	}
	gx, gy := textdetSobel(gray, rows, cols)
	edge := make([]float64, rows*cols) // 1.0 where strong edge, else 0
	for i := range edge {
		if math.Hypot(gx[i], gy[i]) >= magThresh {
			edge[i] = 1
		}
	}
	integ := textdetIntegral(edge, rows, cols)
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			x0 := x - radius
			y0 := y - radius
			x1 := x + radius
			y1 := y + radius
			if x0 < 0 {
				x0 = 0
			}
			if y0 < 0 {
				y0 = 0
			}
			if x1 >= cols {
				x1 = cols - 1
			}
			if y1 >= rows {
				y1 = rows - 1
			}
			sum := textdetRectSum(integ, cols, x0, y0, x1, y1)
			n := float64((x1 - x0 + 1) * (y1 - y0 + 1))
			out.Data[y*cols+x] = sum / n
		}
	}
	return out, nil
}

// LocalizeByEdgeDensity localizes candidate text regions by edge density. It
// builds an [EdgeDensityMap], thresholds it at densityThresh to form a binary
// mask, labels the mask with 8-connectivity, and returns the bounding boxes of
// components whose area is at least minArea, ordered top-to-bottom then
// left-to-right. It returns [ErrInvalidArgument] for radius < 1 or densityThresh
// outside [0,1].
func LocalizeByEdgeDensity(src *cv.Mat, radius int, magThresh, densityThresh float64, minArea int) ([]cv.Rect, error) {
	if densityThresh < 0 || densityThresh > 1 {
		return nil, ErrInvalidArgument
	}
	dm, err := EdgeDensityMap(src, radius, magThresh)
	if err != nil {
		return nil, err
	}
	fg := make([]bool, len(dm.Data))
	for i, v := range dm.Data {
		if v >= densityThresh {
			fg[i] = true
		}
	}
	labels, count := textdetLabelMask(fg, dm.Rows, dm.Cols, Conn8)
	type box struct{ minX, minY, maxX, maxY, area int }
	boxes := make([]box, count+1)
	for l := 1; l <= count; l++ {
		boxes[l] = box{minX: dm.Cols, minY: dm.Rows, maxX: -1, maxY: -1}
	}
	for y := 0; y < dm.Rows; y++ {
		for x := 0; x < dm.Cols; x++ {
			l := labels[y*dm.Cols+x]
			if l == 0 {
				continue
			}
			b := &boxes[l]
			b.area++
			if x < b.minX {
				b.minX = x
			}
			if x > b.maxX {
				b.maxX = x
			}
			if y < b.minY {
				b.minY = y
			}
			if y > b.maxY {
				b.maxY = y
			}
		}
	}
	var rects []cv.Rect
	for l := 1; l <= count; l++ {
		if boxes[l].area < minArea {
			continue
		}
		rects = append(rects, cv.Rect{
			X:      boxes[l].minX,
			Y:      boxes[l].minY,
			Width:  boxes[l].maxX - boxes[l].minX + 1,
			Height: boxes[l].maxY - boxes[l].minY + 1,
		})
	}
	sort.SliceStable(rects, func(i, j int) bool {
		if rects[i].Y != rects[j].Y {
			return rects[i].Y < rects[j].Y
		}
		return rects[i].X < rects[j].X
	})
	return rects, nil
}

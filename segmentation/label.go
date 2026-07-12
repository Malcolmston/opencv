package segmentation

import cv "github.com/malcolmston/opencv"

// LabelMap is a dense, per-pixel region labelling produced by the region-based
// segmenters in this package. Labels are consecutive integers in [0, Count):
// pixel (x, y) — column x, row y — has label Labels[y*Cols+x].
//
// A [cv.Mat] cannot represent an arbitrary labelling because it stores unsigned
// 8-bit samples and a segmentation may contain more than 256 regions. LabelMap
// therefore holds the labels as a flat []int and offers [LabelMap.Colorize] and
// [LabelMap.BoundaryMask] to turn a labelling into a viewable [cv.Mat].
type LabelMap struct {
	// Rows is the image height.
	Rows int
	// Cols is the image width.
	Cols int
	// Count is the number of distinct regions; every label lies in [0, Count).
	Count int
	// Labels holds the region index of each pixel in row-major order, length
	// Rows*Cols.
	Labels []int
}

// newLabelMap allocates a zeroed LabelMap of the given size.
func newLabelMap(rows, cols int) *LabelMap {
	return &LabelMap{Rows: rows, Cols: cols, Labels: make([]int, rows*cols)}
}

// At returns the region label of pixel (x, y). It panics if the coordinates are
// out of range.
func (l *LabelMap) At(x, y int) int {
	if x < 0 || x >= l.Cols || y < 0 || y >= l.Rows {
		panic("segmentation: LabelMap.At out of range")
	}
	return l.Labels[y*l.Cols+x]
}

// RegionSizes returns the pixel count of every region, indexed by label. The
// result has length Count.
func (l *LabelMap) RegionSizes() []int {
	sizes := make([]int, l.Count)
	for _, v := range l.Labels {
		if v >= 0 && v < l.Count {
			sizes[v]++
		}
	}
	return sizes
}

// BoundingRects returns the tight bounding rectangle of every region, indexed by
// label (length Count). The rectangles use the inclusive convention of
// [cv.BoundingRect]: a single-pixel region yields a 1x1 rectangle. Regions with
// no pixels get the zero [cv.Rect].
func (l *LabelMap) BoundingRects() []cv.Rect {
	minX := make([]int, l.Count)
	minY := make([]int, l.Count)
	maxX := make([]int, l.Count)
	maxY := make([]int, l.Count)
	seen := make([]bool, l.Count)
	for y := 0; y < l.Rows; y++ {
		for x := 0; x < l.Cols; x++ {
			v := l.Labels[y*l.Cols+x]
			if v < 0 || v >= l.Count {
				continue
			}
			if !seen[v] {
				seen[v] = true
				minX[v], maxX[v] = x, x
				minY[v], maxY[v] = y, y
				continue
			}
			if x < minX[v] {
				minX[v] = x
			}
			if x > maxX[v] {
				maxX[v] = x
			}
			if y < minY[v] {
				minY[v] = y
			}
			if y > maxY[v] {
				maxY[v] = y
			}
		}
	}
	rects := make([]cv.Rect, l.Count)
	for v := 0; v < l.Count; v++ {
		if !seen[v] {
			continue
		}
		rects[v] = cv.Rect{X: minX[v], Y: minY[v], Width: maxX[v] - minX[v] + 1, Height: maxY[v] - minY[v] + 1}
	}
	return rects
}

// Colorize renders the labelling as a three-channel [cv.Mat], assigning each
// region a distinct, deterministic pseudo-random colour. It is intended for
// visual inspection of a segmentation.
func (l *LabelMap) Colorize() *cv.Mat {
	out := cv.NewMat(l.Rows, l.Cols, 3)
	for i, v := range l.Labels {
		c := regionColor(v)
		b := i * 3
		out.Data[b+0] = c[0]
		out.Data[b+1] = c[1]
		out.Data[b+2] = c[2]
	}
	return out
}

// BoundaryMask returns a single-channel [cv.Mat] whose pixels are 255 on region
// boundaries and 0 elsewhere. A pixel is a boundary when at least one of its
// 4-neighbours carries a different label, so the mask traces the outlines
// between regions.
func (l *LabelMap) BoundaryMask() *cv.Mat {
	out := cv.NewMat(l.Rows, l.Cols, 1)
	for y := 0; y < l.Rows; y++ {
		for x := 0; x < l.Cols; x++ {
			idx := y*l.Cols + x
			lv := l.Labels[idx]
			edge := false
			for _, o := range neighbors4 {
				nx, ny := x+o.dx, y+o.dy
				if nx < 0 || nx >= l.Cols || ny < 0 || ny >= l.Rows {
					continue
				}
				if l.Labels[ny*l.Cols+nx] != lv {
					edge = true
					break
				}
			}
			if edge {
				out.Data[idx] = 255
			}
		}
	}
	return out
}

// regionColor maps a label to a stable colour using a small integer hash, so
// neighbouring labels get visually distinct hues.
func regionColor(label int) [3]uint8 {
	if label < 0 {
		return [3]uint8{0, 0, 0}
	}
	// A simple integer hash (Knuth multiplicative) spreads consecutive labels
	// across the colour cube deterministically.
	h := uint32(label)*2654435761 + 1
	r := uint8(37 + (h>>0)%219)
	g := uint8(37 + (h>>8)%219)
	b := uint8(37 + (h>>16)%219)
	return [3]uint8{r, g, b}
}

// relabelConsecutive maps the arbitrary integer labels in raw (length rows*cols)
// to consecutive labels in [0, count) in order of first appearance, which makes
// the numbering deterministic and independent of the source label values.
func relabelConsecutive(raw []int) (labels []int, count int) {
	remap := make(map[int]int)
	labels = make([]int, len(raw))
	for i, v := range raw {
		nv, ok := remap[v]
		if !ok {
			nv = count
			remap[v] = nv
			count++
		}
		labels[i] = nv
	}
	return labels, count
}

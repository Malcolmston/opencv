package morph2

import (
	"container/heap"

	cv "github.com/malcolmston/opencv"
)

// WatershedRidge is the label assigned to watershed line (ridge) pixels by
// [Watershed].
const WatershedRidge int32 = -1

// LabelMap is a dense row-major grid of integer labels used for markers and
// watershed segmentation. A label of 0 means unlabelled/background, a positive
// value identifies a region, and [WatershedRidge] (-1) marks a watershed line.
// It is a thin helper container and does not duplicate [cv.Mat].
type LabelMap struct {
	// Rows is the grid height.
	Rows int
	// Cols is the grid width.
	Cols int
	// Labels holds Rows*Cols labels in row-major order.
	Labels []int32
}

// NewLabelMap allocates a zero-filled label map of the given size. It panics on
// a non-positive dimension.
func NewLabelMap(rows, cols int) *LabelMap {
	if rows <= 0 || cols <= 0 {
		panic("morph2: NewLabelMap requires positive size")
	}
	return &LabelMap{Rows: rows, Cols: cols, Labels: make([]int32, rows*cols)}
}

// At returns the label at (y, x). It panics on out-of-range coordinates.
func (l *LabelMap) At(y, x int) int32 {
	if y < 0 || y >= l.Rows || x < 0 || x >= l.Cols {
		panic("morph2: LabelMap.At out of range")
	}
	return l.Labels[y*l.Cols+x]
}

// Set stores label at (y, x). It panics on out-of-range coordinates.
func (l *LabelMap) Set(y, x int, label int32) {
	if y < 0 || y >= l.Rows || x < 0 || x >= l.Cols {
		panic("morph2: LabelMap.Set out of range")
	}
	l.Labels[y*l.Cols+x] = label
}

// NumLabels returns the largest positive label present, which for a dense
// 1..N labelling equals the number of regions.
func (l *LabelMap) NumLabels() int {
	m := int32(0)
	for _, v := range l.Labels {
		if v > m {
			m = v
		}
	}
	return int(m)
}

// Clone returns an independent deep copy of the label map.
func (l *LabelMap) Clone() *LabelMap {
	c := &LabelMap{Rows: l.Rows, Cols: l.Cols, Labels: make([]int32, len(l.Labels))}
	copy(c.Labels, l.Labels)
	return c
}

// Ridges returns a binary image marking the watershed line pixels (label
// [WatershedRidge]).
func (l *LabelMap) Ridges() *cv.Mat {
	out := cv.NewMat(l.Rows, l.Cols, 1)
	for i, v := range l.Labels {
		if v == WatershedRidge {
			out.Data[i] = 255
		}
	}
	return out
}

// ToMat renders the label map as a single-channel [cv.Mat] for visualisation:
// background (0) maps to 0, watershed ridges map to 255, and each positive
// label maps to a deterministic value in 1..254.
func (l *LabelMap) ToMat() *cv.Mat {
	out := cv.NewMat(l.Rows, l.Cols, 1)
	for i, v := range l.Labels {
		switch {
		case v == WatershedRidge:
			out.Data[i] = 255
		case v > 0:
			out.Data[i] = uint8((int(v)-1)%254 + 1)
		}
	}
	return out
}

// ConnectedComponentMarkers labels the connected foreground (non-zero)
// components of a binary image with consecutive positive integers starting at
// 1, leaving background at 0. The result is suitable as seed markers for
// [Watershed]. Connectivity selects 4- or 8-connectivity. It panics on
// multi-channel input.
func ConnectedComponentMarkers(src *cv.Mat, conn Connectivity) *LabelMap {
	requireGray(src)
	offs := neighbourOffsets(conn)
	rows, cols := src.Rows, src.Cols
	lm := NewLabelMap(rows, cols)
	var next int32
	queue := make([]int, 0, 64)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := idx(y, x, cols)
			if src.Data[p] == 0 || lm.Labels[p] != 0 {
				continue
			}
			next++
			lm.Labels[p] = next
			queue = queue[:0]
			queue = append(queue, p)
			for head := 0; head < len(queue); head++ {
				cp := queue[head]
				cy, cx := cp/cols, cp%cols
				for _, o := range offs {
					yy, xx := cy+o[0], cx+o[1]
					if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
						continue
					}
					q := idx(yy, xx, cols)
					if src.Data[q] != 0 && lm.Labels[q] == 0 {
						lm.Labels[q] = next
						queue = append(queue, q)
					}
				}
			}
		}
	}
	return lm
}

// wsItem is a priority-queue entry for the flooding order.
type wsItem struct {
	level uint8
	order int
	pos   int
}

type wsHeap []wsItem

// Len reports the number of items in the heap; it is part of heap.Interface.
func (h wsHeap) Len() int { return len(h) }

// Less orders items by ascending intensity level, breaking ties by ascending
// discovery order; it is part of heap.Interface.
func (h wsHeap) Less(i, j int) bool {
	if h[i].level != h[j].level {
		return h[i].level < h[j].level
	}
	return h[i].order < h[j].order
}

// Swap exchanges the items at indices i and j; it is part of heap.Interface.
func (h wsHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// Push appends x, a wsItem, to the heap; it is part of heap.Interface.
func (h *wsHeap) Push(x interface{}) { *h = append(*h, x.(wsItem)) }

// Pop removes and returns the last item of the heap; it is part of
// heap.Interface.
func (h *wsHeap) Pop() interface{} {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// Watershed performs marker-controlled watershed segmentation of a grey-scale
// image using Meyer's flooding algorithm. The image is treated as a
// topographic relief (typically a gradient magnitude); flooding starts from the
// positive-labelled seeds in markers and grows toward higher intensities.
// Pixels equidistant between two catchment basins become watershed lines
// (label [WatershedRidge]).
//
// markers must have the same dimensions as the image: positive labels are
// seeds, 0 is unlabelled. Ordering ties are broken by discovery order, so the
// result is deterministic. It returns a new [LabelMap] and panics on
// multi-channel input or a size mismatch.
func Watershed(image *cv.Mat, markers *LabelMap, conn Connectivity) *LabelMap {
	requireGray(image)
	if markers.Rows != image.Rows || markers.Cols != image.Cols {
		panic("morph2: Watershed marker size mismatch")
	}
	offs := neighbourOffsets(conn)
	rows, cols := image.Rows, image.Cols
	out := markers.Clone()
	lab := out.Labels
	inQueue := make([]bool, rows*cols)

	h := &wsHeap{}
	heap.Init(h)
	order := 0
	push := func(p int) {
		if inQueue[p] {
			return
		}
		inQueue[p] = true
		heap.Push(h, wsItem{level: image.Data[p], order: order, pos: p})
		order++
	}

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := idx(y, x, cols)
			if lab[p] <= 0 {
				continue
			}
			for _, o := range offs {
				yy, xx := y+o[0], x+o[1]
				if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
					continue
				}
				q := idx(yy, xx, cols)
				if lab[q] == 0 {
					push(q)
				}
			}
		}
	}

	for h.Len() > 0 {
		it := heap.Pop(h).(wsItem)
		p := it.pos
		py, px := p/cols, p%cols
		var found int32
		ridge := false
		for _, o := range offs {
			yy, xx := py+o[0], px+o[1]
			if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
				continue
			}
			nl := lab[idx(yy, xx, cols)]
			if nl > 0 {
				if found == 0 {
					found = nl
				} else if found != nl {
					ridge = true
				}
			}
		}
		if ridge {
			lab[p] = WatershedRidge
			continue
		}
		if found == 0 {
			continue
		}
		lab[p] = found
		for _, o := range offs {
			yy, xx := py+o[0], px+o[1]
			if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
				continue
			}
			q := idx(yy, xx, cols)
			if lab[q] == 0 && !inQueue[q] {
				push(q)
			}
		}
	}
	return out
}

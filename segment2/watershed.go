package segment2

import (
	"container/heap"
	"math"

	cv "github.com/malcolmston/opencv"
)

// WatershedLine is the label value assigned to pixels that lie on a watershed
// ridge separating two basins.
const WatershedLine = -1

// GradientMagnitude returns the Sobel gradient magnitude of img as a flat
// row-major []float64 in the image's own coordinate order. A colour image is
// first reduced to luminance. This is the relief that [Watershed] floods.
//
// It panics if img is empty.
func GradientMagnitude(img *cv.Mat) []float64 {
	segment2requireNonEmpty(img, "GradientMagnitude")
	gray := segment2gray(img)
	return segment2sobelMag(gray, img.Rows, img.Cols)
}

// segment2dt1d is the exact 1-D squared-distance transform of Felzenszwalb and
// Huttenlocher. f holds per-column costs; the return is the lower envelope of
// the parabolas rooted at each sample.
func segment2dt1d(f []float64) []float64 {
	n := len(f)
	d := make([]float64, n)
	v := make([]int, n)
	z := make([]float64, n+1)
	k := 0
	v[0] = 0
	z[0] = math.Inf(-1)
	z[1] = math.Inf(1)
	for q := 1; q < n; q++ {
		s := ((f[q] + float64(q*q)) - (f[v[k]] + float64(v[k]*v[k]))) / float64(2*q-2*v[k])
		for s <= z[k] {
			k--
			s = ((f[q] + float64(q*q)) - (f[v[k]] + float64(v[k]*v[k]))) / float64(2*q-2*v[k])
		}
		k++
		v[k] = q
		z[k] = s
		z[k+1] = math.Inf(1)
	}
	k = 0
	for q := 0; q < n; q++ {
		for z[k+1] < float64(q) {
			k++
		}
		dq := float64(q - v[k])
		d[q] = dq*dq + f[v[k]]
	}
	return d
}

// DistanceTransform returns, for every pixel of a binary single-channel image,
// the exact Euclidean distance to the nearest zero (background) pixel. Non-zero
// samples are foreground. The result is a flat row-major []float64 with the same
// layout as the image. Distances are 0 on background pixels.
//
// It panics if img is empty or has more than one channel.
func DistanceTransform(img *cv.Mat) []float64 {
	segment2requireNonEmpty(img, "DistanceTransform")
	if img.Channels != 1 {
		panic("segment2: DistanceTransform requires a single-channel image")
	}
	rows, cols := img.Rows, img.Cols
	const inf = 1e20
	grid := make([]float64, rows*cols)
	for i, v := range img.Data {
		if v == 0 {
			grid[i] = 0
		} else {
			grid[i] = inf
		}
	}
	// Transform along columns.
	col := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			col[y] = grid[y*cols+x]
		}
		dc := segment2dt1d(col)
		for y := 0; y < rows; y++ {
			grid[y*cols+x] = dc[y]
		}
	}
	// Transform along rows.
	row := make([]float64, cols)
	for y := 0; y < rows; y++ {
		copy(row, grid[y*cols:(y+1)*cols])
		dr := segment2dt1d(row)
		for x := 0; x < cols; x++ {
			grid[y*cols+x] = math.Sqrt(dr[x])
		}
	}
	return grid
}

// segment2wsItem is a priority-queue entry for Meyer flooding.
type segment2wsItem struct {
	priority float64
	order    int // FIFO tie-break for determinism
	index    int
}

// segment2wsHeap is a min-heap of segment2wsItem implementing heap.Interface,
// ordering by ascending priority with FIFO tie-breaking for deterministic
// Meyer flooding.
type segment2wsHeap []segment2wsItem

// Len reports the number of items in the heap; it is part of heap.Interface.
func (h segment2wsHeap) Len() int { return len(h) }

// Less orders items by ascending priority, breaking ties by ascending FIFO
// order; it is part of heap.Interface.
func (h segment2wsHeap) Less(i, j int) bool {
	if h[i].priority != h[j].priority {
		return h[i].priority < h[j].priority
	}
	return h[i].order < h[j].order
}

// Swap exchanges the items at indices i and j; it is part of heap.Interface.
func (h segment2wsHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// Push appends x, a segment2wsItem, to the heap; it is part of heap.Interface.
func (h *segment2wsHeap) Push(x interface{}) { *h = append(*h, x.(segment2wsItem)) }

// Pop removes and returns the last item of the heap; it is part of
// heap.Interface.
func (h *segment2wsHeap) Pop() interface{} {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// Watershed performs marker-controlled watershed segmentation with Meyer's
// priority-flooding algorithm over the gradient relief of img. markers is a flat
// row-major labelling the same size as img: positive values are seeds that grow,
// 0 marks unknown pixels to be flooded, and the returned slice replaces every
// unknown pixel with the label of the basin that reached it. Pixels equidistant
// between two basins become [WatershedLine] (-1).
//
// The returned slice is newly allocated; markers is not modified.
//
// It panics if img is empty or len(markers) != img.Rows*img.Cols.
func Watershed(img *cv.Mat, markers []int) []int {
	segment2requireNonEmpty(img, "Watershed")
	rows, cols := img.Rows, img.Cols
	if len(markers) != rows*cols {
		panic("segment2: Watershed markers size mismatch")
	}
	relief := GradientMagnitude(img)
	labels := make([]int, len(markers))
	copy(labels, markers)

	h := &segment2wsHeap{}
	heap.Init(h)
	order := 0
	// Seed the queue with pixels bordering a labelled marker.
	inQueue := make([]bool, len(labels))
	for i := 0; i < len(labels); i++ {
		if labels[i] <= 0 {
			continue
		}
		x := i % cols
		y := i / cols
		for _, o := range segment2neighbors4 {
			nx, ny := x+o.dx, y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			ni := ny*cols + nx
			if labels[ni] == 0 && !inQueue[ni] {
				inQueue[ni] = true
				heap.Push(h, segment2wsItem{priority: relief[ni], order: order, index: ni})
				order++
			}
		}
	}

	for h.Len() > 0 {
		it := heap.Pop(h).(segment2wsItem)
		i := it.index
		x := i % cols
		y := i / cols
		found := 0
		conflict := false
		for _, o := range segment2neighbors4 {
			nx, ny := x+o.dx, y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			nl := labels[ny*cols+nx]
			if nl > 0 {
				if found == 0 {
					found = nl
				} else if nl != found {
					conflict = true
				}
			}
		}
		if conflict {
			labels[i] = WatershedLine
		} else if found > 0 {
			labels[i] = found
			for _, o := range segment2neighbors4 {
				nx, ny := x+o.dx, y+o.dy
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					continue
				}
				ni := ny*cols + nx
				if labels[ni] == 0 && !inQueue[ni] {
					inQueue[ni] = true
					heap.Push(h, segment2wsItem{priority: relief[ni], order: order, index: ni})
					order++
				}
			}
		}
	}
	return labels
}

// WatershedFromMarkers is a convenience wrapper over [Watershed] that takes and
// returns a [LabelMap]. seeds must match the size of img; its positive labels
// are grown, its zeros are flooded. The returned map uses [WatershedLine] (-1)
// for ridge pixels.
//
// It panics if img is empty or seeds does not match img in size.
func WatershedFromMarkers(img *cv.Mat, seeds *LabelMap) *LabelMap {
	segment2requireNonEmpty(img, "WatershedFromMarkers")
	if seeds.Rows != img.Rows || seeds.Cols != img.Cols {
		panic("segment2: WatershedFromMarkers seed size mismatch")
	}
	labels := Watershed(img, seeds.Labels)
	out := &LabelMap{Rows: img.Rows, Cols: img.Cols, Labels: labels}
	maxL := 0
	for _, l := range labels {
		if l > maxL {
			maxL = l
		}
	}
	out.NumLabels = maxL + 1
	return out
}

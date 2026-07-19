package structured_light

import "container/heap"

// qgItem is a candidate pixel waiting to be unwrapped, ordered by quality.
type qgItem struct {
	idx  int     // flat pixel index to unwrap
	from int     // already-unwrapped neighbour that seeds it
	qual float64 // priority: the candidate pixel's own quality
}

// qgQueue is a max-heap of qgItem by quality.
type qgQueue []qgItem

// Len reports the number of items in the queue; it is part of heap.Interface.
func (q qgQueue) Len() int { return len(q) }

// Less orders items by descending quality, so the highest-quality candidate is
// at the root; it is part of heap.Interface.
func (q qgQueue) Less(i, j int) bool { return q[i].qual > q[j].qual }

// Swap exchanges the items at indices i and j; it is part of heap.Interface.
func (q qgQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }

// Push appends x, a qgItem, to the queue; it is part of heap.Interface.
func (q *qgQueue) Push(x interface{}) { *q = append(*q, x.(qgItem)) }

// Pop removes and returns the last item of the queue; it is part of
// heap.Interface.
func (q *qgQueue) Pop() interface{} {
	old := *q
	n := len(old)
	it := old[n-1]
	*q = old[:n-1]
	return it
}

// QualityGuidedUnwrap unwraps a 2-D wrapped phase map by flood-filling outward
// from the highest-quality pixel, always extending the unwrapped region through
// its best-quality frontier pixel next. Following high-quality (smooth) paths
// first and only crossing residues last is what lets this method recover fields
// whose absolute range far exceeds 2π where a naive line-by-line unwrap would
// propagate a single bad step across a whole row.
//
// wrapped and quality are row-major of length rows*cols; higher quality means a
// more trustworthy pixel (see [PhaseGradientQuality]). Each pixel is unwrapped
// relative to the neighbour it was reached from using the nearest-2π rule. The
// result is a new continuous absolute phase map, fixed by setting the seed pixel
// equal to its wrapped value; add a constant to align it to a reference. It
// panics on a size mismatch.
func QualityGuidedUnwrap(wrapped, quality []float64, rows, cols int) []float64 {
	if len(wrapped) != rows*cols || len(quality) != rows*cols {
		panic("structured_light: QualityGuidedUnwrap size mismatch")
	}
	n := rows * cols
	out := make([]float64, n)
	done := make([]bool, n)
	if n == 0 {
		return out
	}

	// Seed at the globally highest-quality pixel.
	seed := 0
	for i := 1; i < n; i++ {
		if quality[i] > quality[seed] {
			seed = i
		}
	}
	out[seed] = wrapped[seed]
	done[seed] = true

	pq := &qgQueue{}
	heap.Init(pq)
	pushNeighbours(pq, seed, quality, done, rows, cols)

	for pq.Len() > 0 {
		it := heap.Pop(pq).(qgItem)
		if done[it.idx] {
			continue
		}
		out[it.idx] = out[it.from] + wrapDelta(wrapped[it.idx]-wrapped[it.from])
		done[it.idx] = true
		pushNeighbours(pq, it.idx, quality, done, rows, cols)
	}
	return out
}

// pushNeighbours enqueues the not-yet-unwrapped 4-connected neighbours of pixel
// idx, each keyed on its own quality and seeded from idx.
func pushNeighbours(pq *qgQueue, idx int, quality []float64, done []bool, rows, cols int) {
	y := idx / cols
	x := idx % cols
	add := func(nx, ny int) {
		if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
			return
		}
		ni := ny*cols + nx
		if done[ni] {
			return
		}
		heap.Push(pq, qgItem{idx: ni, from: idx, qual: quality[ni]})
	}
	add(x+1, y)
	add(x-1, y)
	add(x, y+1)
	add(x, y-1)
}

package segmentation

import "container/heap"

// floodBoundary is the internal sentinel written to a pixel that is claimed by
// two different basins at once during priority flooding. Callers translate it to
// whatever boundary encoding they expose.
const floodBoundary = -1

// priorityFlood runs Meyer's priority-flooding watershed on an arbitrary scalar
// relief seeded by labels. It is the shared engine behind the distance-transform
// watershed and the seeded region growers.
//
// relief and seed both have length rows*cols. In seed, 0 marks an unlabeled
// pixel that flooding may claim and any positive value is a fixed basin label.
// When mask is non-nil a pixel is floodable only where mask is true; pixels with
// mask false keep label 0 and act as barriers. The returned slice copies seed
// and fills each reachable unlabeled pixel with the label of the basin that
// arrives first (lowest relief, ties broken by insertion order); a pixel with
// two disagreeing labelled neighbours becomes [floodBoundary].
func priorityFlood(relief []float64, seed []int, mask []bool, rows, cols int) []int {
	n := rows * cols
	labels := make([]int, n)
	copy(labels, seed)

	floodable := func(idx int) bool { return mask == nil || mask[idx] }

	inQueue := make([]bool, n)
	pq := &pixelHeap{}
	heap.Init(pq)
	seq := 0
	push := func(x, y int) {
		idx := y*cols + x
		if inQueue[idx] || labels[idx] != 0 || !floodable(idx) {
			return
		}
		inQueue[idx] = true
		heap.Push(pq, pixelItem{priority: relief[idx], seq: seq, x: x, y: y})
		seq++
	}

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if labels[y*cols+x] <= 0 {
				continue
			}
			for _, o := range neighbors4 {
				nx, ny := x+o.dx, y+o.dy
				if nx >= 0 && nx < cols && ny >= 0 && ny < rows {
					push(nx, ny)
				}
			}
		}
	}

	for pq.Len() > 0 {
		it := heap.Pop(pq).(pixelItem)
		idx := it.y*cols + it.x

		found := 0
		conflict := false
		for _, o := range neighbors4 {
			nx, ny := it.x+o.dx, it.y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			nl := labels[ny*cols+nx]
			if nl <= 0 {
				continue
			}
			if found == 0 {
				found = nl
			} else if found != nl {
				conflict = true
			}
		}
		if conflict {
			labels[idx] = floodBoundary
			continue
		}
		if found == 0 {
			inQueue[idx] = false
			continue
		}
		labels[idx] = found
		for _, o := range neighbors4 {
			nx, ny := it.x+o.dx, it.y+o.dy
			if nx >= 0 && nx < cols && ny >= 0 && ny < rows {
				push(nx, ny)
			}
		}
	}
	return labels
}

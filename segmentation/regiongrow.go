package segmentation

import (
	"container/heap"

	cv "github.com/malcolmston/opencv"
)

// RegionGrowing performs seeded region growing on img: each seed point starts a
// region and neighbouring pixels are absorbed while their colour stays within
// threshold of the growing region's running mean colour. It returns a [LabelMap]
// where region k (1-based internally, renumbered consecutively in the result)
// grows from seeds[k] and label 0 collects any pixel no region could claim.
//
// Growth is globally ordered by colour similarity: at each step the queued
// candidate whose colour is closest to its region mean is added next (a
// best-first flood driven by a priority queue), which makes the result
// independent of seed order and fully deterministic. A pixel is claimed by the
// first region to reach it. Comparisons use the mean colour of the region as it
// stood when the candidate was queued, mirroring OpenCV-style region growing.
//
// img may have any number of channels. threshold is a colour distance in the
// same units as the samples. It panics if img is empty, seeds is empty, or any
// seed lies outside the image.
func RegionGrowing(img *cv.Mat, seeds []cv.Point, threshold float64) *LabelMap {
	if img.Empty() {
		panic("segmentation: RegionGrowing on empty image")
	}
	if len(seeds) == 0 {
		panic("segmentation: RegionGrowing requires at least one seed")
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	n := rows * cols

	color := func(idx int) []float64 {
		b := idx * ch
		out := make([]float64, ch)
		for c := 0; c < ch; c++ {
			out[c] = float64(img.Data[b+c])
		}
		return out
	}

	labels := make([]int, n)
	// Running per-region colour sum and count for the mean.
	sum := make([][]float64, len(seeds)+1)
	cnt := make([]int, len(seeds)+1)
	for i := range sum {
		sum[i] = make([]float64, ch)
	}

	pq := &growHeap{}
	heap.Init(pq)
	seq := 0

	addSeed := func(region int, p cv.Point) {
		if p.X < 0 || p.X >= cols || p.Y < 0 || p.Y >= rows {
			panic("segmentation: RegionGrowing seed out of bounds")
		}
		idx := p.Y*cols + p.X
		if labels[idx] != 0 {
			return
		}
		labels[idx] = region
		col := color(idx)
		for c := 0; c < ch; c++ {
			sum[region][c] += col[c]
		}
		cnt[region]++
	}
	for k, p := range seeds {
		addSeed(k+1, p)
	}

	mean := func(region int) []float64 {
		m := make([]float64, ch)
		if cnt[region] == 0 {
			return m
		}
		for c := 0; c < ch; c++ {
			m[c] = sum[region][c] / float64(cnt[region])
		}
		return m
	}

	// Queue the frontier of every seed.
	pushNeighbors := func(idx, region int) {
		y, x := idx/cols, idx%cols
		rm := mean(region)
		for _, o := range neighbors4 {
			nx, ny := x+o.dx, y+o.dy
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			nidx := ny*cols + nx
			if labels[nidx] != 0 {
				continue
			}
			d := colorDist(color(nidx), rm)
			heap.Push(pq, growItem{priority: d, seq: seq, x: nx, y: ny, region: region})
			seq++
		}
	}
	for k, p := range seeds {
		pushNeighbors(p.Y*cols+p.X, k+1)
	}

	for pq.Len() > 0 {
		it := heap.Pop(pq).(growItem)
		idx := it.y*cols + it.x
		if labels[idx] != 0 {
			continue
		}
		if it.priority > threshold {
			continue
		}
		region := it.region
		labels[idx] = region
		col := color(idx)
		for c := 0; c < ch; c++ {
			sum[region][c] += col[c]
		}
		cnt[region]++
		pushNeighbors(idx, region)
	}

	out, count := relabelConsecutive(labels)
	return &LabelMap{Rows: rows, Cols: cols, Count: count, Labels: out}
}

// growItem is a region-growing frontier entry: a candidate pixel with the colour
// distance to its region mean as priority and the region it would join.
type growItem struct {
	priority float64
	seq      int
	x, y     int
	region   int
}

// growHeap is a min-heap of growItem ordered by priority, ties broken by
// insertion order so growth is deterministic.
type growHeap []growItem

func (h growHeap) Len() int { return len(h) }
func (h growHeap) Less(i, j int) bool {
	if h[i].priority != h[j].priority {
		return h[i].priority < h[j].priority
	}
	return h[i].seq < h[j].seq
}
func (h growHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *growHeap) Push(x any)   { *h = append(*h, x.(growItem)) }
func (h *growHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

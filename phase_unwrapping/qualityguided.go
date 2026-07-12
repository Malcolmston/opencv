package phase_unwrapping

import (
	"container/heap"
	"math"
)

// QualityGuidedUnwrap unwraps a wrapped phase map by following a quality-guided
// path (Ghiglia and Pritt): pixels are unwrapped most-reliable first, so that any
// unavoidable error is pushed into the least reliable region last. Starting from
// the highest-quality pixel it grows a region using a priority queue keyed by the
// quality of the pixel about to be added, adding to each new pixel the wrapped
// gradient from its already-unwrapped neighbour.
//
// quality is a reliability map where HIGHER is better, such as
// [PseudoCorrelation]; pass nil to use a sensible default (the negated
// [PhaseDerivativeVariance]). When quality is supplied it must match wrapped's
// shape. On a residue-free map the surface is recovered exactly up to a global
// 2*pi constant regardless of the guiding map. Input values are wrapped
// defensively first and neither argument is modified. It returns [ErrEmptyInput]
// for an empty grid and [ErrShapeMismatch] if quality has the wrong shape.
func QualityGuidedUnwrap(wrapped, quality [][]float64) ([][]float64, error) {
	rows, cols, ok := gridDims(wrapped)
	if !ok {
		return nil, ErrEmptyInput
	}
	q, err := resolveQuality(quality, wrapped, rows, cols)
	if err != nil {
		return nil, err
	}
	phase := flatten(wrapped, rows, cols)
	mask := make([]bool, rows*cols)
	for i := range mask {
		mask[i] = true
	}
	u, _ := growUnwrap(phase, q, mask, rows, cols)
	return unflatten(u, rows, cols), nil
}

// MaskedUnwrap unwraps only the pixels selected by mask (mask[i][j] true), using
// the same quality-guided path following as [QualityGuidedUnwrap] but never
// crossing into a masked-out pixel. This isolates a valid region — a measurement
// aperture, or the reliable part of a noisy map — so that invalid data cannot
// corrupt it. Masked-out pixels are returned as NaN; each connected valid
// component is unwrapped independently and therefore carries its own arbitrary
// 2*pi offset.
//
// mask must match wrapped's shape. quality is an optional higher-is-better
// reliability map (nil selects the negated [PhaseDerivativeVariance]); when given
// it must also match the shape. Within a single connected residue-free component
// the surface is recovered exactly up to a constant. Input values are wrapped
// defensively first and no argument is modified. It returns [ErrEmptyInput] for
// an empty grid and [ErrShapeMismatch] on any shape mismatch.
func MaskedUnwrap(wrapped [][]float64, mask [][]bool, quality [][]float64) ([][]float64, error) {
	rows, cols, ok := gridDims(wrapped)
	if !ok {
		return nil, ErrEmptyInput
	}
	if len(mask) != rows {
		return nil, ErrShapeMismatch
	}
	flatMask := make([]bool, rows*cols)
	for i := 0; i < rows; i++ {
		if len(mask[i]) != cols {
			return nil, ErrShapeMismatch
		}
		for j := 0; j < cols; j++ {
			flatMask[i*cols+j] = mask[i][j]
		}
	}
	q, err := resolveQuality(quality, wrapped, rows, cols)
	if err != nil {
		return nil, err
	}
	phase := flatten(wrapped, rows, cols)
	u, visited := growUnwrap(phase, q, flatMask, rows, cols)
	for a := range u {
		if !visited[a] {
			u[a] = math.NaN()
		}
	}
	return unflatten(u, rows, cols), nil
}

// resolveQuality validates a caller-supplied quality map or, when nil, builds the
// default higher-is-better reliability map (negated phase-derivative variance).
func resolveQuality(quality, wrapped [][]float64, rows, cols int) ([]float64, error) {
	q := make([]float64, rows*cols)
	if quality == nil {
		pdv := PhaseDerivativeVariance(wrapped)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				q[i*cols+j] = -pdv[i][j]
			}
		}
		return q, nil
	}
	qr, qc, ok := gridDims(quality)
	if !ok || qr != rows || qc != cols {
		return nil, ErrShapeMismatch
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			q[i*cols+j] = quality[i][j]
		}
	}
	return q, nil
}

// growUnwrap runs the priority-queue region-growing unwrap over the masked-in
// pixels, seeding each connected component from its highest-quality pixel. It
// returns the unwrapped values and a visited mask (false for masked-out pixels).
func growUnwrap(phase, quality []float64, mask []bool, rows, cols int) (u []float64, visited []bool) {
	n := rows * cols
	u = make([]float64, n)
	visited = make([]bool, n)
	pq := &pixelHeap{}
	heap.Init(pq)
	var seq int64

	neighbors := func(a int) [4]int {
		r := a / cols
		c := a % cols
		res := [4]int{-1, -1, -1, -1}
		if r > 0 {
			res[0] = a - cols
		}
		if r < rows-1 {
			res[1] = a + cols
		}
		if c > 0 {
			res[2] = a - 1
		}
		if c < cols-1 {
			res[3] = a + 1
		}
		return res
	}
	push := func(a int) {
		seq++
		heap.Push(pq, pixelItem{index: a, quality: quality[a], seq: seq})
	}

	for {
		// Find the highest-quality unseeded masked pixel to start a new component.
		seed := -1
		bestQ := math.Inf(-1)
		for a := 0; a < n; a++ {
			if !mask[a] || visited[a] {
				continue
			}
			if quality[a] > bestQ {
				bestQ = quality[a]
				seed = a
			}
		}
		if seed < 0 {
			break
		}
		u[seed] = phase[seed]
		visited[seed] = true
		for _, nb := range neighbors(seed) {
			if nb >= 0 && mask[nb] && !visited[nb] {
				push(nb)
			}
		}
		for pq.Len() > 0 {
			it := heap.Pop(pq).(pixelItem)
			a := it.index
			if visited[a] || !mask[a] {
				continue
			}
			// Adopt the offset from the best already-unwrapped neighbour.
			bestNb := -1
			bestNbQ := math.Inf(-1)
			for _, nb := range neighbors(a) {
				if nb >= 0 && visited[nb] && quality[nb] > bestNbQ {
					bestNbQ = quality[nb]
					bestNb = nb
				}
			}
			if bestNb < 0 {
				continue
			}
			u[a] = u[bestNb] + Wrap(phase[a]-phase[bestNb])
			visited[a] = true
			for _, nb := range neighbors(a) {
				if nb >= 0 && mask[nb] && !visited[nb] {
					push(nb)
				}
			}
		}
	}
	return u, visited
}

// pixelItem is a heap entry ordered by descending quality; seq breaks ties by
// insertion order to keep the unwrap deterministic.
type pixelItem struct {
	index   int
	quality float64
	seq     int64
}

// pixelHeap is a max-heap of pixelItem on quality.
type pixelHeap []pixelItem

func (h pixelHeap) Len() int { return len(h) }
func (h pixelHeap) Less(i, j int) bool {
	if h[i].quality != h[j].quality {
		return h[i].quality > h[j].quality
	}
	return h[i].seq < h[j].seq
}
func (h pixelHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *pixelHeap) Push(x any)   { *h = append(*h, x.(pixelItem)) }
func (h *pixelHeap) Pop() any {
	old := *h
	m := len(old)
	it := old[m-1]
	*h = old[:m-1]
	return it
}

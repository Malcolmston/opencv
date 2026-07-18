package inpaint

import (
	"container/heap"
	"math"
)

// Fast Marching Method pixel states.
const (
	fmmKnown   = iota // value settled (source region)
	fmmBand           // in the narrow band (in the priority queue)
	fmmUnknown        // not yet reached
)

// fmmItem is a heap entry ordered by ascending arrival time t.
type fmmItem struct {
	t    float64
	y, x int
}

// fmmHeap is a min-heap of fmmItem by t.
type fmmHeap []fmmItem

func (h fmmHeap) Len() int            { return len(h) }
func (h fmmHeap) Less(i, j int) bool  { return h[i].t < h[j].t }
func (h fmmHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *fmmHeap) Push(x interface{}) { *h = append(*h, x.(fmmItem)) }
func (h *fmmHeap) Pop() interface{} {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// FastMarcher solves the eikonal equation |∇T| = 1 by the Fast Marching Method,
// growing the arrival-time field T outward from a set of source pixels into a
// target region. It underlies [InpaintTelea]'s boundary-ordered fill and the
// [DistanceTransform] helper. Construct with [NewFastMarcher].
type FastMarcher struct {
	rows, cols int
	t          []float64
	flags      []int
}

// NewFastMarcher prepares a march over a mask: the selected (true) pixels of
// mask are the target region to be reached, and the unselected pixels are the
// sources (arrival time 0). The march is not run until [FastMarcher.Solve] is
// called.
func NewFastMarcher(mask *Mask) *FastMarcher {
	if mask == nil {
		panic("inpaint: NewFastMarcher given a nil mask")
	}
	n := mask.Rows * mask.Cols
	f := &FastMarcher{
		rows:  mask.Rows,
		cols:  mask.Cols,
		t:     make([]float64, n),
		flags: make([]int, n),
	}
	for i := 0; i < n; i++ {
		if mask.Data[i] {
			f.flags[i] = fmmUnknown
			f.t[i] = math.Inf(1)
		} else {
			f.flags[i] = fmmKnown
			f.t[i] = 0
		}
	}
	return f
}

// solveEikonal returns the candidate arrival time at (y, x) using the upwind
// finite-difference scheme of Sethian: the smaller settled neighbour along each
// axis feeds the quadratic (T-tX)^2 + (T-tY)^2 = 1.
func (f *FastMarcher) solveEikonal(y, x int) float64 {
	tX := f.axisMin(y, x-1, y, x+1)
	tY := f.axisMin(y-1, x, y+1, x)
	if math.IsInf(tX, 1) && math.IsInf(tY, 1) {
		return math.Inf(1)
	}
	if math.IsInf(tX, 1) {
		return tY + 1
	}
	if math.IsInf(tY, 1) {
		return tX + 1
	}
	// Solve (T-tX)^2 + (T-tY)^2 = 1 for the larger root.
	d := 2 - (tX-tY)*(tX-tY)
	if d < 0 {
		return math.Min(tX, tY) + 1
	}
	return (tX + tY + math.Sqrt(d)) / 2
}

// axisMin returns the smaller settled arrival time among the two neighbours
// (y1,x1) and (y2,x2), or +Inf if neither is settled.
func (f *FastMarcher) axisMin(y1, x1, y2, x2 int) float64 {
	m := math.Inf(1)
	if y1 >= 0 && y1 < f.rows && x1 >= 0 && x1 < f.cols && f.flags[y1*f.cols+x1] != fmmUnknown {
		if f.t[y1*f.cols+x1] < m {
			m = f.t[y1*f.cols+x1]
		}
	}
	if y2 >= 0 && y2 < f.rows && x2 >= 0 && x2 < f.cols && f.flags[y2*f.cols+x2] != fmmUnknown {
		if f.t[y2*f.cols+x2] < m {
			m = f.t[y2*f.cols+x2]
		}
	}
	return m
}

// Solve runs the march and returns the arrival-time field T in row-major order
// (length Rows*Cols). Source pixels have T=0; each target pixel receives the
// geodesic distance (in pixels) to the nearest source under the |∇T|=1 metric.
// The optional visit callback, if non-nil, is invoked once per target pixel in
// the exact order the front reaches them, which is the order [InpaintTelea]
// fills the hole.
func (f *FastMarcher) Solve(visit func(y, x int)) []float64 {
	h := &fmmHeap{}
	heap.Init(h)
	// Seed the band with source pixels adjacent to the target region.
	for y := 0; y < f.rows; y++ {
		for x := 0; x < f.cols; x++ {
			if f.flags[y*f.cols+x] != fmmKnown {
				continue
			}
			for _, d := range neighbors4 {
				ny, nx := y+d[0], x+d[1]
				if ny >= 0 && ny < f.rows && nx >= 0 && nx < f.cols && f.flags[ny*f.cols+nx] == fmmUnknown {
					heap.Push(h, fmmItem{t: 0, y: y, x: x})
					f.flags[y*f.cols+x] = fmmBand
					break
				}
			}
		}
	}
	for h.Len() > 0 {
		it := heap.Pop(h).(fmmItem)
		i := it.y*f.cols + it.x
		f.flags[i] = fmmKnown
		for _, d := range neighbors4 {
			ny, nx := it.y+d[0], it.x+d[1]
			if ny < 0 || ny >= f.rows || nx < 0 || nx >= f.cols {
				continue
			}
			j := ny*f.cols + nx
			if f.flags[j] != fmmUnknown {
				continue
			}
			tv := f.solveEikonal(ny, nx)
			f.t[j] = tv
			f.flags[j] = fmmBand
			if visit != nil {
				visit(ny, nx)
			}
			heap.Push(h, fmmItem{t: tv, y: ny, x: nx})
		}
	}
	return f.t
}

// DistanceTransform returns, for every pixel, an approximate Euclidean distance
// to the nearest unselected pixel of mask, computed by the Fast Marching Method
// (|∇T|=1). Unselected pixels have distance 0; a selected pixel k axis-steps
// from the boundary along an axis direction receives exactly distance k.
func DistanceTransform(mask *Mask) []float64 {
	return NewFastMarcher(mask).Solve(nil)
}

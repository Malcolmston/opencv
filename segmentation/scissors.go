package segmentation

import (
	"container/heap"
	"math"

	cv "github.com/malcolmston/opencv"
)

// IntelligentScissors implements the live-wire boundary tool of Mortensen &
// Barrett ("Intelligent Scissors for Image Composition", 1995), matching the
// behaviour of OpenCV's IntelligentScissorsMB. It precomputes a per-edge cost
// from the image gradient — strong edges are cheap to travel along — and then
// finds the globally minimum-cost path between a seed point and any target with
// Dijkstra's algorithm, so the traced contour snaps to object boundaries.
//
// Typical use is: construct with [NewIntelligentScissors], call
// [IntelligentScissors.BuildMap] once per seed, then [IntelligentScissors.Trace]
// for each target the user hovers over.
type IntelligentScissors struct {
	rows, cols int
	// gcost holds the node cost 1-|grad|/max in [0,1]: low on strong edges. The
	// edge cost between neighbours is the mean of their node costs scaled by the
	// Euclidean step length, which biases the path toward high-gradient pixels.
	gcost []float64
	seed  cv.Point
	prev  []int // predecessor index per node for the current seed, -1 if none
}

// NewIntelligentScissors builds the cost field of img for live-wire tracing.
// img may be single- or three-channel; colour input is reduced to its gradient
// magnitude across channels. It panics if img is empty.
func NewIntelligentScissors(img *cv.Mat) *IntelligentScissors {
	if img.Empty() {
		panic("segmentation: NewIntelligentScissors on empty image")
	}
	rows, cols := img.Rows, img.Cols
	mag := gradientMagnitude(img)
	maxMag := 0.0
	for _, v := range mag {
		if v > maxMag {
			maxMag = v
		}
	}
	gcost := make([]float64, len(mag))
	for i, v := range mag {
		if maxMag > 0 {
			gcost[i] = 1.0 - v/maxMag // 0 on the strongest edge, 1 on flat areas
		} else {
			gcost[i] = 1.0
		}
	}
	return &IntelligentScissors{rows: rows, cols: cols, gcost: gcost, seed: cv.Point{X: -1, Y: -1}}
}

// BuildMap runs Dijkstra from seed over the 8-connected pixel graph, recording
// the minimum-cost predecessor of every reachable pixel so that any number of
// contours can subsequently be traced from this seed with [Trace]. It panics if
// seed lies outside the image.
func (s *IntelligentScissors) BuildMap(seed cv.Point) {
	if seed.X < 0 || seed.X >= s.cols || seed.Y < 0 || seed.Y >= s.rows {
		panic("segmentation: IntelligentScissors seed out of bounds")
	}
	n := s.rows * s.cols
	dist := make([]float64, n)
	for i := range dist {
		dist[i] = math.Inf(1)
	}
	prev := make([]int, n)
	for i := range prev {
		prev[i] = -1
	}
	done := make([]bool, n)

	start := seed.Y*s.cols + seed.X
	dist[start] = 0
	pq := &pixelHeap{}
	heap.Init(pq)
	seq := 0
	heap.Push(pq, pixelItem{priority: 0, seq: seq, x: seed.X, y: seed.Y})
	seq++

	for pq.Len() > 0 {
		it := heap.Pop(pq).(pixelItem)
		idx := it.y*s.cols + it.x
		if done[idx] {
			continue
		}
		done[idx] = true
		for _, o := range neighbors8 {
			nx, ny := it.x+o.dx, it.y+o.dy
			if nx < 0 || nx >= s.cols || ny < 0 || ny >= s.rows {
				continue
			}
			nidx := ny*s.cols + nx
			if done[nidx] {
				continue
			}
			step := 1.0
			if o.dx != 0 && o.dy != 0 {
				step = math.Sqrt2
			}
			w := 0.5 * (s.gcost[idx] + s.gcost[nidx]) * step
			nd := dist[idx] + w
			if nd < dist[nidx] {
				dist[nidx] = nd
				prev[nidx] = idx
				heap.Push(pq, pixelItem{priority: nd, seq: seq, x: nx, y: ny})
				seq++
			}
		}
	}
	s.seed = seed
	s.prev = prev
}

// Trace returns the minimum-cost contour from the current seed (set by
// [BuildMap]) to target, as an ordered slice of points running seed -> target.
// It returns nil if [BuildMap] has not been called or target is unreachable, and
// panics if target lies outside the image.
func (s *IntelligentScissors) Trace(target cv.Point) []cv.Point {
	if s.prev == nil {
		return nil
	}
	if target.X < 0 || target.X >= s.cols || target.Y < 0 || target.Y >= s.rows {
		panic("segmentation: IntelligentScissors target out of bounds")
	}
	idx := target.Y*s.cols + target.X
	start := s.seed.Y*s.cols + s.seed.X
	if idx != start && s.prev[idx] < 0 {
		return nil
	}
	var rev []cv.Point
	for idx != -1 {
		rev = append(rev, cv.Point{X: idx % s.cols, Y: idx / s.cols})
		if idx == start {
			break
		}
		idx = s.prev[idx]
	}
	// Reverse into seed -> target order.
	for i, j := 0, len(rev)-1; i < j; i, j = i+1, j-1 {
		rev[i], rev[j] = rev[j], rev[i]
	}
	return rev
}

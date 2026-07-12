package phase_unwrapping

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// twoPi is 2*pi, the wrapping period.
const twoPi = 2 * math.Pi

// Params configures a [HistogramPhaseUnwrapping]. It mirrors OpenCV's
// HistogramPhaseUnwrapping::Params.
//
// Width and Height describe the size of the wrapped phase map. They are
// informational: [NewHistogramPhaseUnwrapping] stores them, but the unwrapping
// methods derive the working size from the map actually passed in and update
// these fields to match. The histogram fields tune how edges are ordered by
// reliability and never change the result on a residue-free map; they only
// affect how gracefully noisy regions are routed around.
type Params struct {
	// Width is the expected number of columns of the wrapped phase map.
	Width int
	// Height is the expected number of rows of the wrapped phase map.
	Height int
	// HistogramThresh is the reliability value that separates the fine "small"
	// bins (used for reliable, low-value edges) from the coarse "large" bins
	// (used for the long tail of unreliable edges). It is expressed in the same
	// units as the inverse-reliability measure (a sum of squared second
	// differences). The OpenCV default is 3*pi*pi.
	HistogramThresh float64
	// NrOfSmallBins is the number of fine histogram bins covering
	// [0, HistogramThresh). Must be >= 1.
	NrOfSmallBins int
	// NrOfLargeBins is the number of coarse histogram bins covering
	// [HistogramThresh, maxReliability]. Must be >= 1.
	NrOfLargeBins int
}

// DefaultParams returns Params for a map of the given size using the same
// default histogram configuration as OpenCV: HistogramThresh = 3*pi*pi,
// NrOfSmallBins = 10 and NrOfLargeBins = 5.
func DefaultParams(width, height int) Params {
	return Params{
		Width:           width,
		Height:          height,
		HistogramThresh: 3 * math.Pi * math.Pi,
		NrOfSmallBins:   10,
		NrOfLargeBins:   5,
	}
}

// HistogramPhaseUnwrapping is a quality-guided, path-following 2-D phase
// unwrapper. Construct one with [NewHistogramPhaseUnwrapping], call
// [HistogramPhaseUnwrapping.UnwrapPhaseMap] (or
// [HistogramPhaseUnwrapping.UnwrapPhaseMapGrid]) to unwrap a wrapped phase map,
// then optionally [HistogramPhaseUnwrapping.GetInverseReliabilityMap] to
// retrieve the reliability image computed during the last unwrap.
//
// A single instance may be reused for successive maps; each call recomputes the
// reliability map and overwrites any previously cached one. The type is not safe
// for concurrent use.
type HistogramPhaseUnwrapping struct {
	params Params

	rows, cols int
	// invReliability holds the per-pixel inverse-reliability map from the last
	// unwrap, in row-major order (length rows*cols). nil until the first unwrap.
	invReliability []float64
}

// NewHistogramPhaseUnwrapping returns an unwrapper configured by p. Non-positive
// histogram bin counts are replaced by their defaults, and a non-positive
// HistogramThresh is replaced by 3*pi*pi, so a zero Params value yields a usable
// unwrapper.
func NewHistogramPhaseUnwrapping(p Params) *HistogramPhaseUnwrapping {
	if p.NrOfSmallBins <= 0 {
		p.NrOfSmallBins = 10
	}
	if p.NrOfLargeBins <= 0 {
		p.NrOfLargeBins = 5
	}
	if !(p.HistogramThresh > 0) {
		p.HistogramThresh = 3 * math.Pi * math.Pi
	}
	return &HistogramPhaseUnwrapping{params: p}
}

// Params returns the (possibly normalised) parameters in effect. Width and
// Height reflect the size of the most recently unwrapped map once one has been
// processed.
func (h *HistogramPhaseUnwrapping) Params() Params { return h.params }

// ErrEmptyInput is returned when a nil or zero-sized phase map is supplied.
var ErrEmptyInput = errors.New("phase_unwrapping: empty wrapped phase map")

// UnwrapPhaseMapGrid unwraps a wrapped phase map given as a [row][col] grid and
// returns the unwrapped absolute phase as a new grid of the same shape. The
// input is not modified. Values are assumed to lie in (-pi, pi] but are wrapped
// defensively first, so any real-valued input is accepted.
//
// The result matches the true continuous surface up to a single global additive
// constant that is a multiple of 2*pi. It returns [ErrEmptyInput] if wrapped is
// empty or ragged.
func (h *HistogramPhaseUnwrapping) UnwrapPhaseMapGrid(wrapped [][]float64) ([][]float64, error) {
	rows := len(wrapped)
	if rows == 0 || len(wrapped[0]) == 0 {
		return nil, ErrEmptyInput
	}
	cols := len(wrapped[0])
	flat := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		if len(wrapped[i]) != cols {
			return nil, ErrEmptyInput
		}
		for j := 0; j < cols; j++ {
			flat[i*cols+j] = Wrap(wrapped[i][j])
		}
	}
	out := h.unwrap(flat, rows, cols)
	grid := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		grid[i] = out[i*cols : (i+1)*cols : (i+1)*cols]
	}
	return grid, nil
}

// UnwrapPhaseMap unwraps a wrapped phase map stored as a single-channel
// [github.com/malcolmston/opencv.FloatMat] and returns a new FloatMat holding
// the unwrapped absolute phase. The input is not modified. As with
// [HistogramPhaseUnwrapping.UnwrapPhaseMapGrid] the result is defined up to a
// global 2*pi constant, and inputs are wrapped defensively before processing.
//
// It returns [ErrEmptyInput] for a nil or empty matrix.
func (h *HistogramPhaseUnwrapping) UnwrapPhaseMap(wrapped *cv.FloatMat) (*cv.FloatMat, error) {
	if wrapped == nil || wrapped.Rows <= 0 || wrapped.Cols <= 0 || len(wrapped.Data) == 0 {
		return nil, ErrEmptyInput
	}
	rows, cols := wrapped.Rows, wrapped.Cols
	flat := make([]float64, rows*cols)
	for i := range flat {
		flat[i] = Wrap(wrapped.Data[i])
	}
	out := h.unwrap(flat, rows, cols)
	res := cv.NewFloatMat(rows, cols)
	copy(res.Data, out)
	return res, nil
}

// GetInverseReliabilityMap returns the per-pixel inverse-reliability map
// computed during the most recent unwrap, as a single-channel FloatMat. Larger
// values indicate less reliable pixels; the measure is the sum of squared
// second differences of the wrapped phase over the pixel's 3x3 neighbourhood.
// Border pixels, which lack a full neighbourhood, are assigned the maximum
// interior value so that they are treated as the least reliable.
//
// It returns [ErrEmptyInput] if no map has been unwrapped yet.
func (h *HistogramPhaseUnwrapping) GetInverseReliabilityMap() (*cv.FloatMat, error) {
	if h.invReliability == nil {
		return nil, ErrEmptyInput
	}
	res := cv.NewFloatMat(h.rows, h.cols)
	copy(res.Data, h.invReliability)
	return res, nil
}

// unwrap is the core routine operating on a flat, already-wrapped phase slice.
// It fills h.invReliability and returns the unwrapped phase (same layout).
func (h *HistogramPhaseUnwrapping) unwrap(phase []float64, rows, cols int) []float64 {
	h.rows, h.cols = rows, cols
	h.params.Height, h.params.Width = rows, cols

	rel := computeInverseReliability(phase, rows, cols)
	h.invReliability = rel

	n := rows * cols
	// Union-find state with per-pixel 2*pi increments.
	parent := make([]int, n)
	size := make([]int, n)
	members := make([][]int, n)
	k := make([]int, n) // integer multiple of 2*pi to add to each pixel
	for i := 0; i < n; i++ {
		parent[i] = i
		size[i] = 1
		members[i] = []int{i}
	}

	find := func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}

	edges := buildEdges(phase, rel, rows, cols)
	order := histogramOrder(edges, h.params)

	for _, ei := range order {
		e := edges[ei]
		a, b := e.a, e.b
		ra, rb := find(a), find(b)
		if ra == rb {
			continue
		}
		// Desired relation across the edge: k[b] - k[a] == -e.inc.
		if size[ra] < size[rb] {
			// Shift the smaller group A so that k[a] == k[b] + e.inc.
			delta := (k[b] + e.inc) - k[a]
			if delta != 0 {
				for _, m := range members[ra] {
					k[m] += delta
				}
			}
			members[rb] = append(members[rb], members[ra]...)
			size[rb] += size[ra]
			members[ra] = nil
			parent[ra] = rb
		} else {
			// Shift the smaller (or equal) group B so that k[b] == k[a] - e.inc.
			delta := (k[a] - e.inc) - k[b]
			if delta != 0 {
				for _, m := range members[rb] {
					k[m] += delta
				}
			}
			members[ra] = append(members[ra], members[rb]...)
			size[ra] += size[rb]
			members[rb] = nil
			parent[rb] = ra
		}
	}

	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = phase[i] + twoPi*float64(k[i])
	}
	return out
}

// edge connects two neighbouring pixels a and b (flat indices). inc is the
// relative 2*pi increment implied by the wrapped phase difference:
// round((phase[b]-phase[a])/(2*pi)). rel is the combined inverse reliability of
// the endpoints, used to order edge processing.
type edge struct {
	a, b int
	inc  int
	rel  float64
}

// buildEdges creates the horizontal and vertical edges of the pixel lattice in
// a fixed row-major order.
func buildEdges(phase, rel []float64, rows, cols int) []edge {
	edges := make([]edge, 0, 2*rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a := i*cols + j
			if j+1 < cols {
				b := a + 1
				edges = append(edges, edge{a: a, b: b, inc: wrapInc(phase[b] - phase[a]), rel: rel[a] + rel[b]})
			}
			if i+1 < rows {
				b := a + cols
				edges = append(edges, edge{a: a, b: b, inc: wrapInc(phase[b] - phase[a]), rel: rel[a] + rel[b]})
			}
		}
	}
	return edges
}

// wrapInc returns round(d / (2*pi)), the number of 2*pi periods in d. For a
// difference of two wrapped phases (in (-2*pi, 2*pi)) the result is -1, 0 or 1.
func wrapInc(d float64) int {
	return int(math.Round(d / twoPi))
}

// computeInverseReliability computes, for every pixel, the sum of squared second
// differences of the wrapped phase over its 3x3 neighbourhood (horizontal,
// vertical and both diagonals). Higher values mean lower reliability. Border
// pixels, which have no full neighbourhood, are set to the maximum interior
// value so they sort last.
func computeInverseReliability(phase []float64, rows, cols int) []float64 {
	rel := make([]float64, rows*cols)
	at := func(i, j int) float64 { return phase[i*cols+j] }
	maxInterior := 0.0
	for i := 1; i < rows-1; i++ {
		for j := 1; j < cols-1; j++ {
			c := at(i, j)
			hh := Wrap(at(i, j-1)-c) - Wrap(c-at(i, j+1))
			v := Wrap(at(i-1, j)-c) - Wrap(c-at(i+1, j))
			d1 := Wrap(at(i-1, j-1)-c) - Wrap(c-at(i+1, j+1))
			d2 := Wrap(at(i-1, j+1)-c) - Wrap(c-at(i+1, j-1))
			d := hh*hh + v*v + d1*d1 + d2*d2
			rel[i*cols+j] = d
			if d > maxInterior {
				maxInterior = d
			}
		}
	}
	// Assign border pixels the worst interior reliability.
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if i == 0 || j == 0 || i == rows-1 || j == cols-1 {
				rel[i*cols+j] = maxInterior
			}
		}
	}
	return rel
}

// histogramOrder returns the indices of edges ordered by ascending combined
// inverse reliability (most reliable first), using a two-level histogram: fine
// bins below Params.HistogramThresh and coarse bins above it. Ordering within a
// bin preserves the row-major edge order, keeping the whole procedure
// deterministic.
func histogramOrder(edges []edge, p Params) []int {
	if len(edges) == 0 {
		return nil
	}
	small := p.NrOfSmallBins
	large := p.NrOfLargeBins
	thresh := p.HistogramThresh

	maxRel := 0.0
	for i := range edges {
		if edges[i].rel > maxRel {
			maxRel = edges[i].rel
		}
	}

	smallWidth := thresh / float64(small)
	// The large bins cover [thresh, maxRel]. Guard against a zero span.
	largeSpan := maxRel - thresh
	if largeSpan < 0 {
		largeSpan = 0
	}
	largeWidth := largeSpan / float64(large)

	totalBins := small + large
	binOf := func(r float64) int {
		if r < thresh {
			b := int(r / smallWidth)
			if b < 0 {
				b = 0
			}
			if b >= small {
				b = small - 1
			}
			return b
		}
		if largeWidth <= 0 {
			return small // all remaining go into the first large bin
		}
		b := small + int((r-thresh)/largeWidth)
		if b < small {
			b = small
		}
		if b >= totalBins {
			b = totalBins - 1
		}
		return b
	}

	buckets := make([][]int, totalBins)
	for i := range edges {
		b := binOf(edges[i].rel)
		buckets[b] = append(buckets[b], i)
	}
	order := make([]int, 0, len(edges))
	for b := 0; b < totalBins; b++ {
		order = append(order, buckets[b]...)
	}
	return order
}

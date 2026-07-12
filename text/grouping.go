package text

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// GroupingParams controls how [GroupingParams.Group] clusters character boxes
// into text lines. Two boxes join the same line when all three geometric
// heuristics agree: their heights are similar, their vertical centres are
// aligned, and the horizontal gap between them is small relative to their size.
type GroupingParams struct {
	// MaxHeightRatio is the largest allowed ratio between the taller and shorter
	// of two boxes for them to belong to the same line. 1 means identical
	// heights only; typical text tolerates roughly 1.5–2.
	MaxHeightRatio float64
	// MaxCenterYOffset bounds the vertical misalignment of two boxes' centres,
	// as a fraction of the taller box's height. Rows stacked vertically exceed
	// this and are kept apart.
	MaxCenterYOffset float64
	// MaxGapFactor bounds the horizontal gap between two boxes, as a multiple of
	// the wider box's width. Boxes farther apart than this start a new line.
	MaxGapFactor float64
	// MinGroupSize drops groups with fewer than this many boxes. 1 keeps every
	// group, including isolated characters.
	MinGroupSize int
}

// DefaultGroupingParams returns grouping parameters tuned for evenly spaced,
// same-height text such as a printed word or line of digits.
func DefaultGroupingParams() GroupingParams {
	return GroupingParams{
		MaxHeightRatio:   1.7,
		MaxCenterYOffset: 0.5,
		MaxGapFactor:     1.6,
		MinGroupSize:     1,
	}
}

// GroupTextRegions groups character-like boxes into text lines using the default
// heuristics (see [DefaultGroupingParams]). Each returned line is sorted
// left-to-right, and the lines themselves are ordered top-to-bottom. A
// horizontally aligned run of similar-height boxes becomes one line; a second
// row at a different height becomes a separate line.
func GroupTextRegions(regions []cv.Rect) [][]cv.Rect {
	return DefaultGroupingParams().Group(regions)
}

// Group clusters regions into text lines according to p. It is deterministic:
// boxes are compared in a fixed order and every result slice is sorted.
func (p GroupingParams) Group(regions []cv.Rect) [][]cv.Rect {
	n := len(regions)
	if n == 0 {
		return nil
	}
	uf := newIntUnionFind(n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if p.compatible(regions[i], regions[j]) {
				uf.union(i, j)
			}
		}
	}

	// Collect members per line root, preserving deterministic order.
	members := map[int][]int{}
	var roots []int
	for i := 0; i < n; i++ {
		r := uf.find(i)
		if _, ok := members[r]; !ok {
			roots = append(roots, r)
		}
		members[r] = append(members[r], i)
	}

	var lines [][]cv.Rect
	for _, r := range roots {
		idxs := members[r]
		if len(idxs) < p.MinGroupSize {
			continue
		}
		line := make([]cv.Rect, len(idxs))
		for k, idx := range idxs {
			line[k] = regions[idx]
		}
		sort.SliceStable(line, func(a, b int) bool {
			if line[a].X != line[b].X {
				return line[a].X < line[b].X
			}
			return line[a].Y < line[b].Y
		})
		lines = append(lines, line)
	}

	// Order lines top-to-bottom by the mean vertical centre of their boxes.
	sort.SliceStable(lines, func(a, b int) bool {
		ca, cb := meanCenterY(lines[a]), meanCenterY(lines[b])
		if ca != cb {
			return ca < cb
		}
		return lines[a][0].X < lines[b][0].X
	})
	return lines
}

// compatible reports whether boxes a and b can belong to the same text line.
func (p GroupingParams) compatible(a, b cv.Rect) bool {
	if a.Height <= 0 || b.Height <= 0 {
		return false
	}
	hi, lo := a.Height, b.Height
	if hi < lo {
		hi, lo = lo, hi
	}
	if float64(hi)/float64(lo) > p.MaxHeightRatio {
		return false
	}

	cya := float64(a.Y) + float64(a.Height)/2
	cyb := float64(b.Y) + float64(b.Height)/2
	dcy := cya - cyb
	if dcy < 0 {
		dcy = -dcy
	}
	if dcy > p.MaxCenterYOffset*float64(hi) {
		return false
	}

	gap := horizontalGap(a, b)
	widest := a.Width
	if b.Width > widest {
		widest = b.Width
	}
	return float64(gap) <= p.MaxGapFactor*float64(widest)
}

// horizontalGap returns the horizontal distance between two boxes: 0 when their
// x-ranges overlap, otherwise the size of the empty band between them.
func horizontalGap(a, b cv.Rect) int {
	ax2 := a.X + a.Width
	bx2 := b.X + b.Width
	if a.X > bx2 {
		return a.X - bx2
	}
	if b.X > ax2 {
		return b.X - ax2
	}
	return 0
}

func meanCenterY(line []cv.Rect) float64 {
	if len(line) == 0 {
		return 0
	}
	var sum float64
	for _, r := range line {
		sum += float64(r.Y) + float64(r.Height)/2
	}
	return sum / float64(len(line))
}

// ERFilter is a lightweight stand-in for OpenCV's trained Extremal Region
// classifier. Rather than a learned AdaBoost model it applies cheap shape
// heuristics — area, aspect ratio and fill ratio — to reject regions that are
// unlikely to be single characters. Construct one with [NewERFilter] or
// [DefaultERFilter].
type ERFilter struct {
	// MinArea and MaxArea bound the region pixel area (MaxArea <= 0 means no
	// upper bound).
	MinArea int
	MaxArea int
	// MinAspect and MaxAspect bound width/height of the bounding box. Characters
	// are typically taller than wide but rarely extreme in either direction.
	MinAspect float64
	MaxAspect float64
	// MinFillRatio bounds the fraction of the bounding box covered by region
	// pixels, discarding sparse, stroke-free blobs.
	MinFillRatio float64
}

// DefaultERFilter returns a filter with permissive character-shaped defaults.
func DefaultERFilter() ERFilter {
	return ERFilter{
		MinArea:      8,
		MaxArea:      0,
		MinAspect:    0.05,
		MaxAspect:    3.0,
		MinFillRatio: 0.2,
	}
}

// NewERFilter returns an ERFilter with the given area bounds and the default
// aspect-ratio and fill-ratio limits.
func NewERFilter(minArea, maxArea int) ERFilter {
	f := DefaultERFilter()
	f.MinArea = minArea
	f.MaxArea = maxArea
	return f
}

// Keep reports whether a single region passes the filter's shape heuristics.
func (f ERFilter) Keep(r Region) bool {
	if r.Area < f.MinArea {
		return false
	}
	if f.MaxArea > 0 && r.Area > f.MaxArea {
		return false
	}
	if r.Rect.Width <= 0 || r.Rect.Height <= 0 {
		return false
	}
	aspect := float64(r.Rect.Width) / float64(r.Rect.Height)
	if aspect < f.MinAspect || aspect > f.MaxAspect {
		return false
	}
	fill := float64(r.Area) / float64(r.Rect.Width*r.Rect.Height)
	return fill >= f.MinFillRatio
}

// Filter returns the subset of regions that pass the filter, preserving order.
func (f ERFilter) Filter(regions []Region) []Region {
	var out []Region
	for _, r := range regions {
		if f.Keep(r) {
			out = append(out, r)
		}
	}
	return out
}

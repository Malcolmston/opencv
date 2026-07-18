package segment2

import (
	cv "github.com/malcolmston/opencv"
)

// Connectivity selects the pixel neighbourhood used by region-based routines.
type Connectivity int

const (
	// Conn4 considers the four edge-adjacent neighbours (N, S, E, W).
	Conn4 Connectivity = 4
	// Conn8 also considers the four diagonal neighbours.
	Conn8 Connectivity = 8
)

// LabelMap is a dense, row-major integer labelling of an image. Label 0 is used
// by some routines as a background or "unassigned" marker; the meaning of each
// label is documented by the routine that produced the map. Because a
// single-channel [cv.Mat] cannot store more than 256 distinct values, dense
// segmenters return a LabelMap rather than an image.
type LabelMap struct {
	// Rows is the labelling height.
	Rows int
	// Cols is the labelling width.
	Cols int
	// Labels holds Rows*Cols labels in row-major order.
	Labels []int
	// NumLabels is one more than the largest label value, i.e. the number of
	// label slots 0..NumLabels-1. It is not necessarily the count of non-empty
	// regions; call [LabelMap.Compact] to remove gaps.
	NumLabels int
}

// NewLabelMap allocates a zero-filled LabelMap of the given size. It panics if
// either dimension is not positive.
func NewLabelMap(rows, cols int) *LabelMap {
	if rows <= 0 || cols <= 0 {
		panic("segment2: NewLabelMap requires positive dimensions")
	}
	return &LabelMap{
		Rows:      rows,
		Cols:      cols,
		Labels:    make([]int, rows*cols),
		NumLabels: 1,
	}
}

// At returns the label of pixel (x, y). It panics if the coordinates are out of
// range.
func (lm *LabelMap) At(y, x int) int {
	if y < 0 || y >= lm.Rows || x < 0 || x >= lm.Cols {
		panic("segment2: LabelMap.At out of range")
	}
	return lm.Labels[y*lm.Cols+x]
}

// Set stores label at pixel (x, y). It panics if the coordinates are out of
// range. NumLabels is widened to include the new label.
func (lm *LabelMap) Set(y, x, label int) {
	if y < 0 || y >= lm.Rows || x < 0 || x >= lm.Cols {
		panic("segment2: LabelMap.Set out of range")
	}
	lm.Labels[y*lm.Cols+x] = label
	if label+1 > lm.NumLabels {
		lm.NumLabels = label + 1
	}
}

// InBounds reports whether pixel (x, y) lies inside the labelling.
func (lm *LabelMap) InBounds(y, x int) bool {
	return y >= 0 && y < lm.Rows && x >= 0 && x < lm.Cols
}

// Clone returns a deep copy of the LabelMap.
func (lm *LabelMap) Clone() *LabelMap {
	out := &LabelMap{Rows: lm.Rows, Cols: lm.Cols, NumLabels: lm.NumLabels}
	out.Labels = make([]int, len(lm.Labels))
	copy(out.Labels, lm.Labels)
	return out
}

// Compact renumbers the labels so the distinct values present become the
// contiguous range 0..k-1, preserving first-appearance order in row-major scan.
// It updates NumLabels to k and returns k. Negative labels (e.g. watershed
// lines) are left unchanged and do not consume a slot.
func (lm *LabelMap) Compact() int {
	remap := make(map[int]int)
	next := 0
	for i, l := range lm.Labels {
		if l < 0 {
			continue
		}
		nl, ok := remap[l]
		if !ok {
			nl = next
			remap[l] = nl
			next++
		}
		lm.Labels[i] = nl
	}
	lm.NumLabels = next
	if lm.NumLabels < 1 {
		lm.NumLabels = 1
	}
	return next
}

// RegionSizes returns the pixel count of each label slot, indexed by label.
// Negative labels are ignored. The result has length NumLabels.
func (lm *LabelMap) RegionSizes() []int {
	sizes := make([]int, lm.NumLabels)
	for _, l := range lm.Labels {
		if l >= 0 && l < lm.NumLabels {
			sizes[l]++
		}
	}
	return sizes
}

// BoundingRects returns the axis-aligned bounding rectangle of every label slot,
// indexed by label. Empty labels yield a zero rectangle. The result has length
// NumLabels.
func (lm *LabelMap) BoundingRects() []cv.Rect {
	minX := make([]int, lm.NumLabels)
	minY := make([]int, lm.NumLabels)
	maxX := make([]int, lm.NumLabels)
	maxY := make([]int, lm.NumLabels)
	seen := make([]bool, lm.NumLabels)
	for i, l := range lm.Labels {
		if l < 0 || l >= lm.NumLabels {
			continue
		}
		x := i % lm.Cols
		y := i / lm.Cols
		if !seen[l] {
			seen[l] = true
			minX[l], maxX[l] = x, x
			minY[l], maxY[l] = y, y
			continue
		}
		if x < minX[l] {
			minX[l] = x
		}
		if x > maxX[l] {
			maxX[l] = x
		}
		if y < minY[l] {
			minY[l] = y
		}
		if y > maxY[l] {
			maxY[l] = y
		}
	}
	rects := make([]cv.Rect, lm.NumLabels)
	for l := 0; l < lm.NumLabels; l++ {
		if !seen[l] {
			continue
		}
		rects[l] = cv.Rect{
			X:      minX[l],
			Y:      minY[l],
			Width:  maxX[l] - minX[l] + 1,
			Height: maxY[l] - minY[l] + 1,
		}
	}
	return rects
}

// RegionCentroids returns the (x, y) centroid of every label slot as a
// [cv.Point] with rounded integer coordinates, indexed by label. Empty labels
// yield the zero point. The result has length NumLabels.
func (lm *LabelMap) RegionCentroids() []cv.Point {
	sumX := make([]float64, lm.NumLabels)
	sumY := make([]float64, lm.NumLabels)
	cnt := make([]int, lm.NumLabels)
	for i, l := range lm.Labels {
		if l < 0 || l >= lm.NumLabels {
			continue
		}
		sumX[l] += float64(i % lm.Cols)
		sumY[l] += float64(i / lm.Cols)
		cnt[l]++
	}
	out := make([]cv.Point, lm.NumLabels)
	for l := 0; l < lm.NumLabels; l++ {
		if cnt[l] == 0 {
			continue
		}
		out[l] = cv.Point{
			X: int(sumX[l]/float64(cnt[l]) + 0.5),
			Y: int(sumY[l]/float64(cnt[l]) + 0.5),
		}
	}
	return out
}

// MeanColors returns the mean colour of every label slot over img, each a
// channels-length slice, indexed by label. img must match the LabelMap size.
// Empty labels yield a nil slice. It panics on a size mismatch.
func (lm *LabelMap) MeanColors(img *cv.Mat) [][]float64 {
	if img.Rows != lm.Rows || img.Cols != lm.Cols {
		panic("segment2: MeanColors size mismatch")
	}
	ch := img.Channels
	sums := make([][]float64, lm.NumLabels)
	cnt := make([]int, lm.NumLabels)
	for l := range sums {
		sums[l] = make([]float64, ch)
	}
	for i, l := range lm.Labels {
		if l < 0 || l >= lm.NumLabels {
			continue
		}
		b := i * ch
		for c := 0; c < ch; c++ {
			sums[l][c] += float64(img.Data[b+c])
		}
		cnt[l]++
	}
	out := make([][]float64, lm.NumLabels)
	for l := 0; l < lm.NumLabels; l++ {
		if cnt[l] == 0 {
			continue
		}
		m := make([]float64, ch)
		for c := 0; c < ch; c++ {
			m[c] = sums[l][c] / float64(cnt[l])
		}
		out[l] = m
	}
	return out
}

// RegionMask returns a single-channel [cv.Mat] mask that is 255 where the
// labelling equals label and 0 elsewhere.
func (lm *LabelMap) RegionMask(label int) *cv.Mat {
	out := cv.NewMat(lm.Rows, lm.Cols, 1)
	for i, l := range lm.Labels {
		if l == label {
			out.Data[i] = 255
		}
	}
	return out
}

// segment2palette is a fixed, deterministic set of distinct display colours used
// to render label maps. It repeats when labels exceed its length.
var segment2palette = [][3]uint8{
	{230, 25, 75}, {60, 180, 75}, {255, 225, 25}, {0, 130, 200},
	{245, 130, 48}, {145, 30, 180}, {70, 240, 240}, {240, 50, 230},
	{210, 245, 60}, {250, 190, 190}, {0, 128, 128}, {230, 190, 255},
	{170, 110, 40}, {255, 250, 200}, {128, 0, 0}, {170, 255, 195},
	{128, 128, 0}, {255, 215, 180}, {0, 0, 128}, {128, 128, 128},
}

// Colorize renders the labelling as a three-channel [cv.Mat], assigning each
// label a distinct colour from a fixed palette (cycled if there are more labels
// than palette entries). Negative labels are drawn black. The result is
// deterministic for a given labelling.
func (lm *LabelMap) Colorize() *cv.Mat {
	out := cv.NewMat(lm.Rows, lm.Cols, 3)
	for i, l := range lm.Labels {
		b := i * 3
		if l < 0 {
			continue
		}
		col := segment2palette[l%len(segment2palette)]
		out.Data[b] = col[0]
		out.Data[b+1] = col[1]
		out.Data[b+2] = col[2]
	}
	return out
}

// BoundaryMask returns a single-channel [cv.Mat] whose pixels are 255 wherever a
// pixel's label differs from that of its right or bottom neighbour, i.e. on
// region boundaries, and 0 elsewhere.
func (lm *LabelMap) BoundaryMask() *cv.Mat {
	out := cv.NewMat(lm.Rows, lm.Cols, 1)
	for y := 0; y < lm.Rows; y++ {
		for x := 0; x < lm.Cols; x++ {
			l := lm.Labels[y*lm.Cols+x]
			edge := false
			if x+1 < lm.Cols && lm.Labels[y*lm.Cols+x+1] != l {
				edge = true
			}
			if y+1 < lm.Rows && lm.Labels[(y+1)*lm.Cols+x] != l {
				edge = true
			}
			if edge {
				out.Data[y*lm.Cols+x] = 255
			}
		}
	}
	return out
}

// CountRegions returns the number of distinct non-negative labels present.
func (lm *LabelMap) CountRegions() int {
	seen := make(map[int]struct{})
	for _, l := range lm.Labels {
		if l >= 0 {
			seen[l] = struct{}{}
		}
	}
	return len(seen)
}

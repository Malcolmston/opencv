package video

import (
	cv "github.com/malcolmston/opencv"
)

// FlowField is a dense, two-channel float64 motion field: for every pixel it
// stores a horizontal (X) and vertical (Y) displacement from the previous frame
// to the next. It is the two-channel float analogue of the root package's
// single-channel cv.FloatMat, which cannot hold an interleaved (dx, dy) pair.
//
// Samples are stored interleaved in Data with length Rows*Cols*2: the flow for
// row y, column x is at Data[(y*Cols+x)*2] (X) and Data[(y*Cols+x)*2+1] (Y).
type FlowField struct {
	Rows int
	Cols int
	Data []float64
}

// NewFlowField allocates a zero-filled FlowField.
func NewFlowField(rows, cols int) *FlowField {
	return &FlowField{Rows: rows, Cols: cols, Data: make([]float64, rows*cols*2)}
}

// At returns the (dx, dy) displacement stored at row y, column x.
func (f *FlowField) At(y, x int) (dx, dy float64) {
	i := (y*f.Cols + x) * 2
	return f.Data[i], f.Data[i+1]
}

// set stores the (dx, dy) displacement at row y, column x.
func (f *FlowField) set(y, x int, dx, dy float64) {
	i := (y*f.Cols + x) * 2
	f.Data[i] = dx
	f.Data[i+1] = dy
}

// CalcOpticalFlowFarneback computes a dense optical-flow field from prev to
// next. This is a deliberately simplified stand-in for OpenCV's
// polynomial-expansion Farneback algorithm: instead of fitting local quadratic
// polynomials, it performs integer block matching. For every pixel it slides a
// (2*winSize+1) x (2*winSize+1) intensity window over a square search area of
// radius searchRadius in the next frame and picks the integer displacement that
// minimises the sum of squared differences. The winning (dx, dy) is written to
// the corresponding pixel of the returned [FlowField].
//
// Because the search is over integer offsets, the resulting displacements are
// integer-valued; the method is intended for small motions (a few pixels) and
// well-textured images. Untextured regions produce ambiguous matches and may
// yield a zero or arbitrary-but-deterministic displacement (ties are broken by
// preferring the smaller-magnitude offset, then row-major order).
//
// prev and next must have identical dimensions. winSize and searchRadius must
// be >= 1. Multi-channel inputs are converted to grayscale first.
func CalcOpticalFlowFarneback(prev, next *cv.Mat, winSize, searchRadius int) *FlowField {
	if prev == nil || next == nil || prev.Empty() || next.Empty() {
		panic("video: CalcOpticalFlowFarneback requires non-empty images")
	}
	if prev.Rows != next.Rows || prev.Cols != next.Cols {
		panic("video: CalcOpticalFlowFarneback requires equal-sized images")
	}
	if winSize < 1 || searchRadius < 1 {
		panic("video: CalcOpticalFlowFarneback requires winSize >= 1 and searchRadius >= 1")
	}
	pg := gridFromMat(toGray(prev))
	ng := gridFromMat(toGray(next))
	rows, cols := prev.Rows, prev.Cols
	flow := NewFlowField(rows, cols)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			bestSSD := -1.0
			var bestDX, bestDY int
			bestMag := 1 << 30
			for dy := -searchRadius; dy <= searchRadius; dy++ {
				for dx := -searchRadius; dx <= searchRadius; dx++ {
					var ssd float64
					for wy := -winSize; wy <= winSize; wy++ {
						for wx := -winSize; wx <= winSize; wx++ {
							a := pg.atClamp(x+wx, y+wy)
							b := ng.atClamp(x+wx+dx, y+wy+dy)
							d := a - b
							ssd += d * d
						}
					}
					mag := dx*dx + dy*dy
					// Prefer lower SSD; break ties toward smaller displacement
					// magnitude for a stable, deterministic result.
					if bestSSD < 0 || ssd < bestSSD || (ssd == bestSSD && mag < bestMag) {
						bestSSD = ssd
						bestDX = dx
						bestDY = dy
						bestMag = mag
					}
				}
			}
			flow.set(y, x, float64(bestDX), float64(bestDY))
		}
	}
	return flow
}

// MeanFlow returns the average (dx, dy) displacement over the interior of the
// field, excluding a border of the given width where block matching is least
// reliable. It is a small convenience used for summarising a flow field. border
// must be >= 0 and small enough to leave a non-empty interior.
func (f *FlowField) MeanFlow(border int) (dx, dy float64) {
	if border < 0 {
		border = 0
	}
	var sx, sy float64
	var n int
	for y := border; y < f.Rows-border; y++ {
		for x := border; x < f.Cols-border; x++ {
			ix, iy := f.At(y, x)
			sx += ix
			sy += iy
			n++
		}
	}
	if n == 0 {
		return 0, 0
	}
	return sx / float64(n), sy / float64(n)
}

package optflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// grid is a single-channel float64 image used internally for sub-pixel sampling
// of intensities and gradients. Data is stored row-major, length Rows*Cols.
// Keeping the working representation in float64 avoids the repeated 8-bit
// quantisation that would otherwise accumulate across pyramid levels and
// warping iterations.
type grid struct {
	Rows int
	Cols int
	Data []float64
}

// newGrid allocates a zero-filled grid.
func newGrid(rows, cols int) *grid {
	return &grid{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// clampInt clamps v to the inclusive range [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// at returns the sample at integer coordinates (x, y) without bounds checking.
func (g *grid) at(x, y int) float64 { return g.Data[y*g.Cols+x] }

// atClamp returns the sample at integer coordinates (x, y), replicating the
// border for out-of-range coordinates (BORDER_REPLICATE).
func (g *grid) atClamp(x, y int) float64 {
	x = clampInt(x, 0, g.Cols-1)
	y = clampInt(y, 0, g.Rows-1)
	return g.Data[y*g.Cols+x]
}

// bilinear samples the grid at fractional coordinates (x, y) using bilinear
// interpolation, replicating the border for out-of-range neighbours.
func (g *grid) bilinear(x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	dx := x - float64(x0)
	dy := y - float64(y0)
	v00 := g.atClamp(x0, y0)
	v01 := g.atClamp(x0+1, y0)
	v10 := g.atClamp(x0, y0+1)
	v11 := g.atClamp(x0+1, y0+1)
	top := v00*(1-dx) + v01*dx
	bot := v10*(1-dx) + v11*dx
	return top*(1-dy) + bot*dy
}

// gridFromMat converts a single-channel cv.Mat to a float grid.
func gridFromMat(m *cv.Mat) *grid {
	g := newGrid(m.Rows, m.Cols)
	for i := 0; i < len(m.Data); i++ {
		g.Data[i] = float64(m.Data[i])
	}
	return g
}

// toGray returns a single-channel cv.Mat. A one-channel input is cloned; a
// three-channel input is converted with cv.CvtColor (BT.601 luma); any other
// channel count falls back to the first channel. This mirrors the grayscale
// convention used across the root package.
func toGray(m *cv.Mat) *cv.Mat {
	switch m.Channels {
	case 1:
		return m.Clone()
	case 3:
		return cv.CvtColor(m, cv.ColorRGB2Gray)
	default:
		out := cv.NewMat(m.Rows, m.Cols, 1)
		for p := 0; p < m.Total(); p++ {
			out.Data[p] = m.Data[p*m.Channels]
		}
		return out
	}
}

// grayGrid converts any cv.Mat to a grayscale float grid in one step.
func grayGrid(m *cv.Mat) *grid { return gridFromMat(toGray(m)) }

// requirePair validates that two frames are non-empty and identically sized,
// panicking with a message tagged by the calling function name.
func requirePair(prev, next *cv.Mat, fn string) {
	if prev == nil || next == nil || prev.Empty() || next.Empty() {
		panic("optflow: " + fn + " requires non-empty images")
	}
	if prev.Rows != next.Rows || prev.Cols != next.Cols {
		panic("optflow: " + fn + " requires equal-sized images")
	}
}

// sobelX and sobelY are the classic 3x3 Sobel derivative stencils. The raw
// response is 8x the underlying central difference, so callers scale by 1/8 to
// recover a true first derivative (matching video/internal.go's sobelScale3).
func sobelGradients(g *grid) (gx, gy *grid) {
	gx = newGrid(g.Rows, g.Cols)
	gy = newGrid(g.Rows, g.Cols)
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			p00 := g.atClamp(x-1, y-1)
			p01 := g.atClamp(x, y-1)
			p02 := g.atClamp(x+1, y-1)
			p10 := g.atClamp(x-1, y)
			p12 := g.atClamp(x+1, y)
			p20 := g.atClamp(x-1, y+1)
			p21 := g.atClamp(x, y+1)
			p22 := g.atClamp(x+1, y+1)
			sx := (p02 + 2*p12 + p22) - (p00 + 2*p10 + p20)
			sy := (p20 + 2*p21 + p22) - (p00 + 2*p01 + p02)
			gx.Data[y*g.Cols+x] = sx / 8.0
			gy.Data[y*g.Cols+x] = sy / 8.0
		}
	}
	return gx, gy
}

// gaussianSmooth applies a separable 5-tap [1 4 6 4 1]/16 Gaussian to the grid,
// replicating borders. It is the smoothing step used before pyramid
// downsampling.
func gaussianSmooth(g *grid) *grid {
	k := [5]float64{1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16}
	tmp := newGrid(g.Rows, g.Cols)
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			var s float64
			for t := -2; t <= 2; t++ {
				s += k[t+2] * g.atClamp(x+t, y)
			}
			tmp.Data[y*g.Cols+x] = s
		}
	}
	out := newGrid(g.Rows, g.Cols)
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			var s float64
			for t := -2; t <= 2; t++ {
				s += k[t+2] * tmp.atClamp(x, y+t)
			}
			out.Data[y*g.Cols+x] = s
		}
	}
	return out
}

// downsample halves both dimensions of the grid after Gaussian smoothing,
// mirroring OpenCV's pyrDown. The result size is ceil(Rows/2) x ceil(Cols/2)
// with a minimum of 1 in each dimension.
func downsample(g *grid) *grid {
	sm := gaussianSmooth(g)
	nr := (g.Rows + 1) / 2
	nc := (g.Cols + 1) / 2
	if nr < 1 {
		nr = 1
	}
	if nc < 1 {
		nc = 1
	}
	out := newGrid(nr, nc)
	for y := 0; y < nr; y++ {
		for x := 0; x < nc; x++ {
			out.Data[y*nc+x] = sm.atClamp(2*x, 2*y)
		}
	}
	return out
}

// buildPyramid returns a Gaussian pyramid with pyr[0] being the full-resolution
// grid and each higher level roughly half the size of the previous. Building
// stops early once a dimension would drop below minSize. The number of levels
// never exceeds levels+1 (levels additional coarser levels beyond level 0).
func buildPyramid(g *grid, levels, minSize int) []*grid {
	if minSize < 1 {
		minSize = 1
	}
	pyr := []*grid{g}
	for i := 0; i < levels; i++ {
		top := pyr[len(pyr)-1]
		if top.Rows < 2*minSize || top.Cols < 2*minSize {
			break
		}
		pyr = append(pyr, downsample(top))
	}
	return pyr
}

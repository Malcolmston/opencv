package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// trackingGray is an internal single-channel float64 image used by the
// optical-flow and template-matching routines, where 8-bit precision and
// integer sampling are insufficient. Data is row-major with one value per pixel.
type trackingGray struct {
	rows int
	cols int
	data []float64
}

// trackingNewGray allocates a zero-filled float grayscale image.
func trackingNewGray(rows, cols int) *trackingGray {
	return &trackingGray{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

// at returns the value at (y, x), clamping out-of-range coordinates to the
// nearest edge (replicate border).
func (g *trackingGray) at(y, x int) float64 {
	if y < 0 {
		y = 0
	} else if y >= g.rows {
		y = g.rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= g.cols {
		x = g.cols - 1
	}
	return g.data[y*g.cols+x]
}

// set stores value at (y, x); out-of-range coordinates are ignored.
func (g *trackingGray) set(y, x int, v float64) {
	if y < 0 || y >= g.rows || x < 0 || x >= g.cols {
		return
	}
	g.data[y*g.cols+x] = v
}

// bilinear samples the image at sub-pixel coordinate (x, y) with bilinear
// interpolation and edge clamping.
func (g *trackingGray) bilinear(x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	dx := x - float64(x0)
	dy := y - float64(y0)
	v00 := g.at(y0, x0)
	v01 := g.at(y0, x0+1)
	v10 := g.at(y0+1, x0)
	v11 := g.at(y0+1, x0+1)
	top := v00*(1-dx) + v01*dx
	bot := v10*(1-dx) + v11*dx
	return top*(1-dy) + bot*dy
}

// clone returns an independent copy of the image.
func (g *trackingGray) clone() *trackingGray {
	out := trackingNewGray(g.rows, g.cols)
	copy(out.data, g.data)
	return out
}

// trackingToGrayF converts a *cv.Mat into a float grayscale image. A
// single-channel Mat is copied directly; a three- or four-channel Mat is
// reduced with the ITU-R BT.601 luma weights (channels assumed B, G, R as in
// the root package); any other channel count is averaged.
func trackingToGrayF(m *cv.Mat) *trackingGray {
	g := trackingNewGray(m.Rows, m.Cols)
	ch := m.Channels
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			base := (y*m.Cols + x) * ch
			var v float64
			switch {
			case ch == 1:
				v = float64(m.Data[base])
			case ch >= 3:
				b := float64(m.Data[base])
				gg := float64(m.Data[base+1])
				r := float64(m.Data[base+2])
				v = 0.114*b + 0.587*gg + 0.299*r
			default:
				var s float64
				for c := 0; c < ch; c++ {
					s += float64(m.Data[base+c])
				}
				v = s / float64(ch)
			}
			g.data[y*g.cols+x] = v
		}
	}
	return g
}

// toMat converts the float grayscale image back into an 8-bit single-channel
// *cv.Mat, rounding and clamping each value to [0, 255].
func (g *trackingGray) toMat() *cv.Mat {
	m := cv.NewMat(g.rows, g.cols, 1)
	for i, v := range g.data {
		m.Data[i] = clampU8(v)
	}
	return m
}

// clampU8 rounds v to the nearest integer and clamps it into the byte range.
func clampU8(v float64) uint8 {
	r := math.Round(v)
	if r < 0 {
		return 0
	}
	if r > 255 {
		return 255
	}
	return uint8(r)
}

// trackingGaussianBlur3 applies a separable 3x3 Gaussian ([1 2 1]/4) with edge
// replication and returns a new image. It is the pre-filter used before
// pyramid subsampling.
func trackingGaussianBlur3(g *trackingGray) *trackingGray {
	tmp := trackingNewGray(g.rows, g.cols)
	// Horizontal pass.
	for y := 0; y < g.rows; y++ {
		for x := 0; x < g.cols; x++ {
			v := g.at(y, x-1) + 2*g.at(y, x) + g.at(y, x+1)
			tmp.data[y*g.cols+x] = v / 4
		}
	}
	out := trackingNewGray(g.rows, g.cols)
	// Vertical pass.
	for y := 0; y < g.rows; y++ {
		for x := 0; x < g.cols; x++ {
			v := tmp.at(y-1, x) + 2*tmp.at(y, x) + tmp.at(y+1, x)
			out.data[y*g.cols+x] = v / 4
		}
	}
	return out
}

// trackingPyrDown blurs the image and subsamples it by two in each dimension,
// producing an image of size (ceil(rows/2), ceil(cols/2)).
func trackingPyrDown(g *trackingGray) *trackingGray {
	blurred := trackingGaussianBlur3(g)
	nr := (g.rows + 1) / 2
	nc := (g.cols + 1) / 2
	out := trackingNewGray(nr, nc)
	for y := 0; y < nr; y++ {
		for x := 0; x < nc; x++ {
			out.data[y*nc+x] = blurred.at(y*2, x*2)
		}
	}
	return out
}

// gradients computes the horizontal and vertical spatial derivatives of the
// image using centred differences with edge replication.
func (g *trackingGray) gradients() (ix, iy *trackingGray) {
	ix = trackingNewGray(g.rows, g.cols)
	iy = trackingNewGray(g.rows, g.cols)
	for y := 0; y < g.rows; y++ {
		for x := 0; x < g.cols; x++ {
			ix.data[y*g.cols+x] = (g.at(y, x+1) - g.at(y, x-1)) / 2
			iy.data[y*g.cols+x] = (g.at(y+1, x) - g.at(y-1, x)) / 2
		}
	}
	return ix, iy
}

// sobel computes the Sobel horizontal and vertical derivatives with edge
// replication, used by the dense flow solvers for a smoother gradient estimate.
func (g *trackingGray) sobel() (ix, iy *trackingGray) {
	ix = trackingNewGray(g.rows, g.cols)
	iy = trackingNewGray(g.rows, g.cols)
	for y := 0; y < g.rows; y++ {
		for x := 0; x < g.cols; x++ {
			gx := (g.at(y-1, x+1) + 2*g.at(y, x+1) + g.at(y+1, x+1)) -
				(g.at(y-1, x-1) + 2*g.at(y, x-1) + g.at(y+1, x-1))
			gy := (g.at(y+1, x-1) + 2*g.at(y+1, x) + g.at(y+1, x+1)) -
				(g.at(y-1, x-1) + 2*g.at(y-1, x) + g.at(y-1, x+1))
			ix.data[y*g.cols+x] = gx / 8
			iy.data[y*g.cols+x] = gy / 8
		}
	}
	return ix, iy
}

// BuildOpticalFlowPyramid builds a Gaussian image pyramid for coarse-to-fine
// optical-flow tracking. The input may be single- or multi-channel; it is first
// converted to grayscale. The returned slice has at most maxLevel+1 entries:
// element 0 is the full-resolution grayscale image (as an 8-bit single-channel
// *cv.Mat) and each subsequent level is produced from the previous one by a
// Gaussian blur followed by 2x subsampling, halving both dimensions (rounding
// up).
//
// maxLevel must be >= 0. Levels stop early if a dimension would fall below one
// pixel, so the returned slice may contain fewer than maxLevel+1 entries for
// very small images. The result is deterministic.
func BuildOpticalFlowPyramid(img *cv.Mat, maxLevel int) []*cv.Mat {
	if img == nil || img.Empty() {
		panic("tracking: BuildOpticalFlowPyramid requires a non-empty image")
	}
	if maxLevel < 0 {
		panic("tracking: BuildOpticalFlowPyramid requires maxLevel >= 0")
	}
	levels := trackingBuildPyramid(trackingToGrayF(img), maxLevel)
	out := make([]*cv.Mat, len(levels))
	for i, lv := range levels {
		out[i] = lv.toMat()
	}
	return out
}

// trackingBuildPyramid builds a float grayscale Gaussian pyramid.
func trackingBuildPyramid(g *trackingGray, maxLevel int) []*trackingGray {
	pyr := make([]*trackingGray, 0, maxLevel+1)
	pyr = append(pyr, g)
	for l := 1; l <= maxLevel; l++ {
		prev := pyr[l-1]
		if prev.rows < 2 || prev.cols < 2 {
			break
		}
		pyr = append(pyr, trackingPyrDown(prev))
	}
	return pyr
}

package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// sub returns the element-wise difference a-b. The grids must share a shape.
func sub(a, b grid) grid {
	out := newGrid(a.rows, a.cols)
	for i := range out.data {
		out.data[i] = a.data[i] - b.data[i]
	}
	return out
}

// absGrid returns the element-wise absolute value of a.
func absGrid(a grid) grid {
	out := newGrid(a.rows, a.cols)
	for i := range out.data {
		out.data[i] = math.Abs(a.data[i])
	}
	return out
}

// decimate2 subsamples g by taking every second row and column (no averaging),
// matching the ref(1:2:end, 1:2:end) decimation of the VIF reference code. The
// result keeps indices 0, 2, 4, … and is therefore ceil(n/2) on each axis.
func decimate2(g grid) grid {
	r := (g.rows + 1) / 2
	c := (g.cols + 1) / 2
	if r < 1 {
		r = 1
	}
	if c < 1 {
		c = 1
	}
	out := newGrid(r, c)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			out.data[out.idx(y, x)] = g.data[g.idx(2*y, 2*x)]
		}
	}
	return out
}

// chromaIQ extracts the I and Q chrominance channels of the YIQ colour space
// from a three-channel (R,G,B-ordered) image, using the standard NTSC weights.
// The returned hasColor flag is false for single-channel images, in which case
// the I and Q grids are zero and callers should skip the chrominance terms.
func chromaIQ(m *cv.Mat) (iCh, qCh grid, hasColor bool) {
	iCh = newGrid(m.Rows, m.Cols)
	qCh = newGrid(m.Rows, m.Cols)
	if m.Channels != 3 {
		return iCh, qCh, false
	}
	for p := 0; p < m.Total(); p++ {
		b := p * 3
		r := float64(m.Data[b+0])
		g := float64(m.Data[b+1])
		bl := float64(m.Data[b+2])
		iCh.data[p] = 0.596*r - 0.274*g - 0.322*bl
		qCh.data[p] = 0.211*r - 0.523*g + 0.312*bl
	}
	return iCh, qCh, true
}

// similarityMapToMat renders a single-channel float grid to a [cv.Mat] after
// scaling values assumed to lie in [0, 1] into the 8-bit range, clamping any
// that fall outside it.
func similarityMapToMat(g grid) *cv.Mat {
	vis := newGrid(g.rows, g.cols)
	for i, v := range g.data {
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		vis.data[i] = v * 255
	}
	return grayMapToMat(vis)
}

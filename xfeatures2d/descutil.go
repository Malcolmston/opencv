package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// L2Distance returns the Euclidean (L2) distance between two equal-length
// floating-point descriptors. It panics if the descriptors differ in length.
// It is the natural distance for the float descriptors in this package
// ([DAISY], [VGG], [SURF]).
func L2Distance(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("xfeatures2d: L2Distance on descriptors of different length")
	}
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}

// packBit sets bit index bit (little-endian within each byte, matching
// [HammingDistance] and BRISK) of the packed descriptor desc.
func packBit(desc []byte, bit int) {
	desc[bit>>3] |= 1 << uint(bit&7)
}

// gradientMaps computes the horizontal and vertical image gradients of a
// single-channel image with central differences and border replication. The
// returned slices are row-major of length g.Rows*g.Cols.
func gradientMaps(g *cv.Mat) (gx, gy []float64) {
	rows, cols := g.Rows, g.Cols
	gx = make([]float64, rows*cols)
	gy = make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			xm := grayAtClamped(g, x-1, y)
			xp := grayAtClamped(g, x+1, y)
			ym := grayAtClamped(g, x, y-1)
			yp := grayAtClamped(g, x, y+1)
			gx[y*cols+x] = 0.5 * (xp - xm)
			gy[y*cols+x] = 0.5 * (yp - ym)
		}
	}
	return gx, gy
}

// intensityCentroidAngle estimates a keypoint orientation (in radians) from the
// intensity centroid (the first-order image moments) inside a disc of the given
// radius around (cx, cy), using border replication. This is the orientation
// assignment used by ORB and reused here for the rotation-invariant binary
// descriptors. When the neighbourhood is flat the angle is 0.
func intensityCentroidAngle(g *cv.Mat, cx, cy, radius int) float64 {
	if radius < 1 {
		radius = 1
	}
	var m01, m10 float64
	r2 := radius * radius
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			if dx*dx+dy*dy > r2 {
				continue
			}
			v := grayAtClamped(g, cx+dx, cy+dy)
			m10 += float64(dx) * v
			m01 += float64(dy) * v
		}
	}
	if m01 == 0 && m10 == 0 {
		return 0
	}
	return math.Atan2(m01, m10)
}

// floatIntegral is a summed-area table over a floating-point map, the analogue
// of [integral] for gradient or response images.
type floatIntegral struct {
	rows int
	cols int
	data []float64 // (rows+1)*(cols+1), same layout as integral.data
}

// newFloatIntegral builds the summed-area table of a row-major float map of the
// given dimensions.
func newFloatIntegral(src []float64, rows, cols int) *floatIntegral {
	w := cols + 1
	data := make([]float64, (rows+1)*w)
	for y := 0; y < rows; y++ {
		var rowSum float64
		for x := 0; x < cols; x++ {
			rowSum += src[y*cols+x]
			data[(y+1)*w+(x+1)] = data[y*w+(x+1)] + rowSum
		}
	}
	return &floatIntegral{rows: rows, cols: cols, data: data}
}

// sum returns the inclusive sum over the rectangle [x0,x1] × [y0,y1], clamped to
// the map. An empty rectangle returns 0.
func (f *floatIntegral) sum(x0, y0, x1, y1 int) float64 {
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 >= f.cols {
		x1 = f.cols - 1
	}
	if y1 >= f.rows {
		y1 = f.rows - 1
	}
	if x1 < x0 || y1 < y0 {
		return 0
	}
	w := f.cols + 1
	a := f.data[y0*w+x0]
	b := f.data[y0*w+(x1+1)]
	c := f.data[(y1+1)*w+x0]
	d := f.data[(y1+1)*w+(x1+1)]
	return d - b - c + a
}

// boxMean returns the mean value over the rectangle centred at (cx, cy) with the
// given half-width and half-height, clamped to the map.
func (f *floatIntegral) boxMean(cx, cy, hw, hh int) float64 {
	x0, y0 := cx-hw, cy-hh
	x1, y1 := cx+hw, cy+hh
	ex0, ey0, ex1, ey1 := x0, y0, x1, y1
	if ex0 < 0 {
		ex0 = 0
	}
	if ey0 < 0 {
		ey0 = 0
	}
	if ex1 >= f.cols {
		ex1 = f.cols - 1
	}
	if ey1 >= f.rows {
		ey1 = f.rows - 1
	}
	area := (ex1 - ex0 + 1) * (ey1 - ey0 + 1)
	if area <= 0 {
		return 0
	}
	return f.sum(x0, y0, x1, y1) / float64(area)
}

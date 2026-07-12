package face

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// This file extends the basic operators in lbp.go with the "extended" Local
// Binary Pattern family from Ojala, Pietikäinen & Mäenpää (2002): sampling on a
// circle of arbitrary radius and neighbour count with bilinear interpolation,
// and the rotation-invariant uniform (riu2) variant OpenCV exposes through its
// LBPH extended parameters.

// LBPCircular computes the extended, circular Local Binary Pattern code image of
// img with the given radius and neighbour count. The image is reduced to luma;
// for each interior pixel the neighbours are sampled at equal angles on a circle
// of the given radius (bilinearly interpolated between the four surrounding
// pixels, since sample points rarely land on the integer grid) and thresholded
// against the centre, most-significant bit first. neighbours must be between 1
// and 8 so a code still fits in the uint8 output; radius must be at least 1.
//
// The result is a single-channel Mat of size (Rows−2r)×(Cols−2r): the r-pixel
// border, where the circle would fall outside the image, is dropped. With
// radius 1 and 8 neighbours the sampling reduces to the classic 3×3 pattern, up
// to the interpolation of the four diagonal neighbours (which are exact at
// radius 1). It panics on out-of-range parameters or an image too small to hold
// a single sampling neighbourhood.
func LBPCircular(img *cv.Mat, radius, neighbours int) *cv.Mat {
	if radius < 1 {
		panic("face: LBPCircular radius must be >= 1")
	}
	if neighbours < 1 || neighbours > 8 {
		panic("face: LBPCircular neighbours must be in [1,8]")
	}
	g := toGrayMat(img)
	if g.Rows < 2*radius+1 || g.Cols < 2*radius+1 {
		panic("face: LBPCircular image too small for the requested radius")
	}
	offs := circleOffsets(radius, neighbours)
	outRows := g.Rows - 2*radius
	outCols := g.Cols - 2*radius
	out := cv.NewMat(outRows, outCols, 1)
	cols := g.Cols
	for y := radius; y < g.Rows-radius; y++ {
		for x := radius; x < g.Cols-radius; x++ {
			center := float64(g.Data[y*cols+x])
			code := 0
			for k := 0; k < neighbours; k++ {
				val := bilinearSample(g, float64(x)+offs[k][0], float64(y)+offs[k][1])
				// Most-significant bit first: neighbour 0 is the top bit.
				if val >= center {
					code |= 1 << (neighbours - 1 - k)
				}
			}
			out.Data[(y-radius)*outCols+(x-radius)] = uint8(code)
		}
	}
	return out
}

// LBPUniformRotInvariant computes the rotation-invariant uniform LBP (the "riu2"
// operator) of img on the classic radius-1, 8-neighbour circle. Each uniform
// pattern (at most two circular 0↔1 transitions) is labelled by its number of
// set bits, giving labels 0–8; every non-uniform pattern collapses to label 9,
// for 10 labels total. Because the label is the popcount of a uniform code it is
// invariant to rotation of the neighbourhood, so the descriptor is unaffected by
// in-plane rotation of the texture, unlike [LBPUniform].
//
// The result is a single-channel Mat of size (Rows−2)×(Cols−2). It panics if img
// is smaller than 3×3.
func LBPUniformRotInvariant(img *cv.Mat) *cv.Mat {
	g := toGrayMat(img)
	if g.Rows < 3 || g.Cols < 3 {
		panic("face: LBPUniformRotInvariant requires an image of at least 3x3")
	}
	outRows := g.Rows - 2
	outCols := g.Cols - 2
	out := cv.NewMat(outRows, outCols, 1)
	cols := g.Cols
	for y := 1; y < g.Rows-1; y++ {
		for x := 1; x < g.Cols-1; x++ {
			center := g.Data[y*cols+x]
			code := 0
			for k := 0; k < 8; k++ {
				ny := y + lbpOffsets[k][0]
				nx := x + lbpOffsets[k][1]
				if g.Data[ny*cols+nx] >= center {
					code |= lbpWeights[k]
				}
			}
			out.Data[(y-1)*outCols+(x-1)] = uint8(riu2Map[code])
		}
	}
	return out
}

// riu2Map maps each 8-bit LBP code to its rotation-invariant uniform label: the
// popcount for uniform codes (0–8) and 9 for non-uniform codes.
var riu2Map = buildRIU2Map()

func buildRIU2Map() [256]int {
	var table [256]int
	for code := 0; code < 256; code++ {
		if bitTransitions(code) <= 2 {
			table[code] = popcount8(code)
		} else {
			table[code] = 9
		}
	}
	return table
}

// popcount8 counts the set bits of an 8-bit value.
func popcount8(code int) int {
	n := 0
	for code != 0 {
		code &= code - 1
		n++
	}
	return n
}

// circleOffsets returns the (dx, dy) offsets of neighbours sampled at equal
// angles on a circle of the given radius, starting straight up and proceeding
// clockwise. Small rounding noise is snapped to zero so that, at radius 1, the
// axis-aligned neighbours sample exact grid pixels.
func circleOffsets(radius, neighbours int) [][2]float64 {
	offs := make([][2]float64, neighbours)
	for k := 0; k < neighbours; k++ {
		theta := 2 * math.Pi * float64(k) / float64(neighbours)
		dx := float64(radius) * math.Sin(theta)
		dy := -float64(radius) * math.Cos(theta)
		if math.Abs(dx) < 1e-9 {
			dx = 0
		}
		if math.Abs(dy) < 1e-9 {
			dy = 0
		}
		offs[k] = [2]float64{dx, dy}
	}
	return offs
}

// bilinearSample reads img at the fractional location (fx, fy) by bilinear
// interpolation of the four surrounding integer pixels. Coordinates are assumed
// to lie within the valid interior established by the caller.
func bilinearSample(img *cv.Mat, fx, fy float64) float64 {
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	x1 := x0 + 1
	y1 := y0 + 1
	ax := fx - float64(x0)
	ay := fy - float64(y0)
	cols := img.Cols
	// Clamp to the image (the border is excluded by the caller, but the far
	// corner of the circle can round one pixel past the edge).
	if x1 >= img.Cols {
		x1 = img.Cols - 1
	}
	if y1 >= img.Rows {
		y1 = img.Rows - 1
	}
	p00 := float64(img.Data[y0*cols+x0])
	p01 := float64(img.Data[y0*cols+x1])
	p10 := float64(img.Data[y1*cols+x0])
	p11 := float64(img.Data[y1*cols+x1])
	top := p00*(1-ax) + p01*ax
	bot := p10*(1-ax) + p11*ax
	return top*(1-ay) + bot*ay
}

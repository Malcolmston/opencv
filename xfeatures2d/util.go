package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// toGray returns a single-channel view of img. A single-channel Mat is returned
// unchanged (not copied); a three-channel Mat is converted with the BT.601 luma
// weights via cv.CvtColor. It panics if img has neither 1 nor 3 channels.
func toGray(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return img
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		panic("xfeatures2d: expected a 1- or 3-channel image")
	}
}

// grayAtClamped returns the intensity at integer pixel (x, y), clamping
// out-of-range coordinates to the nearest edge (border replication). g must be
// single-channel.
func grayAtClamped(g *cv.Mat, x, y int) float64 {
	if x < 0 {
		x = 0
	} else if x >= g.Cols {
		x = g.Cols - 1
	}
	if y < 0 {
		y = 0
	} else if y >= g.Rows {
		y = g.Rows - 1
	}
	return float64(g.Data[y*g.Cols+x])
}

// bilinear samples g at fractional coordinates (x, y) with bilinear
// interpolation and border replication. g must be single-channel.
func bilinear(g *cv.Mat, x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)
	v00 := grayAtClamped(g, x0, y0)
	v10 := grayAtClamped(g, x0+1, y0)
	v01 := grayAtClamped(g, x0, y0+1)
	v11 := grayAtClamped(g, x0+1, y0+1)
	top := v00*(1-fx) + v10*fx
	bot := v01*(1-fx) + v11*fx
	return top*(1-fy) + bot*fy
}

// integral is a summed-area table of a single-channel image. Sum returns the
// inclusive sum of samples over an axis-aligned rectangle in constant time.
type integral struct {
	rows int
	cols int
	// data has (rows+1)*(cols+1) entries; data[(y+1)*(cols+1)+(x+1)] is the sum
	// of all samples with row < y+1 and column < x+1.
	data []int64
}

// newIntegral builds the summed-area table of a single-channel image.
func newIntegral(g *cv.Mat) *integral {
	rows, cols := g.Rows, g.Cols
	w := cols + 1
	data := make([]int64, (rows+1)*w)
	for y := 0; y < rows; y++ {
		var rowSum int64
		for x := 0; x < cols; x++ {
			rowSum += int64(g.Data[y*cols+x])
			data[(y+1)*w+(x+1)] = data[y*w+(x+1)] + rowSum
		}
	}
	return &integral{rows: rows, cols: cols, data: data}
}

// sum returns the inclusive sum of samples over the rectangle [x0,x1] × [y0,y1].
// The rectangle is clamped to the image; an empty rectangle returns 0.
func (it *integral) sum(x0, y0, x1, y1 int) int64 {
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 >= it.cols {
		x1 = it.cols - 1
	}
	if y1 >= it.rows {
		y1 = it.rows - 1
	}
	if x1 < x0 || y1 < y0 {
		return 0
	}
	w := it.cols + 1
	a := it.data[y0*w+x0]
	b := it.data[y0*w+(x1+1)]
	c := it.data[(y1+1)*w+x0]
	d := it.data[(y1+1)*w+(x1+1)]
	return d - b - c + a
}

// boxMean returns the mean sample value over the (2r+1)×(2r+1) window centred on
// (cx, cy), clamped to the image.
func (it *integral) boxMean(cx, cy, r int) float64 {
	x0, y0 := cx-r, cy-r
	x1, y1 := cx+r, cy+r
	ex0, ey0 := x0, y0
	ex1, ey1 := x1, y1
	if ex0 < 0 {
		ex0 = 0
	}
	if ey0 < 0 {
		ey0 = 0
	}
	if ex1 >= it.cols {
		ex1 = it.cols - 1
	}
	if ey1 >= it.rows {
		ey1 = it.rows - 1
	}
	area := (ex1 - ex0 + 1) * (ey1 - ey0 + 1)
	if area <= 0 {
		return 0
	}
	return float64(it.sum(x0, y0, x1, y1)) / float64(area)
}

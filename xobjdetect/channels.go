package xobjdetect

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// NumOrientBins is the number of unsigned gradient-orientation histogram bins
// used by the oriented-gradient channels.
const NumOrientBins = 6

// NumChannels is the number of integral feature channels produced from an
// image: three colour channels (L*, a*, b*), one gradient-magnitude channel,
// and NumOrientBins oriented-gradient histogram channels.
const NumChannels = 3 + 1 + NumOrientBins

// integralChannels holds one padded integral image per feature channel so that
// the sum of any channel over an axis-aligned rectangle is an O(1) lookup.
//
// Each integral image has (rows+1)*(cols+1) entries; integ[ch][y*(cols+1)+x] is
// the sum of channel ch over the half-open region [0,y) x [0,x).
type integralChannels struct {
	rows, cols int
	stride     int // cols+1
	integ      [][]float64
}

// grayPlane returns the luma of img as a row-major float slice in [0,255].
// Multi-channel images are reduced with the BT.601 weights the root package
// uses; single-channel images are copied directly.
func grayPlane(img *cv.Mat) []float64 {
	rows, cols := img.Rows, img.Cols
	out := make([]float64, rows*cols)
	if img.Channels == 1 {
		for i := 0; i < rows*cols; i++ {
			out[i] = float64(img.Data[i])
		}
		return out
	}
	ch := img.Channels
	for i := 0; i < rows*cols; i++ {
		base := i * ch
		r := float64(img.Data[base])
		g := float64(img.Data[base+1])
		b := float64(img.Data[base+2])
		out[i] = 0.299*r + 0.587*g + 0.114*b
	}
	return out
}

// labPlanes returns the L*, a*, b* channels of img as three row-major float
// slices scaled to [0,1]. It uses the root package's colour conversion; a
// grayscale input is first expanded to RGB so its L* tracks intensity while a*
// and b* stay neutral.
func labPlanes(img *cv.Mat) (l, a, b []float64) {
	rgb := img
	if img.Channels == 1 {
		rgb = cv.CvtColor(img, cv.ColorGray2RGB)
	} else if img.Channels != 3 {
		// Fall back to a grayscale expansion for exotic channel counts.
		g := cv.NewMat(img.Rows, img.Cols, 1)
		gp := grayPlane(img)
		for i := range gp {
			g.Data[i] = uint8(clamp(gp[i], 0, 255))
		}
		rgb = cv.CvtColor(g, cv.ColorGray2RGB)
	}
	lab := cv.CvtColor(rgb, cv.ColorRGB2Lab)
	n := lab.Rows * lab.Cols
	l = make([]float64, n)
	a = make([]float64, n)
	b = make([]float64, n)
	for i := 0; i < n; i++ {
		base := i * 3
		l[i] = float64(lab.Data[base]) / 255
		a[i] = float64(lab.Data[base+1]) / 255
		b[i] = float64(lab.Data[base+2]) / 255
	}
	return l, a, b
}

// computeChannels builds the NumChannels feature planes for img: L*, a*, b*
// (scaled to [0,1]), gradient magnitude (scaled to [0,1]), and NumOrientBins
// oriented-gradient histogram planes holding the gradient magnitude routed to
// the nearest unsigned-orientation bin. Gradients use a centred [-1,0,1]
// difference with border replication.
func computeChannels(img *cv.Mat) [][]float64 {
	rows, cols := img.Rows, img.Cols
	gray := grayPlane(img)
	l, a, b := labPlanes(img)

	planes := make([][]float64, NumChannels)
	for c := range planes {
		planes[c] = make([]float64, rows*cols)
	}
	copy(planes[0], l)
	copy(planes[1], a)
	copy(planes[2], b)

	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return gray[y*cols+x]
	}

	// Normalisation constant so a full-scale gradient maps near 1.
	const gradNorm = 1.0 / 360.624 // ~ sqrt(2)*255
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gx := at(y, x+1) - at(y, x-1)
			gy := at(y+1, x) - at(y-1, x)
			mag := math.Hypot(gx, gy)
			idx := y*cols + x
			planes[3][idx] = mag * gradNorm
			// Unsigned orientation in [0, pi).
			ang := math.Atan2(gy, gx)
			if ang < 0 {
				ang += math.Pi
			}
			bin := int(ang / math.Pi * float64(NumOrientBins))
			if bin >= NumOrientBins {
				bin = NumOrientBins - 1
			}
			planes[4+bin][idx] = mag * gradNorm
		}
	}
	return planes
}

// newIntegralChannels computes the feature channels of img and folds each into
// a padded integral image.
func newIntegralChannels(img *cv.Mat) *integralChannels {
	rows, cols := img.Rows, img.Cols
	planes := computeChannels(img)
	ic := &integralChannels{
		rows:   rows,
		cols:   cols,
		stride: cols + 1,
		integ:  make([][]float64, NumChannels),
	}
	for c := 0; c < NumChannels; c++ {
		integ := make([]float64, (rows+1)*(cols+1))
		plane := planes[c]
		for y := 0; y < rows; y++ {
			var rowSum float64
			off := (y + 1) * ic.stride
			prev := y * ic.stride
			for x := 0; x < cols; x++ {
				rowSum += plane[y*cols+x]
				integ[off+x+1] = integ[prev+x+1] + rowSum
			}
		}
		ic.integ[c] = integ
	}
	return ic
}

// rectSum returns the sum of channel ch over the rectangle whose top-left
// corner is (x, y) and whose size is w x h. The rectangle is clamped to the
// image; a rectangle that is entirely outside yields 0.
func (ic *integralChannels) rectSum(ch, x, y, w, h int) float64 {
	x0, y0 := x, y
	x1, y1 := x+w, y+h
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > ic.cols {
		x1 = ic.cols
	}
	if y1 > ic.rows {
		y1 = ic.rows
	}
	if x1 <= x0 || y1 <= y0 {
		return 0
	}
	integ := ic.integ[ch]
	s := ic.stride
	return integ[y1*s+x1] - integ[y0*s+x1] - integ[y1*s+x0] + integ[y0*s+x0]
}

// rectMean returns the mean value of channel ch over the rectangle (x, y, w, h),
// i.e. rectSum divided by the rectangle area.
func (ic *integralChannels) rectMean(ch, x, y, w, h int) float64 {
	if w <= 0 || h <= 0 {
		return 0
	}
	return ic.rectSum(ch, x, y, w, h) / float64(w*h)
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

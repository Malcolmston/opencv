package segment2

import (
	cv "github.com/malcolmston/opencv"
)

// MeanShiftParams configures the joint spatial-range mean-shift routines.
type MeanShiftParams struct {
	// SpatialRadius is the half-width of the spatial window in pixels.
	SpatialRadius int
	// ColorRadius is the range-domain (colour) bandwidth.
	ColorRadius float64
	// MaxIter caps the mean-shift iterations per pixel.
	MaxIter int
	// Epsilon stops the iteration for a pixel once the mode moves less than
	// this distance (spatial plus colour) in a step.
	Epsilon float64
}

// DefaultMeanShiftParams returns reasonable defaults (SpatialRadius 7,
// ColorRadius 20, MaxIter 5, Epsilon 1).
func DefaultMeanShiftParams() MeanShiftParams {
	return MeanShiftParams{SpatialRadius: 7, ColorRadius: 20, MaxIter: 5, Epsilon: 1}
}

// segment2meanShiftModes runs mean shift for every pixel and returns, for each
// pixel, its converged spatial position and colour in the joint domain.
func segment2meanShiftModes(img *cv.Mat, p MeanShiftParams) (modeX, modeY []float64, modeC [][]float64) {
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	n := rows * cols
	sr := p.SpatialRadius
	if sr < 1 {
		sr = 1
	}
	cr2 := p.ColorRadius * p.ColorRadius
	maxIter := p.MaxIter
	if maxIter < 1 {
		maxIter = 1
	}

	modeX = make([]float64, n)
	modeY = make([]float64, n)
	modeC = make([][]float64, n)

	col := make([]float64, ch)
	for i := 0; i < n; i++ {
		px := float64(i % cols)
		py := float64(i / cols)
		segment2colorInto(img, i%cols, i/cols, col)
		pc := append([]float64(nil), col...)

		for it := 0; it < maxIter; it++ {
			var sx, sy float64
			sc := make([]float64, ch)
			var wsum float64
			cx := int(px + 0.5)
			cy := int(py + 0.5)
			for dy := -sr; dy <= sr; dy++ {
				yy := cy + dy
				if yy < 0 || yy >= rows {
					continue
				}
				for dx := -sr; dx <= sr; dx++ {
					xx := cx + dx
					if xx < 0 || xx >= cols {
						continue
					}
					b := (yy*cols + xx) * ch
					var cd2 float64
					for c := 0; c < ch; c++ {
						d := float64(img.Data[b+c]) - pc[c]
						cd2 += d * d
					}
					if cd2 > cr2 {
						continue
					}
					sx += float64(xx)
					sy += float64(yy)
					for c := 0; c < ch; c++ {
						sc[c] += float64(img.Data[b+c])
					}
					wsum++
				}
			}
			if wsum == 0 {
				break
			}
			nx := sx / wsum
			ny := sy / wsum
			var move float64
			dxm := nx - px
			dym := ny - py
			move += dxm*dxm + dym*dym
			for c := 0; c < ch; c++ {
				nc := sc[c] / wsum
				d := nc - pc[c]
				move += d * d
				pc[c] = nc
			}
			px, py = nx, ny
			if move <= p.Epsilon*p.Epsilon {
				break
			}
		}
		modeX[i] = px
		modeY[i] = py
		modeC[i] = pc
	}
	return
}

// MeanShiftFilterParams performs edge-preserving mean-shift smoothing of img in
// the joint spatial-range domain using the supplied parameters, returning a new
// [cv.Mat] in which each pixel is replaced by the colour of its converged mode.
// This mirrors cv2.pyrMeanShiftFiltering at a single pyramid level.
//
// It panics if img is empty.
func MeanShiftFilterParams(img *cv.Mat, p MeanShiftParams) *cv.Mat {
	segment2requireNonEmpty(img, "MeanShiftFilterParams")
	_, _, modeC := segment2meanShiftModes(img, p)
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	ch := img.Channels
	for i, c := range modeC {
		b := i * ch
		for j := 0; j < ch; j++ {
			out.Data[b+j] = segment2clampU8(c[j])
		}
	}
	return out
}

// MeanShiftFilter is [MeanShiftFilterParams] with the scalar parameters spelled
// out; it smooths img with the given spatial radius, colour radius and
// iteration cap and returns the filtered image.
//
// It panics if img is empty.
func MeanShiftFilter(img *cv.Mat, spatialRadius int, colorRadius float64, maxIter int) *cv.Mat {
	return MeanShiftFilterParams(img, MeanShiftParams{
		SpatialRadius: spatialRadius,
		ColorRadius:   colorRadius,
		MaxIter:       maxIter,
		Epsilon:       1,
	})
}

// MeanShiftSegment performs mean-shift segmentation: every pixel is driven to
// its mode in the joint spatial-range domain and pixels whose modes lie within
// ColorRadius in colour and are spatially connected are merged into one region.
// Regions smaller than minRegionSize pixels are absorbed into the adjacent
// region with the closest mean colour. The result is a [LabelMap].
//
// It panics if img is empty.
func MeanShiftSegment(img *cv.Mat, p MeanShiftParams, minRegionSize int) *LabelMap {
	segment2requireNonEmpty(img, "MeanShiftSegment")
	rows, cols := img.Rows, img.Cols
	_, _, modeC := segment2meanShiftModes(img, p)

	lm := NewLabelMap(rows, cols)
	for i := range lm.Labels {
		lm.Labels[i] = -1
	}
	cr2 := p.ColorRadius * p.ColorRadius
	label := 0
	stack := make([]int, 0, 64)
	for start := 0; start < len(lm.Labels); start++ {
		if lm.Labels[start] != -1 {
			continue
		}
		lm.Labels[start] = label
		seed := modeC[start]
		stack = stack[:0]
		stack = append(stack, start)
		for len(stack) > 0 {
			cur := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			cx := cur % cols
			cy := cur / cols
			for _, o := range segment2neighbors8 {
				nx, ny := cx+o.dx, cy+o.dy
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					continue
				}
				ni := ny*cols + nx
				if lm.Labels[ni] != -1 {
					continue
				}
				if segment2colorDist2(seed, modeC[ni]) <= cr2 {
					lm.Labels[ni] = label
					stack = append(stack, ni)
				}
			}
		}
		label++
	}
	lm.NumLabels = label

	if minRegionSize > 1 {
		segment2absorbSmall(lm, img, minRegionSize)
	}
	return lm
}

// segment2absorbSmall merges regions smaller than minSize into the spatially
// adjacent region with the closest mean colour, iterating until none remain or
// no merge is possible.
func segment2absorbSmall(lm *LabelMap, img *cv.Mat, minSize int) {
	for {
		sizes := lm.RegionSizes()
		means := lm.MeanColors(img)
		merged := false
		for start := 0; start < len(lm.Labels); start++ {
			l := lm.Labels[start]
			if l < 0 || sizes[l] >= minSize || sizes[l] == 0 {
				continue
			}
			// Find the adjacent region with closest mean colour.
			bestL, bestD := -1, 0.0
			for i, ll := range lm.Labels {
				if ll != l {
					continue
				}
				cx := i % lm.Cols
				cy := i / lm.Cols
				for _, o := range segment2neighbors4 {
					nx, ny := cx+o.dx, cy+o.dy
					if nx < 0 || nx >= lm.Cols || ny < 0 || ny >= lm.Rows {
						continue
					}
					nl := lm.Labels[ny*lm.Cols+nx]
					if nl == l || nl < 0 {
						continue
					}
					d := segment2colorDist2(means[l], means[nl])
					if bestL == -1 || d < bestD {
						bestL, bestD = nl, d
					}
				}
			}
			if bestL == -1 {
				continue
			}
			for i, ll := range lm.Labels {
				if ll == l {
					lm.Labels[i] = bestL
				}
			}
			merged = true
			break
		}
		if !merged {
			break
		}
	}
	lm.Compact()
}

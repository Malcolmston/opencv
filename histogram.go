package cv

import "math"

// CalcHist computes a 256-bin intensity histogram of the given channel of src.
// The returned slice has length 256 where entry i counts samples equal to i. It
// panics if channel is out of range.
func CalcHist(src *Mat, channel int) []int {
	if channel < 0 || channel >= src.Channels {
		panic("cv: CalcHist channel out of range")
	}
	hist := make([]int, 256)
	for p := 0; p < src.Total(); p++ {
		hist[src.Data[p*src.Channels+channel]]++
	}
	return hist
}

// EqualizeHist performs global histogram equalisation on a single-channel image
// to spread its intensities across the full range and improve contrast. It
// panics if src is not single-channel.
func EqualizeHist(src *Mat) *Mat {
	requireChannels(src, 1, "EqualizeHist")
	hist := CalcHist(src, 0)
	total := src.Total()

	// Cumulative distribution function.
	cdf := make([]int, 256)
	acc := 0
	cdfMin := 0
	for i := 0; i < 256; i++ {
		acc += hist[i]
		cdf[i] = acc
		if cdfMin == 0 && acc > 0 {
			cdfMin = acc
		}
	}

	// Build the lookup table mapping old intensities to equalised ones.
	lut := make([]uint8, 256)
	denom := total - cdfMin
	if denom <= 0 {
		// Degenerate (single-value) image: identity mapping.
		for i := range lut {
			lut[i] = uint8(i)
		}
	} else {
		for i := 0; i < 256; i++ {
			v := float64(cdf[i]-cdfMin) / float64(denom) * 255
			lut[i] = clampToUint8(v + 0.5)
		}
	}

	dst := NewMat(src.Rows, src.Cols, 1)
	for i, s := range src.Data {
		dst.Data[i] = lut[s]
	}
	return dst
}

// CalcBackProject projects a 256-bin histogram back onto an image: each output
// pixel is set to the histogram value of the corresponding channel intensity in
// src, rescaled so the largest bin maps to 255. The result is a single-channel
// "probability" map highlighting regions whose colour matches the histogram —
// the core of histogram-based tracking. It panics if channel is out of range or
// hist is not 256 bins.
func CalcBackProject(src *Mat, channel int, hist []int) *Mat {
	if channel < 0 || channel >= src.Channels {
		panic("cv: CalcBackProject channel out of range")
	}
	if len(hist) != 256 {
		panic("cv: CalcBackProject requires a 256-bin histogram")
	}
	maxBin := 0
	for _, v := range hist {
		if v > maxBin {
			maxBin = v
		}
	}
	lut := make([]uint8, 256)
	if maxBin > 0 {
		for i, v := range hist {
			lut[i] = clampToUint8(float64(v)/float64(maxBin)*255 + 0.5)
		}
	}
	dst := NewMat(src.Rows, src.Cols, 1)
	for p := 0; p < src.Total(); p++ {
		dst.Data[p] = lut[src.Data[p*src.Channels+channel]]
	}
	return dst
}

// HistCompMethod selects the similarity measure used by [CompareHist].
type HistCompMethod int

const (
	// HistCmpCorrel is the Pearson correlation; higher (up to 1) means more
	// similar.
	HistCmpCorrel HistCompMethod = iota
	// HistCmpChiSqr is the chi-square distance; lower (0) means more similar.
	HistCmpChiSqr
	// HistCmpIntersect is histogram intersection; higher means more similar.
	HistCmpIntersect
	// HistCmpBhattacharyya is the Bhattacharyya distance; lower (0) means more
	// similar.
	HistCmpBhattacharyya
)

// CompareHist measures the similarity of two equal-length histograms under the
// chosen method, matching OpenCV's compareHist. It panics if the histograms
// differ in length.
func CompareHist(h1, h2 []int, method HistCompMethod) float64 {
	if len(h1) != len(h2) {
		panic("cv: CompareHist histograms must have the same length")
	}
	n := len(h1)
	switch method {
	case HistCmpCorrel:
		var mean1, mean2 float64
		for i := 0; i < n; i++ {
			mean1 += float64(h1[i])
			mean2 += float64(h2[i])
		}
		mean1 /= float64(n)
		mean2 /= float64(n)
		var num, d1, d2 float64
		for i := 0; i < n; i++ {
			a := float64(h1[i]) - mean1
			b := float64(h2[i]) - mean2
			num += a * b
			d1 += a * a
			d2 += b * b
		}
		den := math.Sqrt(d1 * d2)
		if den == 0 {
			return 1
		}
		return num / den
	case HistCmpChiSqr:
		var sum float64
		for i := 0; i < n; i++ {
			a := float64(h1[i])
			b := float64(h2[i])
			if a == 0 {
				continue
			}
			sum += (a - b) * (a - b) / a
		}
		return sum
	case HistCmpIntersect:
		var sum float64
		for i := 0; i < n; i++ {
			if h1[i] < h2[i] {
				sum += float64(h1[i])
			} else {
				sum += float64(h2[i])
			}
		}
		return sum
	case HistCmpBhattacharyya:
		var s1, s2, sq float64
		for i := 0; i < n; i++ {
			s1 += float64(h1[i])
			s2 += float64(h2[i])
			sq += math.Sqrt(float64(h1[i]) * float64(h2[i]))
		}
		if s1 == 0 || s2 == 0 {
			return 1
		}
		val := 1 - sq/math.Sqrt(s1*s2)
		if val < 0 {
			val = 0
		}
		return math.Sqrt(val)
	default:
		panic("cv: CompareHist unknown method")
	}
}

// CLAHE applies Contrast-Limited Adaptive Histogram Equalisation to a
// single-channel image. The image is divided into a tileGridSize×tileGridSize
// grid; a clipped, equalised mapping is built per tile (bins are capped at
// clipLimit times the average bin count and the excess is redistributed) and the
// mappings are bilinearly interpolated between tile centres to avoid block
// artefacts. clipLimit <= 0 disables clipping (plain adaptive equalisation). It
// panics if src is not single-channel or tileGridSize < 1.
func CLAHE(src *Mat, clipLimit float64, tileGridSize int) *Mat {
	requireChannels(src, 1, "CLAHE")
	if tileGridSize < 1 {
		panic("cv: CLAHE requires tileGridSize >= 1")
	}
	rows, cols := src.Rows, src.Cols
	tiles := tileGridSize

	// Per-tile mapping LUTs.
	luts := make([][]uint8, tiles*tiles)
	tileIdxX := func(x int) int {
		t := x * tiles / cols
		if t >= tiles {
			t = tiles - 1
		}
		return t
	}
	tileIdxY := func(y int) int {
		t := y * tiles / rows
		if t >= tiles {
			t = tiles - 1
		}
		return t
	}

	// Histograms and pixel counts per tile.
	hists := make([][]int, tiles*tiles)
	counts := make([]int, tiles*tiles)
	for i := range hists {
		hists[i] = make([]int, 256)
	}
	for y := 0; y < rows; y++ {
		ty := tileIdxY(y)
		for x := 0; x < cols; x++ {
			tx := tileIdxX(x)
			t := ty*tiles + tx
			hists[t][src.Data[y*cols+x]]++
			counts[t]++
		}
	}

	for t := 0; t < tiles*tiles; t++ {
		hist := hists[t]
		total := counts[t]
		if total == 0 {
			lut := make([]uint8, 256)
			for i := range lut {
				lut[i] = uint8(i)
			}
			luts[t] = lut
			continue
		}
		if clipLimit > 0 {
			limit := int(clipLimit * float64(total) / 256)
			if limit < 1 {
				limit = 1
			}
			excess := 0
			for i := 0; i < 256; i++ {
				if hist[i] > limit {
					excess += hist[i] - limit
					hist[i] = limit
				}
			}
			// Redistribute the clipped excess uniformly.
			inc := excess / 256
			rem := excess % 256
			for i := 0; i < 256; i++ {
				hist[i] += inc
			}
			for i := 0; i < rem; i++ {
				hist[i]++
			}
		}
		// Build the equalisation LUT from the (clipped) CDF.
		lut := make([]uint8, 256)
		acc := 0
		for i := 0; i < 256; i++ {
			acc += hist[i]
			lut[i] = clampToUint8(float64(acc)/float64(total)*255 + 0.5)
		}
		luts[t] = lut
	}

	dst := NewMat(rows, cols, 1)
	tileW := float64(cols) / float64(tiles)
	tileH := float64(rows) / float64(tiles)
	clampTile := func(t int) int {
		if t < 0 {
			return 0
		}
		if t >= tiles {
			return tiles - 1
		}
		return t
	}
	for y := 0; y < rows; y++ {
		fy := (float64(y)+0.5)/tileH - 0.5
		ty0 := int(math.Floor(fy))
		wy := fy - float64(ty0)
		ty0c := clampTile(ty0)
		ty1c := clampTile(ty0 + 1)
		for x := 0; x < cols; x++ {
			fx := (float64(x)+0.5)/tileW - 0.5
			tx0 := int(math.Floor(fx))
			wx := fx - float64(tx0)
			tx0c := clampTile(tx0)
			tx1c := clampTile(tx0 + 1)
			v := src.Data[y*cols+x]
			v00 := float64(luts[ty0c*tiles+tx0c][v])
			v01 := float64(luts[ty0c*tiles+tx1c][v])
			v10 := float64(luts[ty1c*tiles+tx0c][v])
			v11 := float64(luts[ty1c*tiles+tx1c][v])
			top := v00*(1-wx) + v01*wx
			bot := v10*(1-wx) + v11*wx
			dst.Data[y*cols+x] = clampToUint8(top*(1-wy) + bot*wy + 0.5)
		}
	}
	return dst
}

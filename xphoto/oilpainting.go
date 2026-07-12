package xphoto

import (
	cv "github.com/malcolmston/opencv"
)

// Oilpainting applies an oil-painting stylization, porting
// cv::xphoto::oilPainting. For every pixel it examines the square neighbourhood
// of radius size, buckets the neighbours by quantised intensity
// (intensity/dynRatio), finds the bucket that occurs most often (the local
// intensity mode) and outputs the mean colour of the neighbours that fall in
// that bucket. Collapsing each neighbourhood onto its dominant intensity flat-
// tens texture into painterly patches while preserving edges and overall
// structure, and reduces the number of distinct colours in the image.
//
// size is the neighbourhood radius (values <= 0 default to 1); dynRatio is the
// intensity quantisation step, i.e. the number of grey levels per bucket
// (values <= 0 default to 1). Borders are replicated. src may be single- or
// three-channel; the mean colour is accumulated over every channel.
func Oilpainting(src *cv.Mat, size, dynRatio int) *cv.Mat {
	requireNonEmpty(src, "Oilpainting")
	if size <= 0 {
		size = 1
	}
	if dynRatio <= 0 {
		dynRatio = 1
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	dst := cv.NewMat(rows, cols, ch)

	// Number of intensity buckets after quantisation.
	nBuckets := 256/dynRatio + 1
	counts := make([]int, nBuckets)
	// Per-bucket channel-sum accumulators, laid out bucket-major.
	sums := make([]int, nBuckets*ch)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for i := range counts {
				counts[i] = 0
			}
			for i := range sums {
				sums[i] = 0
			}
			for dy := -size; dy <= size; dy++ {
				for dx := -size; dx <= size; dx++ {
					iy := y + dy
					ix := x + dx
					intensity := int(grayValueRep(src, iy, ix))
					bucket := intensity / dynRatio
					counts[bucket]++
					base := bucket * ch
					for c := 0; c < ch; c++ {
						sums[base+c] += int(atRep(src, iy, ix, c))
					}
				}
			}
			// Find the most frequent bucket (lowest index wins ties, keeping
			// the result deterministic).
			best := 0
			for b := 1; b < nBuckets; b++ {
				if counts[b] > counts[best] {
					best = b
				}
			}
			base := best * ch
			n := counts[best]
			for c := 0; c < ch; c++ {
				dst.Set(y, x, c, uint8((sums[base+c]+n/2)/n))
			}
		}
	}
	return dst
}

// grayValueRep returns the intensity at (y,x) with replicated borders: the
// sample for single-channel input, the BT.601 luma otherwise.
func grayValueRep(m *cv.Mat, y, x int) float64 {
	if m.Channels == 1 {
		return float64(atRep(m, y, x, 0))
	}
	return luma(float64(atRep(m, y, x, 0)), float64(atRep(m, y, x, 1)), float64(atRep(m, y, x, 2)))
}

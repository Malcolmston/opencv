package intensity

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// AutoscaleContrast performs a per-channel min–max contrast stretch: for each
// channel it finds the smallest and largest sample value present and linearly
// maps that observed range onto the full [0,255] range,
//
//	s = 255 · (r − min) / (max − min),
//
// so the darkest sample of a channel becomes 0 and the brightest becomes 255. A
// channel whose samples are all equal (max == min) is copied unchanged, since it
// carries no contrast to stretch. A new Mat is returned.
func AutoscaleContrast(img *cv.Mat) *cv.Mat {
	requireImage(img, "AutoscaleContrast")
	ch := img.Channels
	minv := make([]uint8, ch)
	maxv := make([]uint8, ch)
	for c := 0; c < ch; c++ {
		minv[c] = 255
		maxv[c] = 0
	}
	for p := 0; p < img.Total(); p++ {
		base := p * ch
		for c := 0; c < ch; c++ {
			v := img.Data[base+c]
			if v < minv[c] {
				minv[c] = v
			}
			if v > maxv[c] {
				maxv[c] = v
			}
		}
	}
	dst := cv.NewMat(img.Rows, img.Cols, ch)
	for c := 0; c < ch; c++ {
		lo := float64(minv[c])
		span := float64(maxv[c]) - lo
		for p := 0; p < img.Total(); p++ {
			idx := p*ch + c
			if span <= 0 {
				dst.Data[idx] = img.Data[idx]
				continue
			}
			dst.Data[idx] = clampToUint8((float64(img.Data[idx])-lo)/span*255 + 0.5)
		}
	}
	return dst
}

// HistogramMatching (histogram specification) remaps img so that the cumulative
// distribution of each channel approximates that of the corresponding channel of
// reference, and returns a new Mat with the shape of img. reference supplies only
// the target intensity distribution; it may be any size but must have the same
// channel count as img — it panics otherwise, or if either image is empty.
//
// For each channel the algorithm builds the normalised cumulative histograms of
// img and reference and, for every source level i, selects the smallest target
// level j whose reference CDF is at least the source CDF at i. Applying this
// mapping drives the output histogram toward the reference. Passing an equalised
// (flat) reference reduces to histogram equalisation.
func HistogramMatching(img, reference *cv.Mat) *cv.Mat {
	requireImage(img, "HistogramMatching")
	requireImage(reference, "HistogramMatching reference")
	if img.Channels != reference.Channels {
		panic(fmt.Sprintf("intensity: HistogramMatching channel mismatch %d vs %d",
			img.Channels, reference.Channels))
	}
	ch := img.Channels
	dst := cv.NewMat(img.Rows, img.Cols, ch)
	srcTotal := float64(img.Total())
	refTotal := float64(reference.Total())
	for c := 0; c < ch; c++ {
		srcHist := cv.CalcHist(img, c)
		refHist := cv.CalcHist(reference, c)

		// Normalised cumulative distributions.
		srcCDF := make([]float64, 256)
		refCDF := make([]float64, 256)
		var sAcc, rAcc int
		for i := 0; i < 256; i++ {
			sAcc += srcHist[i]
			rAcc += refHist[i]
			srcCDF[i] = float64(sAcc) / srcTotal
			refCDF[i] = float64(rAcc) / refTotal
		}

		// For each source level find the smallest reference level whose CDF is
		// at least the source CDF. Both CDFs are non-decreasing, so a single
		// forward scan of j suffices.
		lut := make([]uint8, 256)
		j := 0
		for i := 0; i < 256; i++ {
			for j < 255 && refCDF[j] < srcCDF[i] {
				j++
			}
			lut[i] = uint8(j)
		}

		for p := 0; p < img.Total(); p++ {
			idx := p*ch + c
			dst.Data[idx] = lut[img.Data[idx]]
		}
	}
	return dst
}

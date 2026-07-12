package cv

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

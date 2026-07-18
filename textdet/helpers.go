package textdet

// textdetSobel computes the horizontal and vertical Sobel derivatives of a
// single-channel image with 3x3 kernels and replicated borders, returning
// unclamped float results (positive gx points towards increasing intensity to
// the right, positive gy towards increasing intensity downward).
func textdetSobel(gray []uint8, rows, cols int) (gx, gy []float64) {
	gx = make([]float64, rows*cols)
	gy = make([]float64, rows*cols)
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
		return float64(gray[y*cols+x])
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			tl, tc, tr := at(y-1, x-1), at(y-1, x), at(y-1, x+1)
			ml, _, mr := at(y, x-1), at(y, x), at(y, x+1)
			bl, bc, br := at(y+1, x-1), at(y+1, x), at(y+1, x+1)
			gx[y*cols+x] = (tr + 2*mr + br) - (tl + 2*ml + bl)
			gy[y*cols+x] = (bl + 2*bc + br) - (tl + 2*tc + tr)
		}
	}
	return gx, gy
}

// textdetIntegral builds a summed-area table with one extra row and column of
// zeros, so integ has (rows+1)*(cols+1) entries. integ[(y+1)*(cols+1)+(x+1)] is
// the sum of src over the rectangle [0,x] x [0,y].
func textdetIntegral(src []float64, rows, cols int) []float64 {
	w := cols + 1
	integ := make([]float64, (rows+1)*w)
	for y := 0; y < rows; y++ {
		rowSum := 0.0
		for x := 0; x < cols; x++ {
			rowSum += src[y*cols+x]
			integ[(y+1)*w+(x+1)] = integ[y*w+(x+1)] + rowSum
		}
	}
	return integ
}

// textdetRectSum returns the sum of the source over the inclusive rectangle
// [x0,x1] x [y0,y1] using an integral table produced by textdetIntegral.
func textdetRectSum(integ []float64, cols, x0, y0, x1, y1 int) float64 {
	w := cols + 1
	a := integ[y0*w+x0]
	b := integ[y0*w+(x1+1)]
	c := integ[(y1+1)*w+x0]
	d := integ[(y1+1)*w+(x1+1)]
	return d - b - c + a
}

// textdetOtsu computes Otsu's optimal threshold from a 256-bin histogram and
// the total pixel count, returning a grey level in [0,255].
func textdetOtsu(hist [256]int, total int) int {
	if total == 0 {
		return 0
	}
	var sumAll float64
	for t := 0; t < 256; t++ {
		sumAll += float64(t) * float64(hist[t])
	}
	var wB, sumB float64
	bestVar := -1.0
	best := 0
	for t := 0; t < 256; t++ {
		wB += float64(hist[t])
		if wB == 0 {
			continue
		}
		wF := float64(total) - wB
		if wF == 0 {
			break
		}
		sumB += float64(t) * float64(hist[t])
		mB := sumB / wB
		mF := (sumAll - sumB) / wF
		between := wB * wF * (mB - mF) * (mB - mF)
		if between > bestVar {
			bestVar = between
			best = t
		}
	}
	return best
}

// textdetMedianFloat returns the median of xs, mutating a copy. It returns 0 for
// an empty slice.
func textdetMedianFloat(xs []float64) float64 {
	n := len(xs)
	if n == 0 {
		return 0
	}
	cp := make([]float64, n)
	copy(cp, xs)
	// Simple insertion of a full sort is fine for the small slices used here.
	for i := 1; i < n; i++ {
		v := cp[i]
		j := i - 1
		for j >= 0 && cp[j] > v {
			cp[j+1] = cp[j]
			j--
		}
		cp[j+1] = v
	}
	if n%2 == 1 {
		return cp[n/2]
	}
	return (cp[n/2-1] + cp[n/2]) / 2
}

package cv

import "math"

// BayerPattern names the 2x2 colour-filter arrangement of a raw Bayer mosaic
// for [Demosaic]. The two letters give the colours of the top-left and its
// horizontal neighbour on the first row.
type BayerPattern int

const (
	// BayerRG has R,G on the first row and G,B on the second.
	BayerRG BayerPattern = iota
	// BayerGR has G,R on the first row and B,G on the second.
	BayerGR
	// BayerBG has B,G on the first row and G,R on the second.
	BayerBG
	// BayerGB has G,B on the first row and R,G on the second.
	BayerGB
)

// bayerColorAt returns the colour channel index (0=R, 1=G, 2=B) sampled at
// pixel (y, x) for the given Bayer pattern.
func bayerColorAt(p BayerPattern, y, x int) int {
	ey, ex := y&1, x&1
	switch p {
	case BayerRG:
		switch {
		case ey == 0 && ex == 0:
			return 0
		case ey == 1 && ex == 1:
			return 2
		default:
			return 1
		}
	case BayerGR:
		switch {
		case ey == 0 && ex == 1:
			return 0
		case ey == 1 && ex == 0:
			return 2
		default:
			return 1
		}
	case BayerBG:
		switch {
		case ey == 0 && ex == 0:
			return 2
		case ey == 1 && ex == 1:
			return 0
		default:
			return 1
		}
	default: // BayerGB
		switch {
		case ey == 0 && ex == 1:
			return 2
		case ey == 1 && ex == 0:
			return 0
		default:
			return 1
		}
	}
}

// Demosaic reconstructs a 3-channel RGB image from a single-channel Bayer
// mosaic by averaging, for each missing colour, the same-colour samples in the
// 3x3 neighbourhood (replicated borders). This mirrors cv2.cvtColor with the
// COLOR_Bayer*2RGB codes. It requires a single-channel input.
func Demosaic(src *Mat, pattern BayerPattern) *Mat {
	if src.Channels != 1 {
		panic("cv: Demosaic requires a single-channel Bayer image")
	}
	dst := NewMat(src.Rows, src.Cols, 3)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			own := bayerColorAt(pattern, y, x)
			di := dst.index(y, x)
			for c := 0; c < 3; c++ {
				if c == own {
					dst.Data[di+c] = src.At(y, x, 0)
					continue
				}
				var sum, n float64
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						yy, xx := y+dy, x+dx
						if yy < 0 || yy >= src.Rows || xx < 0 || xx >= src.Cols {
							continue
						}
						if bayerColorAt(pattern, yy, xx) == c {
							sum += float64(src.At(yy, xx, 0))
							n++
						}
					}
				}
				if n > 0 {
					dst.Data[di+c] = clampToUint8(sum/n + 0.5)
				}
			}
		}
	}
	return dst
}

// GammaCorrect applies the power-law mapping out = 255*(in/255)^gamma to every
// sample of src via a lookup table and returns the result. Gamma below 1
// brightens, above 1 darkens. It panics on a non-positive gamma.
func GammaCorrect(src *Mat, gamma float64) *Mat {
	if gamma <= 0 {
		panic("cv: GammaCorrect requires a positive gamma")
	}
	var table [256]uint8
	for i := 0; i < 256; i++ {
		table[i] = clampToUint8(math.Pow(float64(i)/255, gamma)*255 + 0.5)
	}
	dst := NewMat(src.Rows, src.Cols, src.Channels)
	for i, v := range src.Data {
		dst.Data[i] = table[v]
	}
	return dst
}

// GaussianKernel2D returns the ksize x ksize separable Gaussian kernel formed
// by the outer product of two [GetGaussianKernel] vectors. It panics on a
// non-positive even ksize.
func GaussianKernel2D(ksize int, sigma float64) [][]float64 {
	k := GetGaussianKernel(ksize, sigma)
	out := make([][]float64, ksize)
	for y := 0; y < ksize; y++ {
		out[y] = make([]float64, ksize)
		for x := 0; x < ksize; x++ {
			out[y][x] = k[y] * k[x]
		}
	}
	return out
}

// RGBToHSVFull converts an 8-bit RGB image to HSV with the hue scaled to the
// full 0..255 range (rather than 0..179), mirroring cv2.cvtColor with
// COLOR_RGB2HSV_FULL. Saturation and value also occupy 0..255.
func RGBToHSVFull(src *Mat) *Mat {
	requireRGB(src, "RGBToHSVFull")
	dst := NewMat(src.Rows, src.Cols, 3)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		b := i * 3
		r := float64(src.Data[b]) / 255
		g := float64(src.Data[b+1]) / 255
		bl := float64(src.Data[b+2]) / 255
		mx := math.Max(r, math.Max(g, bl))
		mn := math.Min(r, math.Min(g, bl))
		d := mx - mn
		var h float64
		switch {
		case d == 0:
			h = 0
		case mx == r:
			h = math.Mod((g-bl)/d, 6)
		case mx == g:
			h = (bl-r)/d + 2
		default:
			h = (r-g)/d + 4
		}
		h *= 60
		if h < 0 {
			h += 360
		}
		var s float64
		if mx > 0 {
			s = d / mx
		}
		dst.Data[b] = clampToUint8(h/360*255 + 0.5)
		dst.Data[b+1] = clampToUint8(s*255 + 0.5)
		dst.Data[b+2] = clampToUint8(mx*255 + 0.5)
	}
	return dst
}

// HSVFullToRGB is the inverse of [RGBToHSVFull], converting a full-range HSV
// image back to 8-bit RGB. It mirrors cv2.cvtColor with COLOR_HSV2RGB_FULL.
func HSVFullToRGB(src *Mat) *Mat {
	requireRGB(src, "HSVFullToRGB")
	dst := NewMat(src.Rows, src.Cols, 3)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		b := i * 3
		h := float64(src.Data[b]) / 255 * 360
		s := float64(src.Data[b+1]) / 255
		v := float64(src.Data[b+2]) / 255
		c := v * s
		x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
		m := v - c
		var r, g, bl float64
		switch {
		case h < 60:
			r, g, bl = c, x, 0
		case h < 120:
			r, g, bl = x, c, 0
		case h < 180:
			r, g, bl = 0, c, x
		case h < 240:
			r, g, bl = 0, x, c
		case h < 300:
			r, g, bl = x, 0, c
		default:
			r, g, bl = c, 0, x
		}
		dst.Data[b] = clampToUint8((r+m)*255 + 0.5)
		dst.Data[b+1] = clampToUint8((g+m)*255 + 0.5)
		dst.Data[b+2] = clampToUint8((bl+m)*255 + 0.5)
	}
	return dst
}

// MinMaxLocMat returns the minimum and maximum sample values of a
// single-channel Mat and their (x, y) locations, mirroring cv2.minMaxLoc. It
// requires a single-channel image.
func MinMaxLocMat(src *Mat) (minVal, maxVal float64, minX, minY, maxX, maxY int) {
	if src.Channels != 1 {
		panic("cv: MinMaxLocMat requires a single-channel image")
	}
	minVal = math.Inf(1)
	maxVal = math.Inf(-1)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			v := float64(src.Data[y*src.Cols+x])
			if v < minVal {
				minVal, minX, minY = v, x, y
			}
			if v > maxVal {
				maxVal, maxX, maxY = v, x, y
			}
		}
	}
	return
}

// MSE returns the mean squared error between two equally shaped 8-bit
// matrices, averaged over all samples.
func MSE(a, b *Mat) float64 {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("cv: MSE requires matrices of equal shape")
	}
	var sse float64
	for i := range a.Data {
		d := float64(a.Data[i]) - float64(b.Data[i])
		sse += d * d
	}
	return sse / float64(len(a.Data))
}

// VarianceMat returns the population variance of all samples in src.
func VarianceMat(src *Mat) float64 {
	n := float64(len(src.Data))
	if n == 0 {
		return 0
	}
	var sum, sumSq float64
	for _, v := range src.Data {
		fv := float64(v)
		sum += fv
		sumSq += fv * fv
	}
	mean := sum / n
	variance := sumSq/n - mean*mean
	if variance < 0 {
		variance = 0
	}
	return variance
}

// StdDevMat returns the population standard deviation of all samples in src.
func StdDevMat(src *Mat) float64 { return math.Sqrt(VarianceMat(src)) }

// Entropy returns the Shannon entropy in bits of the sample distribution of a
// single-channel image, a value in [0, 8] for 8-bit data. It requires a
// single-channel image.
func Entropy(src *Mat) float64 {
	if src.Channels != 1 {
		panic("cv: Entropy requires a single-channel image")
	}
	var hist [256]float64
	for _, v := range src.Data {
		hist[v]++
	}
	n := float64(len(src.Data))
	var e float64
	for _, c := range hist {
		if c > 0 {
			p := c / n
			e -= p * math.Log2(p)
		}
	}
	return e
}

// Median returns the median sample value of src across all channels.
func Median(src *Mat) float64 {
	var hist [256]int
	for _, v := range src.Data {
		hist[v]++
	}
	target := (len(src.Data) + 1) / 2
	cum := 0
	for i := 0; i < 256; i++ {
		cum += hist[i]
		if cum >= target {
			return float64(i)
		}
	}
	return 0
}

// TriangleThreshold binarises a single-channel image using the triangle
// algorithm to pick the threshold automatically, mirroring cv2.threshold with
// THRESH_TRIANGLE. It returns the thresholded image and the chosen level.
func TriangleThreshold(src *Mat) (*Mat, float64) {
	if src.Channels != 1 {
		panic("cv: TriangleThreshold requires a single-channel image")
	}
	var hist [256]int
	for _, v := range src.Data {
		hist[v]++
	}
	// Find the peak and the two extremes of the histogram.
	peak, peakVal := 0, 0
	lo, hi := -1, -1
	for i := 0; i < 256; i++ {
		if hist[i] > peakVal {
			peakVal, peak = hist[i], i
		}
		if hist[i] > 0 {
			if lo < 0 {
				lo = i
			}
			hi = i
		}
	}
	// Use the longer tail from the peak.
	left := peak-lo >= hi-peak
	a, b := lo, peak
	if left {
		a, b = peak, hi
	}
	// Distance from each bin to the line joining (a,0) and (b, peakVal).
	var best int
	var bestDist float64
	dx := float64(b - a)
	dy := float64(hist[b] - hist[a])
	norm := math.Hypot(dx, dy)
	if norm == 0 {
		return Threshold(src, float64(peak), 255, ThreshBinary)
	}
	for i := a; i < b; i++ {
		d := math.Abs(dy*float64(i-a)-dx*float64(hist[i]-hist[a])) / norm
		if d > bestDist {
			bestDist, best = d, i
		}
	}
	out, _ := Threshold(src, float64(best), 255, ThreshBinary)
	return out, float64(best)
}

package cv

import "math"

// GetGaussianKernel returns a normalised 1D Gaussian kernel of length ksize,
// mirroring cv2.getGaussianKernel. When sigma is not positive it is derived
// from ksize with OpenCV's rule sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8. It
// panics if ksize is not a positive odd number.
func GetGaussianKernel(ksize int, sigma float64) []float64 {
	if ksize <= 0 || ksize%2 == 0 {
		panic("cv: GetGaussianKernel requires a positive odd ksize")
	}
	if sigma <= 0 {
		sigma = 0.3*((float64(ksize)-1)*0.5-1) + 0.8
	}
	half := (ksize - 1) / 2
	out := make([]float64, ksize)
	var sum float64
	for i := 0; i < ksize; i++ {
		x := float64(i - half)
		v := math.Exp(-(x * x) / (2 * sigma * sigma))
		out[i] = v
		sum += v
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// getSobelKernel1D builds a 1D kernel of length ksize for the given derivative
// order using binomial smoothing convolved with the finite-difference operator
// [-1, 1]. Order 0 gives a smoothing (binomial) kernel.
func getSobelKernel1D(order, ksize int) []float64 {
	// Binomial (Pascal) row of order (ksize-1-order).
	base := ksize - 1 - order
	kern := []float64{1}
	for i := 0; i < base; i++ {
		next := make([]float64, len(kern)+1)
		for j := 0; j < len(kern); j++ {
			next[j] += kern[j]
			next[j+1] += kern[j]
		}
		kern = next
	}
	// Convolve with [-1, 1] order times to differentiate.
	for i := 0; i < order; i++ {
		next := make([]float64, len(kern)+1)
		for j := 0; j < len(kern); j++ {
			next[j] += -kern[j]
			next[j+1] += kern[j]
		}
		kern = next
	}
	return kern
}

// GetDerivKernels returns the row (kx) and column (ky) kernels used to compute
// image derivatives of order dx and dy with an aperture of ksize, mirroring
// cv2.getDerivKernels. kx is the horizontal kernel and ky the vertical one; a
// Sobel filter convolves rows with kx and columns with ky. It panics on an
// even or non-positive ksize.
func GetDerivKernels(dx, dy, ksize int) (kx, ky []float64) {
	if ksize <= 0 || ksize%2 == 0 {
		panic("cv: GetDerivKernels requires a positive odd ksize")
	}
	return getSobelKernel1D(dx, ksize), getSobelKernel1D(dy, ksize)
}

// SqrBoxFilter returns, for each pixel, the sum (or mean when normalize is
// true) of the squared samples in a ksize x ksize neighbourhood with replicated
// borders, as a FloatMat. It mirrors cv2.sqrBoxFilter and requires a
// single-channel input and a positive odd ksize.
func SqrBoxFilter(src *Mat, ksize int, normalize bool) *FloatMat {
	if src.Channels != 1 {
		panic("cv: SqrBoxFilter requires a single-channel image")
	}
	if ksize <= 0 || ksize%2 == 0 {
		panic("cv: SqrBoxFilter requires a positive odd ksize")
	}
	half := ksize / 2
	out := NewFloatMat(src.Rows, src.Cols)
	area := float64(ksize * ksize)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			var s float64
			for dy := -half; dy <= half; dy++ {
				for dx := -half; dx <= half; dx++ {
					v := float64(src.atReplicate(y+dy, x+dx, 0))
					s += v * v
				}
			}
			if normalize {
				s /= area
			}
			out.Data[y*out.Cols+x] = s
		}
	}
	return out
}

// StackBlur blurs src with a fast separable triangular window of the given odd
// radius-defining size, the same effective kernel as Mario Klingemann's stack
// blur. Larger ksize yields stronger smoothing. It mirrors cv2.stackBlur and
// panics on a non-positive odd ksize.
func StackBlur(src *Mat, ksize int) *Mat {
	if ksize <= 0 || ksize%2 == 0 {
		panic("cv: StackBlur requires a positive odd ksize")
	}
	if ksize == 1 {
		return src.Clone()
	}
	r := ksize / 2
	// Triangular weights 1..r+1..1, normalised.
	w := make([]float64, ksize)
	var sum float64
	for i := 0; i <= r; i++ {
		w[i] = float64(i + 1)
		w[ksize-1-i] = float64(i + 1)
		if i < r {
			sum += 2 * float64(i+1)
		} else {
			sum += float64(i + 1)
		}
	}
	for i := range w {
		w[i] /= sum
	}
	// Separable convolution with replicate borders.
	ch := src.Channels
	tmp := NewMat(src.Rows, src.Cols, ch)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < ch; c++ {
				var acc float64
				for k := -r; k <= r; k++ {
					acc += w[k+r] * float64(src.atReplicate(y, x+k, c))
				}
				tmp.Data[tmp.index(y, x)+c] = clampToUint8(acc + 0.5)
			}
		}
	}
	dst := NewMat(src.Rows, src.Cols, ch)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < ch; c++ {
				var acc float64
				for k := -r; k <= r; k++ {
					acc += w[k+r] * float64(tmp.atReplicate(y+k, x, c))
				}
				dst.Data[dst.index(y, x)+c] = clampToUint8(acc + 0.5)
			}
		}
	}
	return dst
}

// sampleBilinear returns channel c of src at fractional coordinates (fx, fy)
// using bilinear interpolation with replicated borders.
func sampleBilinear(src *Mat, fx, fy float64, c int) float64 {
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	ax := fx - float64(x0)
	ay := fy - float64(y0)
	v00 := float64(src.atReplicate(y0, x0, c))
	v01 := float64(src.atReplicate(y0, x0+1, c))
	v10 := float64(src.atReplicate(y0+1, x0, c))
	v11 := float64(src.atReplicate(y0+1, x0+1, c))
	top := v00*(1-ax) + v01*ax
	bot := v10*(1-ax) + v11*ax
	return top*(1-ay) + bot*ay
}

// GetRectSubPix extracts a width x height patch centred on the fractional point
// (centerX, centerY) using bilinear interpolation with replicated borders,
// mirroring cv2.getRectSubPix. It panics on non-positive dimensions.
func GetRectSubPix(src *Mat, width, height int, centerX, centerY float64) *Mat {
	if width <= 0 || height <= 0 {
		panic("cv: GetRectSubPix requires positive dimensions")
	}
	dst := NewMat(height, width, src.Channels)
	startX := centerX - float64(width-1)/2
	startY := centerY - float64(height-1)/2
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			fx := startX + float64(x)
			fy := startY + float64(y)
			di := dst.index(y, x)
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = clampToUint8(sampleBilinear(src, fx, fy, c) + 0.5)
			}
		}
	}
	return dst
}

// solve3x3 solves the linear system a*x = b for a 3x3 matrix a (row-major) by
// Gaussian elimination, returning the solution and whether a is non-singular.
func solve3x3(a [9]float64, b [3]float64) ([3]float64, bool) {
	m := [3][4]float64{
		{a[0], a[1], a[2], b[0]},
		{a[3], a[4], a[5], b[1]},
		{a[6], a[7], a[8], b[2]},
	}
	for col := 0; col < 3; col++ {
		p := col
		for r := col + 1; r < 3; r++ {
			if math.Abs(m[r][col]) > math.Abs(m[p][col]) {
				p = r
			}
		}
		if m[p][col] == 0 {
			return [3]float64{}, false
		}
		m[p], m[col] = m[col], m[p]
		pv := m[col][col]
		for c := col; c < 4; c++ {
			m[col][c] /= pv
		}
		for r := 0; r < 3; r++ {
			if r == col {
				continue
			}
			f := m[r][col]
			for c := col; c < 4; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	return [3]float64{m[0][3], m[1][3], m[2][3]}, true
}

// GetAffineTransform computes the 2x3 affine transform that maps the three
// source points to the three destination points, mirroring
// cv2.getAffineTransform. It panics when the source points are collinear.
func GetAffineTransform(src, dst [3]Point2f) AffineMatrix {
	a := [9]float64{
		src[0].X, src[0].Y, 1,
		src[1].X, src[1].Y, 1,
		src[2].X, src[2].Y, 1,
	}
	xr, ok1 := solve3x3(a, [3]float64{dst[0].X, dst[1].X, dst[2].X})
	yr, ok2 := solve3x3(a, [3]float64{dst[0].Y, dst[1].Y, dst[2].Y})
	if !ok1 || !ok2 {
		panic("cv: GetAffineTransform requires non-collinear source points")
	}
	return AffineMatrix{xr[0], xr[1], xr[2], yr[0], yr[1], yr[2]}
}

// InvertAffineTransform returns the inverse of a 2x3 affine transform, mirroring
// cv2.invertAffineTransform. It panics when the linear part is singular.
func InvertAffineTransform(m AffineMatrix) AffineMatrix {
	det := m[0]*m[4] - m[1]*m[3]
	if det == 0 {
		panic("cv: InvertAffineTransform on singular matrix")
	}
	id := 1 / det
	ia := m[4] * id
	ib := -m[1] * id
	ic := -m[3] * id
	idd := m[0] * id
	return AffineMatrix{
		ia, ib, -(ia*m[2] + ib*m[5]),
		ic, idd, -(ic*m[2] + idd*m[5]),
	}
}

// WarpPolarMode selects linear or semi-log radial mapping for [WarpPolar].
type WarpPolarMode int

const (
	// WarpPolarLinear maps radius linearly to the output rows.
	WarpPolarLinear WarpPolarMode = iota
	// WarpPolarLog maps radius logarithmically (log-polar).
	WarpPolarLog
)

// WarpPolar remaps src into polar coordinates about (centerX, centerY): the
// output has the given width (angle, 0..2π) and height (radius, 0..maxRadius),
// sampled bilinearly. It mirrors cv2.warpPolar. It panics on non-positive
// dimensions or maxRadius.
func WarpPolar(src *Mat, width, height int, centerX, centerY, maxRadius float64, mode WarpPolarMode) *Mat {
	if width <= 0 || height <= 0 {
		panic("cv: WarpPolar requires positive dimensions")
	}
	if maxRadius <= 0 {
		panic("cv: WarpPolar requires positive maxRadius")
	}
	dst := NewMat(height, width, src.Channels)
	var klog float64
	if mode == WarpPolarLog {
		klog = math.Log(maxRadius) / float64(height)
	}
	for y := 0; y < height; y++ {
		var r float64
		if mode == WarpPolarLog {
			r = math.Exp(float64(y) * klog)
		} else {
			r = float64(y) / float64(height) * maxRadius
		}
		for x := 0; x < width; x++ {
			angle := float64(x) / float64(width) * 2 * math.Pi
			sx := centerX + r*math.Cos(angle)
			sy := centerY + r*math.Sin(angle)
			di := dst.index(y, x)
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = clampToUint8(sampleBilinear(src, sx, sy, c) + 0.5)
			}
		}
	}
	return dst
}

// LinearPolar remaps src to linear polar coordinates, a convenience wrapper for
// [WarpPolar] with WarpPolarLinear, mirroring cv2.linearPolar.
func LinearPolar(src *Mat, width, height int, centerX, centerY, maxRadius float64) *Mat {
	return WarpPolar(src, width, height, centerX, centerY, maxRadius, WarpPolarLinear)
}

// LogPolar remaps src to log-polar coordinates, a convenience wrapper for
// [WarpPolar] with WarpPolarLog, mirroring cv2.logPolar.
func LogPolar(src *Mat, width, height int, centerX, centerY, maxRadius float64) *Mat {
	return WarpPolar(src, width, height, centerX, centerY, maxRadius, WarpPolarLog)
}

// CornerMinEigenVal returns, for each pixel, the smaller eigenvalue of the
// blockSize x blockSize structure tensor built from Sobel derivatives of the
// given aperture, mirroring cv2.cornerMinEigenVal. Larger values mark stronger
// corners. It requires a single-channel input and a Sobel aperture of 1 or 3.
func CornerMinEigenVal(src *Mat, blockSize, ksize int) *FloatMat {
	if src.Channels != 1 {
		panic("cv: CornerMinEigenVal requires a single-channel image")
	}
	if blockSize <= 0 {
		panic("cv: CornerMinEigenVal requires a positive blockSize")
	}
	gx := SobelFloat(src, 1, 0, ksize)[0]
	gy := SobelFloat(src, 0, 1, ksize)[0]
	rows, cols := src.Rows, src.Cols
	out := NewFloatMat(rows, cols)
	half := blockSize / 2
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var a, b, c float64
			for dy := -half; dy <= half; dy++ {
				yy := clampIndex(y+dy, rows)
				for dx := -half; dx <= half; dx++ {
					xx := clampIndex(x+dx, cols)
					ix := gx[yy*cols+xx]
					iy := gy[yy*cols+xx]
					a += ix * ix
					b += ix * iy
					c += iy * iy
				}
			}
			half1 := (a + c) / 2
			d := math.Sqrt(((a-c)/2)*((a-c)/2) + b*b)
			out.Data[y*cols+x] = half1 - d
		}
	}
	return out
}

// clampIndex clamps i into [0, n).
func clampIndex(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

// FloodFill fills the 4-connected region of a single-channel image around the
// seed (seedX, seedY) whose samples stay within loDiff below and upDiff above
// each already-filled neighbour, painting them with newVal. It returns the
// filled copy and the number of pixels changed, mirroring cv2.floodFill with
// the default (floating-range) comparison. It requires a single-channel image.
func FloodFill(src *Mat, seedX, seedY int, newVal float64, loDiff, upDiff float64) (*Mat, int) {
	if src.Channels != 1 {
		panic("cv: FloodFill requires a single-channel image")
	}
	if !src.inBounds(seedY, seedX) {
		panic("cv: FloodFill seed out of range")
	}
	dst := src.Clone()
	nv := clampToUint8(newVal + 0.5)
	rows, cols := src.Rows, src.Cols
	visited := make([]bool, rows*cols)
	type pt struct{ x, y int }
	queue := []pt{{seedX, seedY}}
	visited[seedY*cols+seedX] = true
	dst.Data[dst.index(seedY, seedX)] = nv
	count := 1
	for len(queue) > 0 {
		p := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		ref := float64(src.At(p.y, p.x, 0))
		neigh := [4]pt{{p.x - 1, p.y}, {p.x + 1, p.y}, {p.x, p.y - 1}, {p.x, p.y + 1}}
		for _, q := range neigh {
			if q.x < 0 || q.x >= cols || q.y < 0 || q.y >= rows {
				continue
			}
			idx := q.y*cols + q.x
			if visited[idx] {
				continue
			}
			v := float64(src.At(q.y, q.x, 0))
			if v >= ref-loDiff && v <= ref+upDiff {
				visited[idx] = true
				dst.Data[dst.index(q.y, q.x)] = nv
				count++
				queue = append(queue, q)
			}
		}
	}
	return dst, count
}

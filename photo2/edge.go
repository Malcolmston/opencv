package photo2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// photo2BilateralFloat applies a bilateral filter to a single-channel float
// plane. sigmaSpace sets the spatial Gaussian (radius = ceil(2*sigmaSpace)) and
// sigmaRange the intensity Gaussian. Borders are reflected.
func photo2BilateralFloat(f *cv.FloatMat, sigmaSpace, sigmaRange float64) *cv.FloatMat {
	rows, cols := f.Rows, f.Cols
	if sigmaSpace <= 0 {
		sigmaSpace = 1
	}
	if sigmaRange <= 0 {
		sigmaRange = 1
	}
	radius := int(math.Ceil(2 * sigmaSpace))
	if radius < 1 {
		radius = 1
	}
	// Precompute spatial weights.
	sw := make([]float64, (2*radius+1)*(2*radius+1))
	twoSpace2 := 2 * sigmaSpace * sigmaSpace
	idx := 0
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			sw[idx] = math.Exp(-float64(dy*dy+dx*dx) / twoSpace2)
			idx++
		}
	}
	twoRange2 := 2 * sigmaRange * sigmaRange
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			center := f.Data[y*cols+x]
			var accW, accV float64
			idx = 0
			for dy := -radius; dy <= radius; dy++ {
				yy := photo2Reflect(y+dy, rows)
				for dx := -radius; dx <= radius; dx++ {
					xx := photo2Reflect(x+dx, cols)
					val := f.Data[yy*cols+xx]
					diff := val - center
					w := sw[idx] * math.Exp(-(diff*diff)/twoRange2)
					accW += w
					accV += w * val
					idx++
				}
			}
			out.Data[y*cols+x] = accV / accW
		}
	}
	return out
}

// BilateralFilter applies an edge-preserving bilateral filter to an image. It
// smooths within regions of similar colour while keeping strong edges sharp.
// diameter is the pixel neighbourhood diameter (a non-positive value derives it
// from sigmaSpace); sigmaColor is the range standard deviation in 8-bit units
// and sigmaSpace the spatial standard deviation. Colour images are filtered with
// a joint range kernel over all channels.
func BilateralFilter(img *cv.Mat, diameter int, sigmaColor, sigmaSpace float64) *cv.Mat {
	photo2RequireImage(img, "BilateralFilter")
	if sigmaColor <= 0 {
		sigmaColor = 1
	}
	if sigmaSpace <= 0 {
		sigmaSpace = 1
	}
	radius := diameter / 2
	if diameter <= 0 {
		radius = int(math.Ceil(1.5 * sigmaSpace))
	}
	if radius < 1 {
		radius = 1
	}
	rows, cols, nch := img.Rows, img.Cols, img.Channels
	sw := make([]float64, (2*radius+1)*(2*radius+1))
	twoSpace2 := 2 * sigmaSpace * sigmaSpace
	i := 0
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			sw[i] = math.Exp(-float64(dy*dy+dx*dx) / twoSpace2)
			i++
		}
	}
	twoRange2 := 2 * sigmaColor * sigmaColor
	out := cv.NewMat(rows, cols, nch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			ci := (y*cols + x) * nch
			var accW float64
			acc := make([]float64, nch)
			idx := 0
			for dy := -radius; dy <= radius; dy++ {
				yy := photo2Reflect(y+dy, rows)
				for dx := -radius; dx <= radius; dx++ {
					xx := photo2Reflect(x+dx, cols)
					ni := (yy*cols + xx) * nch
					var dist2 float64
					for c := 0; c < nch; c++ {
						d := float64(img.Data[ni+c]) - float64(img.Data[ci+c])
						dist2 += d * d
					}
					w := sw[idx] * math.Exp(-dist2/twoRange2)
					accW += w
					for c := 0; c < nch; c++ {
						acc[c] += w * float64(img.Data[ni+c])
					}
					idx++
				}
			}
			for c := 0; c < nch; c++ {
				out.Data[ci+c] = photo2Clamp8(acc[c] / accW)
			}
		}
	}
	return out
}

// EdgeFilterMode selects the variant used by [DomainTransformFilter].
type EdgeFilterMode int

const (
	// RecursiveFilter selects the recursive (RF) domain-transform filter: fast,
	// with an infinite impulse response and no explicit window.
	RecursiveFilter EdgeFilterMode = iota
	// NormalizedConvolution selects the normalized-convolution (NC) variant: a
	// finite box filter warped by the domain transform.
	NormalizedConvolution
)

// DomainTransformFilter applies Gastal and Oliveira's (2011) domain-transform
// edge-preserving filter. It smooths the image while respecting edges, using a
// 1-D transform that stretches distances across high-gradient regions so the
// filter does not blur across them. Three separable iterations are applied in
// alternating directions. sigmaS is the spatial standard deviation and sigmaR
// the range standard deviation (in 8-bit units). mode chooses the RF or NC
// variant.
func DomainTransformFilter(img *cv.Mat, mode EdgeFilterMode, sigmaS, sigmaR float64) *cv.Mat {
	photo2RequireImage(img, "DomainTransformFilter")
	if sigmaS <= 0 {
		sigmaS = 1
	}
	if sigmaR <= 0 {
		sigmaR = 1
	}
	rows, cols, nch := img.Rows, img.Cols, img.Channels
	// Work in float [0,255] per channel.
	planes := make([][]float64, nch)
	for c := 0; c < nch; c++ {
		p := make([]float64, rows*cols)
		for i := range p {
			p[i] = float64(img.Data[i*nch+c])
		}
		planes[c] = p
	}
	ratio := sigmaS / sigmaR
	const iterations = 3
	for it := 0; it < iterations; it++ {
		// Per-iteration sigma follows the paper's geometric schedule.
		sigmaHi := sigmaS * math.Sqrt(3) * math.Pow(2, float64(iterations-it-1)) / math.Sqrt(math.Pow(4, float64(iterations))-1)
		// Horizontal derivatives.
		dHdx := make([]float64, rows*cols)
		for y := 0; y < rows; y++ {
			for x := 1; x < cols; x++ {
				var sum float64
				for c := 0; c < nch; c++ {
					sum += math.Abs(planes[c][y*cols+x] - planes[c][y*cols+x-1])
				}
				dHdx[y*cols+x] = 1 + ratio*sum
			}
		}
		for c := 0; c < nch; c++ {
			photo2DTRows(planes[c], dHdx, rows, cols, mode, sigmaHi)
		}
		// Vertical derivatives.
		dVdy := make([]float64, rows*cols)
		for y := 1; y < rows; y++ {
			for x := 0; x < cols; x++ {
				var sum float64
				for c := 0; c < nch; c++ {
					sum += math.Abs(planes[c][y*cols+x] - planes[c][(y-1)*cols+x])
				}
				dVdy[y*cols+x] = 1 + ratio*sum
			}
		}
		for c := 0; c < nch; c++ {
			photo2DTCols(planes[c], dVdy, rows, cols, mode, sigmaHi)
		}
	}
	out := cv.NewMat(rows, cols, nch)
	for c := 0; c < nch; c++ {
		for i := 0; i < rows*cols; i++ {
			out.Data[i*nch+c] = photo2Clamp8(planes[c][i])
		}
	}
	return out
}

// photo2DTRows filters each row of p in place given per-pixel domain-transform
// derivatives.
func photo2DTRows(p, deriv []float64, rows, cols int, mode EdgeFilterMode, sigma float64) {
	if mode == RecursiveFilter {
		a := math.Exp(-math.Sqrt2 / sigma)
		for y := 0; y < rows; y++ {
			base := y * cols
			for x := 1; x < cols; x++ {
				w := math.Pow(a, deriv[base+x])
				p[base+x] += w * (p[base+x-1] - p[base+x])
			}
			for x := cols - 2; x >= 0; x-- {
				w := math.Pow(a, deriv[base+x+1])
				p[base+x] += w * (p[base+x+1] - p[base+x])
			}
		}
		return
	}
	photo2DTNormRows(p, deriv, rows, cols, sigma)
}

// photo2DTCols filters each column of p in place.
func photo2DTCols(p, deriv []float64, rows, cols int, mode EdgeFilterMode, sigma float64) {
	if mode == RecursiveFilter {
		a := math.Exp(-math.Sqrt2 / sigma)
		for x := 0; x < cols; x++ {
			for y := 1; y < rows; y++ {
				w := math.Pow(a, deriv[y*cols+x])
				p[y*cols+x] += w * (p[(y-1)*cols+x] - p[y*cols+x])
			}
			for y := rows - 2; y >= 0; y-- {
				w := math.Pow(a, deriv[(y+1)*cols+x])
				p[y*cols+x] += w * (p[(y+1)*cols+x] - p[y*cols+x])
			}
		}
		return
	}
	photo2DTNormCols(p, deriv, rows, cols, sigma)
}

// photo2DTNormRows performs the normalized-convolution box filter along rows,
// using the cumulative domain-transform distance to bound the window.
func photo2DTNormRows(p, deriv []float64, rows, cols int, sigma float64) {
	radius := sigma * math.Sqrt(3)
	for y := 0; y < rows; y++ {
		base := y * cols
		ct := make([]float64, cols)
		for x := 1; x < cols; x++ {
			ct[x] = ct[x-1] + deriv[base+x]
		}
		res := make([]float64, cols)
		l := 0
		r := 0
		for x := 0; x < cols; x++ {
			for l < cols && ct[l] < ct[x]-radius {
				l++
			}
			if r < x {
				r = x
			}
			for r+1 < cols && ct[r+1] <= ct[x]+radius {
				r++
			}
			var sum float64
			for i := l; i <= r; i++ {
				sum += p[base+i]
			}
			res[x] = sum / float64(r-l+1)
		}
		copy(p[base:base+cols], res)
	}
}

// photo2DTNormCols performs the normalized-convolution box filter along columns.
func photo2DTNormCols(p, deriv []float64, rows, cols int, sigma float64) {
	radius := sigma * math.Sqrt(3)
	for x := 0; x < cols; x++ {
		ct := make([]float64, rows)
		for y := 1; y < rows; y++ {
			ct[y] = ct[y-1] + deriv[y*cols+x]
		}
		res := make([]float64, rows)
		l := 0
		r := 0
		for y := 0; y < rows; y++ {
			for l < rows && ct[l] < ct[y]-radius {
				l++
			}
			if r < y {
				r = y
			}
			for r+1 < rows && ct[r+1] <= ct[y]+radius {
				r++
			}
			var sum float64
			for i := l; i <= r; i++ {
				sum += p[i*cols+x]
			}
			res[y] = sum / float64(r-l+1)
		}
		for y := 0; y < rows; y++ {
			p[y*cols+x] = res[y]
		}
	}
}

// EdgePreservingFilter smooths an image while keeping salient edges crisp, using
// the recursive domain-transform filter. It is a convenience wrapper over
// [DomainTransformFilter] with [RecursiveFilter]. sigmaS controls the spatial
// extent (larger flattens more) and sigmaR the edge sensitivity (larger merges
// more colours).
func EdgePreservingFilter(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat {
	return DomainTransformFilter(img, RecursiveFilter, sigmaS, sigmaR)
}

// GuidedFilter applies He et al.'s (2010) guided filter: an edge-preserving
// smoothing of img whose edges are taken from guide. radius is the box-window
// radius and eps the regularisation (larger eps smooths more; it plays the role
// of range variance). img and guide must be single-channel and share
// dimensions. Passing the same matrix as img and guide gives self-guided
// edge-preserving smoothing.
func GuidedFilter(img, guide *cv.FloatMat, radius int, eps float64) *cv.FloatMat {
	photo2RequireFloat(img, "GuidedFilter")
	photo2RequireFloat(guide, "GuidedFilter")
	if img.Rows != guide.Rows || img.Cols != guide.Cols {
		panic("photo2: GuidedFilter img and guide must share dimensions")
	}
	if radius < 1 {
		radius = 1
	}
	n := img.Rows * img.Cols
	I := guide.Data
	P := img.Data
	meanI := photo2BoxBlurFloat(guide, radius)
	meanP := photo2BoxBlurFloat(img, radius)
	// I*P and I*I.
	ip := cv.NewFloatMat(img.Rows, img.Cols)
	ii := cv.NewFloatMat(img.Rows, img.Cols)
	for i := 0; i < n; i++ {
		ip.Data[i] = I[i] * P[i]
		ii.Data[i] = I[i] * I[i]
	}
	meanIP := photo2BoxBlurFloat(ip, radius)
	meanII := photo2BoxBlurFloat(ii, radius)
	a := cv.NewFloatMat(img.Rows, img.Cols)
	b := cv.NewFloatMat(img.Rows, img.Cols)
	for i := 0; i < n; i++ {
		covIP := meanIP.Data[i] - meanI.Data[i]*meanP.Data[i]
		varI := meanII.Data[i] - meanI.Data[i]*meanI.Data[i]
		av := covIP / (varI + eps)
		a.Data[i] = av
		b.Data[i] = meanP.Data[i] - av*meanI.Data[i]
	}
	meanA := photo2BoxBlurFloat(a, radius)
	meanB := photo2BoxBlurFloat(b, radius)
	out := cv.NewFloatMat(img.Rows, img.Cols)
	for i := 0; i < n; i++ {
		out.Data[i] = meanA.Data[i]*I[i] + meanB.Data[i]
	}
	return out
}

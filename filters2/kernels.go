package filters2

import "math"

// GaussianKernel1D returns a normalised 1-D Gaussian kernel of length
// 2*radius+1 with the given standard deviation, so its samples sum to one. It
// panics if radius is negative or sigma is not positive.
func GaussianKernel1D(radius int, sigma float64) []float64 {
	if radius < 0 {
		panic("filters2: GaussianKernel1D requires a non-negative radius")
	}
	if sigma <= 0 {
		panic("filters2: GaussianKernel1D requires a positive sigma")
	}
	n := 2*radius + 1
	k := make([]float64, n)
	twoSigma2 := 2 * sigma * sigma
	var sum float64
	for i := 0; i < n; i++ {
		d := float64(i - radius)
		v := math.Exp(-d * d / twoSigma2)
		k[i] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// GaussianKernel2D returns a normalised 2-D Gaussian kernel of size size×size
// (size must be a positive odd integer) with the given standard deviation, so
// its samples sum to one. It panics on an even or non-positive size or a
// non-positive sigma.
func GaussianKernel2D(size int, sigma float64) [][]float64 {
	requireOddPositive(size, "GaussianKernel2D")
	if sigma <= 0 {
		panic("filters2: GaussianKernel2D requires a positive sigma")
	}
	r := size / 2
	k := alloc2D(size, size)
	twoSigma2 := 2 * sigma * sigma
	var sum float64
	for y := 0; y < size; y++ {
		dy := float64(y - r)
		for x := 0; x < size; x++ {
			dx := float64(x - r)
			v := math.Exp(-(dx*dx + dy*dy) / twoSigma2)
			k[y][x] = v
			sum += v
		}
	}
	for y := range k {
		for x := range k[y] {
			k[y][x] /= sum
		}
	}
	return k
}

// GaussianDerivativeKernel returns a size×size sampled partial derivative of a
// 2-D Gaussian of the given standard deviation: the (dx-th, dy-th) mixed
// partial with respect to x and y. Supported orders are dx, dy in {0,1,2} with
// dx+dy at most 2. The kernel integrates the analytic derivatives of the
// continuous Gaussian and is the basis for steerable filtering. It panics on an
// even or non-positive size, a non-positive sigma, or an unsupported order.
func GaussianDerivativeKernel(size int, sigma float64, dx, dy int) [][]float64 {
	requireOddPositive(size, "GaussianDerivativeKernel")
	if sigma <= 0 {
		panic("filters2: GaussianDerivativeKernel requires a positive sigma")
	}
	if dx < 0 || dy < 0 || dx+dy > 2 {
		panic("filters2: GaussianDerivativeKernel supports dx,dy in {0,1,2} with dx+dy<=2")
	}
	r := size / 2
	s2 := sigma * sigma
	norm := 1.0 / (2 * math.Pi * s2)
	k := alloc2D(size, size)
	for yy := 0; yy < size; yy++ {
		y := float64(yy - r)
		for xx := 0; xx < size; xx++ {
			x := float64(xx - r)
			g := norm * math.Exp(-(x*x+y*y)/(2*s2))
			// Multiply by the Hermite-style factor for the requested order
			// along each axis: order 1 -> -t/s2, order 2 -> (t^2-s2)/s2^2.
			fx := derivFactor(x, s2, dx)
			fy := derivFactor(y, s2, dy)
			k[yy][xx] = g * fx * fy
		}
	}
	return k
}

// derivFactor returns the polynomial factor multiplying the Gaussian for the
// order-th derivative along one axis at coordinate t with variance s2.
func derivFactor(t, s2 float64, order int) float64 {
	switch order {
	case 0:
		return 1
	case 1:
		// Sign chosen so that correlation (as performed by Convolve) with this
		// kernel yields the true smoothed derivative: a positive response to an
		// intensity increasing along the axis.
		return t / s2
	case 2:
		return (t*t - s2) / (s2 * s2)
	default:
		panic("filters2: unsupported derivative order")
	}
}

// LaplacianOfGaussianKernel returns a size×size sampled Laplacian-of-Gaussian
// (Mexican-hat) kernel of the given standard deviation. The samples are
// mean-corrected so they sum to zero, ensuring the filter gives no response on
// a constant region. It panics on an even or non-positive size or a
// non-positive sigma.
func LaplacianOfGaussianKernel(size int, sigma float64) [][]float64 {
	requireOddPositive(size, "LaplacianOfGaussianKernel")
	if sigma <= 0 {
		panic("filters2: LaplacianOfGaussianKernel requires a positive sigma")
	}
	r := size / 2
	s2 := sigma * sigma
	s4 := s2 * s2
	k := alloc2D(size, size)
	var sum float64
	for yy := 0; yy < size; yy++ {
		y := float64(yy - r)
		for xx := 0; xx < size; xx++ {
			x := float64(xx - r)
			r2 := x*x + y*y
			v := -1.0 / (math.Pi * s4) * (1 - r2/(2*s2)) * math.Exp(-r2/(2*s2))
			k[yy][xx] = v
			sum += v
		}
	}
	mean := sum / float64(size*size)
	for yy := range k {
		for xx := range k[yy] {
			k[yy][xx] -= mean
		}
	}
	return k
}

// DifferenceOfGaussiansKernel returns a size×size difference-of-Gaussians
// kernel, the normalised Gaussian of standard deviation sigma1 minus the
// normalised Gaussian of standard deviation sigma2 (conventionally sigma1 <
// sigma2, giving a band-pass response). The samples sum to zero. It panics on
// an even or non-positive size or a non-positive sigma.
func DifferenceOfGaussiansKernel(size int, sigma1, sigma2 float64) [][]float64 {
	requireOddPositive(size, "DifferenceOfGaussiansKernel")
	if sigma1 <= 0 || sigma2 <= 0 {
		panic("filters2: DifferenceOfGaussiansKernel requires positive sigmas")
	}
	g1 := GaussianKernel2D(size, sigma1)
	g2 := GaussianKernel2D(size, sigma2)
	k := alloc2D(size, size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			k[y][x] = g1[y][x] - g2[y][x]
		}
	}
	return k
}

// GaborParams describes a 2-D Gabor filter. Theta is the orientation of the
// filter's normal in radians, Lambda the sinusoid wavelength in pixels, Sigma
// the Gaussian envelope standard deviation, Gamma the spatial aspect ratio and
// Psi the phase offset in radians.
type GaborParams struct {
	// Sigma is the standard deviation of the Gaussian envelope, in pixels.
	Sigma float64
	// Theta is the orientation of the normal to the parallel stripes, in radians.
	Theta float64
	// Lambda is the wavelength of the cosine factor, in pixels.
	Lambda float64
	// Gamma is the spatial aspect ratio controlling ellipticity.
	Gamma float64
	// Psi is the phase offset of the cosine factor, in radians.
	Psi float64
}

// GaborKernel returns a size×size real Gabor kernel for the given parameters.
// The kernel is the product of a Gaussian envelope and a cosine plane wave. It
// panics on an even or non-positive size or a non-positive Sigma or Lambda.
func GaborKernel(size int, p GaborParams) [][]float64 {
	requireOddPositive(size, "GaborKernel")
	if p.Sigma <= 0 || p.Lambda <= 0 {
		panic("filters2: GaborKernel requires positive Sigma and Lambda")
	}
	return gaborKernelPhase(size, p, p.Psi, math.Cos)
}

// gaborKernelPhase builds a Gabor kernel using the supplied carrier function
// (cosine for the real/even part, sine for the imaginary/odd part).
func gaborKernelPhase(size int, p GaborParams, psi float64, carrier func(float64) float64) [][]float64 {
	r := size / 2
	twoSigma2 := 2 * p.Sigma * p.Sigma
	gamma2 := p.Gamma * p.Gamma
	cosT, sinT := math.Cos(p.Theta), math.Sin(p.Theta)
	k := alloc2D(size, size)
	for yy := 0; yy < size; yy++ {
		y := float64(yy - r)
		for xx := 0; xx < size; xx++ {
			x := float64(xx - r)
			xr := x*cosT + y*sinT
			yr := -x*sinT + y*cosT
			env := math.Exp(-(xr*xr + gamma2*yr*yr) / twoSigma2)
			k[yy][xx] = env * carrier(2*math.Pi*xr/p.Lambda+psi)
		}
	}
	return k
}

// alloc2D allocates a rows×cols float64 grid.
func alloc2D(rows, cols int) [][]float64 {
	g := make([][]float64, rows)
	for i := range g {
		g[i] = make([]float64, cols)
	}
	return g
}

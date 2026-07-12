package face

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// This file implements the Biologically Inspired Features (BIF) descriptor of
// Guo et al. (2009), which OpenCV exposes as cv::face::BIF. BIF models the early
// visual cortex: a bank of Gabor filters at several scales and orientations
// (the "S1" simple-cell responses) is pooled across adjacent scale bands with a
// MAX operation (the "C1" complex-cell responses), and each pooled band is
// summarised over a spatial grid to yield a compact, illumination-tolerant
// feature vector well suited to describing faces (e.g. for age or gender
// estimation).

// BIF computes Biologically Inspired Features. Construct one with [NewBIF]; the
// zero value is not usable.
type BIF struct {
	scales  []float64 // Gabor sigma per scale, ascending
	orients int       // number of equally spaced orientations
	grid    int       // spatial pooling grid side (grid×grid cells per band)
}

// NewBIF returns a BIF descriptor with the given number of scales and
// orientations and a grid×grid spatial pooling grid. Scales use geometrically
// increasing Gabor envelopes; adjacent scales are MAX-pooled into bands, so at
// least two scales are required. It panics on non-positive parameters or fewer
// than two scales.
func NewBIF(scales, orients, grid int) *BIF {
	if scales < 2 {
		panic("face: NewBIF requires at least two scales")
	}
	if orients < 1 || grid < 1 {
		panic("face: NewBIF requires positive orientations and grid")
	}
	sig := make([]float64, scales)
	for i := 0; i < scales; i++ {
		// Sigma grows geometrically from 1.5 px, a compact analogue of the
		// increasing receptive-field sizes in the biological model.
		sig[i] = 1.5 * math.Pow(1.4, float64(i))
	}
	return &BIF{scales: sig, orients: orients, grid: grid}
}

// FeatureLength returns the length of the vector [BIF.Compute] produces: one
// value per (band, orientation, grid cell), where the number of bands is one
// fewer than the number of scales.
func (b *BIF) FeatureLength() int {
	bands := len(b.scales) - 1
	return bands * b.orients * b.grid * b.grid
}

// Compute reduces img to luma and returns its BIF descriptor. Each Gabor
// response magnitude image (S1) is computed, adjacent scales are combined by
// per-pixel MAX into bands (C1), and every band/orientation map is average-
// pooled over a grid×grid layout, concatenating the cell means in
// band-major, orientation, row-major cell order. The descriptor is
// L2-normalised so overall contrast does not dominate the distance between two
// faces. It panics on a nil or empty image.
func (b *BIF) Compute(img *cv.Mat) []float64 {
	g := toGrayMat(img)
	rows, cols := g.Rows, g.Cols

	// S1: Gabor magnitude per scale and orientation.
	// mags[scale][orient] is a rows*cols response magnitude image.
	mags := make([][][]float64, len(b.scales))
	for s, sigma := range b.scales {
		mags[s] = make([][]float64, b.orients)
		for o := 0; o < b.orients; o++ {
			theta := math.Pi * float64(o) / float64(b.orients)
			mags[s][o] = gaborMagnitude(g, sigma, theta)
		}
	}

	bands := len(b.scales) - 1
	feat := make([]float64, 0, b.FeatureLength())
	for band := 0; band < bands; band++ {
		for o := 0; o < b.orients; o++ {
			a := mags[band][o]
			c := mags[band+1][o]
			// C1: per-pixel MAX over the two adjacent scales.
			pooled := make([]float64, rows*cols)
			for i := range pooled {
				if a[i] > c[i] {
					pooled[i] = a[i]
				} else {
					pooled[i] = c[i]
				}
			}
			// Spatial average pooling over the grid.
			for gy := 0; gy < b.grid; gy++ {
				y0 := gy * rows / b.grid
				y1 := (gy + 1) * rows / b.grid
				for gx := 0; gx < b.grid; gx++ {
					x0 := gx * cols / b.grid
					x1 := (gx + 1) * cols / b.grid
					var sum float64
					var n int
					for y := y0; y < y1; y++ {
						for x := x0; x < x1; x++ {
							sum += pooled[y*cols+x]
							n++
						}
					}
					if n > 0 {
						feat = append(feat, sum/float64(n))
					} else {
						feat = append(feat, 0)
					}
				}
			}
		}
	}

	// L2 normalise for illumination/contrast tolerance.
	var norm float64
	for _, v := range feat {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 1e-12 {
		inv := 1 / norm
		for i := range feat {
			feat[i] *= inv
		}
	}
	return feat
}

// gaborMagnitude convolves g with an even (cosine) and odd (sine) Gabor kernel
// of the given envelope sigma and orientation theta, returning the per-pixel
// magnitude sqrt(even² + odd²). Borders are handled by replication. The
// wavelength and aspect ratio track sigma with the conventional constants
// (lambda = 2·sigma, gamma = 0.5).
func gaborMagnitude(g *cv.Mat, sigma, theta float64) []float64 {
	rows, cols := g.Rows, g.Cols
	radius := int(math.Ceil(2.5 * sigma))
	if radius < 1 {
		radius = 1
	}
	lambda := 2 * sigma
	gamma := 0.5
	cosT := math.Cos(theta)
	sinT := math.Sin(theta)
	twoSig2 := 2 * sigma * sigma
	k := 2 * math.Pi / lambda

	// Precompute the separable-free 2D kernels (small windows keep this cheap).
	ksize := 2*radius + 1
	even := make([]float64, ksize*ksize)
	odd := make([]float64, ksize*ksize)
	for j := -radius; j <= radius; j++ {
		for i := -radius; i <= radius; i++ {
			xr := float64(i)*cosT + float64(j)*sinT
			yr := -float64(i)*sinT + float64(j)*cosT
			env := math.Exp(-(xr*xr + gamma*gamma*yr*yr) / twoSig2)
			idx := (j+radius)*ksize + (i + radius)
			even[idx] = env * math.Cos(k*xr)
			odd[idx] = env * math.Sin(k*xr)
		}
	}
	// Zero-mean the even kernel so a flat region gives no response.
	var meanEven float64
	for _, v := range even {
		meanEven += v
	}
	meanEven /= float64(len(even))
	for i := range even {
		even[i] -= meanEven
	}

	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var re, im float64
			for j := -radius; j <= radius; j++ {
				for i := -radius; i <= radius; i++ {
					px := float64(replicateAt(g, y+j, x+i))
					idx := (j+radius)*ksize + (i + radius)
					re += px * even[idx]
					im += px * odd[idx]
				}
			}
			out[y*cols+x] = math.Sqrt(re*re + im*im)
		}
	}
	return out
}

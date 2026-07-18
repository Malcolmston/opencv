package texture

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// GaborParams holds the parameters of a 2-D Gabor filter — a sinusoidal plane
// wave modulated by a Gaussian envelope, the classic model of oriented,
// band-pass texture selectivity.
type GaborParams struct {
	// Sigma is the standard deviation of the Gaussian envelope, in pixels.
	Sigma float64
	// Theta is the orientation of the wave normal, in radians.
	Theta float64
	// Lambda is the wavelength of the sinusoid, in pixels.
	Lambda float64
	// Gamma is the spatial aspect ratio, controlling the ellipticity of the
	// envelope (1 = circular).
	Gamma float64
	// Psi is the phase offset of the sinusoid, in radians.
	Psi float64
}

// GaborKernel builds a size-by-size real Gabor kernel from the given
// parameters and returns it in row-major order. size must be a positive odd
// number and Sigma, Lambda must be > 0. The kernel is centred and follows the
// standard OpenCV parameterisation, so it can be used directly for texture
// energy or convolution.
func GaborKernel(size int, p GaborParams) []float64 {
	if size < 1 || size%2 == 0 {
		panic(fmt.Sprintf("texture: GaborKernel requires positive odd size, got %d", size))
	}
	if p.Sigma <= 0 || p.Lambda <= 0 {
		panic("texture: GaborKernel requires Sigma > 0 and Lambda > 0")
	}
	half := size / 2
	k := make([]float64, size*size)
	ct := math.Cos(p.Theta)
	st := math.Sin(p.Theta)
	sig2 := p.Sigma * p.Sigma
	g2 := p.Gamma * p.Gamma
	idx := 0
	for y := -half; y <= half; y++ {
		for x := -half; x <= half; x++ {
			xr := float64(x)*ct + float64(y)*st
			yr := -float64(x)*st + float64(y)*ct
			env := math.Exp(-(xr*xr + g2*yr*yr) / (2 * sig2))
			wave := math.Cos(2*math.Pi*xr/p.Lambda + p.Psi)
			k[idx] = env * wave
			idx++
		}
	}
	return k
}

// filterField convolves the luminance of img with a square kernel of the given
// side (odd) using reflect-101 borders and returns the response as a field.
func filterField(img *cv.Mat, kernel []float64, side int) *textureField {
	rows, cols := img.Rows, img.Cols
	luma := textureLumaFloat(img)
	half := side / 2
	f := textureNewField(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			ki := 0
			for ky := -half; ky <= half; ky++ {
				yy := textureReflect(y+ky, rows)
				base := yy * cols
				for kx := -half; kx <= half; kx++ {
					xx := textureReflect(x+kx, cols)
					acc += luma[base+xx] * kernel[ki]
					ki++
				}
			}
			f.data[y*cols+x] = acc
		}
	}
	return f
}

// GaborFilter convolves img with the Gabor kernel described by p and returns
// the filtered response as a single-channel [cv.Mat]. The floating-point
// response is shifted and scaled so its full range maps into [0,255] for
// visualisation; use [GaborEnergy] for a scale-independent scalar feature.
func GaborFilter(img *cv.Mat, size int, p GaborParams) *cv.Mat {
	textureRequire(img, "GaborFilter")
	kernel := GaborKernel(size, p)
	f := filterField(img, kernel, size)
	// Normalise to [0,255].
	lo, hi := f.data[0], f.data[0]
	for _, v := range f.data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	span := hi - lo
	dst := cv.NewMat(img.Rows, img.Cols, 1)
	if span == 0 {
		return dst
	}
	for i, v := range f.data {
		dst.Data[i] = textureClampU8((v-lo)/span*255 + 0.5)
	}
	return dst
}

// GaborEnergy returns the mean absolute Gabor response of img for the filter
// described by p with a size-by-size kernel — a single scalar summarising how
// strongly the image contains oriented structure at that frequency and
// orientation. This is the standard Gabor texture-energy measure.
func GaborEnergy(img *cv.Mat, size int, p GaborParams) float64 {
	textureRequire(img, "GaborEnergy")
	kernel := GaborKernel(size, p)
	f := filterField(img, kernel, size)
	return textureMeanAbs(f)
}

// GaborMagnitude returns the local Gabor magnitude field of img: the pixelwise
// square root of the sum of squares of the responses to the even (cosine,
// Psi=0) and odd (sine, Psi=-pi/2) Gabor kernels, which is phase-invariant. The
// magnitude is returned as a rows-by-cols grid.
func GaborMagnitude(img *cv.Mat, size int, p GaborParams) [][]float64 {
	textureRequire(img, "GaborMagnitude")
	even := p
	even.Psi = 0
	odd := p
	odd.Psi = -math.Pi / 2
	fe := filterField(img, GaborKernel(size, even), size)
	fo := filterField(img, GaborKernel(size, odd), size)
	rows, cols := img.Rows, img.Cols
	out := make([][]float64, rows)
	for y := 0; y < rows; y++ {
		row := make([]float64, cols)
		for x := 0; x < cols; x++ {
			e := fe.data[y*cols+x]
			o := fo.data[y*cols+x]
			row[x] = math.Hypot(e, o)
		}
		out[y] = row
	}
	return out
}

// GaborFilterBank builds a bank of Gabor kernels sharing base parameters p but
// spanning nOrient equally spaced orientations in [0, pi) at each of the given
// wavelengths. It returns one kernel (row-major, size-by-size) per
// orientation/wavelength combination, ordered orientation-major. It is the
// front end for multi-channel Gabor texture features.
func GaborFilterBank(size int, p GaborParams, nOrient int, wavelengths []float64) [][]float64 {
	if nOrient < 1 {
		panic(fmt.Sprintf("texture: GaborFilterBank requires nOrient >= 1, got %d", nOrient))
	}
	if len(wavelengths) == 0 {
		panic("texture: GaborFilterBank requires at least one wavelength")
	}
	out := make([][]float64, 0, nOrient*len(wavelengths))
	for o := 0; o < nOrient; o++ {
		theta := math.Pi * float64(o) / float64(nOrient)
		for _, lam := range wavelengths {
			pp := p
			pp.Theta = theta
			pp.Lambda = lam
			out = append(out, GaborKernel(size, pp))
		}
	}
	return out
}

// GaborFeatures runs a bank of nOrient orientations at the given wavelengths
// over img and returns a feature vector holding, for each filter, the mean and
// standard deviation of its magnitude response — the classic Gabor texture
// signature used for segmentation and classification. The vector has length
// 2*nOrient*len(wavelengths), interleaved as [mean0, std0, mean1, std1, ...] in
// the same order as [GaborFilterBank].
func GaborFeatures(img *cv.Mat, size int, p GaborParams, nOrient int, wavelengths []float64) []float64 {
	textureRequire(img, "GaborFeatures")
	bank := GaborFilterBank(size, p, nOrient, wavelengths)
	out := make([]float64, 0, 2*len(bank))
	for _, kernel := range bank {
		f := filterField(img, kernel, size)
		// Use absolute response as the energy field.
		abs := textureNewField(f.rows, f.cols)
		for i, v := range f.data {
			abs.data[i] = math.Abs(v)
		}
		mean := textureMeanAbs(f)
		out = append(out, mean, textureStd(abs))
	}
	return out
}

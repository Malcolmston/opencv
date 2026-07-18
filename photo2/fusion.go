package photo2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MertensParams configures [MertensFusion].
type MertensParams struct {
	// ContrastWeight is the exponent applied to the contrast quality measure
	// (Laplacian magnitude). Larger values favour sharp regions. Default 1.
	ContrastWeight float64
	// SaturationWeight is the exponent applied to the saturation quality
	// measure. Larger values favour vivid regions. Default 1.
	SaturationWeight float64
	// ExposureWeight is the exponent applied to the well-exposedness measure,
	// which prefers pixels near mid-grey. Default 1.
	ExposureWeight float64
	// Levels is the number of multiresolution pyramid levels used for seamless
	// blending; a non-positive value derives it from the image size. More levels
	// blend larger structures.
	Levels int
}

// DefaultMertensParams returns the recommended defaults for [MertensFusion].
func DefaultMertensParams() MertensParams {
	return MertensParams{ContrastWeight: 1, SaturationWeight: 1, ExposureWeight: 1, Levels: 0}
}

// MertensFusion performs Mertens, Kautz and Van Reeth's (2007) exposure fusion.
// Given a bracket of differently exposed images of the same scene, it blends
// them directly into a single well-exposed low-dynamic-range image without ever
// computing an HDR radiance map or requiring exposure times. Each pixel is
// weighted by contrast, saturation and well-exposedness, and the images are
// combined through a Laplacian-pyramid blend so the result is seamless. All
// images must share dimensions and channel count; at least one is required.
func MertensFusion(images []*cv.Mat, params MertensParams) *cv.Mat {
	if len(images) == 0 {
		panic("photo2: MertensFusion given no images")
	}
	photo2RequireImage(images[0], "MertensFusion")
	rows, cols, nch := images[0].Rows, images[0].Cols, images[0].Channels
	for _, m := range images {
		if m == nil || m.Rows != rows || m.Cols != cols || m.Channels != nch {
			panic("photo2: MertensFusion images must share dimensions and channels")
		}
	}
	if params.ContrastWeight < 0 {
		params.ContrastWeight = 0
	}
	if params.SaturationWeight < 0 {
		params.SaturationWeight = 0
	}
	if params.ExposureWeight < 0 {
		params.ExposureWeight = 0
	}
	levels := params.Levels
	if levels <= 0 {
		m := rows
		if cols < m {
			m = cols
		}
		levels = int(math.Log2(float64(m)))
		if levels < 1 {
			levels = 1
		}
	}

	n := len(images)
	total := rows * cols
	// Per-image float planes and weight maps.
	floatImgs := make([][]*cv.FloatMat, n)
	weights := make([]*cv.FloatMat, n)
	weightSum := cv.NewFloatMat(rows, cols)
	for k := 0; k < n; k++ {
		planes := ToFloat(images[k])
		floatImgs[k] = planes
		gray := LuminanceChannels(planes)
		contrast := photo2LaplacianAbs(gray)
		w := cv.NewFloatMat(rows, cols)
		for i := 0; i < total; i++ {
			cQ := math.Pow(contrast.Data[i], params.ContrastWeight)
			sQ := math.Pow(photo2SaturationAt(planes, i), params.SaturationWeight)
			eQ := math.Pow(photo2Exposedness(planes, i), params.ExposureWeight)
			wq := cQ*sQ*eQ + 1e-12
			w.Data[i] = wq
			weightSum.Data[i] += wq
		}
		weights[k] = w
	}
	// Normalise weights.
	for k := 0; k < n; k++ {
		for i := 0; i < total; i++ {
			weights[k].Data[i] /= weightSum.Data[i]
		}
	}

	// Multiresolution blend: accumulate weighted Laplacian pyramids per channel.
	var accum [][]*cv.FloatMat // [channel][level]
	for k := 0; k < n; k++ {
		wPyr := GaussianPyramid(weights[k], levels)
		L := len(wPyr)
		if accum == nil {
			accum = make([][]*cv.FloatMat, nch)
			for c := 0; c < nch; c++ {
				accum[c] = make([]*cv.FloatMat, L)
				for l := 0; l < L; l++ {
					accum[c][l] = cv.NewFloatMat(wPyr[l].Rows, wPyr[l].Cols)
				}
			}
		}
		for c := 0; c < nch; c++ {
			imgPyr := LaplacianPyramid(floatImgs[k][c], levels)
			for l := 0; l < L && l < len(imgPyr); l++ {
				dst := accum[c][l]
				src := imgPyr[l]
				wl := wPyr[l]
				for i := range dst.Data {
					dst.Data[i] += wl.Data[i] * src.Data[i]
				}
			}
		}
	}

	outPlanes := make([]*cv.FloatMat, nch)
	for c := 0; c < nch; c++ {
		outPlanes[c] = ReconstructLaplacianPyramid(accum[c])
	}
	return FromFloat(outPlanes)
}

// photo2LaplacianAbs returns the absolute response of a 4-neighbour Laplacian
// (contrast measure) on a float plane with reflected borders.
func photo2LaplacianAbs(f *cv.FloatMat) *cv.FloatMat {
	rows, cols := f.Rows, f.Cols
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			c := f.Data[y*cols+x]
			up := f.Data[photo2Reflect(y-1, rows)*cols+x]
			dn := f.Data[photo2Reflect(y+1, rows)*cols+x]
			lf := f.Data[y*cols+photo2Reflect(x-1, cols)]
			rt := f.Data[y*cols+photo2Reflect(x+1, cols)]
			out.Data[y*cols+x] = math.Abs(up + dn + lf + rt - 4*c)
		}
	}
	return out
}

// photo2SaturationAt returns the standard deviation across colour channels at
// flat pixel index i (0 for a single channel).
func photo2SaturationAt(planes []*cv.FloatMat, i int) float64 {
	nch := len(planes)
	if nch < 2 {
		return 0
	}
	var mean float64
	for c := 0; c < nch; c++ {
		mean += planes[c].Data[i]
	}
	mean /= float64(nch)
	var v float64
	for c := 0; c < nch; c++ {
		d := planes[c].Data[i] - mean
		v += d * d
	}
	return math.Sqrt(v / float64(nch))
}

// photo2Exposedness returns the well-exposedness weight at flat pixel index i,
// the product over channels of a Gaussian centred at 0.5 (sigma 0.2).
func photo2Exposedness(planes []*cv.FloatMat, i int) float64 {
	const sigma = 0.2
	twoS2 := 2 * sigma * sigma
	w := 1.0
	for _, p := range planes {
		d := p.Data[i] - 0.5
		w *= math.Exp(-(d * d) / twoS2)
	}
	return w
}

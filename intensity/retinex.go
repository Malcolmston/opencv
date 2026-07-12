package intensity

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// DefaultRetinexScales returns the classic three-scale set {15, 80, 250} used
// by [MultiScaleRetinex] and [MSRCR] when no scales are supplied. The scales,
// in pixels, span fine detail, medium structure and global illumination. A
// fresh slice is returned on each call so callers may modify it freely.
func DefaultRetinexScales() []float64 {
	return []float64{15, 80, 250}
}

// ssrPlane returns the single-scale retinex response of one intensity plane,
//
//	R(x) = log(I(x) + 1) − log((G_sigma * I)(x) + 1),
//
// the log-domain difference between the plane and its Gaussian-blurred surround.
// The +1 offsets keep the logarithm finite at zero.
func ssrPlane(plane []float64, rows, cols int, sigma float64) []float64 {
	blur := blurPlaneFloat(plane, rows, cols, sigma)
	out := make([]float64, len(plane))
	for i := range plane {
		out[i] = math.Log(plane[i]+1) - math.Log(blur[i]+1)
	}
	return out
}

// msrPlane returns the multi-scale retinex response of one plane, the unweighted
// mean of the single-scale responses over the given sigmas.
func msrPlane(plane []float64, rows, cols int, sigmas []float64) []float64 {
	acc := make([]float64, len(plane))
	for _, s := range sigmas {
		r := ssrPlane(plane, rows, cols, s)
		for i := range acc {
			acc[i] += r[i]
		}
	}
	inv := 1.0 / float64(len(sigmas))
	for i := range acc {
		acc[i] *= inv
	}
	return acc
}

// stretchFloatToUint8 linearly maps v onto [0,255] after discarding the lowest
// lowFrac and highest highFrac of its (sorted) mass, the "simplest colour
// balance" stretch. A degenerate range maps everything to mid-grey.
func stretchFloatToUint8(v []float64, lowFrac, highFrac float64) []uint8 {
	n := len(v)
	out := make([]uint8, n)
	s := make([]float64, n)
	copy(s, v)
	sort.Float64s(s)
	loIdx := int(lowFrac * float64(n))
	hiIdx := n - 1 - int(highFrac*float64(n))
	if loIdx < 0 {
		loIdx = 0
	}
	if hiIdx >= n {
		hiIdx = n - 1
	}
	if hiIdx <= loIdx {
		loIdx, hiIdx = 0, n-1
	}
	lo, hi := s[loIdx], s[hiIdx]
	span := hi - lo
	if span <= 0 {
		for i := range out {
			out[i] = 128
		}
		return out
	}
	scale := 255 / span
	for i, x := range v {
		out[i] = clampToUint8((x-lo)*scale + 0.5)
	}
	return out
}

// requireScales panics unless sigmas is non-empty and every scale is positive.
func requireScales(sigmas []float64, name string) {
	if len(sigmas) == 0 {
		panic("intensity: " + name + " requires at least one scale")
	}
	for _, s := range sigmas {
		if !(s > 0) || math.IsInf(s, 0) {
			panic("intensity: " + name + " requires positive, finite scales")
		}
	}
}

// SingleScaleRetinex enhances img with the single-scale retinex (SSR) of Jobson,
// Rahman and Woodell (1997). Each channel is replaced by the log-domain
// difference between the channel and its Gaussian surround of standard deviation
// sigma, which discounts the smoothly varying illumination and keeps the local
// reflectance; the result is stretched back into [0,255] with a 1% tail clip per
// channel. Small sigma favours local detail and dynamic-range compression; large
// sigma favours tonal rendition. sigma must be positive; it panics otherwise.
//
// Because retinex normalises away the global illumination, a dark image is
// lifted toward a balanced mid-range while its local structure — the ordering
// and texture of reflectances — is preserved. The output is deterministic.
func SingleScaleRetinex(img *cv.Mat, sigma float64) *cv.Mat {
	requireImage(img, "SingleScaleRetinex")
	requireScales([]float64{sigma}, "SingleScaleRetinex")
	return MultiScaleRetinex(img, []float64{sigma})
}

// MultiScaleRetinex enhances img with multi-scale retinex (MSR): the per-channel
// retinex responses at each scale in sigmas are averaged, then stretched into
// [0,255] with a 1% tail clip per channel. Combining several scales balances the
// dynamic-range compression of small scales against the tonal fidelity of large
// ones. Pass [DefaultRetinexScales] for the classic {15,80,250}. sigmas must be
// non-empty with positive, finite entries; it panics otherwise. The output is
// deterministic.
func MultiScaleRetinex(img *cv.Mat, sigmas []float64) *cv.Mat {
	requireImage(img, "MultiScaleRetinex")
	requireScales(sigmas, "MultiScaleRetinex")
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	dst := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		plane := channelFloat(img, c)
		msr := msrPlane(plane, rows, cols, sigmas)
		stretched := stretchFloatToUint8(msr, 0.01, 0.01)
		for p := range stretched {
			dst.Data[p*ch+c] = stretched[p]
		}
	}
	return dst
}

// MSRCR constants after Jobson et al.: alpha weights the colour-restoration
// logarithm, beta scales it.
const (
	msrcrAlpha = 125.0
	msrcrBeta  = 46.0
)

// MSRCR enhances img with multi-scale retinex with colour restoration (MSRCR),
// adding a chromatic term that counteracts the greying-out MSR causes on
// strongly coloured regions. For each channel the MSR response is multiplied by
// the colour-restoration factor
//
//	C_c = beta · log(1 + alpha · I_c / Σ_k I_k),
//
// which rewards channels that dominate a pixel, before a per-channel 1% tail
// stretch to [0,255]. On a single-channel image the restoration factor is
// constant and MSRCR reduces to [MultiScaleRetinex]. sigmas must be non-empty
// with positive, finite entries; it panics otherwise. The output is
// deterministic.
func MSRCR(img *cv.Mat, sigmas []float64) *cv.Mat {
	requireImage(img, "MSRCR")
	requireScales(sigmas, "MSRCR")
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	n := img.Total()

	// Per-pixel channel sum for the colour-restoration denominator.
	sum := make([]float64, n)
	for p := 0; p < n; p++ {
		base := p * ch
		var s float64
		for c := 0; c < ch; c++ {
			s += float64(img.Data[base+c])
		}
		sum[p] = s
	}

	dst := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		plane := channelFloat(img, c)
		msr := msrPlane(plane, rows, cols, sigmas)
		combined := make([]float64, n)
		for p := 0; p < n; p++ {
			restore := msrcrBeta * math.Log(1+msrcrAlpha*plane[p]/(sum[p]+1))
			combined[p] = restore * msr[p]
		}
		stretched := stretchFloatToUint8(combined, 0.01, 0.01)
		for p := range stretched {
			dst.Data[p*ch+c] = stretched[p]
		}
	}
	return dst
}

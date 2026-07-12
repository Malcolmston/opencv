package xphoto

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// estimateIllumMinkowski estimates a per-channel illuminant from src using the
// general Minkowski-norm colour-constancy framework: for norm order p the
// illuminant of channel c is ( mean( f_c^p ) )^(1/p). p == 1 recovers the
// gray-world assumption (the plain channel mean) and p -> infinity recovers the
// white-patch / max-RGB assumption. Only pixels whose maximum channel is below
// satLevel contribute, so clipped highlights do not bias the estimate. The
// returned vector is normalised to unit L2 length; a degenerate (all-dark)
// image yields the neutral illuminant {1,1,1}/sqrt(3).
func estimateIllumMinkowski(src *cv.Mat, p, satLevel float64) [3]float64 {
	total := src.Total()
	var acc [3]float64
	var n float64
	infinite := math.IsInf(p, 1)
	for i := 0; i < total; i++ {
		r := float64(src.Data[i*3+0])
		g := float64(src.Data[i*3+1])
		b := float64(src.Data[i*3+2])
		if math.Max(r, math.Max(g, b)) >= satLevel {
			continue
		}
		if infinite {
			acc[0] = math.Max(acc[0], r)
			acc[1] = math.Max(acc[1], g)
			acc[2] = math.Max(acc[2], b)
		} else {
			acc[0] += math.Pow(r, p)
			acc[1] += math.Pow(g, p)
			acc[2] += math.Pow(b, p)
		}
		n++
	}
	if n == 0 {
		return [3]float64{1, 1, 1}
	}
	var e [3]float64
	if infinite {
		e = acc
	} else {
		for c := 0; c < 3; c++ {
			e[c] = math.Pow(acc[c]/n, 1.0/p)
		}
	}
	return normalizeL2(e)
}

// normalizeL2 scales v to unit Euclidean length, returning the neutral
// illuminant for a (near-)zero vector.
func normalizeL2(v [3]float64) [3]float64 {
	norm := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
	if norm < 1e-9 {
		s := 1.0 / math.Sqrt(3)
		return [3]float64{s, s, s}
	}
	return [3]float64{v[0] / norm, v[1] / norm, v[2] / norm}
}

// gainsFromIllum converts a per-channel illuminant estimate into neutralising
// channel gains that map the illuminant onto grey. The gains are scaled so the
// green channel is left unchanged (gain 1), which keeps overall brightness close
// to the original the way OpenCV's balancers do.
func gainsFromIllum(illum [3]float64) [3]float64 {
	var gains [3]float64
	ref := illum[1]
	if ref <= 1e-9 {
		ref = (illum[0] + illum[1] + illum[2]) / 3
	}
	for c := 0; c < 3; c++ {
		if illum[c] > 1e-9 {
			gains[c] = ref / illum[c]
		} else {
			gains[c] = 1
		}
	}
	return gains
}

// ShadesOfGray white-balances src using the general "shades of gray" colour
// constancy method (Finlayson & Trezzi, 2004): it estimates the scene
// illuminant with the Minkowski p-norm of each channel and applies neutralising
// gains. p is the Minkowski norm order; p == 1 is the gray-world assumption,
// larger p weights bright pixels more, and p == +Inf is the white-patch
// (max-RGB) assumption. Non-finite p other than +Inf, and p <= 0, default to 6
// (a common robust choice). Pixels at or above 99.6% of full scale are treated
// as clipped and excluded. src must be a three-channel RGB image; the input is
// not modified.
func ShadesOfGray(src *cv.Mat, p float64) *cv.Mat {
	requireNonEmpty(src, "ShadesOfGray")
	requireChannels(src, 3, "ShadesOfGray")
	if !math.IsInf(p, 1) && (p <= 0 || math.IsNaN(p)) {
		p = 6
	}
	illum := estimateIllumMinkowski(src, p, 254.0)
	gains := gainsFromIllum(illum)
	return ApplyChannelGains(src, gains[0], gains[1], gains[2])
}

// WhitePatchWB white-balances src with the white-patch (max-RGB / Retinex)
// assumption: the brightest value found in each channel is assumed to be the
// scene's white point, so each channel is scaled to bring those maxima into
// balance. It is the p -> infinity limit of [ShadesOfGray]. src must be a
// three-channel RGB image; the input is not modified.
func WhitePatchWB(src *cv.Mat) *cv.Mat {
	requireNonEmpty(src, "WhitePatchWB")
	requireChannels(src, 3, "WhitePatchWB")
	illum := estimateIllumMinkowski(src, math.Inf(1), 256.0)
	gains := gainsFromIllum(illum)
	return ApplyChannelGains(src, gains[0], gains[1], gains[2])
}

// GrayEdgeWB white-balances src with the first-order gray-edge assumption (van
// de Weijer et al.): the average reflectance of scene *edges* is achromatic, so
// the illuminant is estimated from the Minkowski p-norm of the per-channel
// gradient magnitudes rather than the pixel values. This is often more robust
// than gray-world on scenes with a large uniform coloured object. p is the
// Minkowski order (p <= 0 defaults to 6); src must be three-channel RGB and is
// not modified.
func GrayEdgeWB(src *cv.Mat, p float64) *cv.Mat {
	requireNonEmpty(src, "GrayEdgeWB")
	requireChannels(src, 3, "GrayEdgeWB")
	if !math.IsInf(p, 1) && (p <= 0 || math.IsNaN(p)) {
		p = 6
	}
	rows, cols := src.Rows, src.Cols
	var acc [3]float64
	var n float64
	infinite := math.IsInf(p, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < 3; c++ {
				gmag := channelGradient(src, y, x, c)
				if infinite {
					acc[c] = math.Max(acc[c], gmag)
				} else {
					acc[c] += math.Pow(gmag, p)
				}
			}
			n++
		}
	}
	var e [3]float64
	if infinite || n == 0 {
		e = acc
	} else {
		for c := 0; c < 3; c++ {
			e[c] = math.Pow(acc[c]/n, 1.0/p)
		}
	}
	illum := normalizeL2(e)
	gains := gainsFromIllum(illum)
	return ApplyChannelGains(src, gains[0], gains[1], gains[2])
}

// AutoWhiteBalance white-balances src with no tuning required. It estimates the
// illuminant with a moderate Minkowski norm (shades-of-gray, p = 6), which
// behaves like gray-world on well-balanced scenes but resists being dragged by a
// single large flat colour, and applies the neutralising gains. It is a
// convenience entry point for callers that just want "make this look neutral"
// without choosing an algorithm. src must be three-channel RGB; the input is not
// modified.
func AutoWhiteBalance(src *cv.Mat) *cv.Mat {
	requireNonEmpty(src, "AutoWhiteBalance")
	requireChannels(src, 3, "AutoWhiteBalance")
	return ShadesOfGray(src, 6)
}

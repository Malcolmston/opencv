package ximgproc

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// NiBlackVariant selects the local-threshold formula used by
// [NiBlackThreshold]. Pass the corresponding int value as the variant argument.
type NiBlackVariant int

const (
	// NiBlackNiblack is the original Niblack (1986) rule T = m + k·s, where m
	// and s are the local mean and standard deviation. k is typically negative
	// (around −0.2) when the foreground is darker than the background.
	NiBlackNiblack NiBlackVariant = iota
	// NiBlackSauvola is the Sauvola–Pietikäinen (2000) rule
	// T = m·(1 + k·(s/R − 1)) with dynamic range R = 128, which suppresses the
	// noise Niblack produces in uniform background regions. k is typically ~0.5.
	NiBlackSauvola
	// NiBlackWolf is the Wolf–Jolion (2004) rule
	// T = m + k·(s/R − 1)·(m − M), where R is the maximum local standard
	// deviation over the whole image and M is its global minimum intensity. It
	// normalises contrast across the image. k is typically ~0.5.
	NiBlackWolf
	// NiBlackNick is the NICK rule T = m + k·√((Σp² − m²)/N), which shifts the
	// threshold to cope with very low-contrast images. k is typically negative
	// (−0.2..−0.1).
	NiBlackNick
)

// NiBlackThreshold binarizes a single-channel image with a threshold computed
// independently for each pixel from the mean and standard deviation of its
// blockSize×blockSize neighbourhood, and returns a new single-channel Mat.
// Samples strictly greater than their local threshold become 255, the rest 0
// (equivalent to [cv.ThreshBinary]).
//
// Because the threshold adapts to local illumination, the method binarizes
// images with uneven lighting or shading gradients far better than a single
// global threshold such as Otsu. variant selects the threshold formula — see
// [NiBlackVariant] for the four supported rules and the meaning of k in each.
//
// blockSize must be a positive odd integer; img must be single-channel. It
// panics otherwise or on an unknown variant. Window statistics use edge-clamped
// neighbourhoods (border pixels are normalised by the number of valid samples).
func NiBlackThreshold(img *cv.Mat, k float64, blockSize, variant int) *cv.Mat {
	if img.Channels != 1 {
		panic("ximgproc: NiBlackThreshold requires a single-channel image")
	}
	if blockSize <= 0 || blockSize%2 == 0 {
		panic(fmt.Sprintf("ximgproc: NiBlackThreshold requires a positive odd blockSize, got %d", blockSize))
	}
	rows, cols := img.Rows, img.Cols
	r := blockSize / 2

	src := make([]float64, rows*cols)
	for i, v := range img.Data {
		src[i] = float64(v)
	}
	mean := boxMean(src, rows, cols, r)
	meanSq := boxMean(mul(src, src), rows, cols, r)

	std := make([]float64, len(mean))
	maxStd := 0.0
	for i := range std {
		v := meanSq[i] - mean[i]*mean[i]
		if v < 0 {
			v = 0
		}
		std[i] = math.Sqrt(v)
		if std[i] > maxStd {
			maxStd = std[i]
		}
	}

	// Global minimum intensity, used by the Wolf variant.
	minVal := 255.0
	for _, v := range src {
		if v < minVal {
			minVal = v
		}
	}

	if maxStd == 0 {
		maxStd = 1 // avoid division by zero on a perfectly flat image
	}

	out := cv.NewMat(rows, cols, 1)
	for i := range src {
		m := mean[i]
		s := std[i]
		var t float64
		switch NiBlackVariant(variant) {
		case NiBlackNiblack:
			t = m + k*s
		case NiBlackSauvola:
			t = m * (1 + k*(s/128.0-1))
		case NiBlackWolf:
			t = m + k*(s/maxStd-1)*(m-minVal)
		case NiBlackNick:
			// meanSq = Σp²/N, so Σp²/N − m²/N = meanSq − m²/N.
			n := float64(windowCount(rows, cols, r, i))
			b := meanSq[i] - m*m/n
			if b < 0 {
				b = 0
			}
			t = m + k*math.Sqrt(b)
		default:
			panic(fmt.Sprintf("ximgproc: NiBlackThreshold unknown variant %d", variant))
		}
		if src[i] > t {
			out.Data[i] = 255
		}
	}
	return out
}

// windowCount returns the number of valid pixels in the clamped blockSize
// window (radius r) centred on flat index i of a rows×cols image.
func windowCount(rows, cols, r, i int) int {
	y := i / cols
	x := i % cols
	y0 := y - r
	if y0 < 0 {
		y0 = 0
	}
	y1 := y + r
	if y1 > rows-1 {
		y1 = rows - 1
	}
	x0 := x - r
	if x0 < 0 {
		x0 = 0
	}
	x1 := x + r
	if x1 > cols-1 {
		x1 = cols - 1
	}
	return (y1 - y0 + 1) * (x1 - x0 + 1)
}

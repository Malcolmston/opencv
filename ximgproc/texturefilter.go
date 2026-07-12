package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BilateralTextureFilter applies the bilateral texture filter of Cho, Lee, Kim
// and Lee ("Bilateral Texture Filtering", 2014) to src and returns a new Mat of
// the same shape. It separates structure from texture: strong texture (fine,
// oscillatory detail) is removed while the main object boundaries are kept
// sharp, even when texture and structure have comparable contrast — a regime in
// which ordinary bilateral or guided filtering fails.
//
// For each pixel the method searches the (2·radius+1)² candidate patches that
// contain it and picks the one whose content is most "flat" according to the
// modified relative total variation
//
//	mRTV = maxGradient · Σ|gradient| / (Σ gradient² + ε),
//
// which is small for structural (edge-dominated) patches and large for busy
// texture. The mean colour of the selected patch becomes the pixel's value in a
// guidance image G; src is then joint-bilateral filtered using G as the guide.
// Repeating this for iters iterations progressively strips texture.
//
// radius is the patch radius in pixels (must be positive); iters is the number
// of texture-removal iterations (typically 3–5, and must be ≥ 1). src may be 1-
// or 3-channel. It panics on a non-positive radius or iters < 1. The filter is
// deterministic.
func BilateralTextureFilter(src *cv.Mat, radius, iters int) *cv.Mat {
	if radius <= 0 {
		panic("ximgproc: BilateralTextureFilter requires a positive radius")
	}
	if iters < 1 {
		panic("ximgproc: BilateralTextureFilter requires iters >= 1")
	}
	sigmaSpace := float64(2*radius + 1)
	sigmaColor := math.Sqrt(3.0) * 40.0

	cur := src.Clone()
	for it := 0; it < iters; it++ {
		guide := textureGuidance(cur, radius)
		cur = jointBilateralCore(guide, cur, 2*radius+1, sigmaColor, sigmaSpace)
	}
	return cur
}

// textureGuidance builds the guidance image G: every pixel is replaced by the
// mean colour of the least-textured patch (minimum mRTV) that contains it.
func textureGuidance(img *cv.Mat, radius int) *cv.Mat {
	rows, cols := img.Rows, img.Cols
	ch := img.Channels

	// Grayscale gradient magnitude for the mRTV measure.
	gray := channelPlane(toGray(img), 0)
	gmag := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gx := gray[y*cols+reflect(x+1, cols)] - gray[y*cols+reflect(x-1, cols)]
			gy := gray[reflect(y+1, rows)*cols+x] - gray[reflect(y-1, rows)*cols+x]
			gmag[y*cols+x] = math.Hypot(gx, gy)
		}
	}

	// mRTV of the patch centred at each pixel.
	mrtv := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var maxG, sumG, sumG2 float64
			for dy := -radius; dy <= radius; dy++ {
				yy := reflect(y+dy, rows)
				for dx := -radius; dx <= radius; dx++ {
					xx := reflect(x+dx, cols)
					g := gmag[yy*cols+xx]
					if g > maxG {
						maxG = g
					}
					sumG += g
					sumG2 += g * g
				}
			}
			mrtv[y*cols+x] = maxG * sumG / (sumG2 + 1e-6)
		}
	}

	// Patch means per centre.
	means := make([][]float64, rows*cols)
	area := float64((2*radius + 1) * (2*radius + 1))
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m := make([]float64, ch)
			for dy := -radius; dy <= radius; dy++ {
				yy := reflect(y+dy, rows)
				for dx := -radius; dx <= radius; dx++ {
					xx := reflect(x+dx, cols)
					pi := (yy*cols + xx) * ch
					for c := 0; c < ch; c++ {
						m[c] += float64(img.Data[pi+c])
					}
				}
			}
			for c := 0; c < ch; c++ {
				m[c] /= area
			}
			means[y*cols+x] = m
		}
	}

	// For each pixel choose the containing patch with the smallest mRTV.
	out := cv.NewMat(rows, cols, ch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			best := math.MaxFloat64
			bestC := y*cols + x
			for dy := -radius; dy <= radius; dy++ {
				cy := y + dy
				if cy < 0 || cy >= rows {
					continue
				}
				for dx := -radius; dx <= radius; dx++ {
					cx := x + dx
					if cx < 0 || cx >= cols {
						continue
					}
					ci := cy*cols + cx
					if mrtv[ci] < best {
						best = mrtv[ci]
						bestC = ci
					}
				}
			}
			m := means[bestC]
			oi := (y*cols + x) * ch
			for c := 0; c < ch; c++ {
				out.Data[oi+c] = clampU8(m[c])
			}
		}
	}
	return out
}

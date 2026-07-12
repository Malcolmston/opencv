package photo

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PencilSketch renders img as a pencil drawing, returning both a grayscale
// sketch and a colour-tinted sketch, mirroring OpenCV's pencilSketch. The effect
// is the classic colour-dodge blend: the luma is inverted, edge-preservingly
// blurred, and used to divide the original luma, which brightens flat regions to
// near-white and leaves only the darkened pencil strokes along edges. The
// grayscale result is that stroke image; the colour result multiplies the
// original colours by the stroke brightness so the drawing keeps a faint tint.
//
// sigmaS and sigmaR control the edge-preserving blur (see
// [DomainTransformFilter]); shadeFactor in roughly [0,0.1] darkens the strokes
// (default 0.02 when non-positive). img may be single- or three-channel; gray is
// always single-channel and color has img's channel count.
func PencilSketch(img *cv.Mat, sigmaS, sigmaR, shadeFactor float64) (gray *cv.Mat, color *cv.Mat) {
	if img == nil || img.Empty() {
		panic("photo: PencilSketch given an empty image")
	}
	if shadeFactor <= 0 {
		shadeFactor = 0.02
	}
	rows, cols := img.Rows, img.Cols
	luma := grayOf(img)

	// Inverted luma, edge-preservingly smoothed.
	inv := cv.NewMat(rows, cols, 1)
	for i := range inv.Data {
		inv.Data[i] = 255 - luma.Data[i]
	}
	blur := DomainTransformFilter(inv, NormconvFilter, sigmaS, sigmaR)

	// Colour dodge: result = base * 255 / (255 - blurredInvert). The strokes
	// (dodge below 1) are then darkened by shadeFactor.
	stroke := make([]float64, rows*cols) // in [0,1]
	gray = cv.NewMat(rows, cols, 1)
	for i := range gray.Data {
		base := float64(luma.Data[i])
		denom := 255 - float64(blur.Data[i])
		var dodge float64
		if denom <= 0 {
			dodge = 255
		} else {
			dodge = base * 255 / denom
		}
		s := dodge / 255 * (1 - shadeFactor)
		if s > 1 {
			s = 1
		}
		stroke[i] = s
		gray.Data[i] = clampU8(s * 255)
	}

	color = cv.NewMat(rows, cols, img.Channels)
	for i := 0; i < rows*cols; i++ {
		for c := 0; c < img.Channels; c++ {
			color.Data[i*img.Channels+c] = clampU8(stroke[i] * float64(img.Data[i*img.Channels+c]))
		}
	}
	return gray, color
}

// OilPainting applies an oil-painting stylisation, the algorithm from OpenCV's
// xphoto module. For each pixel it examines a (2*size+1)² neighbourhood, buckets
// the neighbours by intensity into 256/dynRatio bins, finds the most populated
// bin and outputs the average colour of the pixels falling in it. This collapses
// local variation onto its dominant tone, giving the flat, brush-stroked look of
// an oil painting while respecting region boundaries.
//
// size is the neighbourhood radius (minimum 1) and dynRatio the intensity
// quantisation step (minimum 1; larger means coarser, more posterised output).
// img may be single- or three-channel; the output has the same shape. The
// original is not modified.
func OilPainting(img *cv.Mat, size, dynRatio int) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: OilPainting given an empty image")
	}
	if size < 1 {
		size = 1
	}
	if dynRatio < 1 {
		dynRatio = 1
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	out := cv.NewMat(rows, cols, ch)
	bins := 256/dynRatio + 1

	gray := grayOf(img)
	count := make([]int, bins)
	sums := make([][]float64, bins)
	for b := range sums {
		sums[b] = make([]float64, ch)
	}

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for b := range count {
				count[b] = 0
				for c := 0; c < ch; c++ {
					sums[b][c] = 0
				}
			}
			for dy := -size; dy <= size; dy++ {
				for dx := -size; dx <= size; dx++ {
					ny, nx := y+dy, x+dx
					if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
						continue
					}
					b := int(gray.Data[ny*cols+nx]) / dynRatio
					count[b]++
					for c := 0; c < ch; c++ {
						sums[b][c] += float64(img.Data[(ny*cols+nx)*ch+c])
					}
				}
			}
			best := 0
			for b := 1; b < bins; b++ {
				if count[b] > count[best] {
					best = b
				}
			}
			denom := float64(count[best])
			for c := 0; c < ch; c++ {
				out.Data[(y*cols+x)*ch+c] = clampU8(sums[best][c] / denom)
			}
		}
	}
	return out
}

// Cartoonify gives img a cartoon look by combining edge-preserving colour
// smoothing with bold dark outlines. The colours are flattened with
// [DomainTransformFilter] so regions become uniform, an edge mask is built by
// thresholding the local gradient magnitude, and the outline pixels are darkened
// over the flattened colours. The result resembles cel-shaded artwork.
//
// sigmaS and sigmaR control the colour smoothing (see [DomainTransformFilter]).
// img may be single- or three-channel; the output has the same shape. The
// original is not modified.
func Cartoonify(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: Cartoonify given an empty image")
	}
	smooth := DomainTransformFilter(img, RecursFilter, sigmaS, sigmaR)
	mag := gradientMagnitude(grayOf(img))
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	out := cv.NewMat(rows, cols, ch)
	// Edge strength scaled to a [0,1] darkening factor.
	const edgeScale = 24.0
	for i := 0; i < rows*cols; i++ {
		w := math.Exp(-mag[i] / edgeScale) // ~1 in flat areas, ~0 on strong edges
		for c := 0; c < ch; c++ {
			out.Data[i*ch+c] = clampU8(w * float64(smooth.Data[i*ch+c]))
		}
	}
	return out
}

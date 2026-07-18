package texture

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// textureSlope returns the least-squares slope of ys against xs. It returns 0
// if fewer than two points are given or the xs have no spread.
func textureSlope(xs, ys []float64) float64 {
	n := float64(len(xs))
	if n < 2 {
		return 0
	}
	var sx, sy, sxy, sxx float64
	for i := range xs {
		sx += xs[i]
		sy += ys[i]
		sxy += xs[i] * ys[i]
		sxx += xs[i] * xs[i]
	}
	den := n*sxx - sx*sx
	if den == 0 {
		return 0
	}
	return (n*sxy - sx*sy) / den
}

// BoxCountingDimension estimates the Minkowski–Bouligand (box-counting) fractal
// dimension of the foreground of img. The luminance is thresholded at
// threshold (pixels with luminance >= threshold are foreground); then for a
// sequence of box sizes s = 1, 2, 4, ... the image is tiled into s-by-s boxes
// and N(s), the number of boxes containing at least one foreground pixel, is
// counted. The dimension is the slope of log N(s) against log(1/s). A value
// near 1 indicates line-like structure and a value near 2 a plane-filling
// texture. It returns 0 if there is no foreground.
func BoxCountingDimension(img *cv.Mat, threshold uint8) float64 {
	textureRequire(img, "BoxCountingDimension")
	rows, cols := img.Rows, img.Cols
	luma := textureLuma(img)
	fg := make([]bool, len(luma))
	var any bool
	for i, v := range luma {
		if v >= threshold {
			fg[i] = true
			any = true
		}
	}
	if !any {
		return 0
	}
	minDim := rows
	if cols < minDim {
		minDim = cols
	}
	var xs, ys []float64
	for s := 1; s <= minDim; s *= 2 {
		nbx := (cols + s - 1) / s
		nby := (rows + s - 1) / s
		occupied := make([]bool, nbx*nby)
		var count int
		for y := 0; y < rows; y++ {
			by := y / s
			for x := 0; x < cols; x++ {
				if !fg[y*cols+x] {
					continue
				}
				bi := by*nbx + x/s
				if !occupied[bi] {
					occupied[bi] = true
					count++
				}
			}
		}
		if count > 0 {
			xs = append(xs, math.Log(1/float64(s)))
			ys = append(ys, math.Log(float64(count)))
		}
	}
	return textureSlope(xs, ys)
}

// DifferentialBoxCounting estimates the fractal dimension of a grayscale image
// with the Sarkar–Chaudhuri differential box-counting method, which treats the
// intensity surface as a 3-D landscape. For each grid size s the image is
// partitioned into s-by-s columns; within a column the number of stacked boxes
// of height s' = s*G/M (G = 256 gray levels, M = the larger image dimension)
// spanned by the local gray-level range is accumulated into N(s). The dimension
// is the slope of log N(s) against log(1/s) with s = 2..M/2. Unlike
// [BoxCountingDimension] it uses the full gray-level information and needs no
// threshold.
func DifferentialBoxCounting(img *cv.Mat) float64 {
	textureRequire(img, "DifferentialBoxCounting")
	rows, cols := img.Rows, img.Cols
	luma := textureLuma(img)
	M := rows
	if cols > M {
		M = cols
	}
	const G = 256
	minDim := rows
	if cols < minDim {
		minDim = cols
	}
	var xs, ys []float64
	for s := 2; s <= minDim/2; s++ {
		boxH := float64(s) * G / float64(M)
		if boxH < 1 {
			boxH = 1
		}
		nbx := (cols + s - 1) / s
		nby := (rows + s - 1) / s
		var nSum float64
		for by := 0; by < nby; by++ {
			y0 := by * s
			y1 := y0 + s
			if y1 > rows {
				y1 = rows
			}
			for bx := 0; bx < nbx; bx++ {
				x0 := bx * s
				x1 := x0 + s
				if x1 > cols {
					x1 = cols
				}
				gmin := 255
				gmax := 0
				for yy := y0; yy < y1; yy++ {
					row := yy * cols
					for xx := x0; xx < x1; xx++ {
						v := int(luma[row+xx])
						if v < gmin {
							gmin = v
						}
						if v > gmax {
							gmax = v
						}
					}
				}
				l := math.Floor(float64(gmin) / boxH)
				k := math.Floor(float64(gmax) / boxH)
				nSum += k - l + 1
			}
		}
		if nSum > 0 {
			xs = append(xs, math.Log(1/float64(s)))
			ys = append(ys, math.Log(nSum))
		}
	}
	return textureSlope(xs, ys)
}

// FractalDimension is a convenience alias for [DifferentialBoxCounting]: it
// returns a grayscale fractal-dimension estimate of img requiring no threshold,
// suitable as a single scalar texture-roughness descriptor.
func FractalDimension(img *cv.Mat) float64 {
	return DifferentialBoxCounting(img)
}

// Lacunarity computes the gliding-box lacunarity of the foreground of img at the
// given box size, a measure of how "gappy" or heterogeneous a fractal texture
// is (it distinguishes textures that share a fractal dimension but differ in
// clustering). The luminance is thresholded at threshold; a box-size-by-box-size
// window is slid over every position; and the lacunarity is
// E[m^2]/E[m]^2 where m is the foreground mass (pixel count) in the window.
// A value of 1 means perfectly uniform mass; larger values mean more clustered.
// boxSize must be >= 1 and no larger than either image dimension; it panics
// otherwise. It returns 0 if no window fits or the mean mass is 0.
func Lacunarity(img *cv.Mat, threshold uint8, boxSize int) float64 {
	textureRequire(img, "Lacunarity")
	rows, cols := img.Rows, img.Cols
	if boxSize < 1 || boxSize > rows || boxSize > cols {
		panic("texture: Lacunarity requires 1 <= boxSize <= min(rows,cols)")
	}
	luma := textureLuma(img)
	// Integral image of the binary foreground.
	fg := make([]float64, len(luma))
	for i, v := range luma {
		if v >= threshold {
			fg[i] = 1
		}
	}
	ii := textureIntegral(fg, rows, cols)
	w := cols + 1
	var sum, sum2, count float64
	for y0 := 0; y0+boxSize <= rows; y0++ {
		for x0 := 0; x0+boxSize <= cols; x0++ {
			y1 := y0 + boxSize
			x1 := x0 + boxSize
			m := ii[y1*w+x1] - ii[y0*w+x1] - ii[y1*w+x0] + ii[y0*w+x0]
			sum += m
			sum2 += m * m
			count++
		}
	}
	if count == 0 {
		return 0
	}
	mean := sum / count
	if mean == 0 {
		return 0
	}
	meanSq := sum2 / count
	return meanSq / (mean * mean)
}

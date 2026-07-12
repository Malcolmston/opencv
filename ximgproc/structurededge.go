package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// StructuredEdgeDetectionLite estimates a per-pixel edge probability map for img
// and returns it as a [cv.FloatMat] with values in [0,1]. It is a lightweight,
// training-free stand-in for OpenCV's StructuredEdgeDetection (which relies on a
// pre-trained random forest): instead of a learned model it combines multiscale
// oriented gradient energy with gradient-direction non-maximum suppression to
// produce thin, well-localised edge responses.
//
// The detector computes the Sobel gradient at the native resolution and at a
// half-resolution copy (up-sampled back), sums their magnitudes to capture both
// fine and coarse structure, thins the result by suppressing any pixel that is
// not a local maximum along its gradient direction, and normalises the surviving
// energy to [0,1]. The output is suitable both for visualisation and as the edge
// input to [EdgeBoxes].
//
// img may be 1- or 3-channel; colour is reduced to luma. The map is
// deterministic. This is an approximation, not a port of the trained model — see
// the package documentation.
func StructuredEdgeDetectionLite(img *cv.Mat) *cv.FloatMat {
	rows, cols := img.Rows, img.Cols
	g := channelPlane(toGray(img), 0)

	mag, ori := gradientMagOri(g, rows, cols)

	// Coarse scale: blur, subsample by 2, gradient, and read back bilinearly.
	blurred := gaussianBlurFloat(g, rows, cols, 1.0)
	crows, ccols := (rows+1)/2, (cols+1)/2
	small := make([]float64, crows*ccols)
	for y := 0; y < crows; y++ {
		for x := 0; x < ccols; x++ {
			small[y*ccols+x] = blurred[reflect(2*y, rows)*cols+reflect(2*x, cols)]
		}
	}
	smag, _ := gradientMagOri(small, crows, ccols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			sy := y / 2
			sx := x / 2
			if sy >= crows {
				sy = crows - 1
			}
			if sx >= ccols {
				sx = ccols - 1
			}
			mag[y*cols+x] += 0.5 * smag[sy*ccols+sx]
		}
	}

	// Non-maximum suppression along the gradient direction.
	thin := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			c := mag[i]
			dx := math.Cos(ori[i])
			dy := math.Sin(ori[i])
			f := bilinearSample(mag, rows, cols, float64(x)+dx, float64(y)+dy)
			b := bilinearSample(mag, rows, cols, float64(x)-dx, float64(y)-dy)
			if c >= f && c >= b {
				thin[i] = c
			}
		}
	}

	// Normalise to [0,1].
	var maxV float64
	for _, v := range thin {
		if v > maxV {
			maxV = v
		}
	}
	out := cv.NewFloatMat(rows, cols)
	if maxV > 0 {
		for i, v := range thin {
			out.Data[i] = v / maxV
		}
	}
	return out
}

// gradientMagOri returns the Sobel gradient magnitude and orientation (radians)
// of a row-major plane using reflect-101 borders.
func gradientMagOri(p []float64, rows, cols int) (mag, ori []float64) {
	mag = make([]float64, rows*cols)
	ori = make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var gx, gy float64
			for dy := -1; dy <= 1; dy++ {
				yy := reflect(y+dy, rows)
				for dx := -1; dx <= 1; dx++ {
					xx := reflect(x+dx, cols)
					v := p[yy*cols+xx]
					gx += sobelKX[dy+1][dx+1] * v
					gy += sobelKY[dy+1][dx+1] * v
				}
			}
			mag[y*cols+x] = math.Hypot(gx, gy)
			ori[y*cols+x] = math.Atan2(gy, gx)
		}
	}
	return mag, ori
}

var sobelKX = [3][3]float64{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}
var sobelKY = [3][3]float64{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}

// bilinearSample reads a row-major plane at fractional (x,y) with clamped
// borders, returning 0 outside a small guard band via clamping to the edge.
func bilinearSample(p []float64, rows, cols int, x, y float64) float64 {
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x > float64(cols-1) {
		x = float64(cols - 1)
	}
	if y > float64(rows-1) {
		y = float64(rows - 1)
	}
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 > cols-1 {
		x1 = cols - 1
	}
	if y1 > rows-1 {
		y1 = rows - 1
	}
	fx := x - float64(x0)
	fy := y - float64(y0)
	v00 := p[y0*cols+x0]
	v01 := p[y0*cols+x1]
	v10 := p[y1*cols+x0]
	v11 := p[y1*cols+x1]
	top := v00 + fx*(v01-v00)
	bot := v10 + fx*(v11-v10)
	return top + fy*(bot-top)
}

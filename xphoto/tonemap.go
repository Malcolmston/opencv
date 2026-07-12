package xphoto

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TonemapDurand implements Durand and Dorsey's fast bilateral-filtering tone
// mapping operator, porting cv::xphoto::TonemapDurand (historically
// cv::TonemapDurand). It decomposes the log-luminance of an image into a
// piecewise-smooth base layer (obtained with an edge-preserving bilateral
// filter) and a detail layer, compresses the contrast of the base layer only,
// and recombines. Because the bilateral filter respects edges, large-scale
// contrast is reduced without haloing while fine detail is preserved, which is
// exactly what makes a high-dynamic-range scene displayable.
//
// The Go port operates on the package's 8-bit [cv.Mat], treating the input as a
// low-dynamic-range radiance proxy: it is therefore a local-contrast /
// detail-preserving operator rather than a true HDR-to-LDR mapper, but the
// algorithm structure is faithful. Construct with [NewTonemapDurand].
type TonemapDurand struct {
	// Gamma is the gamma applied to the final result (1.0 = none). Values <= 0
	// default to 1.
	Gamma float64
	// Contrast is the target contrast (in log10 units) the base layer is
	// compressed to; smaller values flatten the image more. Values <= 0 default
	// to 4.
	Contrast float64
	// Saturation scales colour saturation of the result; 1 keeps it, <1
	// desaturates, >1 boosts. Values < 0 default to 1.
	Saturation float64
	// SigmaSpace is the spatial standard deviation (in pixels) of the bilateral
	// filter. Values <= 0 default to 2.
	SigmaSpace float64
	// SigmaColor is the range (log-luminance) standard deviation of the
	// bilateral filter. Values <= 0 default to 2.
	SigmaColor float64
}

// NewTonemapDurand returns a TonemapDurand with OpenCV's default parameters:
// gamma 1.0, contrast 4.0, saturation 1.0, sigmaSpace 2.0 and sigmaColor 2.0.
func NewTonemapDurand() *TonemapDurand {
	return &TonemapDurand{Gamma: 1.0, Contrast: 4.0, Saturation: 1.0, SigmaSpace: 2.0, SigmaColor: 2.0}
}

// Process tone-maps src and returns a new image of the same shape. src must be a
// three-channel RGB image; the input is not modified.
func (t *TonemapDurand) Process(src *cv.Mat) *cv.Mat {
	requireNonEmpty(src, "TonemapDurand.Process")
	requireChannels(src, 3, "TonemapDurand.Process")

	gamma := t.Gamma
	if gamma <= 0 {
		gamma = 1
	}
	contrast := t.Contrast
	if contrast <= 0 {
		contrast = 4
	}
	saturation := t.Saturation
	if saturation < 0 {
		saturation = 1
	}
	sigmaSpace := t.SigmaSpace
	if sigmaSpace <= 0 {
		sigmaSpace = 2
	}
	sigmaColor := t.SigmaColor
	if sigmaColor <= 0 {
		sigmaColor = 2
	}

	rows, cols := src.Rows, src.Cols
	n := rows * cols

	// Luminance and log-luminance planes (input normalised to [0,1]).
	const eps = 1e-4
	lum := make([]float64, n)
	logLum := make([]float64, n)
	for i := 0; i < n; i++ {
		r := float64(src.Data[i*3+0]) / 255.0
		g := float64(src.Data[i*3+1]) / 255.0
		b := float64(src.Data[i*3+2]) / 255.0
		l := luma(r, g, b)
		if l < eps {
			l = eps
		}
		lum[i] = l
		logLum[i] = math.Log10(l)
	}

	// Base layer via bilateral filter of log-luminance; detail is the residual.
	base := bilateralFilterPlane(logLum, rows, cols, sigmaSpace, sigmaColor)
	minB, maxB := base[0], base[0]
	for i := 1; i < n; i++ {
		if base[i] < minB {
			minB = base[i]
		}
		if base[i] > maxB {
			maxB = base[i]
		}
	}
	rangeB := maxB - minB
	if rangeB < 1e-6 {
		rangeB = 1e-6
	}
	compress := contrast / rangeB

	dst := cv.NewMat(rows, cols, 3)
	invGamma := 1.0 / gamma
	for i := 0; i < n; i++ {
		detail := logLum[i] - base[i]
		// Compress the base layer, anchoring the maximum at 0 so highlights map
		// to ~1 after exponentiation.
		newLogLum := (base[i]-maxB)*compress + detail
		newLum := math.Pow(10, newLogLum)
		for c := 0; c < 3; c++ {
			in := float64(src.Data[i*3+c]) / 255.0
			// Recolour from the ratio to luminance, with saturation control.
			ratio := math.Pow(in/lum[i], saturation)
			out := ratio * newLum
			out = math.Pow(math.Max(out, 0), invGamma)
			dst.Data[i*3+c] = clampU8(out * 255.0)
		}
	}
	return dst
}

// bilateralFilterPlane applies an edge-preserving bilateral filter to a single
// float plane. sigmaSpace is the spatial standard deviation (pixels) and
// sigmaColor the range standard deviation (same units as the plane). The window
// radius is ceil(2*sigmaSpace).
func bilateralFilterPlane(plane []float64, rows, cols int, sigmaSpace, sigmaColor float64) []float64 {
	radius := int(math.Ceil(2 * sigmaSpace))
	if radius < 1 {
		radius = 1
	}
	// Precompute the spatial Gaussian kernel.
	spatial := make([][]float64, 2*radius+1)
	for dy := -radius; dy <= radius; dy++ {
		row := make([]float64, 2*radius+1)
		for dx := -radius; dx <= radius; dx++ {
			row[dx+radius] = math.Exp(-float64(dx*dx+dy*dy) / (2 * sigmaSpace * sigmaSpace))
		}
		spatial[dy+radius] = row
	}
	colorDen := 2 * sigmaColor * sigmaColor
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			center := plane[y*cols+x]
			var wsum, vsum float64
			for dy := -radius; dy <= radius; dy++ {
				yy := y + dy
				if yy < 0 {
					yy = 0
				} else if yy >= rows {
					yy = rows - 1
				}
				srow := spatial[dy+radius]
				for dx := -radius; dx <= radius; dx++ {
					xx := x + dx
					if xx < 0 {
						xx = 0
					} else if xx >= cols {
						xx = cols - 1
					}
					v := plane[yy*cols+xx]
					diff := v - center
					w := srow[dx+radius] * math.Exp(-diff*diff/colorDen)
					wsum += w
					vsum += w * v
				}
			}
			out[y*cols+x] = vsum / wsum
		}
	}
	return out
}

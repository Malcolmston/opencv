package hdr

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// MergeDebevec merges a bracket of LDR exposures into a linear radiance map
// using the calibrated camera response and the per-image exposure times. For
// each pixel and channel it forms the hat-weighted average of the
// response-linearised log radiance minus the log exposure time, then
// exponentiates, exactly as in Debevec & Malik.
//
// resp must have been calibrated for the same channel count as the images (see
// [CalibrateDebevec] / [CalibrateRobertson]); pass the result of
// [LinearResponse] when the camera is known to be linear. The recovered
// radiance is defined up to the same global scale as resp, so radiance ratios
// between pixels are meaningful while absolute values are not.
func MergeDebevec(images []*cv.Mat, times []float64, resp *CameraResponse) (*Radiance, error) {
	if err := validateStack(images, times); err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, errors.New("hdr: nil camera response")
	}
	rows, cols, ch := images[0].Rows, images[0].Cols, images[0].Channels
	if resp.Channels != ch {
		return nil, errors.New("hdr: response channel count does not match images")
	}
	logTimes := make([]float64, len(times))
	for j, t := range times {
		logTimes[j] = math.Log(t)
	}
	// Precompute the log response per channel.
	logCurve := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		logCurve[c] = resp.logCurve(c)
	}

	out := NewRadiance(rows, cols, ch)
	nImg := len(images)
	total := rows * cols
	for p := 0; p < total; p++ {
		for c := 0; c < ch; c++ {
			var num, den float64
			for j := 0; j < nImg; j++ {
				z := int(images[j].Data[p*ch+c])
				w := hat(z)
				num += w * (logCurve[c][z] - logTimes[j])
				den += w
			}
			var e float64
			if den > 0 {
				e = math.Exp(num / den)
			}
			out.Data[p*ch+c] = e
		}
	}
	return out, nil
}

// MergeMertensParams configures [MergeMertens]. The three exponents weight the
// contrast, saturation and well-exposedness quality measures; the defaults
// (all 1) match OpenCV. A non-positive value in [NewMergeMertens] selects the
// default for that field.
type MergeMertensParams struct {
	// ContrastWeight raises the contrast measure to this power.
	ContrastWeight float64
	// SaturationWeight raises the saturation measure to this power.
	SaturationWeight float64
	// ExposednessWeight raises the well-exposedness measure to this power.
	ExposednessWeight float64
}

// NewMergeMertensParams returns the default Mertens fusion parameters (all
// exponents equal to 1).
func NewMergeMertensParams() MergeMertensParams {
	return MergeMertensParams{ContrastWeight: 1, SaturationWeight: 1, ExposednessWeight: 1}
}

// MergeMertens performs Mertens–Kautz–Van Reeth exposure fusion: it blends the
// LDR stack directly into a single well-exposed, displayable 8-bit image
// without any camera response or exposure times. Each source pixel is weighted
// by the product of three quality measures — contrast (Laplacian magnitude),
// saturation (per-pixel channel standard deviation) and well-exposedness
// (closeness of every channel to mid-grey) — and the weighted images are
// combined with a Laplacian-pyramid blend so seams do not appear.
//
// All images must share dimensions and channel count. The result is a [cv.Mat]
// with samples in [0,255]; conceptually it is a [0,1] fusion scaled to 8 bits.
func MergeMertens(images []*cv.Mat, params MergeMertensParams) (*cv.Mat, error) {
	if len(images) < 2 {
		return nil, errors.New("hdr: need at least two exposures")
	}
	rows, cols, ch := images[0].Rows, images[0].Cols, images[0].Channels
	for _, m := range images {
		if m == nil || m.Empty() {
			return nil, errors.New("hdr: nil or empty image in stack")
		}
		if m.Rows != rows || m.Cols != cols || m.Channels != ch {
			return nil, errors.New("hdr: all images must share dimensions and channel count")
		}
	}
	wc := params.ContrastWeight
	ws := params.SaturationWeight
	we := params.ExposednessWeight
	if wc <= 0 {
		wc = 1
	}
	if ws <= 0 {
		ws = 1
	}
	if we <= 0 {
		we = 1
	}

	n := len(images)
	// Normalised float channels per image and the weight map per image.
	chans := make([][]*plane, n) // chans[i][c]
	weights := make([]*plane, n)
	weightSum := newPlane(rows, cols)
	for i, m := range images {
		chans[i] = toFloatPlanes(m)
		gray := grayPlane(chans[i])
		contrast := gray.laplacianAbs()
		w := newPlane(rows, cols)
		for p := 0; p < rows*cols; p++ {
			ct := math.Pow(contrast.data[p]+1e-12, wc)
			sat := math.Pow(saturationAt(chans[i], p)+1e-12, ws)
			exp := math.Pow(exposednessAt(chans[i], p)+1e-12, we)
			val := ct * sat * exp
			w.data[p] = val
			weightSum.data[p] += val
		}
		weights[i] = w
	}
	// Normalise weights so they sum to one per pixel.
	for i := 0; i < n; i++ {
		for p := 0; p < rows*cols; p++ {
			if weightSum.data[p] > 0 {
				weights[i].data[p] /= weightSum.data[p]
			} else {
				weights[i].data[p] = 1.0 / float64(n)
			}
		}
	}

	levels := pyramidLevels(rows, cols)
	// Accumulate the blended Laplacian pyramid, one pyramid per channel.
	blendedCh := make([][]*plane, ch)
	for c := 0; c < ch; c++ {
		blendedCh[c] = make([]*plane, levels)
		for l := 0; l < levels; l++ {
			blendedCh[c][l] = newPlane(pyrRows(rows, l), pyrCols(cols, l))
		}
	}
	for i := 0; i < n; i++ {
		wPyr := gaussianPyramid(weights[i], levels)
		for c := 0; c < ch; c++ {
			lPyr := laplacianPyramid(chans[i][c], levels)
			for l := 0; l < levels; l++ {
				dst := blendedCh[c][l].data
				src := lPyr[l].data
				gw := wPyr[l].data
				for p := range dst {
					dst[p] += gw[p] * src[p]
				}
			}
		}
	}

	out := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		collapsed := collapsePyramid(blendedCh[c])
		for p := 0; p < rows*cols; p++ {
			out.Data[p*ch+c] = clamp8(clamp01(collapsed.data[p]) * 255)
		}
	}
	return out, nil
}

// toFloatPlanes splits a Mat into per-channel float planes scaled to [0,1].
func toFloatPlanes(m *cv.Mat) []*plane {
	planes := make([]*plane, m.Channels)
	total := m.Rows * m.Cols
	for c := 0; c < m.Channels; c++ {
		p := newPlane(m.Rows, m.Cols)
		for i := 0; i < total; i++ {
			p.data[i] = float64(m.Data[i*m.Channels+c]) / 255.0
		}
		planes[c] = p
	}
	return planes
}

// grayPlane returns the mean of the channel planes (a simple luma proxy in
// [0,1]) used for the contrast measure.
func grayPlane(planes []*plane) *plane {
	if len(planes) == 1 {
		return planes[0].clone()
	}
	g := newPlane(planes[0].rows, planes[0].cols)
	for p := range g.data {
		var s float64
		for c := range planes {
			s += planes[c].data[p]
		}
		g.data[p] = s / float64(len(planes))
	}
	return g
}

// saturationAt returns the standard deviation across channels at pixel p.
func saturationAt(planes []*plane, p int) float64 {
	if len(planes) == 1 {
		return 0
	}
	var mean float64
	for c := range planes {
		mean += planes[c].data[p]
	}
	mean /= float64(len(planes))
	var v float64
	for c := range planes {
		d := planes[c].data[p] - mean
		v += d * d
	}
	return math.Sqrt(v / float64(len(planes)))
}

// exposednessAt returns the product over channels of a Gaussian centred on
// mid-grey (0.5), the Mertens well-exposedness measure.
func exposednessAt(planes []*plane, p int) float64 {
	const sigma = 0.2
	prod := 1.0
	for c := range planes {
		d := planes[c].data[p] - 0.5
		prod *= math.Exp(-(d * d) / (2 * sigma * sigma))
	}
	return prod
}

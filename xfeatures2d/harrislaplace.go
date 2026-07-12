package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// HarrisLaplace detects scale-adapted corners, a simplified port of OpenCV's
// cv::xfeatures2d::HarrisLaplaceFeatureDetector.
//
// The image is smoothed at a geometric series of scales. At each scale the
// Harris corner response (cv.CornerHarris) locates spatially strong corners; a
// corner is retained only if the scale-normalised Laplacian (sigma²·∇²) attains
// a local maximum in magnitude at that scale, which selects the characteristic
// scale of the feature. Surviving points are reported with a Size proportional
// to their selected scale.
type HarrisLaplace struct {
	// NumOctaves is the number of scale levels examined.
	NumOctaves int
	// InitSigma is the smallest smoothing standard deviation.
	InitSigma float64
	// ScaleFactor multiplies the sigma from one level to the next (> 1).
	ScaleFactor float64
	// HarrisThreshold is the minimum Harris response for a spatial corner.
	HarrisThreshold float64
	// HarrisK is the Harris free parameter.
	HarrisK float64
}

// NewHarrisLaplace returns a HarrisLaplace detector with sensible defaults.
func NewHarrisLaplace() *HarrisLaplace {
	return &HarrisLaplace{
		NumOctaves:      4,
		InitSigma:       1.0,
		ScaleFactor:     1.4,
		HarrisThreshold: 1e6,
		HarrisK:         0.04,
	}
}

// Detect finds scale-adapted Harris corners in img and returns them as
// keypoints. Each keypoint's Size is proportional to its selected scale, its
// Response is the Harris response, and Angle is -1. img may be single- or
// three-channel; a colour image is converted to gray.
func (h *HarrisLaplace) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	nOct := h.NumOctaves
	if nOct < 1 {
		nOct = 1
	}
	factor := h.ScaleFactor
	if factor <= 1 {
		factor = 1.4
	}

	sigmas := make([]float64, nOct)
	harris := make([]*cv.FloatMat, nOct)
	lap := make([][]float64, nOct)
	sigma := h.InitSigma
	if sigma <= 0 {
		sigma = 1
	}
	for o := 0; o < nOct; o++ {
		sigmas[o] = sigma
		ksize := gaussianKSize(sigma)
		blurred := cv.GaussianBlur(gray, ksize, sigma)
		harris[o] = cv.CornerHarris(blurred, 3, 3, h.HarrisK)
		// Scale-normalised Laplacian magnitude via the Sobel second derivatives.
		xx := cv.SobelFloat(blurred, 2, 0, 3)[0]
		yy := cv.SobelFloat(blurred, 0, 2, 3)[0]
		l := make([]float64, rows*cols)
		norm := sigma * sigma
		for i := range l {
			l[i] = math.Abs(norm * (xx[i] + yy[i]))
		}
		lap[o] = l
		sigma *= factor
	}

	var kps []KeyPoint
	for o := 0; o < nOct; o++ {
		hr := harris[o]
		for y := 1; y < rows-1; y++ {
			for x := 1; x < cols-1; x++ {
				v := hr.At(y, x)
				if v < h.HarrisThreshold {
					continue
				}
				// Spatial non-maximum suppression of the Harris response.
				if !isHarrisMax(hr, x, y) {
					continue
				}
				// Scale selection: the Laplacian magnitude must peak at this
				// octave relative to its neighbours in scale.
				lv := lap[o][y*cols+x]
				if o > 0 && lap[o-1][y*cols+x] > lv {
					continue
				}
				if o < nOct-1 && lap[o+1][y*cols+x] > lv {
					continue
				}
				kps = append(kps, KeyPoint{
					Pt:       cv.Point{X: x, Y: y},
					Size:     sigmas[o] * 2,
					Angle:    -1,
					Response: v,
				})
			}
		}
	}
	return kps
}

// isHarrisMax reports whether the Harris response at (x, y) is a strict maximum
// in its 3×3 neighbourhood.
func isHarrisMax(hr *cv.FloatMat, x, y int) bool {
	v := hr.At(y, x)
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			if hr.At(y+dy, x+dx) > v {
				return false
			}
		}
	}
	return true
}

// gaussianKSize returns an odd Gaussian kernel size that covers ±3σ.
func gaussianKSize(sigma float64) int {
	k := int(math.Ceil(sigma*3))*2 + 1
	if k < 3 {
		k = 3
	}
	return k
}

package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// StarDetector detects center–surround extrema, approximating OpenCV's
// cv::xfeatures2d::StarDetector (the CenSurE detector).
//
// CenSurE approximates the scale-normalised Laplacian of Gaussian with a
// bi-level center–surround filter: a bright inner region minus the surrounding
// annulus. This implementation uses nested square (box) filters evaluated in
// constant time through an integral image — the simplest CenSurE bi-level
// filter, a fast stand-in for the octagon/hexagon filters of the original. The
// response is computed at a range of scales; a pixel that is a local maximum of
// the absolute response over both space and scale, and exceeds ResponseThreshold,
// becomes a keypoint whose Size encodes the selected scale.
type StarDetector struct {
	// MaxSize bounds the largest inner box half-size (scale) considered.
	MaxSize int
	// ResponseThreshold is the minimum absolute center–surround response for a
	// pixel to be reported.
	ResponseThreshold float64
}

// NewStarDetector returns a StarDetector with a default maximum scale and
// response threshold.
func NewStarDetector() *StarDetector {
	return &StarDetector{MaxSize: 8, ResponseThreshold: 12}
}

// starScales lists the inner box half-sizes (in pixels) at which the response is
// evaluated. Larger scales respond to larger blobs.
func (s *StarDetector) scales() []int {
	maxSize := s.MaxSize
	if maxSize < 2 {
		maxSize = 2
	}
	var out []int
	for r := 1; r <= maxSize; r++ {
		out = append(out, r)
	}
	return out
}

// Detect finds center–surround extrema in img and returns them as keypoints.
// Each keypoint's Response is the absolute normalised response, Size is the
// selected scale's outer diameter and Angle is -1. img may be single- or
// three-channel; a colour image is converted to gray.
func (s *StarDetector) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	it := newIntegral(gray)
	scales := s.scales()
	ns := len(scales)

	// responses[k] is the normalised center–surround response at scale k.
	responses := make([][]float64, ns)
	for k := range responses {
		responses[k] = make([]float64, rows*cols)
		inner := scales[k]
		outer := inner * 2
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				innerMean := it.boxMean(x, y, inner)
				outerMean := it.boxMean(x, y, outer)
				innerArea := float64((2*inner + 1) * (2*inner + 1))
				outerArea := float64((2*outer + 1) * (2*outer + 1))
				// Surround = outer box minus inner box.
				surroundArea := outerArea - innerArea
				surroundMean := (outerMean*outerArea - innerMean*innerArea) / surroundArea
				responses[k][y*cols+x] = innerMean - surroundMean
			}
		}
	}

	var kps []KeyPoint
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Pick the scale with the largest absolute response at this pixel.
			bestK := 0
			bestAbs := math.Abs(responses[0][y*cols+x])
			for k := 1; k < ns; k++ {
				a := math.Abs(responses[k][y*cols+x])
				if a > bestAbs {
					bestAbs = a
					bestK = k
				}
			}
			if bestAbs < s.ResponseThreshold {
				continue
			}
			// Border margin: the outer box must stay inside the image.
			outer := scales[bestK] * 2
			if x < outer || y < outer || x+outer >= cols || y+outer >= rows {
				continue
			}
			if !isSpatialScaleMax(responses, bestK, x, y, cols, rows, bestAbs) {
				continue
			}
			kps = append(kps, KeyPoint{
				Pt:       cv.Point{X: x, Y: y},
				Size:     float64(2*outer + 1),
				Angle:    -1,
				Response: bestAbs,
			})
		}
	}
	return kps
}

// isSpatialScaleMax reports whether |response| at (x,y,k) is at least as large as
// every neighbour in the 3×3×3 space-scale neighbourhood, and strictly larger
// than at least one, making it a strict local extremum.
func isSpatialScaleMax(responses [][]float64, k, x, y, cols, rows int, val float64) bool {
	ns := len(responses)
	strictlyGreater := false
	for dk := -1; dk <= 1; dk++ {
		kk := k + dk
		if kk < 0 || kk >= ns {
			continue
		}
		for dy := -1; dy <= 1; dy++ {
			ny := y + dy
			if ny < 0 || ny >= rows {
				continue
			}
			for dx := -1; dx <= 1; dx++ {
				nx := x + dx
				if nx < 0 || nx >= cols {
					continue
				}
				if dk == 0 && dx == 0 && dy == 0 {
					continue
				}
				a := math.Abs(responses[kk][ny*cols+nx])
				if a > val {
					return false
				}
				if a < val {
					strictlyGreater = true
				}
			}
		}
	}
	return strictlyGreater
}

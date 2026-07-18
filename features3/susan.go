package features3

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// features3circularMask returns the (dx, dy) offsets of a filled circular mask
// of the given radius, together with the offset count.
func features3circularMask(radius int) [][2]int {
	var mask [][2]int
	r2 := radius * radius
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			if dx*dx+dy*dy <= r2 {
				mask = append(mask, [2]int{dx, dy})
			}
		}
	}
	return mask
}

// SUSANResponse computes the SUSAN (Smallest Univalue Segment Assimilating
// Nucleus) corner response of an image. For each pixel it measures the USAN
// area — the smoothly-weighted count of nearby pixels within intensity t of the
// nucleus using the exponential similarity exp(-((I-I0)/t)^6) over a circular
// mask of the given radius — and returns geometricThreshold-USAN where that is
// positive, else 0. Small USAN areas (corners) give large responses. When
// geometricThreshold <= 0 it defaults to half the mask area. Colour input is
// converted to grayscale first.
func SUSANResponse(img *cv.Mat, radius int, t, geometricThreshold float64) *cv.FloatMat {
	if radius < 1 {
		radius = 3
	}
	if t <= 0 {
		t = 27
	}
	g := features3ToGray(img)
	mask := features3circularMask(radius)
	if geometricThreshold <= 0 {
		geometricThreshold = 0.5 * float64(len(mask))
	}
	res := cv.NewFloatMat(g.Rows, g.Cols)
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			i0 := g.at(x, y)
			var usan float64
			for _, o := range mask {
				d := (g.atClamped(x+o[0], y+o[1]) - i0) / t
				d2 := d * d
				usan += math.Exp(-(d2 * d2 * d2))
			}
			if usan < geometricThreshold {
				res.Data[y*g.Cols+x] = geometricThreshold - usan
			}
		}
	}
	return res
}

// SUSANCorners detects corners with the SUSAN operator. It computes
// [SUSANResponse], applies 3×3 non-maximum suppression and returns the local
// maxima whose response is at least qualityLevel times the maximum response, as
// keypoints sorted by descending response. When maxCorners > 0 only the
// strongest maxCorners are returned. Colour input is converted to grayscale.
func SUSANCorners(img *cv.Mat, radius int, t, geometricThreshold, qualityLevel float64, maxCorners int) []KeyPoint {
	resp := SUSANResponse(img, radius, t, geometricThreshold)
	return features3peaks(resp, qualityLevel, maxCorners)
}

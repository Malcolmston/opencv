package features3

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// features3circle16 holds the radius-3 Bresenham circle offsets used by FAST and
// AGAST, listed clockwise from the top pixel.
var features3circle16 = [16][2]int{
	{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
	{0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3},
}

// IsFASTCorner reports whether pixel (x, y) passes the FAST-9 segment test with
// the given intensity threshold: at least nine contiguous pixels on the radius-3
// Bresenham circle are all brighter than centre+threshold or all darker than
// centre-threshold. Colour input is converted to grayscale first. Pixels within
// three of the border always return false.
func IsFASTCorner(img *cv.Mat, x, y, threshold int) bool {
	g := features3ToGray(img)
	if x < 3 || y < 3 || x >= g.Cols-3 || y >= g.Rows-3 {
		return false
	}
	return features3isFast(g, x, y, threshold)
}

// features3isFast is the core FAST-9 test on the working grayscale buffer.
func features3isFast(g *features3gray, x, y, threshold int) bool {
	center := g.at(x, y)
	hi := center + float64(threshold)
	lo := center - float64(threshold)
	var ring [16]float64
	for i, o := range features3circle16 {
		ring[i] = g.at(x+o[0], y+o[1])
	}
	// Look for 9 contiguous brighter or 9 contiguous darker pixels, allowing the
	// arc to wrap around the 16-pixel ring.
	brighter := 0
	darker := 0
	for i := 0; i < 16+9; i++ {
		v := ring[i%16]
		if v > hi {
			brighter++
			darker = 0
		} else if v < lo {
			darker++
			brighter = 0
		} else {
			brighter = 0
			darker = 0
		}
		if brighter >= 9 || darker >= 9 {
			return true
		}
	}
	return false
}

// FASTScore returns the FAST cornerness score at (x, y): the sum of absolute
// differences between the centre and the sixteen circle pixels. It is 0 for
// pixels within three of the border. Colour input is converted to grayscale.
func FASTScore(img *cv.Mat, x, y int) float64 {
	g := features3ToGray(img)
	return features3fastScore(g, x, y)
}

// features3fastScore computes the summed absolute contrast score on the working
// buffer.
func features3fastScore(g *features3gray, x, y int) float64 {
	if x < 3 || y < 3 || x >= g.Cols-3 || y >= g.Rows-3 {
		return 0
	}
	center := g.at(x, y)
	var s float64
	for _, o := range features3circle16 {
		s += math.Abs(g.at(x+o[0], y+o[1]) - center)
	}
	return s
}

// FASTKeyPoints detects corners with the FAST-9 accelerated segment test. Every
// interior pixel passing [IsFASTCorner] with the given threshold becomes a
// keypoint whose Response is its [FASTScore]. When nonmaxSuppression is true,
// keypoints that are not a strict maximum of the score among their eight
// neighbours are removed. Colour input is converted to grayscale first. Results
// are sorted by descending response.
func FASTKeyPoints(img *cv.Mat, threshold int, nonmaxSuppression bool) []KeyPoint {
	g := features3ToGray(img)
	rows, cols := g.Rows, g.Cols
	score := make([]float64, rows*cols)
	corner := make([]bool, rows*cols)
	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			if features3isFast(g, x, y, threshold) {
				corner[y*cols+x] = true
				score[y*cols+x] = features3fastScore(g, x, y)
			}
		}
	}
	var kps []KeyPoint
	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			i := y*cols + x
			if !corner[i] {
				continue
			}
			if nonmaxSuppression {
				v := score[i]
				isMax := true
				for dy := -1; dy <= 1 && isMax; dy++ {
					for dx := -1; dx <= 1; dx++ {
						if dx == 0 && dy == 0 {
							continue
						}
						if corner[(y+dy)*cols+(x+dx)] && score[(y+dy)*cols+(x+dx)] > v {
							isMax = false
							break
						}
					}
				}
				if !isMax {
					continue
				}
			}
			kps = append(kps, NewKeyPoint(float64(x), float64(y), score[i]))
		}
	}
	SortKeyPointsByResponse(kps)
	return kps
}

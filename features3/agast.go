package features3

import cv "github.com/malcolmston/opencv"

// features3isArc reports whether the sixteen circle samples in ring contain at
// least contiguity consecutive values all above center+t (brighter) or all
// below center-t (darker), allowing the arc to wrap around the ring.
func features3isArc(ring *[16]float64, center float64, t, contiguity int) bool {
	hi := center + float64(t)
	lo := center - float64(t)
	brighter := 0
	darker := 0
	for i := 0; i < 16+contiguity; i++ {
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
		if brighter >= contiguity || darker >= contiguity {
			return true
		}
	}
	return false
}

// features3ring loads the sixteen radius-3 circle samples around (x, y).
func features3ring(g *features3gray, x, y int) [16]float64 {
	var ring [16]float64
	for i, o := range features3circle16 {
		ring[i] = g.at(x+o[0], y+o[1])
	}
	return ring
}

// IsAGASTCorner reports whether pixel (x, y) is a corner under the AGAST
// (Adaptive and Generic Accelerated Segment Test) OAST-16 criterion: at least
// nine contiguous pixels on the radius-3 circle are all brighter than
// centre+threshold or all darker than centre-threshold. It shares the FAST
// circle but is the entry point for the AGAST detector and its adaptive score.
// Colour input is converted to grayscale first; border pixels return false.
func IsAGASTCorner(img *cv.Mat, x, y, threshold int) bool {
	g := features3ToGray(img)
	if x < 3 || y < 3 || x >= g.Cols-3 || y >= g.Rows-3 {
		return false
	}
	ring := features3ring(g, x, y)
	return features3isArc(&ring, g.at(x, y), threshold, 9)
}

// AGASTScore returns the adaptive AGAST cornerness score at (x, y): the largest
// intensity threshold for which the pixel still satisfies the OAST-16 corner
// test, found by binary search over [threshold, 255]. A higher score means a
// more robust corner. It is 0 for border pixels or points that are not corners
// at the base threshold. Colour input is converted to grayscale first.
func AGASTScore(img *cv.Mat, x, y, threshold int) int {
	g := features3ToGray(img)
	return features3agastScore(g, x, y, threshold)
}

// features3agastScore computes the adaptive score on the working buffer.
func features3agastScore(g *features3gray, x, y, threshold int) int {
	if x < 3 || y < 3 || x >= g.Cols-3 || y >= g.Rows-3 {
		return 0
	}
	ring := features3ring(g, x, y)
	center := g.at(x, y)
	if !features3isArc(&ring, center, threshold, 9) {
		return 0
	}
	lo, hi := threshold, 255
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if features3isArc(&ring, center, mid, 9) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// AGASTKeyPoints detects corners with the AGAST OAST-16 test. Every interior
// pixel passing [IsAGASTCorner] at the given threshold becomes a keypoint whose
// Response is its adaptive [AGASTScore]. When nonmaxSuppression is true,
// keypoints that are not a strict maximum of the score among their eight
// neighbours are removed. Colour input is converted to grayscale first. Results
// are sorted by descending response.
func AGASTKeyPoints(img *cv.Mat, threshold int, nonmaxSuppression bool) []KeyPoint {
	g := features3ToGray(img)
	rows, cols := g.Rows, g.Cols
	score := make([]float64, rows*cols)
	corner := make([]bool, rows*cols)
	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			s := features3agastScore(g, x, y, threshold)
			if s > 0 {
				corner[y*cols+x] = true
				score[y*cols+x] = float64(s)
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

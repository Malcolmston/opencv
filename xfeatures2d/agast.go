package xfeatures2d

import cv "github.com/malcolmston/opencv"

// agastCircle holds the 16 Bresenham-circle offsets (radius 3) scanned by the
// segment test, in clockwise order starting at the top. It matches the ring used
// by FAST.
var agastCircle = [16][2]int{
	{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
	{0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3},
}

// AGAST detects corners with an adaptive-threshold FAST segment test. It is a
// self-contained variant of OpenCV's cv::AgastFeatureDetector.
//
// The classic FAST-9 test marks a pixel as a corner when at least nine
// contiguous pixels on the radius-3 Bresenham ring are all brighter than the
// centre plus a threshold, or all darker than the centre minus the threshold.
// AGAST makes the threshold adaptive: for every pixel it finds, by binary
// search, the largest threshold at which the pixel still passes the segment test
// (its AGAST score). Pixels whose score reaches Threshold are corners, and the
// score serves both to rank them and to drive non-maximum suppression.
type AGAST struct {
	// Threshold is the minimum AGAST score (adaptive contrast) for a pixel to be
	// reported as a corner.
	Threshold int
	// NonmaxSuppression, when true, discards corners that are not the strongest
	// in their 3×3 neighbourhood by score.
	NonmaxSuppression bool
}

// NewAGAST returns an AGAST detector with the given segment-test threshold and
// non-maximum suppression enabled.
func NewAGAST(threshold int) *AGAST {
	return &AGAST{Threshold: threshold, NonmaxSuppression: true}
}

// Detect finds corners in img and returns them as keypoints. Each keypoint's
// Response is its AGAST score, Size is the ring diameter (7) and Angle is -1.
// img may be single- or three-channel; a colour image is converted to gray.
func (a *AGAST) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	threshold := a.Threshold
	if threshold < 1 {
		threshold = 1
	}
	score := make([]int, rows*cols)

	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			p := int(gray.Data[y*cols+x])
			var ring [16]int
			for k := 0; k < 16; k++ {
				ring[k] = int(gray.Data[(y+agastCircle[k][1])*cols+(x+agastCircle[k][0])])
			}
			s := agastScore(ring, p)
			if s >= threshold {
				score[y*cols+x] = s
			}
		}
	}

	var kps []KeyPoint
	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			s := score[y*cols+x]
			if s == 0 {
				continue
			}
			if a.NonmaxSuppression {
				suppressed := false
				for dy := -1; dy <= 1 && !suppressed; dy++ {
					for dx := -1; dx <= 1; dx++ {
						if dx == 0 && dy == 0 {
							continue
						}
						if score[(y+dy)*cols+(x+dx)] > s {
							suppressed = true
							break
						}
					}
				}
				if suppressed {
					continue
				}
			}
			kps = append(kps, KeyPoint{
				Pt:       cv.Point{X: x, Y: y},
				Size:     7,
				Angle:    -1,
				Response: float64(s),
			})
		}
	}
	return kps
}

// agastScore returns the largest threshold t (in [0,255]) at which the pixel of
// intensity p, surrounded by the 16 ring samples, still passes the FAST-9
// segment test. A score of 0 means the pixel is not a corner at any positive
// threshold.
func agastScore(ring [16]int, p int) int {
	if !segmentTest(ring, p, 1) {
		return 0
	}
	lo, hi := 1, 255
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if segmentTest(ring, p, mid) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// segmentTest reports whether at least nine contiguous ring samples are all
// brighter than p+t or all darker than p-t, wrapping around the ring.
func segmentTest(ring [16]int, p, t int) bool {
	hi := p + t
	lo := p - t
	for _, bright := range [2]bool{true, false} {
		run := 0
		for k := 0; k < 24; k++ {
			v := ring[k%16]
			ok := false
			if bright {
				ok = v > hi
			} else {
				ok = v < lo
			}
			if ok {
				run++
				if run >= 9 {
					return true
				}
			} else {
				run = 0
			}
		}
	}
	return false
}

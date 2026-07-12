package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TBMR detects Tree-Based Morse Regions, a port in the spirit of OpenCV's
// cv::xfeatures2d::TBMR.
//
// TBMRs are extremal regions of the image's morphological component tree (the
// max-tree for bright regions and the min-tree for dark ones) that are
// topologically stable across a range of grey levels. This implementation
// realises that idea with the well-known component-tree stability criterion
// shared with MSER: the image is thresholded at a sweep of levels, connected
// components are extracted at each level, and a component that changes area only
// slowly as the threshold varies (small area variation over a delta of levels)
// is reported as a keypoint. Both the max-tree (bright) and min-tree (dark)
// are processed. The exact region set differs from OpenCV's incremental
// max-tree construction, but follows the same extremal-region-stability
// principle (documented approximation).
type TBMR struct {
	// Delta is the grey-level offset (in level-steps) over which area stability
	// is measured.
	Delta int
	// LevelStep is the grey-level increment between successive thresholds.
	LevelStep int
	// MinArea and MaxArea bound the accepted region area in pixels.
	MinArea int
	MaxArea int
	// MaxVariation is the largest relative area change for a region to count as
	// stable.
	MaxVariation float64
}

// NewTBMR returns a TBMR detector with sensible defaults.
func NewTBMR() *TBMR {
	return &TBMR{
		Delta:        1,
		LevelStep:    8,
		MinArea:      12,
		MaxArea:      100000,
		MaxVariation: 0.5,
	}
}

// levelData holds the connected-component labelling of one thresholded level.
type levelData struct {
	labels []int
	area   []int
	cx     []float64
	cy     []float64
}

// Detect finds stable extremal regions in img and returns them as keypoints.
// Each keypoint's Pt is the region centroid, Size is the equivalent-circle
// diameter, Response is 1-variation (higher is more stable) and Angle is -1.
// img may be single- or three-channel; a colour image is converted to gray.
func (t *TBMR) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	step := t.LevelStep
	if step < 1 {
		step = 1
	}
	// Threshold levels 0,step,2*step,...,255.
	var levels []int
	for v := 0; v <= 255; v += step {
		levels = append(levels, v)
	}

	var kps []KeyPoint
	kps = append(kps, t.detectTree(gray, levels, true)...)
	kps = append(kps, t.detectTree(gray, levels, false)...)
	return kps
}

// detectTree runs the stability sweep for the bright (max-tree) or dark
// (min-tree) component tree.
func (t *TBMR) detectTree(gray *cv.Mat, levels []int, bright bool) []KeyPoint {
	cols := gray.Cols
	n := len(levels)
	data := make([]*levelData, n)
	for i, lv := range levels {
		data[i] = buildLevel(gray, lv, bright)
	}

	// Seeds are regional extrema: local maxima (bright) or minima (dark).
	seeds := regionalExtrema(gray, bright)

	var kps []KeyPoint
	delta := t.Delta
	if delta < 1 {
		delta = 1
	}
	seen := make(map[int]bool)
	for _, s := range seeds {
		sx, sy := s%cols, s/cols
		// Area of the component containing the seed at each level (0 when the
		// seed is not part of the foreground at that level).
		areaAt := make([]int, n)
		labelAt := make([]int, n)
		for i := 0; i < n; i++ {
			lbl := data[i].labels[sy*cols+sx]
			labelAt[i] = lbl
			if lbl > 0 {
				areaAt[i] = data[i].area[lbl]
			}
		}
		// Find the stability-optimal level: minimal relative area variation.
		bestVar := math.Inf(1)
		bestLevel := -1
		for i := delta; i < n-delta; i++ {
			a := areaAt[i]
			if a < t.MinArea || a > t.MaxArea {
				continue
			}
			ap := areaAt[i-delta] // higher threshold index? levels increase, so foreground shrinks as level rises for bright
			an := areaAt[i+delta]
			if a == 0 {
				continue
			}
			variation := math.Abs(float64(ap-an)) / float64(a)
			if variation < bestVar {
				bestVar = variation
				bestLevel = i
			}
		}
		if bestLevel < 0 || bestVar > t.MaxVariation {
			continue
		}
		lbl := labelAt[bestLevel]
		if lbl <= 0 {
			continue
		}
		key := bestLevel*1000003 + lbl
		if bright {
			key = -key - 1
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		ld := data[bestLevel]
		area := ld.area[lbl]
		diameter := 2 * math.Sqrt(float64(area)/math.Pi)
		kps = append(kps, KeyPoint{
			Pt:       cv.Point{X: int(math.Round(ld.cx[lbl])), Y: int(math.Round(ld.cy[lbl]))},
			Size:     diameter,
			Angle:    -1,
			Response: 1 - bestVar,
		})
	}
	return kps
}

// buildLevel thresholds gray at level lv and labels its connected components.
func buildLevel(gray *cv.Mat, lv int, bright bool) *levelData {
	var bin *cv.Mat
	if bright {
		// Foreground = pixels with intensity >= lv.
		bin, _ = cv.Threshold(gray, float64(lv-1), 255, cv.ThreshBinary)
	} else {
		// Foreground = pixels with intensity <= lv.
		bin, _ = cv.Threshold(gray, float64(lv), 255, cv.ThreshBinaryInv)
	}
	labels, count, stats := cv.ConnectedComponentsWithStats(bin, cv.Connectivity8)
	area := make([]int, count)
	cx := make([]float64, count)
	cy := make([]float64, count)
	for _, st := range stats {
		if st.Label >= 0 && st.Label < count {
			area[st.Label] = st.Area
			cx[st.Label] = st.CentroidX
			cy[st.Label] = st.CentroidY
		}
	}
	return &levelData{labels: labels, area: area, cx: cx, cy: cy}
}

// regionalExtrema returns the flat indices of pixels that are local maxima
// (bright) or minima (dark) in their 8-neighbourhood, the seeds of the tree.
func regionalExtrema(gray *cv.Mat, bright bool) []int {
	rows, cols := gray.Rows, gray.Cols
	var out []int
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			v := gray.Data[y*cols+x]
			isExtreme := true
			for dy := -1; dy <= 1 && isExtreme; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					nv := gray.Data[(y+dy)*cols+(x+dx)]
					if bright && nv > v {
						isExtreme = false
						break
					}
					if !bright && nv < v {
						isExtreme = false
						break
					}
				}
			}
			if isExtreme {
				out = append(out, y*cols+x)
			}
		}
	}
	return out
}

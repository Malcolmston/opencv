package stitching

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// feature is a detected keypoint together with its patch descriptor. The
// descriptor is a mean-subtracted, L2-normalised intensity window, which makes
// the squared Euclidean distance between two descriptors equal to 2·(1 − NCC)
// and therefore invariant to local brightness and contrast changes.
type feature struct {
	x, y int
	desc []float64
}

// toGray returns a single-channel view of img suitable for corner detection.
// A grayscale image is returned unchanged (a shallow reference); a three-channel
// image is converted with the BT.601 luma weights. Other channel counts panic.
func toGray(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return img
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		panic("stitching: images must have 1 or 3 channels")
	}
}

// detectAndDescribe finds Shi–Tomasi corners in gray and computes a normalised
// intensity-patch descriptor for each corner that lies far enough from the
// border to sample a full (2·radius+1)² window. It returns the surviving
// features in detector (descending strength) order.
func detectAndDescribe(gray *cv.Mat, maxCorners int, quality, minDist float64, blockSize, radius int) []feature {
	corners := cv.GoodFeaturesToTrack(gray, maxCorners, quality, minDist, blockSize)
	feats := make([]feature, 0, len(corners))
	for _, p := range corners {
		d, ok := describePatch(gray, p.X, p.Y, radius)
		if !ok {
			continue
		}
		feats = append(feats, feature{x: p.X, y: p.Y, desc: d})
	}
	return feats
}

// describePatch builds the normalised intensity descriptor for the window
// centred on (x, y). It returns false when the window would leave the image or
// the patch is flat (zero variance), in which case no stable descriptor exists.
func describePatch(gray *cv.Mat, x, y, radius int) ([]float64, bool) {
	if x-radius < 0 || y-radius < 0 || x+radius >= gray.Cols || y+radius >= gray.Rows {
		return nil, false
	}
	side := 2*radius + 1
	desc := make([]float64, side*side)
	var mean float64
	i := 0
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			v := float64(gray.Data[(y+dy)*gray.Cols+(x+dx)])
			desc[i] = v
			mean += v
			i++
		}
	}
	mean /= float64(len(desc))
	var norm float64
	for i := range desc {
		desc[i] -= mean
		norm += desc[i] * desc[i]
	}
	norm = math.Sqrt(norm)
	if norm < 1e-6 {
		return nil, false
	}
	for i := range desc {
		desc[i] /= norm
	}
	return desc, true
}

// match is a putative correspondence between a feature in the query set and a
// feature in the train set, with the squared descriptor distance.
type match struct {
	queryIdx int
	trainIdx int
	dist2    float64
}

// descDist2 returns the squared Euclidean distance between two equal-length
// descriptors.
func descDist2(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// matchFeatures matches every query feature to its nearest train feature and
// keeps the pair only when it passes Lowe's ratio test — the nearest distance
// must be below ratio times the second-nearest — which rejects ambiguous
// matches in repetitive texture. Results are sorted by ascending distance so
// the strongest matches come first. Matching is exhaustive and deterministic.
func matchFeatures(query, train []feature, ratio float64) []match {
	var matches []match
	ratio2 := ratio * ratio
	for qi := range query {
		best, second := math.Inf(1), math.Inf(1)
		bestIdx := -1
		for ti := range train {
			d := descDist2(query[qi].desc, train[ti].desc)
			if d < best {
				second = best
				best = d
				bestIdx = ti
			} else if d < second {
				second = d
			}
		}
		if bestIdx < 0 {
			continue
		}
		// Ratio test; when there is no second neighbour the match is accepted.
		if math.IsInf(second, 1) || best <= ratio2*second {
			matches = append(matches, match{queryIdx: qi, trainIdx: bestIdx, dist2: best})
		}
	}
	sort.SliceStable(matches, func(i, j int) bool { return matches[i].dist2 < matches[j].dist2 })
	return matches
}

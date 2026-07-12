package features2d

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Detector is implemented by any type that can find keypoints in an image, such
// as [ORB], [SIFT], [KAZE], [AKAZE], [FastFeatureDetector],
// [AgastFeatureDetector], [GFTTDetector] and [SimpleBlobDetector]. It is the
// input type accepted by [EvaluateFeatureDetector].
type Detector interface {
	Detect(img *cv.Mat) []KeyPoint
}

// KeyPointsFilter provides the keypoint-list filtering utilities from OpenCV's
// cv::KeyPointsFilter. The zero value is ready to use; the methods have no
// receiver state and never modify their input slice, returning a new slice
// instead (OpenCV filters in place).
type KeyPointsFilter struct{}

// RunByImageBorder removes keypoints whose neighbourhood (a disc of radius
// borderSize, or the keypoint centre when borderSize <= 0) falls outside the
// rows×cols image. Keypoints exactly borderSize pixels from an edge are kept.
func (KeyPointsFilter) RunByImageBorder(kps []KeyPoint, rows, cols, borderSize int) []KeyPoint {
	if borderSize < 0 {
		borderSize = 0
	}
	out := make([]KeyPoint, 0, len(kps))
	for _, kp := range kps {
		if kp.Pt.X >= borderSize && kp.Pt.X < cols-borderSize &&
			kp.Pt.Y >= borderSize && kp.Pt.Y < rows-borderSize {
			out = append(out, kp)
		}
	}
	return out
}

// RunByKeypointSize keeps only keypoints whose Size lies in the half-open range
// [minSize, maxSize). Pass a very large maxSize to impose only a lower bound.
func (KeyPointsFilter) RunByKeypointSize(kps []KeyPoint, minSize, maxSize float64) []KeyPoint {
	out := make([]KeyPoint, 0, len(kps))
	for _, kp := range kps {
		if kp.Size >= minSize && kp.Size < maxSize {
			out = append(out, kp)
		}
	}
	return out
}

// RetainBest returns the n keypoints with the strongest Response, keeping every
// keypoint whose response ties the n-th strongest (so the result may exceed n,
// matching OpenCV). A non-positive n, or n >= len(kps), returns a copy of the
// whole slice. The relative order of the survivors is not preserved: they are
// returned in descending response order with position tie-breaking for
// determinism.
func (KeyPointsFilter) RetainBest(kps []KeyPoint, n int) []KeyPoint {
	sorted := make([]KeyPoint, len(kps))
	copy(sorted, kps)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Response != sorted[j].Response {
			return sorted[i].Response > sorted[j].Response
		}
		if sorted[i].Pt.Y != sorted[j].Pt.Y {
			return sorted[i].Pt.Y < sorted[j].Pt.Y
		}
		return sorted[i].Pt.X < sorted[j].Pt.X
	})
	if n <= 0 || n >= len(sorted) {
		return sorted
	}
	// Include ties with the n-th strongest response.
	threshold := sorted[n-1].Response
	keep := n
	for keep < len(sorted) && sorted[keep].Response == threshold {
		keep++
	}
	return sorted[:keep]
}

// RemoveDuplicated removes keypoints that share the same integer position, Size
// and Angle, keeping the first occurrence of each. Unlike OpenCV it does not
// require the input to be pre-sorted.
func (KeyPointsFilter) RemoveDuplicated(kps []KeyPoint) []KeyPoint {
	type key struct {
		x, y        int
		size, angle float64
	}
	seen := make(map[key]struct{}, len(kps))
	out := make([]KeyPoint, 0, len(kps))
	for _, kp := range kps {
		k := key{kp.Pt.X, kp.Pt.Y, kp.Size, kp.Angle}
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		out = append(out, kp)
	}
	return out
}

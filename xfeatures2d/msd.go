package xfeatures2d

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// MSDDetector detects Maximal Self-Dissimilarity interest points, a port of
// OpenCV's cv::xfeatures2d::MSDDetector.
//
// The MSD saliency of a pixel is its contextual self-dissimilarity: the average
// sum-of-squared-differences between the patch centred on the pixel and the K
// least dissimilar patches found within a surrounding search area. Points that
// look unlike their neighbourhood (corners, blobs, junctions) score highly.
// Keypoints are the local maxima of this saliency above a threshold, after
// non-maximum suppression.
type MSDDetector struct {
	// PatchRadius is the half side of the compared patches.
	PatchRadius int
	// SearchRadius is the half side of the search area scanned for similar
	// patches.
	SearchRadius int
	// NMSRadius is the half side of the non-maximum-suppression window.
	NMSRadius int
	// KNN is the number of least-dissimilar patches averaged into the saliency.
	KNN int
	// Threshold is the minimum saliency for a pixel to be reported.
	Threshold float64
}

// NewMSDDetector returns an MSDDetector with sensible defaults.
func NewMSDDetector() *MSDDetector {
	return &MSDDetector{
		PatchRadius:  3,
		SearchRadius: 5,
		NMSRadius:    5,
		KNN:          4,
		Threshold:    250,
	}
}

// patchSSD returns the sum of squared differences between the patches centred at
// (ax, ay) and (bx, by) of gray, using border replication.
func msdPatchSSD(gray *cv.Mat, ax, ay, bx, by, r int) float64 {
	var s float64
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			d := grayAtClamped(gray, ax+dx, ay+dy) - grayAtClamped(gray, bx+dx, by+dy)
			s += d * d
		}
	}
	return s
}

// Detect finds MSD keypoints in img. Each keypoint's Response is its saliency,
// Size is the patch diameter and Angle is -1. img may be single- or
// three-channel; a colour image is converted to gray.
func (m *MSDDetector) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	pr := m.PatchRadius
	sr := m.SearchRadius
	knn := m.KNN
	if knn < 1 {
		knn = 1
	}
	saliency := make([]float64, rows*cols)

	border := pr
	for y := border; y < rows-border; y++ {
		for x := border; x < cols-border; x++ {
			dists := make([]float64, 0, (2*sr+1)*(2*sr+1))
			for dy := -sr; dy <= sr; dy++ {
				for dx := -sr; dx <= sr; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					dists = append(dists, msdPatchSSD(gray, x, y, x+dx, y+dy, pr))
				}
			}
			sort.Float64s(dists)
			k := knn
			if k > len(dists) {
				k = len(dists)
			}
			var sum float64
			for i := 0; i < k; i++ {
				sum += dists[i]
			}
			saliency[y*cols+x] = sum / float64(k)
		}
	}

	var kps []KeyPoint
	nr := m.NMSRadius
	for y := border; y < rows-border; y++ {
		for x := border; x < cols-border; x++ {
			s := saliency[y*cols+x]
			if s < m.Threshold {
				continue
			}
			if !isLocalMax(saliency, x, y, cols, rows, nr, s) {
				continue
			}
			kps = append(kps, KeyPoint{
				Pt:       cv.Point{X: x, Y: y},
				Size:     float64(2*pr + 1),
				Angle:    -1,
				Response: s,
			})
		}
	}
	return kps
}

// isLocalMax reports whether saliency at (x, y) is the strict maximum in its
// (2r+1)² window (ties are broken so only one representative survives).
func isLocalMax(saliency []float64, x, y, cols, rows, r int, val float64) bool {
	for dy := -r; dy <= r; dy++ {
		ny := y + dy
		if ny < 0 || ny >= rows {
			continue
		}
		for dx := -r; dx <= r; dx++ {
			nx := x + dx
			if nx < 0 || nx >= cols {
				continue
			}
			if dx == 0 && dy == 0 {
				continue
			}
			nv := saliency[ny*cols+nx]
			if nv > val {
				return false
			}
			// Break ties by position so a plateau yields a single keypoint.
			if nv == val && (ny < y || (ny == y && nx < x)) {
				return false
			}
		}
	}
	return true
}

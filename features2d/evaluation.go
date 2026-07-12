package features2d

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// RecallPrecisionPoint is one sample of a recall/precision curve produced by
// [ComputeRecallPrecisionCurve]. Recall is the fraction of correct correspondences
// recovered and Precision the fraction of accepted matches that are correct, both
// in [0, 1]. (OpenCV returns Point2f(1-precision, recall); this exposes the two
// quantities directly.)
type RecallPrecisionPoint struct {
	Recall    float64
	Precision float64
}

// ComputeRecallPrecisionCurve builds a recall/precision curve from k-nearest
// matches and a mask marking which of those matches are geometrically correct,
// mirroring OpenCV's cv::computeRecallPrecisionCurve. matches and correct are
// parallel: matches[q] holds the candidate matches for query q (typically a
// k=1..n KnnMatch row) and correct[q][j] reports whether matches[q][j] is a true
// correspondence.
//
// Every candidate is treated as a detection whose confidence is the negated
// descriptor distance. Sweeping a threshold from the most to the least confident
// detection, the function accumulates true and false positives and emits a
// RecallPrecisionPoint at each step. The result is sorted by ascending recall. It
// panics if the two arguments have mismatched shapes.
func ComputeRecallPrecisionCurve(matches [][]DMatch, correct [][]bool) []RecallPrecisionPoint {
	if len(matches) != len(correct) {
		panic("features2d: ComputeRecallPrecisionCurve shape mismatch")
	}
	type det struct {
		dist    float64
		correct bool
	}
	var dets []det
	totalCorrect := 0
	for q := range matches {
		if len(matches[q]) != len(correct[q]) {
			panic("features2d: ComputeRecallPrecisionCurve row shape mismatch")
		}
		for j := range matches[q] {
			dets = append(dets, det{matches[q][j].Distance, correct[q][j]})
			if correct[q][j] {
				totalCorrect++
			}
		}
	}
	if len(dets) == 0 {
		return nil
	}
	// Most confident first = smallest distance first.
	sort.SliceStable(dets, func(i, j int) bool { return dets[i].dist < dets[j].dist })

	var curve []RecallPrecisionPoint
	tp, fp := 0, 0
	for _, d := range dets {
		if d.correct {
			tp++
		} else {
			fp++
		}
		var recall, precision float64
		if totalCorrect > 0 {
			recall = float64(tp) / float64(totalCorrect)
		}
		if tp+fp > 0 {
			precision = float64(tp) / float64(tp+fp)
		}
		curve = append(curve, RecallPrecisionPoint{Recall: recall, Precision: precision})
	}
	return curve
}

// applyHomography maps an integer point through a 3×3 projective transform,
// returning the transformed coordinates in floating point.
func applyHomography(h cv.PerspectiveMatrix, p cv.Point) (float64, float64) {
	x, y := float64(p.X), float64(p.Y)
	w := h[6]*x + h[7]*y + h[8]
	if w == 0 {
		w = 1e-12
	}
	nx := (h[0]*x + h[1]*y + h[2]) / w
	ny := (h[3]*x + h[4]*y + h[5]) / w
	return nx, ny
}

// EvaluateFeatureDetector measures the repeatability of a [Detector] across an
// image pair related by the homography h1to2 (mapping img1 coordinates into
// img2), following the intent of OpenCV's cv::evaluateFeatureDetector. It detects
// keypoints in both images, projects each img1 keypoint into img2 through h1to2,
// and counts a correspondence when a projected keypoint lands within tolerance
// pixels of an img2 keypoint (each img2 keypoint matched at most once, nearest
// first). Repeatability is the correspondence count divided by the number of
// img1 keypoints whose projection falls inside img2.
//
// It returns the repeatability in [0, 1] and the absolute correspondence count.
// Unlike OpenCV, which measures overlap of the elliptic keypoint regions, this
// uses a simpler point-distance criterion; on scenes without large scale change
// the two agree closely. A tolerance <= 0 defaults to 3 pixels.
func EvaluateFeatureDetector(img1, img2 *cv.Mat, h1to2 cv.PerspectiveMatrix, det Detector, tolerance float64) (repeatability float64, correspondences int) {
	if tolerance <= 0 {
		tolerance = 3
	}
	kp1 := det.Detect(img1)
	kp2 := det.Detect(img2)

	// Project img1 keypoints that fall inside img2.
	type proj struct {
		x, y float64
	}
	var projected []proj
	for _, kp := range kp1 {
		px, py := applyHomography(h1to2, kp.Pt)
		if px < 0 || py < 0 || px >= float64(img2.Cols) || py >= float64(img2.Rows) {
			continue
		}
		projected = append(projected, proj{px, py})
	}
	if len(projected) == 0 || len(kp2) == 0 {
		return 0, 0
	}

	tol2 := tolerance * tolerance
	used := make([]bool, len(kp2))
	for _, p := range projected {
		best, bestD := -1, tol2
		for j, kp := range kp2 {
			if used[j] {
				continue
			}
			dx := p.x - float64(kp.Pt.X)
			dy := p.y - float64(kp.Pt.Y)
			d := dx*dx + dy*dy
			if d <= bestD {
				bestD, best = d, j
			}
		}
		if best >= 0 {
			used[best] = true
			correspondences++
		}
	}
	repeatability = float64(correspondences) / float64(len(projected))
	return repeatability, correspondences
}

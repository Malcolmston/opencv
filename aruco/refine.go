package aruco

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// RefineDetectedMarkers recovers board markers that the initial detection
// missed. Given the markers already found in image (detectedCorners and
// detectedIds, the parallel slices from [DetectMarkers]) and the [GridBoard]
// they belong to, it fits the board-to-image homography from the known
// detections, predicts where each still-missing board marker should appear, and
// searches the image's quadrilateral candidates near those predictions. A
// candidate close to a prediction is accepted: its identifier is confirmed from
// its bit grid when readable, and otherwise assigned from the board layout.
//
// It returns the merged corners and ids (the originals followed by any
// recoveries) and recovered, the number of markers added. When the board pose
// cannot be established from the existing detections (fewer than one board
// marker present) the inputs are returned unchanged. This mirrors OpenCV's
// cv::aruco::refineDetectedMarkers, which uses the board to rescue markers under
// glare, blur or partial occlusion.
func RefineDetectedMarkers(image *cv.Mat, board *GridBoard, detectedCorners [][4]cv.Point, detectedIds []int) (corners [][4]cv.Point, ids []int, recovered int) {
	corners = make([][4]cv.Point, len(detectedCorners))
	copy(corners, detectedCorners)
	ids = make([]int, len(detectedIds))
	copy(ids, detectedIds)
	if image == nil || board == nil {
		return corners, ids, 0
	}

	var src, dst [][2]float64
	present := make(map[int]bool)
	for i, id := range ids {
		if i >= len(corners) {
			break
		}
		present[id] = true
		obj, ok := board.objectCornersForID(id)
		if !ok {
			continue
		}
		for j := 0; j < 4; j++ {
			src = append(src, [2]float64{obj[j][0], obj[j][1]})
			dst = append(dst, [2]float64{float64(corners[i][j].X), float64(corners[i][j].Y)})
		}
	}
	if len(src) < 4 {
		return corners, ids, 0
	}
	h, ok := homographyFromPoints(src, dst)
	if !ok {
		return corners, ids, 0
	}

	// Collect all quadrilateral candidates in the image once.
	gray := toGray(image)
	bin := thresholdForContours(gray)
	contours, _ := cv.FindContours(bin, cv.RetrExternal, cv.ChainApproxNone)
	var cands [][4]cv.Point
	for _, c := range contours {
		if quad, ok := candidateQuad(c); ok {
			cands = append(cands, quad)
		}
	}
	used := make([]bool, len(cands))

	for _, id := range board.ids {
		if present[id] {
			continue
		}
		obj, _ := board.objectCornersForID(id)
		var pred [4]cv.Point
		okAll := true
		for j := 0; j < 4; j++ {
			u, v, ok := applyH(h, obj[j][0], obj[j][1])
			if !ok {
				okAll = false
				break
			}
			pred[j] = cv.Point{X: int(math.Round(u)), Y: int(math.Round(v))}
		}
		if !okAll {
			continue
		}
		pcx, pcy := quadCenter(pred)
		pside := quadSide(pred)

		best := -1
		bestDist := 0.5 * pside
		for ci, cand := range cands {
			if used[ci] {
				continue
			}
			ccx, ccy := quadCenter(cand)
			d := math.Hypot(ccx-pcx, ccy-pcy)
			if d < bestDist {
				bestDist = d
				best = ci
			}
		}
		if best < 0 {
			continue
		}
		cand := cands[best]
		used[best] = true

		var ordered [4]cv.Point
		if read, ok := readMarkerGrid(gray, cand, board.dict.bitsPerSide); ok {
			if mid, k, mok := matchGrid(board.dict, read); mok && mid == id {
				ordered = rotateCorners(cand, k)
			} else {
				ordered = orderQuadToReference(cand, pred)
			}
		} else {
			ordered = orderQuadToReference(cand, pred)
		}
		corners = append(corners, ordered)
		ids = append(ids, id)
		present[id] = true
		recovered++
	}
	return corners, ids, recovered
}

// orderQuadToReference reorders quad so that its corners best align with the
// reference corners ref (which are in the desired TL, TR, BR, BL order): each
// reference corner is greedily paired with the nearest still-unused quad corner.
func orderQuadToReference(quad, ref [4]cv.Point) [4]cv.Point {
	var out [4]cv.Point
	usedC := [4]bool{}
	for r := 0; r < 4; r++ {
		best := -1
		bestD := math.MaxFloat64
		for c := 0; c < 4; c++ {
			if usedC[c] {
				continue
			}
			d := math.Hypot(float64(quad[c].X-ref[r].X), float64(quad[c].Y-ref[r].Y))
			if d < bestD {
				bestD = d
				best = c
			}
		}
		usedC[best] = true
		out[r] = quad[best]
	}
	return out
}

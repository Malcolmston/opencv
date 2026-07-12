package aruco

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// DrawCharucoDiamond renders a ChArUco "diamond": a 3x3 chessboard carrying four
// ArUco markers in its white squares, the analogue of OpenCV's
// cv::aruco::drawCharucoDiamond. The four identifiers ids are placed in the top,
// left, right and bottom white squares in that order. squareLength and
// markerLength are the chessboard square and marker side lengths in pixels;
// markerLength must be smaller than squareLength. The result is a single-channel
// image 3*squareLength pixels on a side.
//
// A diamond is a compact, self-identifying target: its four marker ids act as a
// signature, so several diamonds in one scene can be told apart while its
// chessboard corners give a precise local pose.
func DrawCharucoDiamond(dict *Dictionary, ids [4]int, squareLength, markerLength int) *cv.Mat {
	if dict == nil {
		panic("aruco: DrawCharucoDiamond nil dictionary")
	}
	if squareLength <= 0 || markerLength <= 0 {
		panic("aruco: DrawCharucoDiamond requires positive lengths")
	}
	if markerLength >= squareLength {
		panic("aruco: DrawCharucoDiamond requires markerLength < squareLength")
	}
	for _, id := range ids {
		if id < 0 || id >= dict.Size() {
			panic(fmt.Sprintf("aruco: DrawCharucoDiamond id %d out of range [0,%d)", id, dict.Size()))
		}
	}
	size := 3 * squareLength
	canvas := cv.NewMat(size, size, 1)
	canvas.SetTo(255)
	black := cv.NewScalar(0)
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			if (col+row)%2 != 0 {
				continue
			}
			x0 := col * squareLength
			y0 := row * squareLength
			cv.Rectangle(canvas, cv.Point{X: x0, Y: y0}, cv.Point{X: x0 + squareLength - 1, Y: y0 + squareLength - 1}, black, cv.Filled)
		}
	}
	// White squares in (col,row): top (1,0), left (0,1), right (2,1), bottom (1,2).
	squares := [4][2]int{{1, 0}, {0, 1}, {2, 1}, {1, 2}}
	off := (squareLength - markerLength) / 2
	for i, sq := range squares {
		x0 := sq[0]*squareLength + off
		y0 := sq[1]*squareLength + off
		GenerateMarker(dict, ids[i], markerLength).CopyTo(canvas, y0, x0)
	}
	return canvas
}

// DetectCharucoDiamond groups already-detected ArUco markers into ChArUco
// diamonds, the analogue of OpenCV's cv::aruco::detectCharucoDiamond.
// markerCorners and markerIds are the parallel slices from [DetectMarkers];
// squareLength and markerLength are the diamond's square and marker side lengths
// (any consistent unit, used only as a ratio to place the corners). For every
// set of four markers whose centres form a small square, it returns the four
// image corners of the diamond's central chessboard square and the four marker
// identifiers in canonical top, left, right, bottom order.
//
// The diamondCorners and diamondIds results are parallel, one entry per diamond
// found, and are nil when no diamond is present. Corners are obtained from the
// board-to-image homography fitted to the four markers, so they are subpixel
// accurate for a planar diamond.
func DetectCharucoDiamond(image *cv.Mat, markerCorners [][4]cv.Point, markerIds []int, dict *Dictionary, squareLength, markerLength float64) (diamondCorners [][4]cv.Point, diamondIds [][4]int) {
	_ = image // reserved for optional corner refinement; not required here.
	n := len(markerIds)
	if n < 4 || squareLength <= 0 || markerLength <= 0 {
		return nil, nil
	}
	centers := make([][2]float64, n)
	for i := range markerCorners {
		cx, cy := quadCenter(markerCorners[i])
		centers[i] = [2]float64{cx, cy}
	}
	used := make([]bool, n)

	// Object model (board frame, +Y up) of a unit 3x3 diamond.
	sq := squareLength
	boardH := 3 * sq
	half := markerLength / 2
	whiteCenters := [4][2]float64{
		{1.5 * sq, boardH - 0.5*sq}, // top    (1,0)
		{0.5 * sq, boardH - 1.5*sq}, // left   (0,1)
		{2.5 * sq, boardH - 1.5*sq}, // right  (2,1)
		{1.5 * sq, boardH - 2.5*sq}, // bottom (1,2)
	}
	centralSquare := [4][2]float64{
		{1 * sq, 2 * sq}, // TL
		{2 * sq, 2 * sq}, // TR
		{2 * sq, 1 * sq}, // BR
		{1 * sq, 1 * sq}, // BL
	}

	for a := 0; a < n; a++ {
		if used[a] {
			continue
		}
		for b := a + 1; b < n; b++ {
			if used[b] {
				continue
			}
			for c := b + 1; c < n; c++ {
				if used[c] {
					continue
				}
				for d := c + 1; d < n; d++ {
					if used[d] {
						continue
					}
					idx := [4]int{a, b, c, d}
					order, ok := diamondArrangement([4][2]float64{centers[a], centers[b], centers[c], centers[d]})
					if !ok {
						continue
					}
					// order maps canonical slot (top,left,right,bottom) -> local index 0..3.
					var src, dst [][2]float64
					var dids [4]int
					for slot := 0; slot < 4; slot++ {
						local := idx[order[slot]]
						dids[slot] = markerIds[local]
						mc := whiteCenters[slot]
						objC := [4][2]float64{
							{mc[0] - half, mc[1] + half},
							{mc[0] + half, mc[1] + half},
							{mc[0] + half, mc[1] - half},
							{mc[0] - half, mc[1] - half},
						}
						for j := 0; j < 4; j++ {
							src = append(src, objC[j])
							dst = append(dst, [2]float64{float64(markerCorners[local][j].X), float64(markerCorners[local][j].Y)})
						}
					}
					h, ok := homographyFromPoints(src, dst)
					if !ok {
						continue
					}
					var corners [4]cv.Point
					okAll := true
					for j := 0; j < 4; j++ {
						u, v, ok := applyH(h, centralSquare[j][0], centralSquare[j][1])
						if !ok {
							okAll = false
							break
						}
						corners[j] = cv.Point{X: int(math.Round(u)), Y: int(math.Round(v))}
					}
					if !okAll {
						continue
					}
					used[a], used[b], used[c], used[d] = true, true, true, true
					diamondCorners = append(diamondCorners, corners)
					diamondIds = append(diamondIds, dids)
				}
			}
		}
	}
	return diamondCorners, diamondIds
}

// diamondArrangement reports whether the four marker centres form a diamond (a
// square rotated so its vertices point up/down/left/right) and returns, for each
// canonical slot (0 top, 1 left, 2 right, 3 bottom), the index into pts of the
// marker occupying it. ok is false when the points are not a plausible diamond.
func diamondArrangement(pts [4][2]float64) (order [4]int, ok bool) {
	var cx, cy float64
	for _, p := range pts {
		cx += p[0]
		cy += p[1]
	}
	cx /= 4
	cy /= 4
	// All four must be roughly equidistant from the centroid.
	var rs [4]float64
	rmin, rmax := math.MaxFloat64, 0.0
	for i, p := range pts {
		rs[i] = math.Hypot(p[0]-cx, p[1]-cy)
		rmin = math.Min(rmin, rs[i])
		rmax = math.Max(rmax, rs[i])
	}
	if rmin < 1e-6 || rmax > 1.35*rmin {
		return order, false
	}
	// Slots by extreme coordinate: top=min y, bottom=max y, left=min x, right=max x.
	top, bottom, left, right := -1, -1, -1, -1
	minY, maxY, minX, maxX := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
	for i, p := range pts {
		if p[1] < minY {
			minY, top = p[1], i
		}
		if p[1] > maxY {
			maxY, bottom = p[1], i
		}
		if p[0] < minX {
			minX, left = p[0], i
		}
		if p[0] > maxX {
			maxX, right = p[0], i
		}
	}
	order = [4]int{top, left, right, bottom}
	// The four slots must be four distinct markers.
	seen := map[int]bool{}
	for _, i := range order {
		if i < 0 || seen[i] {
			return [4]int{}, false
		}
		seen[i] = true
	}
	return order, true
}

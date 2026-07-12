package calib3d

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// FindChessboardCorners locates the inner corners of a chessboard calibration
// pattern in an image. patternSize is (cols, rows): the number of inner corners
// per chessboard row and column (one less than the number of squares in each
// direction). The board is assumed to be roughly axis-aligned and to sit on a
// contrasting quiet zone, as in a standard calibration capture.
//
// The detector responds to the X-junction (saddle) structure where four squares
// meet: at each pixel it compares the two diagonal pairs of neighbours a fixed
// radius away, which is large at a true corner and near zero along edges and flat
// regions. Candidate maxima are found by non-maximum suppression, refined to
// sub-pixel accuracy by a response-weighted centroid, and ordered row by row. It
// returns the corners in row-major order and found == true only when exactly
// cols·rows corners are recovered; otherwise the best-effort corners are returned
// with found == false.
func FindChessboardCorners(img *cv.Mat, patternSize [2]int) (corners [][2]float64, found bool) {
	if img == nil || img.Empty() {
		return nil, false
	}
	cols, rows := patternSize[0], patternSize[1]
	if cols < 2 || rows < 2 {
		return nil, false
	}
	w, h := img.Cols, img.Rows
	gray := toGray(img)
	// Estimate the square size in pixels from the pattern and image extent.
	sq := math.Min(float64(w)/float64(cols+1), float64(h)/float64(rows+1))
	// The saddle response saturates over a flat plateau of half-width rr around
	// each corner (the diagonal-neighbour offset). Keeping rr modest gives a
	// narrow plateau; r is the border margin skipped during scanning.
	rr := int(math.Max(2, math.Round(sq*0.2)))
	r := rr + 1
	minDist := math.Max(3, sq*0.6)
	// A centroid window of radius 2·rr fully contains the plateau regardless of
	// which plateau pixel seeded it, while staying clear of neighbouring corners,
	// so the weighted centroid lands on the corner centre.
	refineRad := 2 * rr

	// A checkerboard X-junction has two diagonally-opposite neighbour pairs that
	// are internally consistent (one pair bright, the other dark). Reward that
	// while penalising L- and T-junctions along the board border, whose diagonal
	// pairs are internally inconsistent.
	resp := make([]float64, w*h)
	maxResp := 0.0
	for y := r; y < h-r; y++ {
		for x := r; x < w-r; x++ {
			ne := gray[(y-rr)*w+(x+rr)]
			nw := gray[(y-rr)*w+(x-rr)]
			se := gray[(y+rr)*w+(x+rr)]
			sw := gray[(y+rr)*w+(x-rr)]
			diag1 := ne + sw
			diag2 := nw + se
			s := math.Abs(diag1-diag2) - (math.Abs(ne-sw) + math.Abs(nw-se))
			if s < 0 {
				s = 0
			}
			resp[y*w+x] = s
			if s > maxResp {
				maxResp = s
			}
		}
	}
	if maxResp < 1e-9 {
		return nil, false
	}
	thresh := 0.3 * maxResp

	type cand struct {
		x, y, s float64
	}
	var raw []cand
	for y := r; y < h-r; y++ {
		for x := r; x < w-r; x++ {
			if resp[y*w+x] >= thresh {
				raw = append(raw, cand{float64(x), float64(y), resp[y*w+x]})
			}
		}
	}
	// Greedy non-maximum suppression by minimum separation.
	sort.Slice(raw, func(i, j int) bool { return raw[i].s > raw[j].s })
	var cands []cand
	for _, c := range raw {
		ok := true
		for _, a := range cands {
			if math.Hypot(a.x-c.x, a.y-c.y) < minDist {
				ok = false
				break
			}
		}
		if ok {
			cx, cy := refineCentroid(resp, w, h, int(c.x), int(c.y), refineRad)
			cands = append(cands, cand{cx, cy, c.s})
		}
	}
	need := cols * rows
	if len(cands) < need {
		return nil, false
	}
	if len(cands) > need {
		cands = cands[:need]
	}
	// Order row by row: sort by y, chunk into rows, sort each chunk by x.
	sort.Slice(cands, func(i, j int) bool { return cands[i].y < cands[j].y })
	corners = make([][2]float64, 0, need)
	for row := 0; row < rows; row++ {
		chunk := cands[row*cols : (row+1)*cols]
		sort.Slice(chunk, func(i, j int) bool { return chunk[i].x < chunk[j].x })
		for _, c := range chunk {
			corners = append(corners, [2]float64{c.x, c.y})
		}
	}
	return corners, true
}

// toGray returns the image as a float grayscale buffer (row-major), averaging
// channels for multi-channel input.
func toGray(img *cv.Mat) []float64 {
	w, h, ch := img.Cols, img.Rows, img.Channels
	g := make([]float64, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if ch == 1 {
				g[y*w+x] = float64(img.At(y, x, 0))
				continue
			}
			var s float64
			for c := 0; c < ch; c++ {
				s += float64(img.At(y, x, c))
			}
			g[y*w+x] = s / float64(ch)
		}
	}
	return g
}

// refineCentroid computes a response-weighted centroid in a small window for
// sub-pixel corner localisation.
func refineCentroid(resp []float64, w, h, x, y, rad int) (float64, float64) {
	var sw, sx, sy float64
	for dy := -rad; dy <= rad; dy++ {
		yy := y + dy
		if yy < 0 || yy >= h {
			continue
		}
		for dx := -rad; dx <= rad; dx++ {
			xx := x + dx
			if xx < 0 || xx >= w {
				continue
			}
			ww := resp[yy*w+xx]
			sw += ww
			sx += ww * float64(xx)
			sy += ww * float64(yy)
		}
	}
	if sw < 1e-12 {
		return float64(x), float64(y)
	}
	return sx / sw, sy / sw
}

// DrawChessboardCorners renders detected chessboard corners onto img in place,
// mirroring OpenCV's visualisation. patternSize is (cols, rows). When found is
// true the corners are drawn as small connected circles, coloured per row so the
// detected ordering is visible; when found is false each corner is drawn as an
// isolated red circle. The image must have three channels for the colours to be
// meaningful.
func DrawChessboardCorners(img *cv.Mat, patternSize [2]int, corners [][2]float64, found bool) {
	if img == nil || img.Empty() || len(corners) == 0 {
		return
	}
	cols := patternSize[0]
	radius := 4
	pt := func(i int) cv.Point {
		return cv.Point{X: int(math.Round(corners[i][0])), Y: int(math.Round(corners[i][1]))}
	}
	if !found {
		red := cv.NewScalar(255, 0, 0)
		for i := range corners {
			cv.Circle(img, pt(i), radius, red, 1)
		}
		return
	}
	palette := []cv.Scalar{
		cv.NewScalar(255, 0, 0), cv.NewScalar(0, 255, 0), cv.NewScalar(0, 0, 255),
		cv.NewScalar(255, 255, 0), cv.NewScalar(255, 0, 255), cv.NewScalar(0, 255, 255),
	}
	for i := range corners {
		color := palette[(i/cols)%len(palette)]
		cv.Circle(img, pt(i), radius, color, 1)
		if i%cols != 0 {
			cv.Line(img, pt(i-1), pt(i), color, 1)
		}
	}
}

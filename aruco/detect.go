package aruco

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// cellPixels is the number of canonical pixels rendered per marker cell when a
// candidate quadrilateral is unwarped for reading. A modest value keeps the
// intermediate images small while leaving enough samples to read each cell
// reliably from its centre.
const cellPixels = 10

// minMarkerArea is the smallest polygon area, in square pixels, that a candidate
// quadrilateral may have. Anything smaller cannot hold a readable grid and is
// almost always noise.
const minMarkerArea = 64.0

// DetectMarkers finds every marker from dict that appears in img and returns,
// for each detection, its four image corners and its identifier. The two
// returned slices are parallel: corners[i] carries the id ids[i].
//
// img may be single-channel (grayscale) or three-channel (RGB); it is not
// modified. The pipeline adaptively thresholds the image, traces external
// contours, keeps convex four-vertex quadrilaterals of sufficient area,
// perspectively unwarps each to a canonical square, Otsu-thresholds it, reads
// the cell grid, and matches it against the dictionary under all four
// rotations within the dictionary's Hamming tolerance.
//
// The returned corners are ordered clockwise starting from the marker's own
// top-left cell, so the ordering is invariant to how the marker is rotated in
// the image: rotating a marker in the scene yields the same id with the corners
// cyclically shifted. Markers whose grid does not match any dictionary entry, or
// whose surrounding border is not black, are rejected. If dict is nil or no
// marker is found, both results are nil.
func DetectMarkers(img *cv.Mat, dict *Dictionary) (corners [][4]cv.Point, ids []int) {
	if img == nil || dict == nil || img.Empty() {
		return nil, nil
	}

	gray := toGray(img)
	bin := thresholdForContours(gray)
	// ChainApproxNone keeps every boundary point, which cv.ApproxPolyDP needs to
	// resolve the four corners reliably (a pre-collapsed run can lose a corner).
	contours, _ := cv.FindContours(bin, cv.RetrExternal, cv.ChainApproxNone)

	for _, c := range contours {
		quad, ok := candidateQuad(c)
		if !ok {
			continue
		}
		read, ok := readMarkerGrid(gray, quad, dict.bitsPerSide)
		if !ok {
			continue
		}
		id, k, ok := matchGrid(dict, read)
		if !ok {
			continue
		}
		ordered := rotateCorners(quad, k)
		if isDuplicate(corners, ids, ordered, id) {
			continue
		}
		corners = append(corners, ordered)
		ids = append(ids, id)
	}
	return corners, ids
}

// toGray returns a single-channel copy of img, converting from RGB when needed.
func toGray(img *cv.Mat) *cv.Mat {
	if img.Channels == 1 {
		return img.Clone()
	}
	return cv.CvtColor(img, cv.ColorRGB2Gray)
}

// thresholdForContours adaptively binarises a grayscale image so that dark
// marker borders become foreground (255) and the lighter background becomes 0,
// which is what cv.FindContours expects. The block size scales with the image so
// the same routine works across a range of marker sizes.
func thresholdForContours(gray *cv.Mat) *cv.Mat {
	minDim := gray.Rows
	if gray.Cols < minDim {
		minDim = gray.Cols
	}
	block := minDim / 8
	if block%2 == 0 {
		block++
	}
	if block < 7 {
		block = 7
	}
	if block >= minDim {
		block = minDim - 1
		if block%2 == 0 {
			block--
		}
		if block < 3 {
			block = 3
		}
	}
	return cv.AdaptiveThreshold(gray, 255, cv.AdaptiveThreshMeanC, cv.ThreshBinaryInv, block, 7)
}

// candidateQuad approximates a contour to a polygon and returns its four corners
// in clockwise winding when the contour is a convex quadrilateral of sufficient
// area. The returned corners keep whatever vertex the approximation started
// from; the true top-left is resolved later from the dictionary match.
func candidateQuad(c cv.Contour) ([4]cv.Point, bool) {
	pts := []cv.Point(c)
	if len(pts) < 4 {
		return [4]cv.Point{}, false
	}
	peri := cv.ArcLength(pts, true)
	approx := cv.ApproxPolyDP(pts, 0.05*peri, true)
	if len(approx) != 4 {
		return [4]cv.Point{}, false
	}
	if !isConvex(approx) {
		return [4]cv.Point{}, false
	}
	if polygonArea(approx) < minMarkerArea {
		return [4]cv.Point{}, false
	}
	var quad [4]cv.Point
	copy(quad[:], approx)
	orientClockwise(&quad)
	return quad, true
}

// isConvex reports whether the polygon's vertices turn consistently in one
// direction, i.e. every consecutive edge pair has the same cross-product sign.
func isConvex(p []cv.Point) bool {
	n := len(p)
	if n < 3 {
		return false
	}
	var sign int
	for i := 0; i < n; i++ {
		a := p[i]
		b := p[(i+1)%n]
		c := p[(i+2)%n]
		cross := (b.X-a.X)*(c.Y-b.Y) - (b.Y-a.Y)*(c.X-b.X)
		if cross == 0 {
			continue
		}
		s := 1
		if cross < 0 {
			s = -1
		}
		if sign == 0 {
			sign = s
		} else if s != sign {
			return false
		}
	}
	return sign != 0
}

// polygonArea returns the absolute area of a polygon via the shoelace formula.
func polygonArea(p []cv.Point) float64 {
	n := len(p)
	var a float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += float64(p[i].X)*float64(p[j].Y) - float64(p[j].X)*float64(p[i].Y)
	}
	return math.Abs(a) / 2
}

// signedArea returns the signed shoelace area of the four corners. In image
// coordinates (y downward) a clockwise polygon has negative signed area.
func signedArea(q [4]cv.Point) float64 {
	var a float64
	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		a += float64(q[i].X)*float64(q[j].Y) - float64(q[j].X)*float64(q[i].Y)
	}
	return a / 2
}

// orientClockwise reorders the quad in place, keeping its first vertex, so that
// the winding is clockwise in image coordinates. This makes the unwarp mapping
// consistent across candidates (never mirrored).
func orientClockwise(q *[4]cv.Point) {
	if signedArea(*q) > 0 {
		// Counter-clockwise: reverse the trailing three vertices.
		q[1], q[3] = q[3], q[1]
	}
}

// readMarkerGrid unwarps the candidate quadrilateral to a canonical square,
// Otsu-thresholds it, verifies the black border, and reads the inner bit grid.
// It returns the flat side*side inner grid (1 white, 0 black) or ok=false when
// the border is not sufficiently black.
func readMarkerGrid(gray *cv.Mat, quad [4]cv.Point, side int) ([]byte, bool) {
	cells := side + 2
	s := cells * cellPixels
	// dst lists the canonical corners in the same (clockwise) winding as quad, so
	// the warp is a pure rotation of the marker and never a mirror image:
	// quad[0]->top-left, quad[1]->bottom-left, quad[2]->bottom-right,
	// quad[3]->top-right of the canonical square.
	dst := [4]cv.Point{
		{X: 0, Y: 0},
		{X: 0, Y: s - 1},
		{X: s - 1, Y: s - 1},
		{X: s - 1, Y: 0},
	}
	square, ok := unwarpSquare(gray, quad, dst, s)
	if !ok {
		return nil, false
	}
	bw, _ := cv.Threshold(square, 0, 255, cv.ThreshBinary|cv.ThreshOtsu)

	// Read every cell of the bordered grid by majority vote over its centre.
	full := make([]byte, cells*cells)
	for r := 0; r < cells; r++ {
		for cIdx := 0; cIdx < cells; cIdx++ {
			if cellWhite(bw, r, cIdx) {
				full[r*cells+cIdx] = 1
			}
		}
	}

	// The border ring must be black; tolerate at most one stray cell so that a
	// single warp artefact does not sink a genuine marker, while random quads
	// (with roughly half-white borders) are still rejected.
	borderErrors := 0
	for r := 0; r < cells; r++ {
		for cIdx := 0; cIdx < cells; cIdx++ {
			if r != 0 && r != cells-1 && cIdx != 0 && cIdx != cells-1 {
				continue
			}
			if full[r*cells+cIdx] == 1 {
				borderErrors++
			}
		}
	}
	if borderErrors > 1 {
		return nil, false
	}

	inner := make([]byte, side*side)
	for i := 0; i < side; i++ {
		for j := 0; j < side; j++ {
			inner[i*side+j] = full[(i+1)*cells+(j+1)]
		}
	}
	return inner, true
}

// unwarpSquare computes the homography from the candidate quad to the canonical
// square and warps gray through it, returning the s-by-s canonical image. It
// recovers from the panic that cv.GetPerspectiveTransform or cv.WarpPerspective
// raise on a degenerate (near-collinear) quad, reporting ok=false instead so a
// bad candidate is skipped rather than crashing detection.
func unwarpSquare(gray *cv.Mat, quad, dst [4]cv.Point, s int) (square *cv.Mat, ok bool) {
	defer func() {
		if recover() != nil {
			square, ok = nil, false
		}
	}()
	h := cv.GetPerspectiveTransform(quad, dst)
	return cv.WarpPerspective(gray, h, s, s, cv.InterLinear), true
}

// cellWhite reports whether cell (row, col) of the canonical square reads as
// white, by averaging the binarised samples in the cell's central region (the
// outer quarter on each side is skipped to avoid bleed from neighbours).
func cellWhite(bw *cv.Mat, row, col int) bool {
	margin := cellPixels / 4
	y0 := row*cellPixels + margin
	y1 := (row+1)*cellPixels - margin
	x0 := col*cellPixels + margin
	x1 := (col+1)*cellPixels - margin
	white, total := 0, 0
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			if y < 0 || y >= bw.Rows || x < 0 || x >= bw.Cols {
				continue
			}
			total++
			if bw.At(y, x, 0) > 127 {
				white++
			}
		}
	}
	return total > 0 && white*2 > total
}

// rotateCorners reorders the candidate corners into the marker's own frame,
// returning them clockwise starting from the marker's top-left cell, so the
// order is invariant to how the marker is rotated in the image.
//
// The unwarp maps quad[0..3] to the canonical top-left, bottom-left,
// bottom-right and top-right corners, so the canonical corners in clockwise
// order (TL, TR, BR, BL) are quad[cwSeq]. When the reading equals the
// dictionary marker rotated clockwise k times, the marker's own clockwise corner
// sequence maps to the canonical clockwise sequence starting at offset k.
func rotateCorners(quad [4]cv.Point, k int) [4]cv.Point {
	cwSeq := [4]int{0, 3, 2, 1} // canonical TL, TR, BR, BL as quad indices
	var out [4]cv.Point
	for m := 0; m < 4; m++ {
		out[m] = quad[cwSeq[(k+m)%4]]
	}
	return out
}

// isDuplicate reports whether an accepted detection with the same id already
// covers nearly the same location, which happens when a marker yields more than
// one qualifying contour. Proximity is judged by centre distance relative to the
// marker's own size.
func isDuplicate(corners [][4]cv.Point, ids []int, cand [4]cv.Point, id int) bool {
	ccx, ccy := quadCenter(cand)
	side := quadSide(cand)
	for i := range corners {
		if ids[i] != id {
			continue
		}
		ex, ey := quadCenter(corners[i])
		if math.Hypot(ccx-ex, ccy-ey) < 0.5*side {
			return true
		}
	}
	return false
}

// quadCenter returns the average of the four corner coordinates.
func quadCenter(q [4]cv.Point) (x, y float64) {
	for _, p := range q {
		x += float64(p.X)
		y += float64(p.Y)
	}
	return x / 4, y / 4
}

// quadSide returns the mean edge length of the quadrilateral.
func quadSide(q [4]cv.Point) float64 {
	var sum float64
	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		sum += math.Hypot(float64(q[j].X-q[i].X), float64(q[j].Y-q[i].Y))
	}
	return sum / 4
}

package objdetect

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// DetectFinderPatterns scans img and returns the centres of every QR finder
// pattern it can locate, ordered strongest-first (by the number of supporting
// row/column crossings). Unlike [QRCodeDetector.Detect], which keeps only the
// three strongest, this returns all candidates and is the basis for locating
// several QR symbols at once with [QRCodeDetector.DetectMulti].
func (d *QRCodeDetector) DetectFinderPatterns(img *cv.Mat) []cv.Point {
	g := matToGray(img)
	thr := d.darkThreshold()
	dark := make([]bool, g.w*g.h)
	for i, v := range g.pix {
		dark[i] = v < thr
	}

	var cands []candidate
	for y := 0; y < g.h; y++ {
		runs, starts, colors := rowRuns(dark, g.w, y)
		for i := 0; i+5 <= len(runs); i++ {
			if !colors[i] {
				continue
			}
			five := runs[i : i+5]
			if !checkFinderRatio(five) {
				continue
			}
			cx := starts[i] + runs[i] + runs[i+1] + runs[i+2]/2
			if !d.confirmVertical(dark, g.w, g.h, cx, y) {
				continue
			}
			total := 0
			for _, r := range five {
				total += r
			}
			cands = append(cands, candidate{
				x: float64(cx), y: float64(y), size: float64(total) / 7.0,
			})
		}
	}
	return clusterCandidates(cands)
}

// DetectMulti locates every QR symbol in img and returns one quadrilateral per
// symbol, the multi-code analogue of OpenCV's
// QRCodeDetector::detectMulti. It first finds all finder patterns with
// [QRCodeDetector.DetectFinderPatterns], then forms QR symbols from triples of
// patterns arranged as a right angle with two equal-length legs (a QR's
// top-left, top-right and bottom-left finders). For each accepted triple it
// returns the four corners in order top-left, top-right, bottom-right,
// bottom-left, with the fourth (bottom-right) corner inferred from the other
// three. found is true when at least one symbol is located.
//
// Quadrilaterals are returned in a deterministic order (by top-left corner,
// top-to-bottom then left-to-right). As with [QRCodeDetector.Detect], the
// payload is not decoded.
func (d *QRCodeDetector) DetectMulti(img *cv.Mat) (quads [][]cv.Point, found bool) {
	centres := d.DetectFinderPatterns(img)
	n := len(centres)
	if n < 3 {
		return nil, false
	}

	type fpt struct{ x, y float64 }
	pts := make([]fpt, n)
	for i, c := range centres {
		pts[i] = fpt{float64(c.X), float64(c.Y)}
	}

	// Enumerate triples and test each vertex as the right-angle corner.
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			for k := j + 1; k < n; k++ {
				triple := [3]int{i, j, k}
				for ci := 0; ci < 3; ci++ {
					a := pts[triple[ci]]
					b := pts[triple[(ci+1)%3]]
					c := pts[triple[(ci+2)%3]]
					v1x, v1y := b.x-a.x, b.y-a.y
					v2x, v2y := c.x-a.x, c.y-a.y
					l1 := math.Hypot(v1x, v1y)
					l2 := math.Hypot(v2x, v2y)
					if l1 < 1 || l2 < 1 {
						continue
					}
					ratio := l1 / l2
					if ratio < 0.75 || ratio > 1.333 {
						continue
					}
					cosang := (v1x*v2x + v1y*v2y) / (l1 * l2)
					if math.Abs(cosang) > 0.25 { // not close to perpendicular
						continue
					}
					// Fourth corner opposite the right angle.
					fx, fy := b.x+v2x, b.y+v2y // = a + v1 + v2
					quad := []cv.Point{
						{X: int(a.x + 0.5), Y: int(a.y + 0.5)},
						{X: int(b.x + 0.5), Y: int(b.y + 0.5)},
						{X: int(fx + 0.5), Y: int(fy + 0.5)},
						{X: int(c.x + 0.5), Y: int(c.y + 0.5)},
					}
					quads = append(quads, quad)
					break // one right-angle corner per triple
				}
			}
		}
	}
	if len(quads) == 0 {
		return nil, false
	}
	sort.Slice(quads, func(a, b int) bool {
		if quads[a][0].Y != quads[b][0].Y {
			return quads[a][0].Y < quads[b][0].Y
		}
		return quads[a][0].X < quads[b][0].X
	})
	return quads, true
}

package ximgproc

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Box is an axis-aligned object-proposal rectangle with a confidence Score. X,Y
// is the top-left corner and W,H the size, all in pixels.
type Box struct {
	X, Y, W, H int
	Score      float64
}

// EdgeBoxes generates object bounding-box proposals from an edge-response map,
// following the objectness idea of Zitnick and Dollár ("Edge Boxes: Locating
// Object Proposals from Edges", 2014): a good object box wholly encloses many
// edges while few edges straddle (cross) its boundary. It returns up to maxBoxes
// candidate [Box] proposals sorted by descending score.
//
// The score of a candidate box is
//
//	(interior edge energy − beta · boundary-band edge energy) / (perimeter^κ),
//
// where the interior energy is the summed edge magnitude strictly inside the
// box, the boundary-band energy is the summed magnitude in a thin frame just
// inside the box edges (edges that leak across the border, penalising boxes that
// merely clip an object), and the perimeter normalisation (κ = 1.5) prevents a
// bias toward ever-larger boxes. Candidates are enumerated over a range of
// positions and sizes and greedily non-maximum suppressed so returned boxes
// overlap by less than 0.5 IoU.
//
// edges is an edge-magnitude map such as the output of
// [StructuredEdgeDetectionLite]; larger values mean stronger edges. maxBoxes
// caps the number of proposals (must be positive). It panics if maxBoxes ≤ 0 or
// edges is empty. The result is deterministic.
func EdgeBoxes(edges *cv.FloatMat, maxBoxes int) []Box {
	if maxBoxes <= 0 {
		panic("ximgproc: EdgeBoxes requires maxBoxes > 0")
	}
	rows, cols := edges.Rows, edges.Cols
	if rows == 0 || cols == 0 {
		panic("ximgproc: EdgeBoxes given an empty edge map")
	}

	// Integral image of edge magnitude for O(1) rectangle sums.
	ii := integral(edges.Data, rows, cols)
	sum := func(y0, x0, y1, x1 int) float64 {
		if x1 < x0 || y1 < y0 {
			return 0
		}
		return rectSum(ii, cols, y0, x0, y1, x1)
	}

	const (
		beta = 1.0
		band = 2 // boundary-band thickness in pixels
	)
	// Candidate sizes and step, scaled to the image.
	minSide := rows / 6
	if c := cols / 6; c < minSide {
		minSide = c
	}
	if minSide < 4 {
		minSide = 4
	}
	step := minSide / 2
	if step < 2 {
		step = 2
	}

	var cands []Box
	for h := minSide; h <= rows; h = h * 3 / 2 {
		for w := minSide; w <= cols; w = w * 3 / 2 {
			for y := 0; y+h <= rows; y += step {
				for x := 0; x+w <= cols; x += step {
					x1 := x + w - 1
					y1 := y + h - 1
					interior := sum(y, x, y1, x1)
					// Inner rectangle after removing the boundary band.
					iy0, ix0 := y+band, x+band
					iy1, ix1 := y1-band, x1-band
					var inner float64
					if iy1 >= iy0 && ix1 >= ix0 {
						inner = sum(iy0, ix0, iy1, ix1)
					}
					boundary := interior - inner
					perim := float64(2 * (w + h))
					score := (inner - beta*boundary) / math.Pow(perim, 1.5)
					if score <= 0 {
						continue
					}
					cands = append(cands, Box{X: x, Y: y, W: w, H: h, Score: score})
				}
			}
		}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].Score > cands[j].Score })

	// Greedy non-maximum suppression by IoU.
	var out []Box
	for _, c := range cands {
		keep := true
		for _, k := range out {
			if iou(c, k) > 0.5 {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, c)
			if len(out) >= maxBoxes {
				break
			}
		}
	}
	return out
}

// iou returns the intersection-over-union of two boxes.
func iou(a, b Box) float64 {
	ax1, ay1 := a.X+a.W, a.Y+a.H
	bx1, by1 := b.X+b.W, b.Y+b.H
	ix0 := maxInt(a.X, b.X)
	iy0 := maxInt(a.Y, b.Y)
	ix1 := minInt(ax1, bx1)
	iy1 := minInt(ay1, by1)
	iw := ix1 - ix0
	ih := iy1 - iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	union := float64(a.W*a.H+b.W*b.H) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

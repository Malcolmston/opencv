package ximgproc

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// LineSeg is a detected straight line segment with floating-point endpoints
// (x1,y1)–(x2,y2) in image pixel coordinates.
type LineSeg struct {
	X1, Y1 float64
	X2, Y2 float64
}

// Length returns the Euclidean length of the segment in pixels.
func (s LineSeg) Length() float64 {
	dx := s.X2 - s.X1
	dy := s.Y2 - s.Y1
	return math.Hypot(dx, dy)
}

// FastLineDetector detects straight line segments in img and returns them as a
// slice of [LineSeg]. It first extracts a binary edge map with [cv.Canny]
// (low/high thresholds 50/150), accumulates the edge points into a Hough
// (ρ,θ) table, selects the strongest, well-separated peaks, and for each peak
// walks the supporting edge pixels to recover the segment's actual endpoints
// (splitting on large gaps). Segments shorter than 10 pixels are discarded.
//
// img may be 1- or 3-channel; colour is reduced to luma. The result is sorted
// by descending length so the most prominent segments come first. This is a
// lightweight detector intended for clear, high-contrast lines rather than a
// full LSD implementation.
func FastLineDetector(img *cv.Mat) []LineSeg {
	gray := toGray(img)
	edges := cv.Canny(gray, 50, 150)
	rows, cols := edges.Rows, edges.Cols

	// Collect edge points.
	type pt struct{ x, y int }
	var pts []pt
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if edges.Data[y*cols+x] != 0 {
				pts = append(pts, pt{x, y})
			}
		}
	}
	if len(pts) == 0 {
		return nil
	}

	const numTheta = 180
	thetaStep := math.Pi / numTheta
	diag := math.Hypot(float64(rows), float64(cols))
	rhoMax := int(math.Ceil(diag))
	numRho := 2*rhoMax + 1

	cosT := make([]float64, numTheta)
	sinT := make([]float64, numTheta)
	for t := 0; t < numTheta; t++ {
		cosT[t] = math.Cos(float64(t) * thetaStep)
		sinT[t] = math.Sin(float64(t) * thetaStep)
	}

	acc := make([]int, numTheta*numRho)
	for _, p := range pts {
		for t := 0; t < numTheta; t++ {
			rho := float64(p.x)*cosT[t] + float64(p.y)*sinT[t]
			ri := int(math.Round(rho)) + rhoMax
			if ri < 0 || ri >= numRho {
				continue
			}
			acc[t*numRho+ri]++
		}
	}

	// Threshold on votes: at least a quarter of the peak, and never below 20.
	maxVotes := 0
	for _, v := range acc {
		if v > maxVotes {
			maxVotes = v
		}
	}
	thresh := maxVotes / 4
	if thresh < 20 {
		thresh = 20
	}

	type peak struct {
		t, r, votes int
	}
	var peaks []peak
	for t := 0; t < numTheta; t++ {
		for r := 0; r < numRho; r++ {
			v := acc[t*numRho+r]
			if v < thresh {
				continue
			}
			// 3×3 local maximum in the accumulator.
			if isLocalMax(acc, numTheta, numRho, t, r) {
				peaks = append(peaks, peak{t, r, v})
			}
		}
	}
	sort.Slice(peaks, func(i, j int) bool { return peaks[i].votes > peaks[j].votes })

	var segs []LineSeg
	for _, pk := range peaks {
		rho := float64(pk.r - rhoMax)
		ct, stv := cosT[pk.t], sinT[pk.t]
		// Direction along the line (perpendicular to the (cosθ,sinθ) normal).
		dxL, dyL := -stv, ct

		// Supporting edge points within 1.5px of the line, keyed by projection.
		type proj struct {
			s    float64
			x, y int
		}
		var support []proj
		for _, p := range pts {
			d := float64(p.x)*ct + float64(p.y)*stv - rho
			if math.Abs(d) > 1.5 {
				continue
			}
			s := float64(p.x)*dxL + float64(p.y)*dyL
			support = append(support, proj{s, p.x, p.y})
		}
		if len(support) < 10 {
			continue
		}
		sort.Slice(support, func(i, j int) bool { return support[i].s < support[j].s })

		// Split into runs wherever the gap along the line exceeds 5 px.
		runStart := 0
		for i := 1; i <= len(support); i++ {
			gap := i == len(support) || support[i].s-support[i-1].s > 5
			if !gap {
				continue
			}
			a := support[runStart]
			b := support[i-1]
			seg := LineSeg{X1: float64(a.x), Y1: float64(a.y), X2: float64(b.x), Y2: float64(b.y)}
			if seg.Length() >= 10 {
				segs = append(segs, seg)
			}
			runStart = i
		}
	}

	sort.Slice(segs, func(i, j int) bool { return segs[i].Length() > segs[j].Length() })
	return segs
}

// isLocalMax reports whether accumulator cell (t,r) is >= all of its 3×3
// neighbours (with rho clamped, theta not wrapped).
func isLocalMax(acc []int, numTheta, numRho, t, r int) bool {
	v := acc[t*numRho+r]
	for dt := -1; dt <= 1; dt++ {
		tt := t + dt
		if tt < 0 || tt >= numTheta {
			continue
		}
		for dr := -1; dr <= 1; dr++ {
			rr := r + dr
			if rr < 0 || rr >= numRho {
				continue
			}
			if acc[tt*numRho+rr] > v {
				return false
			}
		}
	}
	return true
}

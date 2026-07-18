package shapefit

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// HoughLine is a line detected by the standard Hough transform, parameterized
// in polar (rho, theta) form: the line is the set of points where
// x·cos(Theta) + y·sin(Theta) = Rho.
type HoughLine struct {
	// Rho is the signed distance from the origin to the line.
	Rho float64
	// Theta is the angle of the line normal in radians, in [0, π).
	Theta float64
	// Votes is the number of edge pixels that voted for the line.
	Votes int
}

// ToLine converts the polar Hough line to the normalized normal-form [Line].
func (h HoughLine) ToLine() Line {
	return Line{A: math.Cos(h.Theta), B: math.Sin(h.Theta), C: -h.Rho}
}

// HoughLines runs the standard Hough line transform over the foreground pixels
// of a binary image (a pixel is foreground when its first-channel sample is
// nonzero). rhoStep is the distance resolution in pixels and thetaStep the
// angular resolution in radians. Every accumulator bin that is a local maximum
// with at least threshold votes yields one [HoughLine]; the result is sorted by
// descending vote count.
func HoughLines(src *cv.Mat, rhoStep, thetaStep float64, threshold int) []HoughLine {
	if src == nil || src.Empty() || rhoStep <= 0 || thetaStep <= 0 {
		return nil
	}
	w, h := src.Cols, src.Rows
	diag := math.Hypot(float64(w), float64(h))
	numTheta := int(math.Round(math.Pi / thetaStep))
	if numTheta < 1 {
		numTheta = 1
	}
	numRho := int(math.Round(2*diag/rhoStep)) + 1
	cosT := make([]float64, numTheta)
	sinT := make([]float64, numTheta)
	for t := 0; t < numTheta; t++ {
		ang := float64(t) * thetaStep
		cosT[t] = math.Cos(ang)
		sinT[t] = math.Sin(ang)
	}
	acc := make([]int, numTheta*numRho)
	at := func(t, r int) int { return t*numRho + r }
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if src.At(y, x, 0) == 0 {
				continue
			}
			fx, fy := float64(x), float64(y)
			for t := 0; t < numTheta; t++ {
				rho := fx*cosT[t] + fy*sinT[t]
				ri := int(math.Round((rho + diag) / rhoStep))
				if ri >= 0 && ri < numRho {
					acc[at(t, ri)]++
				}
			}
		}
	}
	var out []HoughLine
	for t := 0; t < numTheta; t++ {
		for r := 0; r < numRho; r++ {
			v := acc[at(t, r)]
			if v < threshold {
				continue
			}
			if !shapefitIsPeak2D(acc, t, r, numTheta, numRho) {
				continue
			}
			out = append(out, HoughLine{
				Rho:   float64(r)*rhoStep - diag,
				Theta: float64(t) * thetaStep,
				Votes: v,
			})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Votes > out[j].Votes })
	return out
}

// shapefitIsPeak2D reports whether accumulator bin (t, r) is at least as large
// as its eight neighbours in the (theta, rho) grid.
func shapefitIsPeak2D(acc []int, t, r, nt, nr int) bool {
	v := acc[t*nr+r]
	for dt := -1; dt <= 1; dt++ {
		for dr := -1; dr <= 1; dr++ {
			if dt == 0 && dr == 0 {
				continue
			}
			tt := t + dt
			rr := r + dr
			if tt < 0 || tt >= nt || rr < 0 || rr >= nr {
				continue
			}
			if acc[tt*nr+rr] > v {
				return false
			}
		}
	}
	return true
}

// Segment is a finite line segment between two endpoints.
type Segment struct {
	// A and B are the segment endpoints.
	A, B cv.Point2f
}

// Length returns the Euclidean length of the segment.
func (s Segment) Length() float64 {
	return math.Hypot(s.B.X-s.A.X, s.B.Y-s.A.Y)
}

// HoughLinesP detects line segments in a binary image. It first finds the
// dominant lines with the standard Hough transform, then, for each line,
// gathers the foreground pixels lying within rhoStep of it, orders them along
// the line and splits them into maximal runs whose gaps do not exceed
// maxLineGap. Runs at least minLineLength long are returned as segments. The
// procedure is fully deterministic. rhoStep and thetaStep set the Hough
// resolution and threshold the minimum line votes.
func HoughLinesP(src *cv.Mat, rhoStep, thetaStep float64, threshold, minLineLength, maxLineGap int) []Segment {
	if src == nil || src.Empty() {
		return nil
	}
	lines := HoughLines(src, rhoStep, thetaStep, threshold)
	pts := PointsFromMat(src, 0)
	used := make([]bool, len(pts))
	var out []Segment
	for _, hl := range lines {
		line := hl.ToLine()
		dir := line.Direction()
		// Collect unused inliers, projecting each onto the line direction.
		type proj struct {
			i int
			t float64
			p cv.Point2f
		}
		var members []proj
		for i, p := range pts {
			if used[i] {
				continue
			}
			if line.Distance(p) <= rhoStep {
				t := p.X*dir.X + p.Y*dir.Y
				members = append(members, proj{i, t, p})
			}
		}
		if len(members) < 2 {
			continue
		}
		sort.Slice(members, func(a, b int) bool { return members[a].t < members[b].t })
		// Split into runs separated by gaps larger than maxLineGap.
		runStart := 0
		flush := func(lo, hi int) {
			if hi-lo < 1 {
				return
			}
			seg := Segment{A: members[lo].p, B: members[hi].p}
			if seg.Length() >= float64(minLineLength) {
				for k := lo; k <= hi; k++ {
					used[members[k].i] = true
				}
				out = append(out, seg)
			}
		}
		for k := 1; k < len(members); k++ {
			if members[k].t-members[k-1].t > float64(maxLineGap) {
				flush(runStart, k-1)
				runStart = k
			}
		}
		flush(runStart, len(members)-1)
	}
	return out
}

// HoughCircle is a circle detected by the Hough transform.
type HoughCircle struct {
	// Center is the detected circle center.
	Center cv.Point2f
	// Radius is the detected radius.
	Radius float64
	// Votes is the accumulator support for the circle.
	Votes int
}

// ToCircle converts the detection to a plain [Circle].
func (h HoughCircle) ToCircle() Circle {
	return Circle{Center: h.Center, Radius: h.Radius}
}

// HoughCircles detects circles in a binary edge image by voting into a
// per-radius center accumulator. For each integer radius r in
// [minRadius, maxRadius] and each foreground pixel, every possible center on the
// circle of radius r around that pixel casts a vote. Accumulator peaks with at
// least threshold votes become detections; detections are suppressed if a
// stronger one lies within minDist pixels. Results are sorted by descending
// votes. This routine is O(radii · edges · angular-samples) and is intended for
// modest images.
func HoughCircles(src *cv.Mat, minRadius, maxRadius, threshold int, minDist float64) []HoughCircle {
	if src == nil || src.Empty() || minRadius < 1 || maxRadius < minRadius {
		return nil
	}
	w, h := src.Cols, src.Rows
	pts := PointsFromMat(src, 0)
	var cands []HoughCircle
	for r := minRadius; r <= maxRadius; r++ {
		acc := make([]int, w*h)
		// Angular sampling proportional to circumference keeps the ring dense.
		steps := int(2 * math.Pi * float64(r))
		if steps < 24 {
			steps = 24
		}
		cosv := make([]float64, steps)
		sinv := make([]float64, steps)
		for s := 0; s < steps; s++ {
			ang := 2 * math.Pi * float64(s) / float64(steps)
			cosv[s] = math.Cos(ang)
			sinv[s] = math.Sin(ang)
		}
		fr := float64(r)
		for _, p := range pts {
			for s := 0; s < steps; s++ {
				cx := int(math.Round(p.X - fr*cosv[s]))
				cy := int(math.Round(p.Y - fr*sinv[s]))
				if cx >= 0 && cx < w && cy >= 0 && cy < h {
					acc[cy*w+cx]++
				}
			}
		}
		for cy := 0; cy < h; cy++ {
			for cx := 0; cx < w; cx++ {
				v := acc[cy*w+cx]
				if v < threshold {
					continue
				}
				if !shapefitIsPeakGrid(acc, cx, cy, w, h) {
					continue
				}
				cands = append(cands, HoughCircle{
					Center: cv.Point2f{X: float64(cx), Y: float64(cy)},
					Radius: fr,
					Votes:  v,
				})
			}
		}
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].Votes > cands[j].Votes })
	// Non-maximum suppression across radii by center proximity.
	var out []HoughCircle
	for _, c := range cands {
		keep := true
		for _, k := range out {
			if math.Hypot(c.Center.X-k.Center.X, c.Center.Y-k.Center.Y) < minDist {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, c)
		}
	}
	return out
}

// shapefitIsPeakGrid reports whether accumulator cell (x, y) is at least as
// large as its eight neighbours in a w×h grid.
func shapefitIsPeakGrid(acc []int, x, y, w, h int) bool {
	v := acc[y*w+x]
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			nx := x + dx
			ny := y + dy
			if nx < 0 || nx >= w || ny < 0 || ny >= h {
				continue
			}
			if acc[ny*w+nx] > v {
				return false
			}
		}
	}
	return true
}

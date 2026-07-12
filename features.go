package cv

import (
	"math"
	"sort"
)

// boxSum sums a per-pixel float field over a blockSize×blockSize window centred
// on each pixel, replicating the border.
func boxSum(field []float64, rows, cols, blockSize int) []float64 {
	out := make([]float64, rows*cols)
	a := blockSize / 2
	clamp := func(v, hi int) int {
		if v < 0 {
			return 0
		}
		if v >= hi {
			return hi - 1
		}
		return v
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for dy := -a; dy <= a; dy++ {
				ry := clamp(y+dy, rows)
				for dx := -a; dx <= a; dx++ {
					rx := clamp(x+dx, cols)
					s += field[ry*cols+rx]
				}
			}
			out[y*cols+x] = s
		}
	}
	return out
}

// structureTensor computes the windowed sums of the gradient products
// (Sxx, Syy, Sxy) used by both Harris and Shi–Tomasi corner measures.
func structureTensor(src *Mat, blockSize, ksize int) (sxx, syy, sxy []float64) {
	requireChannels(src, 1, "corner detection")
	ix := SobelFloat(src, 1, 0, ksize)[0]
	iy := SobelFloat(src, 0, 1, ksize)[0]
	n := src.Total()
	ixx := make([]float64, n)
	iyy := make([]float64, n)
	ixy := make([]float64, n)
	for i := 0; i < n; i++ {
		ixx[i] = ix[i] * ix[i]
		iyy[i] = iy[i] * iy[i]
		ixy[i] = ix[i] * iy[i]
	}
	sxx = boxSum(ixx, src.Rows, src.Cols, blockSize)
	syy = boxSum(iyy, src.Rows, src.Cols, blockSize)
	sxy = boxSum(ixy, src.Rows, src.Cols, blockSize)
	return
}

// CornerHarris computes the Harris corner response of a single-channel image.
// For each pixel it forms the structure tensor M summed over a blockSize window
// of Sobel gradients (aperture ksize) and returns R = det(M) - k*trace(M)^2 as a
// [FloatMat]. Large positive responses indicate corners. It panics if src is
// not single-channel.
func CornerHarris(src *Mat, blockSize, ksize int, k float64) *FloatMat {
	sxx, syy, sxy := structureTensor(src, blockSize, ksize)
	res := NewFloatMat(src.Rows, src.Cols)
	for i := range res.Data {
		det := sxx[i]*syy[i] - sxy[i]*sxy[i]
		trace := sxx[i] + syy[i]
		res.Data[i] = det - k*trace*trace
	}
	return res
}

// GoodFeaturesToTrack finds strong corners with the Shi–Tomasi measure (the
// smaller eigenvalue of the windowed structure tensor). Corners weaker than
// qualityLevel times the strongest are discarded, the survivors are taken in
// descending strength, and each accepted corner suppresses others within
// minDistance. At most maxCorners points are returned (all of them when
// maxCorners <= 0). blockSize is the tensor window size. It panics if src is not
// single-channel.
func GoodFeaturesToTrack(src *Mat, maxCorners int, qualityLevel, minDistance float64, blockSize int) []Point {
	sxx, syy, sxy := structureTensor(src, blockSize, 3)
	rows, cols := src.Rows, src.Cols
	resp := make([]float64, rows*cols)
	var maxResp float64
	for i := range resp {
		a := sxx[i]
		b := sxy[i]
		c := syy[i]
		// Smaller eigenvalue of [[a,b],[b,c]].
		tr := a + c
		disc := math.Sqrt((a-c)*(a-c) + 4*b*b)
		lambda := (tr - disc) / 2
		resp[i] = lambda
		if lambda > maxResp {
			maxResp = lambda
		}
	}
	if maxResp <= 0 {
		return nil
	}
	threshold := qualityLevel * maxResp

	type cand struct {
		p Point
		v float64
	}
	var cands []cand
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			i := y*cols + x
			v := resp[i]
			if v < threshold {
				continue
			}
			// 3×3 non-maximum suppression.
			isMax := true
			for dy := -1; dy <= 1 && isMax; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if resp[(y+dy)*cols+(x+dx)] > v {
						isMax = false
						break
					}
				}
			}
			if isMax {
				cands = append(cands, cand{Point{X: x, Y: y}, v})
			}
		}
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].v > cands[j].v })

	var out []Point
	minD2 := minDistance * minDistance
	for _, c := range cands {
		ok := true
		for _, p := range out {
			dx := float64(c.p.X - p.X)
			dy := float64(c.p.Y - p.Y)
			if dx*dx+dy*dy < minD2 {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		out = append(out, c.p)
		if maxCorners > 0 && len(out) >= maxCorners {
			break
		}
	}
	return out
}

// PolarLine is a line in Hesse normal form: the set of points (x, y) with
// x*cos(Theta) + y*sin(Theta) = Rho. Rho is in pixels and Theta in radians,
// matching the output of [HoughLines].
type PolarLine struct {
	Rho   float64
	Theta float64
}

// HoughLines detects lines in a binary edge image with the standard Hough
// transform. It accumulates votes over a (rho, theta) grid quantised by rhoStep
// (pixels) and thetaStep (radians), then returns every accumulator peak — a
// local maximum whose vote count is at least threshold — as a [PolarLine],
// sorted by descending votes. It panics if src is not single-channel.
func HoughLines(src *Mat, rhoStep, thetaStep float64, threshold int) []PolarLine {
	requireChannels(src, 1, "HoughLines")
	rows, cols := src.Rows, src.Cols
	maxRho := math.Hypot(float64(rows), float64(cols))
	numRho := int(math.Round(2*maxRho/rhoStep)) + 1
	numTheta := int(math.Round(math.Pi / thetaStep))
	if numTheta < 1 {
		numTheta = 1
	}
	cosT := make([]float64, numTheta)
	sinT := make([]float64, numTheta)
	for t := 0; t < numTheta; t++ {
		theta := float64(t) * thetaStep
		cosT[t] = math.Cos(theta)
		sinT[t] = math.Sin(theta)
	}
	acc := make([]int, numTheta*numRho)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if src.Data[y*cols+x] == 0 {
				continue
			}
			for t := 0; t < numTheta; t++ {
				rho := float64(x)*cosT[t] + float64(y)*sinT[t]
				r := int(math.Round((rho + maxRho) / rhoStep))
				if r < 0 || r >= numRho {
					continue
				}
				acc[t*numRho+r]++
			}
		}
	}

	type peak struct {
		votes int
		line  PolarLine
	}
	var peaks []peak
	for t := 0; t < numTheta; t++ {
		for r := 0; r < numRho; r++ {
			v := acc[t*numRho+r]
			if v < threshold {
				continue
			}
			// Local maximum in the (theta, rho) neighbourhood.
			isMax := true
			for dt := -1; dt <= 1 && isMax; dt++ {
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
						isMax = false
						break
					}
				}
			}
			if !isMax {
				continue
			}
			peaks = append(peaks, peak{
				votes: v,
				line:  PolarLine{Rho: float64(r)*rhoStep - maxRho, Theta: float64(t) * thetaStep},
			})
		}
	}
	sort.SliceStable(peaks, func(i, j int) bool { return peaks[i].votes > peaks[j].votes })
	out := make([]PolarLine, len(peaks))
	for i, p := range peaks {
		out[i] = p.line
	}
	return out
}

// LineSegment is a finite line segment between two end points, as returned by
// [HoughLinesP].
type LineSegment struct {
	Pt1 Point
	Pt2 Point
}

// HoughLinesP detects line segments in a binary edge image. It runs the standard
// Hough transform to find candidate line directions, then, for each accumulator
// peak (at least threshold votes), gathers the edge pixels lying on that line,
// orders them, and splits them into segments, bridging gaps up to maxLineGap and
// emitting segments at least minLineLength long. The traversal is deterministic
// (raster order), unlike OpenCV's randomised implementation. It panics if src is
// not single-channel.
func HoughLinesP(src *Mat, rhoStep, thetaStep float64, threshold, minLineLength, maxLineGap int) []LineSegment {
	requireChannels(src, 1, "HoughLinesP")
	rows, cols := src.Rows, src.Cols
	lines := HoughLines(src, rhoStep, thetaStep, threshold)
	used := make([]bool, rows*cols)
	var segs []LineSegment
	minLen2 := float64(minLineLength * minLineLength)

	for _, ln := range lines {
		ct, st := math.Cos(ln.Theta), math.Sin(ln.Theta)
		// Collect edge pixels within half a pixel of the line, keyed by their
		// position along the line direction (-sin, cos).
		type onLine struct {
			t float64
			p Point
		}
		var pts []onLine
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				idx := y*cols + x
				if src.Data[idx] == 0 || used[idx] {
					continue
				}
				d := float64(x)*ct + float64(y)*st - ln.Rho
				if d < -1.0 || d > 1.0 {
					continue
				}
				t := -float64(x)*st + float64(y)*ct
				pts = append(pts, onLine{t: t, p: Point{X: x, Y: y}})
			}
		}
		if len(pts) < 2 {
			continue
		}
		sort.SliceStable(pts, func(i, j int) bool { return pts[i].t < pts[j].t })
		start := 0
		for i := 1; i <= len(pts); i++ {
			gap := false
			if i == len(pts) {
				gap = true
			} else if pts[i].t-pts[i-1].t > float64(maxLineGap) {
				gap = true
			}
			if !gap {
				continue
			}
			p1 := pts[start].p
			p2 := pts[i-1].p
			dx := float64(p2.X - p1.X)
			dy := float64(p2.Y - p1.Y)
			if dx*dx+dy*dy >= minLen2 {
				segs = append(segs, LineSegment{Pt1: p1, Pt2: p2})
				for j := start; j < i; j++ {
					used[pts[j].p.Y*cols+pts[j].p.X] = true
				}
			}
			start = i
		}
	}
	return segs
}

// HoughCircle is a detected circle with an integer centre and radius, as
// returned by [HoughCircles].
type HoughCircle struct {
	X      int
	Y      int
	Radius int
}

// HoughCircles detects circles in a single-channel (typically grayscale) image
// using the Hough gradient method. Edges are found with [Canny] at high
// threshold param1 (low threshold param1/2); every edge pixel then votes for
// centre candidates along its gradient direction across the radius range
// [minRadius, maxRadius]. Accumulator peaks of at least param2 votes become
// circles, and detections closer than minDist are merged. It panics if src is
// not single-channel or the radius range is invalid.
func HoughCircles(src *Mat, minDist, param1, param2 float64, minRadius, maxRadius int) []HoughCircle {
	requireChannels(src, 1, "HoughCircles")
	if minRadius < 1 || maxRadius < minRadius {
		panic("cv: HoughCircles requires 1 <= minRadius <= maxRadius")
	}
	rows, cols := src.Rows, src.Cols
	edges := Canny(src, param1/2, param1)
	gx := SobelFloat(src, 1, 0, 3)[0]
	gy := SobelFloat(src, 0, 1, 3)[0]

	acc := make([]int, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if edges.Data[y*cols+x] == 0 {
				continue
			}
			g := math.Hypot(gx[y*cols+x], gy[y*cols+x])
			if g == 0 {
				continue
			}
			ux := gx[y*cols+x] / g
			uy := gy[y*cols+x] / g
			for _, sgn := range []float64{1, -1} {
				for r := minRadius; r <= maxRadius; r++ {
					cx := int(math.Round(float64(x) + sgn*ux*float64(r)))
					cy := int(math.Round(float64(y) + sgn*uy*float64(r)))
					if cx < 0 || cx >= cols || cy < 0 || cy >= rows {
						continue
					}
					acc[cy*cols+cx]++
				}
			}
		}
	}

	// Smooth the accumulator (3×3 sum) so votes spread by edge thickness and
	// radius quantisation coalesce into a single peak near the true centre.
	sm := make([]int, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			s := 0
			for dy := -1; dy <= 1; dy++ {
				ny := y + dy
				if ny < 0 || ny >= rows {
					continue
				}
				for dx := -1; dx <= 1; dx++ {
					nx := x + dx
					if nx < 0 || nx >= cols {
						continue
					}
					s += acc[ny*cols+nx]
				}
			}
			sm[y*cols+x] = s
		}
	}

	type cand struct {
		votes int
		x, y  int
	}
	var cands []cand
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			v := sm[y*cols+x]
			if float64(v) < param2 {
				continue
			}
			isMax := true
			for dy := -1; dy <= 1 && isMax; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if sm[(y+dy)*cols+(x+dx)] > v {
						isMax = false
						break
					}
				}
			}
			if isMax {
				cands = append(cands, cand{v, x, y})
			}
		}
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].votes > cands[j].votes })

	var out []HoughCircle
	minDist2 := minDist * minDist
	for _, c := range cands {
		tooClose := false
		for _, o := range out {
			dx := float64(c.x - o.X)
			dy := float64(c.y - o.Y)
			if dx*dx+dy*dy < minDist2 {
				tooClose = true
				break
			}
		}
		if tooClose {
			continue
		}
		// Estimate the radius as the most common edge-to-centre distance.
		radius := estimateRadius(edges, c.x, c.y, minRadius, maxRadius)
		out = append(out, HoughCircle{X: c.x, Y: c.y, Radius: radius})
	}
	return out
}

// estimateRadius picks the radius in [minR,maxR] with the most edge pixels at
// that distance from (cx, cy).
func estimateRadius(edges *Mat, cx, cy, minR, maxR int) int {
	rows, cols := edges.Rows, edges.Cols
	hist := make([]int, maxR+1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if edges.Data[y*cols+x] == 0 {
				continue
			}
			d := int(math.Round(math.Hypot(float64(x-cx), float64(y-cy))))
			if d >= minR && d <= maxR {
				hist[d]++
			}
		}
	}
	best, bestR := -1, minR
	for r := minR; r <= maxR; r++ {
		if hist[r] > best {
			best = hist[r]
			bestR = r
		}
	}
	return bestR
}

// fastCircle holds the 16 Bresenham-circle offsets (radius 3) scanned by FAST,
// in clockwise order starting at the top.
var fastCircle = [16][2]int{
	{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
	{0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3},
}

// FASTCorners detects corners with the FAST-9 algorithm on a single-channel
// image: a pixel is a corner when at least 9 contiguous pixels on the radius-3
// Bresenham circle are all brighter than the centre plus threshold or all
// darker than the centre minus threshold. When nonmaxSuppression is true,
// corners that are not the strongest (by summed absolute contrast) within their
// 3×3 neighbourhood are discarded. It panics if src is not single-channel.
func FASTCorners(src *Mat, threshold int, nonmaxSuppression bool) []Point {
	requireChannels(src, 1, "FASTCorners")
	rows, cols := src.Rows, src.Cols
	score := make([]float64, rows*cols)
	isCorner := make([]bool, rows*cols)
	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			p := int(src.Data[y*cols+x])
			hi := p + threshold
			lo := p - threshold
			var vals [16]int
			brighter := 0
			darker := 0
			for k := 0; k < 16; k++ {
				v := int(src.Data[(y+fastCircle[k][1])*cols+(x+fastCircle[k][0])])
				vals[k] = v
				if v > hi {
					brighter++
				} else if v < lo {
					darker++
				}
			}
			if brighter < 9 && darker < 9 {
				continue
			}
			if !fastContiguous(vals, hi, lo) {
				continue
			}
			isCorner[y*cols+x] = true
			var s float64
			for k := 0; k < 16; k++ {
				s += math.Abs(float64(vals[k] - p))
			}
			score[y*cols+x] = s
		}
	}

	var out []Point
	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			i := y*cols + x
			if !isCorner[i] {
				continue
			}
			if nonmaxSuppression {
				s := score[i]
				suppressed := false
				for dy := -1; dy <= 1 && !suppressed; dy++ {
					for dx := -1; dx <= 1; dx++ {
						if dx == 0 && dy == 0 {
							continue
						}
						if score[(y+dy)*cols+(x+dx)] > s {
							suppressed = true
							break
						}
					}
				}
				if suppressed {
					continue
				}
			}
			out = append(out, Point{X: x, Y: y})
		}
	}
	return out
}

// fastContiguous reports whether the 16 circle samples contain a run of at least
// 9 consecutive entries all above hi or all below lo (wrapping around the ring).
func fastContiguous(vals [16]int, hi, lo int) bool {
	for _, test := range []struct {
		bright bool
	}{{true}, {false}} {
		run := 0
		// Scan 16+8 to allow the run to wrap around the circle.
		for k := 0; k < 24; k++ {
			v := vals[k%16]
			ok := false
			if test.bright {
				ok = v > hi
			} else {
				ok = v < lo
			}
			if ok {
				run++
				if run >= 9 {
					return true
				}
			} else {
				run = 0
			}
		}
	}
	return false
}

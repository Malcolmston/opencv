package edges2

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Line is a straight line in the normal (Hesse) parametrisation
// x*cos(Theta) + y*sin(Theta) = Rho, as returned by [HoughLines]. Rho is the
// signed distance from the origin in pixels and Theta the normal angle in
// radians in [0, pi). Votes is the number of edge pixels that supported the
// line in the Hough accumulator.
type Line struct {
	// Rho is the signed distance from the origin to the line, in pixels.
	Rho float64
	// Theta is the angle of the line normal, in radians in [0, pi).
	Theta float64
	// Votes is the accumulator support for the line.
	Votes int
}

// ToSegment clips the infinite line to the rows×cols image rectangle and
// returns the visible portion as a [Segment]. The second result is false when
// the line does not intersect the image.
func (l Line) ToSegment(rows, cols int) (Segment, bool) {
	return LineToSegment(l, rows, cols)
}

// Segment is a finite line segment with floating-point endpoints
// (X1,Y1)-(X2,Y2), as returned by [HoughLinesP] and [LSD].
type Segment struct {
	// X1, Y1 are the coordinates of the first endpoint.
	X1, Y1 float64
	// X2, Y2 are the coordinates of the second endpoint.
	X2, Y2 float64
}

// Length returns the Euclidean length of the segment.
func (s Segment) Length() float64 {
	return math.Hypot(s.X2-s.X1, s.Y2-s.Y1)
}

// Angle returns the orientation of the segment atan2(dy,dx) in radians in the
// range (-pi, pi].
func (s Segment) Angle() float64 {
	return math.Atan2(s.Y2-s.Y1, s.X2-s.X1)
}

// Midpoint returns the coordinates of the segment centre.
func (s Segment) Midpoint() (x, y float64) {
	return (s.X1 + s.X2) / 2, (s.Y1 + s.Y2) / 2
}

// Circle is a circle with centre (X,Y) and the given Radius, as returned by
// [HoughCircles]. Votes is the accumulator support for the circle.
type Circle struct {
	// X is the centre column coordinate.
	X float64
	// Y is the centre row coordinate.
	Y float64
	// Radius is the circle radius in pixels.
	Radius float64
	// Votes is the accumulator support for the circle.
	Votes int
}

// edges2HoughAccumulate builds the (rho, theta) Hough accumulator of a binary
// edge image and returns it together with the parametrisation it used.
func edges2HoughAccumulate(edges *cv.Mat, rhoRes, thetaRes float64) (acc []int, numRho, numTheta, rhoOff int, cosT, sinT []float64) {
	edges2RequireGray(edges, "HoughLines")
	if rhoRes <= 0 {
		rhoRes = 1
	}
	if thetaRes <= 0 {
		thetaRes = math.Pi / 180
	}
	rows, cols := edges.Rows, edges.Cols
	diag := math.Hypot(float64(rows), float64(cols))
	rhoOff = int(math.Ceil(diag / rhoRes))
	numRho = 2*rhoOff + 1
	numTheta = int(math.Ceil(math.Pi / thetaRes))
	cosT = make([]float64, numTheta)
	sinT = make([]float64, numTheta)
	for t := 0; t < numTheta; t++ {
		a := float64(t) * thetaRes
		cosT[t] = math.Cos(a)
		sinT[t] = math.Sin(a)
	}
	acc = make([]int, numRho*numTheta)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if edges.Data[y*cols+x] == 0 {
				continue
			}
			for t := 0; t < numTheta; t++ {
				r := (float64(x)*cosT[t] + float64(y)*sinT[t]) / rhoRes
				ri := int(math.Round(r)) + rhoOff
				if ri >= 0 && ri < numRho {
					acc[ri*numTheta+t]++
				}
			}
		}
	}
	return acc, numRho, numTheta, rhoOff, cosT, sinT
}

// HoughLineAccumulator computes the raw (rho, theta) Hough voting accumulator
// of a binary edge image and returns it as a [FloatGrid] with one row per rho
// bin and one column per theta bin. The theta of column t is t*thetaRes; the
// rho of row r is (r - (Rows-1)/2)*rhoRes. It panics on multi-channel input.
func HoughLineAccumulator(edges *cv.Mat, rhoRes, thetaRes float64) *FloatGrid {
	acc, numRho, numTheta, _, _, _ := edges2HoughAccumulate(edges, rhoRes, thetaRes)
	g := NewFloatGrid(numRho, numTheta)
	for i, v := range acc {
		g.Data[i] = float64(v)
	}
	return g
}

// HoughLines detects straight lines in a binary edge image with the standard
// Hough transform. rhoRes is the distance resolution in pixels and thetaRes the
// angular resolution in radians; a bin is reported as a line when its vote count
// is at least threshold and is a local maximum in the accumulator. The lines are
// returned sorted by descending votes. It panics on multi-channel input.
func HoughLines(edges *cv.Mat, rhoRes, thetaRes float64, threshold int) []Line {
	acc, numRho, numTheta, rhoOff, _, _ := edges2HoughAccumulate(edges, rhoRes, thetaRes)
	if thetaRes <= 0 {
		thetaRes = math.Pi / 180
	}
	if rhoRes <= 0 {
		rhoRes = 1
	}
	var lines []Line
	for ri := 0; ri < numRho; ri++ {
		for t := 0; t < numTheta; t++ {
			v := acc[ri*numTheta+t]
			if v < threshold {
				continue
			}
			if !edges2IsLocalMax(acc, numRho, numTheta, ri, t) {
				continue
			}
			lines = append(lines, Line{
				Rho:   float64(ri-rhoOff) * rhoRes,
				Theta: float64(t) * thetaRes,
				Votes: v,
			})
		}
	}
	sort.SliceStable(lines, func(i, j int) bool { return lines[i].Votes > lines[j].Votes })
	return lines
}

// edges2IsLocalMax reports whether accumulator cell (ri,t) is >= all of its
// eight neighbours, with theta wrapping modulo pi.
func edges2IsLocalMax(acc []int, numRho, numTheta, ri, t int) bool {
	v := acc[ri*numTheta+t]
	for dr := -1; dr <= 1; dr++ {
		nr := ri + dr
		if nr < 0 || nr >= numRho {
			continue
		}
		for dt := -1; dt <= 1; dt++ {
			if dr == 0 && dt == 0 {
				continue
			}
			nt := (t + dt + numTheta) % numTheta
			if acc[nr*numTheta+nt] > v {
				return false
			}
		}
	}
	return true
}

// HoughLinesP extracts finite line segments from a binary edge image. It first
// locates dominant lines with the Hough accumulator (using threshold), then
// walks each line collecting maximal runs of edge pixels, splitting a run when a
// gap longer than maxLineGap pixels is met and keeping only segments at least
// minLineLength pixels long. The procedure is fully deterministic. Segments are
// returned sorted by descending length. It panics on multi-channel input.
func HoughLinesP(edges *cv.Mat, rhoRes, thetaRes float64, threshold int, minLineLength, maxLineGap float64) []Segment {
	edges2RequireGray(edges, "HoughLinesP")
	lines := HoughLines(edges, rhoRes, thetaRes, threshold)
	rows, cols := edges.Rows, edges.Cols
	diag := math.Hypot(float64(rows), float64(cols))
	var segs []Segment
	seen := make(map[[4]int]bool)
	for _, ln := range lines {
		ct := math.Cos(ln.Theta)
		st := math.Sin(ln.Theta)
		// Base point on the line closest to the origin.
		x0 := ct * ln.Rho
		y0 := st * ln.Rho
		// Direction along the line (perpendicular to the normal).
		dx := -st
		dy := ct
		inRun := false
		gap := 0.0
		var sx, sy, ex, ey float64
		flush := func() {
			if inRun {
				seg := Segment{X1: sx, Y1: sy, X2: ex, Y2: ey}
				if seg.Length() >= minLineLength {
					key := [4]int{int(math.Round(sx)), int(math.Round(sy)), int(math.Round(ex)), int(math.Round(ey))}
					if !seen[key] {
						seen[key] = true
						segs = append(segs, seg)
					}
				}
			}
			inRun = false
			gap = 0
		}
		for t := -diag; t <= diag; t += 1.0 {
			x := x0 + dx*t
			y := y0 + dy*t
			xi := int(math.Round(x))
			yi := int(math.Round(y))
			if xi < 0 || xi >= cols || yi < 0 || yi >= rows {
				if inRun {
					gap += 1
					if gap > maxLineGap {
						flush()
					}
				}
				continue
			}
			if edges2EdgeNear(edges, yi, xi) {
				if !inRun {
					inRun = true
					sx, sy = x, y
				}
				ex, ey = x, y
				gap = 0
			} else if inRun {
				gap += 1
				if gap > maxLineGap {
					flush()
				}
			}
		}
		flush()
	}
	sort.SliceStable(segs, func(i, j int) bool { return segs[i].Length() > segs[j].Length() })
	return segs
}

// edges2EdgeNear reports whether (y,x) or one of its 4-neighbours is an edge
// pixel, tolerating the rounding of the line walk.
func edges2EdgeNear(edges *cv.Mat, y, x int) bool {
	rows, cols := edges.Rows, edges.Cols
	if edges.Data[y*cols+x] != 0 {
		return true
	}
	for _, d := range [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} {
		ny := y + d[0]
		nx := x + d[1]
		if ny >= 0 && ny < rows && nx >= 0 && nx < cols && edges.Data[ny*cols+nx] != 0 {
			return true
		}
	}
	return false
}

// LineToSegment clips the infinite line l to the rows×cols image rectangle and
// returns the visible portion as a [Segment]. The second result is false when
// the line does not cross the image.
func LineToSegment(l Line, rows, cols int) (Segment, bool) {
	ct := math.Cos(l.Theta)
	st := math.Sin(l.Theta)
	x0 := ct * l.Rho
	y0 := st * l.Rho
	dx := -st
	dy := ct
	// Intersect the parametric line with the four rectangle borders and keep
	// the extreme parameter values that lie inside the image.
	tMin := math.Inf(1)
	tMax := math.Inf(-1)
	consider := func(t float64) {
		x := x0 + dx*t
		y := y0 + dy*t
		if x >= -1e-6 && x <= float64(cols-1)+1e-6 && y >= -1e-6 && y <= float64(rows-1)+1e-6 {
			if t < tMin {
				tMin = t
			}
			if t > tMax {
				tMax = t
			}
		}
	}
	if math.Abs(dx) > 1e-12 {
		consider((0 - x0) / dx)
		consider((float64(cols-1) - x0) / dx)
	}
	if math.Abs(dy) > 1e-12 {
		consider((0 - y0) / dy)
		consider((float64(rows-1) - y0) / dy)
	}
	if math.IsInf(tMin, 1) || tMax <= tMin {
		return Segment{}, false
	}
	return Segment{
		X1: x0 + dx*tMin, Y1: y0 + dy*tMin,
		X2: x0 + dx*tMax, Y2: y0 + dy*tMax,
	}, true
}

// HoughCircles detects circles in a single-channel image with the gradient
// Hough transform. The image is smoothed and its Canny edges and Sobel gradient
// directions are computed internally; every edge pixel then votes for centre
// candidates a distance r along its gradient, for each integer radius r in
// [minRadius, maxRadius]. A candidate is reported when its votes reach threshold
// and it is a local maximum in the (x,y,r) accumulator; nearer, stronger circles
// suppress weaker ones within minDist pixels. Circles are returned sorted by
// descending votes. It panics on multi-channel input.
func HoughCircles(src *cv.Mat, minRadius, maxRadius, threshold int, cannyLow, cannyHigh, minDist float64) []Circle {
	edges2RequireGray(src, "HoughCircles")
	if minRadius < 1 {
		minRadius = 1
	}
	if maxRadius < minRadius {
		maxRadius = minRadius
	}
	rows, cols := src.Rows, src.Cols
	blurred := edges2Blur(src, 1.2)
	gf := Sobel(blurred)
	edges := CannyField(gf, cannyLow, cannyHigh)
	nR := maxRadius - minRadius + 1
	acc := make([]int, nR*rows*cols)
	idx := func(ri, y, x int) int { return (ri*rows+y)*cols + x }
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if edges.Data[y*cols+x] == 0 {
				continue
			}
			gx, gy := gf.At(y, x)
			norm := math.Hypot(gx, gy)
			if norm == 0 {
				continue
			}
			ux := gx / norm
			uy := gy / norm
			for ri := 0; ri < nR; ri++ {
				r := float64(minRadius + ri)
				for _, s := range [2]float64{1, -1} {
					cx := int(math.Round(float64(x) - s*ux*r))
					cy := int(math.Round(float64(y) - s*uy*r))
					if cx >= 0 && cx < cols && cy >= 0 && cy < rows {
						acc[idx(ri, cy, cx)]++
					}
				}
			}
		}
	}
	var circles []Circle
	for ri := 0; ri < nR; ri++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				v := acc[idx(ri, y, x)]
				if v < threshold {
					continue
				}
				if !edges2IsCircleMax(acc, nR, rows, cols, ri, y, x) {
					continue
				}
				circles = append(circles, Circle{
					X: float64(x), Y: float64(y),
					Radius: float64(minRadius + ri), Votes: v,
				})
			}
		}
	}
	sort.SliceStable(circles, func(i, j int) bool { return circles[i].Votes > circles[j].Votes })
	return edges2SuppressCircles(circles, minDist)
}

// edges2IsCircleMax reports whether accumulator cell (ri,y,x) dominates its
// 3×3×3 neighbourhood in the circle accumulator.
func edges2IsCircleMax(acc []int, nR, rows, cols, ri, y, x int) bool {
	v := acc[(ri*rows+y)*cols+x]
	for dr := -1; dr <= 1; dr++ {
		nr := ri + dr
		if nr < 0 || nr >= nR {
			continue
		}
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
				if dr == 0 && dy == 0 && dx == 0 {
					continue
				}
				if acc[(nr*rows+ny)*cols+nx] > v {
					return false
				}
			}
		}
	}
	return true
}

// edges2SuppressCircles removes circles whose centre lies within minDist pixels
// of an already-accepted, stronger circle.
func edges2SuppressCircles(circles []Circle, minDist float64) []Circle {
	if minDist <= 0 {
		return circles
	}
	var kept []Circle
	for _, c := range circles {
		ok := true
		for _, k := range kept {
			if math.Hypot(c.X-k.X, c.Y-k.Y) < minDist {
				ok = false
				break
			}
		}
		if ok {
			kept = append(kept, c)
		}
	}
	return kept
}

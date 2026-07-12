package linedescriptor

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// EDLinesDetector detects straight line segments with the Edge Drawing Lines
// (EDLines) algorithm of Akinlar and Topal (2011), an alternative front end to
// the LSD detector offered by some builds of the upstream line_descriptor
// module. Where [LSDDetector] grows blobby line-support regions from gradient
// maxima, EDLines first traces clean, one-pixel-wide edge chains and then fits
// straight segments to those chains, which tends to give crisper endpoints on
// well-defined edges.
//
// The exported fields tune the three stages — anchor selection, edge routing
// and line fitting; [NewEDLinesDetector] fills them with sensible defaults.
type EDLinesDetector struct {
	// GradThreshold is the minimum gradient magnitude for a pixel to be part of
	// any edge; pixels below it are treated as flat and terminate a trace.
	GradThreshold float64
	// AnchorThreshold is the extra margin, above GradThreshold, by which a pixel
	// must exceed its neighbours along the gradient direction to be chosen as an
	// anchor (a trace seed). Larger values yield fewer, stronger anchors.
	AnchorThreshold float64
	// MinLineLength discards fitted segments shorter than this many pixels.
	MinLineLength float64
	// FitError is the maximum root-mean-square perpendicular distance, in
	// pixels, between an edge chain and the straight line fitted through it; a
	// chain is split into a new segment wherever this tolerance is exceeded.
	FitError float64
}

// NewEDLinesDetector returns an EDLines detector with defaults matching the
// common EDLines configuration: a gradient floor of 12, an anchor margin of 4,
// a minimum segment length of 10 pixels and a 1.4-pixel line-fit tolerance.
func NewEDLinesDetector() *EDLinesDetector {
	return &EDLinesDetector{
		GradThreshold:   12,
		AnchorThreshold: 4,
		MinLineLength:   10,
		FitError:        1.4,
	}
}

// Detect finds straight line segments in img and returns them as a slice of
// [KeyLine] sorted by descending [KeyLine.Response] (segment length). img may be
// 1- or 3-channel; colour is reduced to luma first. Every returned segment has
// Octave 0.
//
// The algorithm proceeds in four deterministic stages:
//
//  1. Gradients. Per-pixel gx, gy come from the 3×3 Sobel operator; each
//     pixel's magnitude is hypot(gx, gy). A pixel is classified as belonging to
//     a horizontal edge (walk left/right) when |gx| ≥ |gy|, otherwise a vertical
//     edge (walk up/down).
//
//  2. Anchors. A pixel is an anchor when its magnitude exceeds GradThreshold +
//     AnchorThreshold and is a local maximum across the gradient direction. All
//     anchors are visited in order of descending magnitude, ties by pixel index.
//
//  3. Edge drawing. From each unused anchor the trace walks in both directions
//     along the edge, at each step choosing the strongest of the three forward
//     neighbours whose magnitude clears GradThreshold, until it runs off an edge
//     or into already-traced pixels. This yields a one-pixel-wide ordered chain.
//
//  4. Line fitting. Each chain is scanned and approximated by straight segments
//     via an incremental total-least-squares fit: points are accumulated while
//     the perpendicular distance to the running fit stays within FitError, and a
//     segment is emitted (subject to MinLineLength) whenever the tolerance is
//     exceeded or the chain ends.
func (e *EDLinesDetector) Detect(img *cv.Mat) []KeyLine {
	if e.GradThreshold <= 0 || e.MinLineLength <= 0 {
		panic("linedescriptor: EDLinesDetector requires positive GradThreshold and MinLineLength")
	}
	gray := toGray(img)
	gx, gy, rows, cols := gradients(gray)
	n := rows * cols

	mag := make([]float64, n)
	horizontal := make([]bool, n) // true => edge runs horizontally (walk left/right)
	for i := 0; i < n; i++ {
		mag[i] = math.Hypot(gx[i], gy[i])
		// A horizontal edge (walk left/right) has a vertical gradient, i.e. gy
		// dominates gx; a vertical edge (walk up/down) has gx dominant.
		horizontal[i] = math.Abs(gy[i]) > math.Abs(gx[i])
	}

	anchors := e.findAnchors(mag, horizontal, rows, cols)

	used := make([]bool, n)
	var lines []KeyLine
	for _, a := range anchors {
		if used[a] {
			continue
		}
		chain := e.trace(a, mag, horizontal, used, rows, cols)
		if len(chain) < int(math.Ceil(e.MinLineLength)) {
			continue
		}
		lines = append(lines, e.fitChain(chain)...)
	}

	sort.SliceStable(lines, func(i, j int) bool {
		return lines[i].Response > lines[j].Response
	})
	return lines
}

// findAnchors returns the anchor pixel indices sorted by descending magnitude
// (ties by index). A pixel is an anchor when its magnitude clears the anchor
// threshold and dominates its two neighbours across the gradient direction.
func (e *EDLinesDetector) findAnchors(mag []float64, horizontal []bool, rows, cols int) []int {
	thresh := e.GradThreshold + e.AnchorThreshold
	var anchors []int
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			i := y*cols + x
			if mag[i] < thresh {
				continue
			}
			var a, b float64
			if horizontal[i] {
				// Edge horizontal => gradient vertical => compare up/down.
				a = mag[i-cols]
				b = mag[i+cols]
			} else {
				a = mag[i-1]
				b = mag[i+1]
			}
			if mag[i] >= a && mag[i] >= b {
				anchors = append(anchors, i)
			}
		}
	}
	sort.SliceStable(anchors, func(p, q int) bool {
		return mag[anchors[p]] > mag[anchors[q]]
	})
	return anchors
}

// trace walks an edge chain outward from the anchor in both directions and
// returns the ordered pixel coordinates. Pixels are marked used as they are
// consumed so a chain is never traced twice.
func (e *EDLinesDetector) trace(anchor int, mag []float64, horizontal []bool, used []bool, rows, cols int) []cv.Point {
	forward := e.walk(anchor, +1, mag, horizontal, used, rows, cols)
	backward := e.walk(anchor, -1, mag, horizontal, used, rows, cols)
	// backward is anchor-exclusive and ordered from the anchor outward; reverse
	// it and prepend so the chain reads end-to-end.
	chain := make([]cv.Point, 0, len(forward)+len(backward))
	for i := len(backward) - 1; i >= 0; i-- {
		chain = append(chain, backward[i])
	}
	chain = append(chain, forward...)
	return chain
}

// walk traces from the anchor in a single sense (dir = +1 or -1 along the edge
// direction) and returns the visited pixels. The forward call (dir +1) includes
// the anchor itself; the backward call (dir -1) excludes it to avoid a
// duplicate.
func (e *EDLinesDetector) walk(anchor, dir int, mag []float64, horizontal []bool, used []bool, rows, cols int) []cv.Point {
	var pts []cv.Point
	x := anchor % cols
	y := anchor / cols
	if dir > 0 {
		used[anchor] = true
		pts = append(pts, cv.Point{X: x, Y: y})
	}
	for {
		i := y*cols + x
		var candidates [3]int
		var cx, cy [3]int
		if horizontal[i] {
			// Walk left/right; the three forward neighbours differ in row.
			nx := x + dir
			cx[0], cy[0] = nx, y-1
			cx[1], cy[1] = nx, y
			cx[2], cy[2] = nx, y+1
		} else {
			// Walk up/down; the three forward neighbours differ in column.
			ny := y + dir
			cx[0], cy[0] = x-1, ny
			cx[1], cy[1] = x, ny
			cx[2], cy[2] = x+1, ny
		}
		best := -1
		bestMag := e.GradThreshold
		for k := 0; k < 3; k++ {
			if cx[k] < 0 || cx[k] >= cols || cy[k] < 0 || cy[k] >= rows {
				continue
			}
			j := cy[k]*cols + cx[k]
			if used[j] {
				continue
			}
			if mag[j] > bestMag {
				bestMag = mag[j]
				best = k
			}
			candidates[k] = j
		}
		if best < 0 {
			break
		}
		x, y = cx[best], cy[best]
		used[candidates[best]] = true
		pts = append(pts, cv.Point{X: x, Y: y})
	}
	return pts
}

// fitChain approximates an ordered edge chain by one or more straight segments
// using an incremental total-least-squares fit, splitting the chain wherever the
// perpendicular fit error would exceed FitError. Segments shorter than
// MinLineLength are discarded.
func (e *EDLinesDetector) fitChain(chain []cv.Point) []KeyLine {
	var out []KeyLine
	var acc []cv.Point
	flush := func() {
		if len(acc) < 2 {
			acc = acc[:0]
			return
		}
		if kl, ok := segmentFromPoints(acc, e.MinLineLength); ok {
			out = append(out, kl)
		}
		acc = acc[:0]
	}
	for _, p := range chain {
		if len(acc) < 3 {
			acc = append(acc, p)
			continue
		}
		if perpDistance(acc, p) <= e.FitError {
			acc = append(acc, p)
			continue
		}
		flush()
		acc = append(acc, p)
	}
	flush()
	return out
}

// perpDistance returns the perpendicular distance from candidate point p to the
// total-least-squares line fitted through the accumulated points.
func perpDistance(pts []cv.Point, p cv.Point) float64 {
	cx, cy, dirX, dirY := fitLine(pts)
	dx := float64(p.X) - cx
	dy := float64(p.Y) - cy
	return math.Abs(dx*(-dirY) + dy*dirX)
}

// fitLine returns the centroid and unit direction of the total-least-squares
// line through pts (principal eigenvector of the point covariance matrix).
func fitLine(pts []cv.Point) (cx, cy, dirX, dirY float64) {
	n := float64(len(pts))
	for _, p := range pts {
		cx += float64(p.X)
		cy += float64(p.Y)
	}
	cx /= n
	cy /= n
	var sxx, syy, sxy float64
	for _, p := range pts {
		dx := float64(p.X) - cx
		dy := float64(p.Y) - cy
		sxx += dx * dx
		syy += dy * dy
		sxy += dx * dy
	}
	theta := 0.5 * math.Atan2(2*sxy, sxx-syy)
	return cx, cy, math.Cos(theta), math.Sin(theta)
}

// segmentFromPoints fits a line to pts and projects them onto it to obtain the
// two endpoints, returning a KeyLine when the resulting segment is at least
// minLength long.
func segmentFromPoints(pts []cv.Point, minLength float64) (KeyLine, bool) {
	cx, cy, dirX, dirY := fitLine(pts)
	minS, maxS := math.Inf(1), math.Inf(-1)
	for _, p := range pts {
		s := (float64(p.X)-cx)*dirX + (float64(p.Y)-cy)*dirY
		if s < minS {
			minS = s
		}
		if s > maxS {
			maxS = s
		}
	}
	if maxS-minS < minLength {
		return KeyLine{}, false
	}
	return newKeyLine(
		cx+dirX*minS, cy+dirY*minS,
		cx+dirX*maxS, cy+dirY*maxS,
	), true
}

package linedescriptor

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// LSDDetector detects straight line segments with a Line Segment Detector in
// the style of Grompone von Gioi, Jakubowicz, Morel and Randall (2010). Its
// exported fields tune the detection and may be overridden after construction;
// [NewLSDDetector] returns a detector with sensible defaults.
type LSDDetector struct {
	// AngleTol is the level-line orientation tolerance, in radians, used when
	// growing a line-support region: a candidate pixel joins the region only
	// when its level-line angle is within AngleTol of the region's running mean
	// angle. The upstream LSD default is 22.5°.
	AngleTol float64
	// GradThreshold is the minimum gradient magnitude for a pixel to be
	// considered part of any line-support region. Pixels below it are treated
	// as flat. The default derives from AngleTol as 2/sin(AngleTol), matching
	// LSD's gradient-quantisation reasoning.
	GradThreshold float64
	// MinLength discards line-support regions whose fitted segment is shorter
	// than this many pixels.
	MinLength float64
	// MinDensity rejects regions whose aligned-pixel density inside the fitted
	// rectangle (pixels / (length × width)) is below this value; it keeps the
	// detector from accepting blobs as lines. LSD uses 0.7.
	MinDensity float64
	// MinRegionSize discards line-support regions with fewer than this many
	// pixels before any rectangle is fitted.
	MinRegionSize int
}

// NewLSDDetector returns a detector configured with the defaults used by the
// upstream LSD algorithm: a 22.5° orientation tolerance, the derived gradient
// threshold, a minimum segment length of 8 pixels and a minimum aligned-pixel
// density of 0.7.
func NewLSDDetector() *LSDDetector {
	tol := 22.5 * math.Pi / 180
	return &LSDDetector{
		AngleTol:      tol,
		GradThreshold: 2.0 / math.Sin(tol),
		MinLength:     8,
		MinDensity:    0.7,
		MinRegionSize: 5,
	}
}

// Detect finds straight line segments in img and returns them as a slice of
// [KeyLine], sorted by descending [KeyLine.Response] (i.e. by length) so the
// most prominent segments come first. img may be 1- or 3-channel; colour is
// reduced to luma first.
//
// The algorithm is a faithful but simplified LSD:
//
//  1. Gradients. The image is smoothed implicitly by the 3×3 Sobel operator to
//     obtain per-pixel gx, gy. Each pixel's gradient magnitude is hypot(gx,gy)
//     and its "level-line" orientation — the direction of the edge itself,
//     perpendicular to the gradient — is atan2(gx, -gy).
//
//  2. Region growing. Pixels with a magnitude above GradThreshold are visited
//     in order of decreasing magnitude (ties broken by pixel index, for
//     determinism). Starting from an unused pixel as a seed, the region grows
//     breadth-first over 8-connected neighbours, admitting a neighbour when its
//     level-line orientation is within AngleTol of the region's running mean
//     orientation. The mean is tracked as the argument of the summed unit
//     vectors so it stays numerically stable and wraps correctly.
//
//  3. Rectangle approximation. Each region is reduced to a segment by a
//     magnitude-weighted principal-component fit: the centroid and the 2×2
//     second-moment matrix give the elongation direction (the eigenvector of
//     the larger eigenvalue); the region pixels are projected onto that axis to
//     yield the two endpoints, and their spread perpendicular to it gives the
//     rectangle width.
//
//  4. Validation. A region is accepted only when it has at least MinRegionSize
//     pixels, its fitted segment is at least MinLength long, and its
//     aligned-pixel density inside the rectangle is at least MinDensity. This
//     rejects short or fat regions that are not genuine lines. (The upstream
//     NFA / a-contrario validation with rectangle refinement is replaced by
//     this simpler density test.)
//
// Multi-octave detection over a scale pyramid is not implemented; every
// returned segment has Octave 0.
func (d *LSDDetector) Detect(img *cv.Mat) []KeyLine {
	gray := toGray(img)
	gx, gy, rows, cols := gradients(gray)
	n := rows * cols

	mag := make([]float64, n)
	llAngle := make([]float64, n)
	for i := 0; i < n; i++ {
		mag[i] = math.Hypot(gx[i], gy[i])
		// Level-line orientation: perpendicular to the gradient.
		llAngle[i] = math.Atan2(gx[i], -gy[i])
	}

	// Candidate pixels sorted by descending magnitude, ties by index.
	cand := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if mag[i] > d.GradThreshold {
			cand = append(cand, i)
		}
	}
	sort.SliceStable(cand, func(a, b int) bool {
		return mag[cand[a]] > mag[cand[b]]
	})

	used := make([]bool, n)
	var lines []KeyLine

	// Reusable BFS scratch to avoid per-region allocation churn.
	queue := make([]int, 0, 256)
	region := make([]int, 0, 256)

	for _, seed := range cand {
		if used[seed] {
			continue
		}
		// Grow a region from the seed.
		region = region[:0]
		queue = queue[:0]
		used[seed] = true
		region = append(region, seed)
		queue = append(queue, seed)
		sumSin := math.Sin(llAngle[seed])
		sumCos := math.Cos(llAngle[seed])
		regAngle := llAngle[seed]

		for len(queue) > 0 {
			p := queue[len(queue)-1]
			queue = queue[:len(queue)-1]
			py := p / cols
			px := p % cols
			for dy := -1; dy <= 1; dy++ {
				ny := py + dy
				if ny < 0 || ny >= rows {
					continue
				}
				for dx := -1; dx <= 1; dx++ {
					nx := px + dx
					if nx < 0 || nx >= cols {
						continue
					}
					q := ny*cols + nx
					if used[q] || mag[q] <= d.GradThreshold {
						continue
					}
					if angleDiff(llAngle[q], regAngle) > d.AngleTol {
						continue
					}
					used[q] = true
					region = append(region, q)
					queue = append(queue, q)
					sumSin += math.Sin(llAngle[q])
					sumCos += math.Cos(llAngle[q])
					regAngle = math.Atan2(sumSin, sumCos)
				}
			}
		}

		if len(region) < d.MinRegionSize {
			continue
		}
		if line, ok := d.fitRegion(region, mag, cols); ok {
			lines = append(lines, line)
		}
	}

	sort.SliceStable(lines, func(i, j int) bool {
		return lines[i].Response > lines[j].Response
	})
	return lines
}

// fitRegion approximates a line-support region (pixel indices into a cols-wide
// image, weighted by mag) with a segment, and validates it against the
// detector's length and density thresholds. The bool reports acceptance.
func (d *LSDDetector) fitRegion(region []int, mag []float64, cols int) (KeyLine, bool) {
	// Magnitude-weighted centroid.
	var wSum, cx, cy float64
	for _, p := range region {
		w := mag[p]
		x := float64(p % cols)
		y := float64(p / cols)
		wSum += w
		cx += w * x
		cy += w * y
	}
	if wSum == 0 {
		return KeyLine{}, false
	}
	cx /= wSum
	cy /= wSum

	// Weighted second-moment (covariance) matrix.
	var ixx, iyy, ixy float64
	for _, p := range region {
		w := mag[p]
		dx := float64(p%cols) - cx
		dy := float64(p/cols) - cy
		ixx += w * dx * dx
		iyy += w * dy * dy
		ixy += w * dx * dy
	}
	// Principal axis: eigenvector of the larger eigenvalue of [[ixx,ixy],[ixy,iyy]].
	theta := 0.5 * math.Atan2(2*ixy, ixx-iyy)
	dirX, dirY := math.Cos(theta), math.Sin(theta)
	perpX, perpY := -dirY, dirX

	// Project pixels onto the axis (for endpoints) and perpendicular (for width).
	minS, maxS := math.Inf(1), math.Inf(-1)
	maxW := 0.0
	for _, p := range region {
		dx := float64(p%cols) - cx
		dy := float64(p/cols) - cy
		s := dx*dirX + dy*dirY
		if s < minS {
			minS = s
		}
		if s > maxS {
			maxS = s
		}
		if o := math.Abs(dx*perpX + dy*perpY); o > maxW {
			maxW = o
		}
	}

	length := maxS - minS
	if length < d.MinLength {
		return KeyLine{}, false
	}
	// Density of aligned pixels inside the fitted rectangle.
	width := 2*maxW + 1
	area := length * width
	if area > 0 {
		if density := float64(len(region)) / area; density < d.MinDensity {
			return KeyLine{}, false
		}
	}

	x1 := cx + dirX*minS
	y1 := cy + dirY*minS
	x2 := cx + dirX*maxS
	y2 := cy + dirY*maxS
	return newKeyLine(x1, y1, x2, y2), true
}

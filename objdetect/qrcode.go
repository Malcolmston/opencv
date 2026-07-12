package objdetect

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// QRCodeDetector locates the finder patterns of a QR code. A QR symbol carries
// three identical square finder patterns in its top-left, top-right and
// bottom-left corners; each is a set of nested squares whose cross-section has
// the distinctive 1:1:3:1:1 dark:light:dark:light:dark run-length ratio. This
// detector finds those three patterns; it does not decode the payload.
//
// # Deferred
//
// Full decoding — perspective rectification from the finder (and alignment)
// patterns, module sampling, format/version information, and Reed–Solomon
// error correction — is out of scope. [QRCodeDetector.Detect] returns only the
// finder-pattern centres.
type QRCodeDetector struct {
	// Threshold is the luma value below which a pixel is considered dark.
	// Zero selects the default of 128.
	Threshold float64
}

// NewQRCodeDetector returns a detector with the default binarisation threshold.
func NewQRCodeDetector() *QRCodeDetector {
	return &QRCodeDetector{Threshold: 128}
}

func (d *QRCodeDetector) darkThreshold() float64 {
	if d.Threshold <= 0 {
		return 128
	}
	return d.Threshold
}

// candidate is a confirmed finder-pattern crossing with an estimated module
// size, used to cluster many per-row/column hits into distinct patterns.
type candidate struct {
	x, y float64
	size float64 // estimated module size in pixels
}

// Detect scans img for QR finder patterns and returns the centres of the (up
// to three) strongest patterns. found is true when at least three patterns are
// located. When more than three candidates survive, the three largest clusters
// are returned. Centres are returned in a deterministic order (top-to-bottom,
// then left-to-right).
func (d *QRCodeDetector) Detect(img *cv.Mat) (corners []cv.Point, found bool) {
	g := matToGray(img)
	thr := d.darkThreshold()
	dark := make([]bool, g.w*g.h)
	for i, v := range g.pix {
		dark[i] = v < thr
	}

	var cands []candidate
	// Horizontal scan: rows carrying a 1:1:3:1:1 profile, confirmed vertically.
	for y := 0; y < g.h; y++ {
		runs, starts, colors := rowRuns(dark, g.w, y)
		for i := 0; i+5 <= len(runs); i++ {
			if !colors[i] { // pattern must begin on a dark run
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

	centers := clusterCandidates(cands)
	sort.Slice(centers, func(i, j int) bool {
		if centers[i].Y != centers[j].Y {
			return centers[i].Y < centers[j].Y
		}
		return centers[i].X < centers[j].X
	})
	if len(centers) < 3 {
		return centers, false
	}
	return centers[:3], true
}

// confirmVertical checks that the column through (cx,y) also shows a 1:1:3:1:1
// profile centred on the dark run that contains row y.
func (d *QRCodeDetector) confirmVertical(dark []bool, w, h, cx, y int) bool {
	if cx < 0 || cx >= w {
		return false
	}
	runs, starts, colors := colRuns(dark, w, h, cx)
	// Find the run that contains row y.
	ri := -1
	for i := range runs {
		if y >= starts[i] && y < starts[i]+runs[i] {
			ri = i
			break
		}
	}
	if ri < 2 || ri+2 >= len(runs) {
		return false
	}
	if !colors[ri] { // centre run must be dark
		return false
	}
	return checkFinderRatio(runs[ri-2 : ri+3])
}

// checkFinderRatio reports whether five consecutive dark/light run lengths
// match the 1:1:3:1:1 finder-pattern proportion within tolerance.
func checkFinderRatio(five []int) bool {
	total := 0
	for _, r := range five {
		if r <= 0 {
			return false
		}
		total += r
	}
	if total < 7 {
		return false
	}
	module := float64(total) / 7.0
	maxVar := module / 2.0
	return math.Abs(module-float64(five[0])) <= maxVar &&
		math.Abs(module-float64(five[1])) <= maxVar &&
		math.Abs(3*module-float64(five[2])) <= 3*maxVar &&
		math.Abs(module-float64(five[3])) <= maxVar &&
		math.Abs(module-float64(five[4])) <= maxVar
}

// rowRuns returns the run lengths, start columns and colours (dark=true) of the
// horizontal run-length encoding of row y.
func rowRuns(dark []bool, w, y int) (runs, starts []int, colors []bool) {
	base := y * w
	x := 0
	for x < w {
		col := dark[base+x]
		start := x
		for x < w && dark[base+x] == col {
			x++
		}
		runs = append(runs, x-start)
		starts = append(starts, start)
		colors = append(colors, col)
	}
	return runs, starts, colors
}

// colRuns is the vertical analogue of rowRuns for column cx.
func colRuns(dark []bool, w, h, cx int) (runs, starts []int, colors []bool) {
	y := 0
	for y < h {
		col := dark[y*w+cx]
		start := y
		for y < h && dark[y*w+cx] == col {
			y++
		}
		runs = append(runs, y-start)
		starts = append(starts, start)
		colors = append(colors, col)
	}
	return runs, starts, colors
}

// clusterCandidates merges candidates that lie within half a pattern width of
// each other (a pattern spans 7 modules) and returns the averaged centre of
// each cluster, ordered by descending member count so the strongest patterns
// come first.
func clusterCandidates(cands []candidate) []cv.Point {
	type cluster struct {
		sx, sy, ss float64
		n          int
	}
	var clusters []cluster
	for _, c := range cands {
		merged := false
		for i := range clusters {
			cl := &clusters[i]
			cxAvg := cl.sx / float64(cl.n)
			cyAvg := cl.sy / float64(cl.n)
			sizeAvg := cl.ss / float64(cl.n)
			maxDist := 3.5 * sizeAvg // half of a 7-module pattern
			if math.Hypot(c.x-cxAvg, c.y-cyAvg) <= maxDist {
				cl.sx += c.x
				cl.sy += c.y
				cl.ss += c.size
				cl.n++
				merged = true
				break
			}
		}
		if !merged {
			clusters = append(clusters, cluster{sx: c.x, sy: c.y, ss: c.size, n: 1})
		}
	}
	sort.Slice(clusters, func(i, j int) bool { return clusters[i].n > clusters[j].n })
	out := make([]cv.Point, len(clusters))
	for i, cl := range clusters {
		out[i] = cv.Point{
			X: int(cl.sx/float64(cl.n) + 0.5),
			Y: int(cl.sy/float64(cl.n) + 0.5),
		}
	}
	return out
}

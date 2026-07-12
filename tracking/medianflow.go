package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TrackerMedianFlow implements the Median-Flow tracker. It seeds a regular grid
// of points inside the box, tracks each with local Lucas-Kanade optical flow
// ([lkTrack]), and keeps only the reliable half of them, judged by two errors:
// the forward-backward error (track a point to the next frame and back, and
// measure how far it lands from its start) and the appearance error (1 minus the
// NCC between the point's neighbourhood before and after). From the surviving
// points it estimates translation as the median displacement and scale as the
// median ratio of pairwise point distances, then moves and resizes the box about
// its centre.
//
// Median-Flow adapts to scale but not rotation and has no re-detection, so it
// reports failure (and freezes the box) when too few points survive — the usual
// symptom of occlusion or the object leaving the frame.
//
// Construct it with [NewTrackerMedianFlow].
type TrackerMedianFlow struct {
	// GridSize is the number of tracked points per side (GridSize² points).
	GridSize int
	// WinRadius is the half-size of the Lucas-Kanade and NCC windows in pixels.
	WinRadius int
	// Iters is the maximum number of Lucas-Kanade iterations per point.
	Iters int
	// MinPoints is the fewest surviving points for which Update trusts the
	// estimate; below it Update returns the previous box with ok false.
	MinPoints int

	prev   *cv.Mat
	box    cv.Rect
	inited bool
}

// NewTrackerMedianFlow returns a TrackerMedianFlow with sensible defaults
// (GridSize 10, WinRadius 4, Iters 20, MinPoints 4).
func NewTrackerMedianFlow() *TrackerMedianFlow {
	return &TrackerMedianFlow{GridSize: 10, WinRadius: 4, Iters: 20, MinPoints: 4}
}

// Init stores the first luma frame and the object box.
func (t *TrackerMedianFlow) Init(frame *cv.Mat, bbox cv.Rect) {
	gray := toGray(frame)
	t.prev = gray
	t.box = clampRect(bbox, gray.Rows, gray.Cols)
	t.inited = true
}

// gridPoints returns a GridSize×GridSize grid of points spread across box, each
// point at the centre of its cell.
func (t *TrackerMedianFlow) gridPoints() []pt {
	n := t.GridSize
	pts := make([]pt, 0, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x := float64(t.box.X) + (float64(j)+0.5)*float64(t.box.Width)/float64(n)
			y := float64(t.box.Y) + (float64(i)+0.5)*float64(t.box.Height)/float64(n)
			pts = append(pts, pt{x, y})
		}
	}
	return pts
}

// Update tracks the point grid into the new frame, filters unreliable points,
// and returns the box moved and scaled by the median motion. It panics if called
// before Init.
func (t *TrackerMedianFlow) Update(frame *cv.Mat) (cv.Rect, bool) {
	if !t.inited {
		panic("tracking: TrackerMedianFlow.Update called before Init")
	}
	cur := toGray(frame)
	r := t.WinRadius

	type cand struct {
		p0, p1  pt
		fb, app float64
	}
	var cands []cand
	for _, p := range t.gridPoints() {
		fx, fy, ok1 := lkTrack(t.prev, cur, p.x, p.y, r, t.Iters)
		if !ok1 {
			continue
		}
		bx, by, ok2 := lkTrack(cur, t.prev, fx, fy, r, t.Iters)
		if !ok2 {
			continue
		}
		fb := math.Hypot(p.x-bx, p.y-by)
		app := 1 - nccScore(t.prev, cur, p.x, p.y, fx, fy, r)
		cands = append(cands, cand{p0: p, p1: pt{fx, fy}, fb: fb, app: app})
	}
	if len(cands) < t.MinPoints {
		t.prev = cur
		return t.box, false
	}

	// Keep points whose forward-backward and appearance errors are both at or
	// below the median (the standard Median-Flow filtering).
	fbs := make([]float64, len(cands))
	apps := make([]float64, len(cands))
	for i, c := range cands {
		fbs[i] = c.fb
		apps[i] = c.app
	}
	medFB := median(fbs)
	medApp := median(apps)
	var p0, p1 []pt
	for _, c := range cands {
		if c.fb <= medFB && c.app <= medApp {
			p0 = append(p0, c.p0)
			p1 = append(p1, c.p1)
		}
	}
	if len(p0) < t.MinPoints {
		t.prev = cur
		return t.box, false
	}

	// Median translation.
	dxs := make([]float64, len(p0))
	dys := make([]float64, len(p0))
	for i := range p0 {
		dxs[i] = p1[i].x - p0[i].x
		dys[i] = p1[i].y - p0[i].y
	}
	dx := median(dxs)
	dy := median(dys)

	// Median scale from the ratio of pairwise distances before and after.
	var ratios []float64
	for i := 0; i < len(p0); i++ {
		for j := i + 1; j < len(p0); j++ {
			d0 := math.Hypot(p0[i].x-p0[j].x, p0[i].y-p0[j].y)
			d1 := math.Hypot(p1[i].x-p1[j].x, p1[i].y-p1[j].y)
			if d0 > 1e-6 {
				ratios = append(ratios, d1/d0)
			}
		}
	}
	scale := 1.0
	if len(ratios) > 0 {
		scale = median(ratios)
	}

	cx, cy := rectCenter(t.box)
	cx += dx
	cy += dy
	nw := float64(t.box.Width) * scale
	nh := float64(t.box.Height) * scale
	box := cv.Rect{
		X:      int(math.Round(cx - nw/2)),
		Y:      int(math.Round(cy - nh/2)),
		Width:  int(math.Round(nw)),
		Height: int(math.Round(nh)),
	}
	box = clampRect(box, cur.Rows, cur.Cols)
	t.box = box
	t.prev = cur
	return box, true
}

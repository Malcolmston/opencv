package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FlowField is a dense optical-flow field storing a (u, v) displacement per
// pixel, where u is the horizontal and v the vertical component measured in
// pixels from the first frame to the second. It is the result type of the dense
// flow solvers [CalcOpticalFlowHornSchunck] and [CalcOpticalFlowFarneback].
type FlowField struct {
	// Rows is the field height.
	Rows int
	// Cols is the field width.
	Cols int
	u    []float64
	v    []float64
}

// NewFlowField allocates a zero (no-motion) flow field of the given size. It
// panics if either dimension is not positive.
func NewFlowField(rows, cols int) *FlowField {
	if rows <= 0 || cols <= 0 {
		panic("tracking: NewFlowField requires positive dimensions")
	}
	return &FlowField{Rows: rows, Cols: cols, u: make([]float64, rows*cols), v: make([]float64, rows*cols)}
}

// At returns the displacement vector at pixel (y, x) as a Point2f whose X is the
// horizontal and Y the vertical flow. It panics if the coordinates are out of
// range.
func (f *FlowField) At(y, x int) Point2f {
	i := f.idx(y, x)
	return Point2f{X: f.u[i], Y: f.v[i]}
}

// SetFlow stores the displacement (du, dv) at pixel (y, x). It panics if the
// coordinates are out of range.
func (f *FlowField) SetFlow(y, x int, du, dv float64) {
	i := f.idx(y, x)
	f.u[i] = du
	f.v[i] = dv
}

// idx bounds-checks and returns the flat offset of pixel (y, x).
func (f *FlowField) idx(y, x int) int {
	if y < 0 || y >= f.Rows || x < 0 || x >= f.Cols {
		panic("tracking: FlowField coordinate out of range")
	}
	return y*f.Cols + x
}

// MagnitudeAt returns the Euclidean length of the flow vector at pixel (y, x).
func (f *FlowField) MagnitudeAt(y, x int) float64 {
	i := f.idx(y, x)
	return math.Hypot(f.u[i], f.v[i])
}

// MeanFlow returns the average displacement over the whole field.
func (f *FlowField) MeanFlow() Point2f {
	var su, sv float64
	n := float64(len(f.u))
	for i := range f.u {
		su += f.u[i]
		sv += f.v[i]
	}
	if n == 0 {
		return Point2f{}
	}
	return Point2f{X: su / n, Y: sv / n}
}

// MaxMagnitude returns the largest flow-vector length in the field.
func (f *FlowField) MaxMagnitude() float64 {
	var m float64
	for i := range f.u {
		if d := math.Hypot(f.u[i], f.v[i]); d > m {
			m = d
		}
	}
	return m
}

// LKParams configures the Lucas-Kanade solvers.
type LKParams struct {
	// WindowRadius is the half-size of the square integration window; the window
	// is (2*WindowRadius+1) pixels on a side. A value of 0 is treated as 3.
	WindowRadius int
	// MaxIterations bounds the per-level refinement iterations. Zero is treated
	// as 10.
	MaxIterations int
	// Epsilon stops refinement once the update step falls below this many pixels.
	// Zero is treated as 0.01.
	Epsilon float64
	// MinEigenThreshold rejects a point whose gradient structure tensor has a
	// minimum eigenvalue below this value (an untrackable, poorly textured
	// point). Zero is treated as 1e-4.
	MinEigenThreshold float64
}

// DefaultLKParams returns the default Lucas-Kanade parameters (window radius 3,
// 10 iterations, epsilon 0.01, min-eigen threshold 1e-4).
func DefaultLKParams() LKParams {
	return LKParams{WindowRadius: 3, MaxIterations: 10, Epsilon: 0.01, MinEigenThreshold: 1e-4}
}

// normalized fills in default values for zero fields.
func (p LKParams) normalized() LKParams {
	if p.WindowRadius <= 0 {
		p.WindowRadius = 3
	}
	if p.MaxIterations <= 0 {
		p.MaxIterations = 10
	}
	if p.Epsilon <= 0 {
		p.Epsilon = 0.01
	}
	if p.MinEigenThreshold <= 0 {
		p.MinEigenThreshold = 1e-4
	}
	return p
}

// CalcOpticalFlowLK computes sparse Lucas-Kanade optical flow at a single image
// resolution. For each point in pts it estimates the displacement that maps its
// neighbourhood in prev onto the corresponding neighbourhood in next by
// iteratively solving the 2x2 Lucas-Kanade normal equations
//
//	[ ΣIx²  ΣIxIy ] [du]   [ ΣIx·It ]
//	[ ΣIxIy ΣIy²  ] [dv] = [ ΣIy·It ]
//
// over a square window, where Ix, Iy are the spatial gradients of prev and It
// is the temporal difference next(p+d) − prev(p) at the current estimate.
//
// It returns the tracked points (prev point plus estimated displacement) and a
// parallel status slice; status[i] is true when point i was tracked
// successfully — the structure tensor was well conditioned and the point stayed
// in bounds. prev and next must be non-empty and identically sized. Multi-channel
// inputs are converted to grayscale. The computation is deterministic.
func CalcOpticalFlowLK(prev, next *cv.Mat, pts []Point2f, params LKParams) (nextPts []Point2f, status []bool) {
	requirePair(prev, next, "CalcOpticalFlowLK")
	gp := trackingToGrayF(prev)
	gn := trackingToGrayF(next)
	return trackingLKLevel(gp, gn, pts, cloneInitial(pts), params.normalized())
}

// CalcOpticalFlowPyrLK computes sparse Lucas-Kanade optical flow with a
// coarse-to-fine Gaussian image pyramid, the standard technique for handling
// displacements larger than the integration window. Flow is first estimated on
// the coarsest level and propagated down, doubling the estimate at each step to
// account for the change in resolution, and refined at every level.
//
// maxLevel is the index of the coarsest pyramid level (0 disables pyramiding and
// makes this equivalent to [CalcOpticalFlowLK]). It returns the tracked points
// and a parallel success status slice. prev and next must be non-empty and
// identically sized; multi-channel inputs are converted to grayscale. The
// computation is deterministic.
func CalcOpticalFlowPyrLK(prev, next *cv.Mat, pts []Point2f, maxLevel int, params LKParams) (nextPts []Point2f, status []bool) {
	requirePair(prev, next, "CalcOpticalFlowPyrLK")
	if maxLevel < 0 {
		panic("tracking: CalcOpticalFlowPyrLK requires maxLevel >= 0")
	}
	p := params.normalized()
	pyrPrev := trackingBuildPyramid(trackingToGrayF(prev), maxLevel)
	pyrNext := trackingBuildPyramid(trackingToGrayF(next), maxLevel)
	top := len(pyrPrev) - 1

	// Initial guess at the coarsest level: scale points down.
	scale := math.Pow(2, float64(top))
	guess := make([]Point2f, len(pts))
	scaledPts := make([]Point2f, len(pts))
	for i, pt := range pts {
		guess[i] = Point2f{X: pt.X / scale, Y: pt.Y / scale}
		scaledPts[i] = guess[i]
	}

	status = make([]bool, len(pts))
	for i := range status {
		status[i] = true
	}

	for lvl := top; lvl >= 0; lvl-- {
		lp := pyrPrev[lvl]
		ln := pyrNext[lvl]
		s := math.Pow(2, float64(lvl))
		levelPts := make([]Point2f, len(pts))
		for i, pt := range pts {
			levelPts[i] = Point2f{X: pt.X / s, Y: pt.Y / s}
		}
		refined, st := trackingLKLevel(lp, ln, levelPts, guess, p)
		for i := range refined {
			if !st[i] {
				status[i] = false
			}
			guess[i] = refined[i]
		}
		if lvl > 0 {
			// Propagate to the finer level: the finer level is twice the size, so
			// the displacement doubles.
			for i := range guess {
				disp := guess[i].Sub(levelPts[i])
				finerPt := Point2f{X: pts[i].X / (s / 2), Y: pts[i].Y / (s / 2)}
				guess[i] = finerPt.Add(disp.Scale(2))
			}
		}
	}
	return guess, status
}

// trackingLKLevel runs the iterative LK refinement on one resolution. prevPts
// are the source points in this level's coordinates; initGuess are the current
// estimates of their positions in next (this level's coordinates).
func trackingLKLevel(gp, gn *trackingGray, prevPts, initGuess []Point2f, p LKParams) ([]Point2f, []bool) {
	ix, iy := gp.sobel()
	w := p.WindowRadius
	out := make([]Point2f, len(prevPts))
	status := make([]bool, len(prevPts))

	for k := range prevPts {
		px := prevPts[k].X
		py := prevPts[k].Y
		// Build the gradient structure tensor over the window around the source
		// point in prev.
		var g11, g12, g22 float64
		for dy := -w; dy <= w; dy++ {
			for dx := -w; dx <= w; dx++ {
				sx := px + float64(dx)
				sy := py + float64(dy)
				gx := ix.bilinear(sx, sy)
				gy := iy.bilinear(sx, sy)
				g11 += gx * gx
				g12 += gx * gy
				g22 += gy * gy
			}
		}
		det := g11*g22 - g12*g12
		// Minimum eigenvalue of the 2x2 symmetric tensor.
		trace := g11 + g22
		disc := math.Sqrt(math.Max(0, trace*trace-4*det))
		minEig := (trace - disc) / 2
		npix := float64((2*w + 1) * (2*w + 1))
		if det <= 1e-12 || minEig/npix < p.MinEigenThreshold {
			out[k] = initGuess[k]
			status[k] = false
			continue
		}

		curX := initGuess[k].X
		curY := initGuess[k].Y
		ok := true
		for iter := 0; iter < p.MaxIterations; iter++ {
			var b1, b2 float64
			for dy := -w; dy <= w; dy++ {
				for dx := -w; dx <= w; dx++ {
					sx := px + float64(dx)
					sy := py + float64(dy)
					gx := ix.bilinear(sx, sy)
					gy := iy.bilinear(sx, sy)
					it := gp.bilinear(sx, sy) - gn.bilinear(curX+float64(dx), curY+float64(dy))
					b1 += gx * it
					b2 += gy * it
				}
			}
			// Solve G * eta = b.
			du := (g22*b1 - g12*b2) / det
			dv := (g11*b2 - g12*b1) / det
			curX += du
			curY += dv
			if math.Hypot(du, dv) < p.Epsilon {
				break
			}
			if curX < -float64(gn.cols) || curX > 2*float64(gn.cols) || curY < -float64(gn.rows) || curY > 2*float64(gn.rows) {
				ok = false
				break
			}
		}
		out[k] = Point2f{X: curX, Y: curY}
		status[k] = ok
	}
	return out, status
}

// cloneInitial returns a copy of pts to seed the single-level solver's initial
// guess (identity displacement).
func cloneInitial(pts []Point2f) []Point2f {
	out := make([]Point2f, len(pts))
	copy(out, pts)
	return out
}

// requirePair validates that two frames are non-empty and identically sized.
func requirePair(prev, next *cv.Mat, who string) {
	if prev == nil || prev.Empty() || next == nil || next.Empty() {
		panic("tracking: " + who + " requires two non-empty images")
	}
	if prev.Rows != next.Rows || prev.Cols != next.Cols {
		panic("tracking: " + who + " requires identically sized images")
	}
}

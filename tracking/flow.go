package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// pt is a fractional 2D image point used by the flow and Median-Flow code.
type pt struct {
	x, y float64
}

// lkTrack estimates the displacement of the point (px, py) from prev to cur with
// a single-level iterative Lucas-Kanade solver over a (2*win+1)² window. It
// returns the tracked position in cur and whether the solve was well
// conditioned; an ill-conditioned window (a flat or purely 1D-textured
// neighbourhood, where the 2×2 gradient matrix is near-singular) yields ok
// false so the caller can discard the point.
//
// Each iteration accumulates the spatial-gradient structure matrix G and the
// temporal-mismatch vector b over the window (all sampled bilinearly at
// sub-pixel positions), solves G·v = b, and applies the Newton step d -= v. It
// stops early once a step is sub-0.01-pixel. Both inputs must be single-channel.
func lkTrack(prev, cur *cv.Mat, px, py float64, win, iters int) (float64, float64, bool) {
	dx, dy := 0.0, 0.0
	for it := 0; it < iters; it++ {
		var g11, g12, g22, b1, b2 float64
		for wy := -win; wy <= win; wy++ {
			for wx := -win; wx <= win; wx++ {
				x := px + float64(wx)
				y := py + float64(wy)
				ix := (sampleBilinear(prev, x+1, y) - sampleBilinear(prev, x-1, y)) / 2
				iy := (sampleBilinear(prev, x, y+1) - sampleBilinear(prev, x, y-1)) / 2
				dt := sampleBilinear(cur, x+dx, y+dy) - sampleBilinear(prev, x, y)
				g11 += ix * ix
				g12 += ix * iy
				g22 += iy * iy
				b1 += dt * ix
				b2 += dt * iy
			}
		}
		det := g11*g22 - g12*g12
		// Scale the conditioning threshold with the window size so it is
		// independent of how many samples were summed.
		n := float64((2*win + 1) * (2*win + 1))
		if math.Abs(det) < 1e-3*n*n {
			return px + dx, py + dy, false
		}
		vx := (g22*b1 - g12*b2) / det
		vy := (g11*b2 - g12*b1) / det
		dx -= vx
		dy -= vy
		if vx*vx+vy*vy < 1e-4 {
			break
		}
	}
	return px + dx, py + dy, true
}

// nccScore returns the normalised cross-correlation in [-1,1] between the
// (2*r+1)² windows of a centred at (ax, ay) and b centred at (bx, by), sampled
// bilinearly. A value of 1 means identical appearance; it is used as an
// appearance-consistency measure (1-ncc is the error). Both inputs must be
// single-channel.
func nccScore(a, b *cv.Mat, ax, ay, bx, by float64, r int) float64 {
	var sa, sb, n float64
	for wy := -r; wy <= r; wy++ {
		for wx := -r; wx <= r; wx++ {
			sa += sampleBilinear(a, ax+float64(wx), ay+float64(wy))
			sb += sampleBilinear(b, bx+float64(wx), by+float64(wy))
			n++
		}
	}
	ma := sa / n
	mb := sb / n
	var cov, va, vb float64
	for wy := -r; wy <= r; wy++ {
		for wx := -r; wx <= r; wx++ {
			da := sampleBilinear(a, ax+float64(wx), ay+float64(wy)) - ma
			db := sampleBilinear(b, bx+float64(wx), by+float64(wy)) - mb
			cov += da * db
			va += da * da
			vb += db * db
		}
	}
	den := math.Sqrt(va * vb)
	if den == 0 {
		return 0
	}
	return cov / den
}

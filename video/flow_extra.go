package video

import (
	cv "github.com/malcolmston/opencv"
)

// CalcOpticalFlowPyrLKF is the sub-pixel variant of [CalcOpticalFlowPyrLK]. It
// accepts and returns floating-point [PointF] locations, so both the input
// feature positions and the recovered ones keep sub-pixel precision instead of
// being rounded to the nearest pixel. The pyramidal Lucas-Kanade machinery is
// identical: a Gaussian pyramid is built for each frame and each point is
// refined coarse-to-fine using the structure tensor assembled from the previous
// frame's Sobel gradients.
//
// For each input point it returns the estimated new location, a status flag
// (false when the structure tensor is singular or the track diverges off the
// image), and the root-mean-square intensity residual over the window at the
// final estimate. winSize is rounded up to an odd value; maxLevel must be >= 0;
// prev and next must have identical dimensions.
func CalcOpticalFlowPyrLKF(prev, next *cv.Mat, prevPts []PointF, winSize, maxLevel int) (nextPts []PointF, status []bool, errs []float64) {
	if prev == nil || next == nil || prev.Empty() || next.Empty() {
		panic("video: CalcOpticalFlowPyrLKF requires non-empty images")
	}
	if prev.Rows != next.Rows || prev.Cols != next.Cols {
		panic("video: CalcOpticalFlowPyrLKF requires equal-sized images")
	}
	if maxLevel < 0 {
		panic("video: CalcOpticalFlowPyrLKF requires maxLevel >= 0")
	}
	if winSize < 1 {
		panic("video: CalcOpticalFlowPyrLKF requires winSize >= 1")
	}
	if winSize%2 == 0 {
		winSize++
	}
	half := winSize / 2

	prevPyr := BuildOpticalFlowPyramid(prev, maxLevel)
	nextPyr := BuildOpticalFlowPyramid(next, maxLevel)
	levels := buildPrevLevels(prevPyr)
	nextGrids := make([]*grid, len(nextPyr))
	for i, m := range nextPyr {
		nextGrids[i] = gridFromMat(m)
	}
	top := len(levels) - 1

	const (
		maxIter = 30
		epsSq   = 0.0001
		minDet  = 1e-6
	)

	nextPts = make([]PointF, len(prevPts))
	status = make([]bool, len(prevPts))
	errs = make([]float64, len(prevPts))

	for pi, pt := range prevPts {
		var vx, vy float64
		ok := true
		var lastErr float64

		for l := top; l >= 0; l-- {
			lv := levels[l]
			ng := nextGrids[l]
			scale := 1.0 / float64(int(1)<<uint(l))
			cx := pt.X * scale
			cy := pt.Y * scale

			var gxx, gxy, gyy float64
			for wy := -half; wy <= half; wy++ {
				for wx := -half; wx <= half; wx++ {
					sx := cx + float64(wx)
					sy := cy + float64(wy)
					ix := lv.gx.bilinear(sx, sy)
					iy := lv.gy.bilinear(sx, sy)
					gxx += ix * ix
					gxy += ix * iy
					gyy += iy * iy
				}
			}
			det := gxx*gyy - gxy*gxy
			if det < minDet {
				ok = false
				break
			}

			for iter := 0; iter < maxIter; iter++ {
				var bx, by, e2 float64
				n := 0
				for wy := -half; wy <= half; wy++ {
					for wx := -half; wx <= half; wx++ {
						sx := cx + float64(wx)
						sy := cy + float64(wy)
						it := lv.img.bilinear(sx, sy) - ng.bilinear(sx+vx, sy+vy)
						ix := lv.gx.bilinear(sx, sy)
						iy := lv.gy.bilinear(sx, sy)
						bx += ix * it
						by += iy * it
						e2 += it * it
						n++
					}
				}
				dx := (gyy*bx - gxy*by) / det
				dy := (gxx*by - gxy*bx) / det
				vx += dx
				vy += dy
				lastErr = sqrt(e2 / float64(n))
				if dx*dx+dy*dy < epsSq {
					break
				}
			}

			if l > 0 {
				vx *= 2
				vy *= 2
			}
		}

		if ok {
			nx := pt.X + vx
			ny := pt.Y + vy
			if nx < -float64(prev.Cols) || nx > 2*float64(prev.Cols) ||
				ny < -float64(prev.Rows) || ny > 2*float64(prev.Rows) {
				ok = false
			}
			nextPts[pi] = PointF{X: nx, Y: ny}
		}
		if !ok {
			nextPts[pi] = pt
		}
		status[pi] = ok
		errs[pi] = lastErr
	}
	return nextPts, status, errs
}

// parabolaVertex fits a parabola through three equally spaced samples
// (ym at -1, y0 at 0, yp at +1) and returns the sub-pixel offset of its
// extremum, clamped to [-1, 1]. It is used to refine the integer minimum of a
// matching cost to sub-pixel precision.
func parabolaVertex(ym, y0, yp float64) float64 {
	denom := ym - 2*y0 + yp
	if denom == 0 {
		return 0
	}
	off := 0.5 * (ym - yp) / denom
	if off < -1 {
		off = -1
	}
	if off > 1 {
		off = 1
	}
	return off
}

// CalcOpticalFlowFarnebackSubpixel refines [CalcOpticalFlowFarneback] to
// sub-pixel accuracy. It performs the same integer block-matching search, but
// after locating the best integer displacement it fits a 1-D parabola to the
// sum-of-squared-difference cost along each axis around the minimum and shifts
// the reported displacement by the parabola's vertex. The result is a dense
// [FlowField] with fractional displacements, which is markedly more accurate
// than the integer-only base routine for sub-pixel motions.
//
// prev and next must have identical dimensions; winSize and searchRadius must be
// >= 1. Multi-channel inputs are converted to grayscale first.
func CalcOpticalFlowFarnebackSubpixel(prev, next *cv.Mat, winSize, searchRadius int) *FlowField {
	if prev == nil || next == nil || prev.Empty() || next.Empty() {
		panic("video: CalcOpticalFlowFarnebackSubpixel requires non-empty images")
	}
	if prev.Rows != next.Rows || prev.Cols != next.Cols {
		panic("video: CalcOpticalFlowFarnebackSubpixel requires equal-sized images")
	}
	if winSize < 1 || searchRadius < 1 {
		panic("video: CalcOpticalFlowFarnebackSubpixel requires winSize >= 1 and searchRadius >= 1")
	}
	pg := gridFromMat(toGray(prev))
	ng := gridFromMat(toGray(next))
	rows, cols := prev.Rows, prev.Cols
	flow := NewFlowField(rows, cols)

	// ssdAt returns the block SSD between the prev window at (x,y) and the next
	// window displaced by (dx, dy), sampled bilinearly so fractional offsets are
	// meaningful.
	ssdAt := func(x, y int, dx, dy float64) float64 {
		var s float64
		for wy := -winSize; wy <= winSize; wy++ {
			for wx := -winSize; wx <= winSize; wx++ {
				a := pg.atClamp(x+wx, y+wy)
				b := ng.bilinear(float64(x+wx)+dx, float64(y+wy)+dy)
				d := a - b
				s += d * d
			}
		}
		return s
	}

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			bestSSD := -1.0
			var bestDX, bestDY int
			bestMag := 1 << 30
			for dy := -searchRadius; dy <= searchRadius; dy++ {
				for dx := -searchRadius; dx <= searchRadius; dx++ {
					ssd := ssdAt(x, y, float64(dx), float64(dy))
					mag := dx*dx + dy*dy
					if bestSSD < 0 || ssd < bestSSD || (ssd == bestSSD && mag < bestMag) {
						bestSSD = ssd
						bestDX = dx
						bestDY = dy
						bestMag = mag
					}
				}
			}
			// Parabolic sub-pixel refinement along each axis around the integer
			// minimum.
			cxm := ssdAt(x, y, float64(bestDX-1), float64(bestDY))
			cx0 := bestSSD
			cxp := ssdAt(x, y, float64(bestDX+1), float64(bestDY))
			subX := parabolaVertex(cxm, cx0, cxp)
			cym := ssdAt(x, y, float64(bestDX), float64(bestDY-1))
			cyp := ssdAt(x, y, float64(bestDX), float64(bestDY+1))
			subY := parabolaVertex(cym, cx0, cyp)
			flow.set(y, x, float64(bestDX)+subX, float64(bestDY)+subY)
		}
	}
	return flow
}

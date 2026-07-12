package video

import (
	cv "github.com/malcolmston/opencv"
)

// BuildOpticalFlowPyramid builds a Gaussian image pyramid for optical-flow
// tracking. The input may be single- or multi-channel; it is first converted to
// grayscale. The returned slice has maxLevel+1 entries: element 0 is the
// full-resolution grayscale image and each subsequent level is produced from
// the previous one with cv.PyrDown, halving both dimensions (rounding up). This
// is the pyramid consumed by [CalcOpticalFlowPyrLK].
//
// maxLevel must be >= 0. Levels stop early if a dimension would fall below one
// pixel, so the returned slice may contain fewer than maxLevel+1 entries for
// very small images.
func BuildOpticalFlowPyramid(img *cv.Mat, maxLevel int) []*cv.Mat {
	if img == nil || img.Empty() {
		panic("video: BuildOpticalFlowPyramid requires a non-empty image")
	}
	if maxLevel < 0 {
		panic("video: BuildOpticalFlowPyramid requires maxLevel >= 0")
	}
	pyr := make([]*cv.Mat, 0, maxLevel+1)
	pyr = append(pyr, toGray(img))
	for l := 1; l <= maxLevel; l++ {
		prev := pyr[l-1]
		if prev.Rows < 2 || prev.Cols < 2 {
			break
		}
		pyr = append(pyr, cv.PyrDown(prev))
	}
	return pyr
}

// lkLevel bundles the per-pyramid-level data the tracker needs for the previous
// frame: the intensity grid and its two normalised gradient grids.
type lkLevel struct {
	img *grid
	gx  *grid
	gy  *grid
}

// buildLevels converts a grayscale pyramid into per-level intensity and
// gradient grids for the previous frame.
func buildPrevLevels(pyr []*cv.Mat) []lkLevel {
	levels := make([]lkLevel, len(pyr))
	for i, m := range pyr {
		gx, gy := gradients(m)
		levels[i] = lkLevel{img: gridFromMat(m), gx: gx, gy: gy}
	}
	return levels
}

// CalcOpticalFlowPyrLK computes sparse optical flow with the pyramidal
// Lucas-Kanade method. It tracks each point in prevPts from the prev frame to
// the next frame and returns, for every input point, its estimated new location,
// a status flag reporting whether the flow was found, and a tracking error (the
// root-mean-square intensity residual over the window at the final estimate).
//
// The algorithm builds a Gaussian pyramid (see [BuildOpticalFlowPyramid]) for
// each frame and processes points coarse-to-fine. At every level it assembles
// the 2x2 spatial structure tensor
//
//	G = sum_W [ Ix*Ix  Ix*Iy ; Ix*Iy  Iy*Iy ]
//
// over a winSize x winSize window from the Sobel gradients (Ix, Iy) of the
// previous frame, then iterates the Lucas-Kanade update d = G^-1 * b, where
// b = sum_W [ Ix*It ; Iy*It ] and It is the intensity difference between the
// previous window and the (sub-pixel, bilinearly sampled) next window at the
// current displacement. The refined displacement is doubled and passed to the
// next finer level. Points whose structure tensor is singular (untextured
// regions, the aperture problem) are marked with status=false.
//
// winSize must be a positive odd integer (it is rounded up to odd if even);
// maxLevel must be >= 0. Input points are treated as integer pixel coordinates
// and the returned points are rounded to the nearest pixel. prev and next must
// have identical dimensions.
func CalcOpticalFlowPyrLK(prev, next *cv.Mat, prevPts []cv.Point, winSize, maxLevel int) (nextPts []cv.Point, status []bool, errs []float64) {
	if prev == nil || next == nil || prev.Empty() || next.Empty() {
		panic("video: CalcOpticalFlowPyrLK requires non-empty images")
	}
	if prev.Rows != next.Rows || prev.Cols != next.Cols {
		panic("video: CalcOpticalFlowPyrLK requires equal-sized images")
	}
	if maxLevel < 0 {
		panic("video: CalcOpticalFlowPyrLK requires maxLevel >= 0")
	}
	if winSize < 1 {
		panic("video: CalcOpticalFlowPyrLK requires winSize >= 1")
	}
	if winSize%2 == 0 {
		winSize++
	}
	half := winSize / 2

	prevPyr := BuildOpticalFlowPyramid(prev, maxLevel)
	nextPyr := BuildOpticalFlowPyramid(next, maxLevel)
	// The two pyramids share the same depth because the images are equal-sized.
	levels := buildPrevLevels(prevPyr)
	nextGrids := make([]*grid, len(nextPyr))
	for i, m := range nextPyr {
		nextGrids[i] = gridFromMat(m)
	}
	top := len(levels) - 1

	const (
		maxIter   = 30
		epsSq     = 0.001 // squared step threshold for early convergence
		minDet    = 1e-6  // structure-tensor determinant floor
		errFactor = 255.0 // scale of intensity residuals
	)

	nextPts = make([]cv.Point, len(prevPts))
	status = make([]bool, len(prevPts))
	errs = make([]float64, len(prevPts))

	for pi, pt := range prevPts {
		// Displacement accumulated in the coordinate system of the current
		// level. Starts at zero on the coarsest level.
		var vx, vy float64
		ok := true
		var lastErr float64

		for l := top; l >= 0; l-- {
			lv := levels[l]
			ng := nextGrids[l]
			scale := 1.0 / float64(int(1)<<uint(l))
			// Centre of the window at this level.
			cx := float64(pt.X) * scale
			cy := float64(pt.Y) * scale

			// Assemble the structure tensor once per level; it depends only on
			// the previous frame's gradients, not on the displacement.
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

			// Iterative refinement of the displacement at this level.
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
				// Solve G * d = b.
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
				// Propagate the displacement to the next finer level.
				vx *= 2
				vy *= 2
			}
		}

		if ok {
			nx := float64(pt.X) + vx
			ny := float64(pt.Y) + vy
			// Reject clearly diverged tracks that leave the image.
			if nx < -float64(prev.Cols) || nx > 2*float64(prev.Cols) ||
				ny < -float64(prev.Rows) || ny > 2*float64(prev.Rows) {
				ok = false
			}
			nextPts[pi] = cv.Point{X: roundInt(nx), Y: roundInt(ny)}
		}
		if !ok {
			nextPts[pi] = pt
		}
		status[pi] = ok
		errs[pi] = lastErr
	}
	return nextPts, status, errs
}

// roundInt rounds a float to the nearest integer (half away from zero).
func roundInt(v float64) int {
	if v < 0 {
		return -int(-v + 0.5)
	}
	return int(v + 0.5)
}

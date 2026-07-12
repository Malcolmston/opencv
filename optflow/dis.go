package optflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// CalcOpticalFlowDIS computes a dense optical-flow field from prev to next using
// a simplified Dense Inverse Search (DIS) scheme: coarse-to-fine patch matching
// on a Gaussian pyramid.
//
// The algorithm builds an image pyramid for each frame and processes it from the
// coarsest level to the finest. At every level each pixel carries a flow
// estimate inherited (and scaled) from the coarser level above; a small local
// search over integer offsets within searchRadius refines that estimate by
// minimising the sum of squared differences of a (2·patchRadius+1)² intensity
// patch between prev and the bilinearly-sampled next frame. The refined field is
// lightly smoothed before being propagated down to the next finer level. Working
// coarse-to-fine lets the method recover displacements far larger than
// searchRadius alone would allow, while the per-level search keeps each step
// cheap.
//
// patchRadius and searchRadius must be >= 1 and levels >= 0 (levels is the
// number of extra coarser pyramid levels above the full-resolution image;
// pyramid construction stops early when a level would become too small).
// Reasonable defaults are patchRadius=4, searchRadius=2, levels=3.
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. Untextured patches yield ambiguous matches; ties
// are broken toward the smaller-magnitude offset then row-major order, so the
// result is fully deterministic.
func CalcOpticalFlowDIS(prev, next *cv.Mat, patchRadius, searchRadius, levels int) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowDIS")
	if patchRadius < 1 || searchRadius < 1 {
		panic("optflow: CalcOpticalFlowDIS requires patchRadius >= 1 and searchRadius >= 1")
	}
	if levels < 0 {
		panic("optflow: CalcOpticalFlowDIS requires levels >= 0")
	}

	pg := grayGrid(prev)
	ng := grayGrid(next)

	// Build both pyramids; keep them aligned by construction (identical sizes).
	minSize := 2*patchRadius + 1
	pPyr := buildPyramid(pg, levels, minSize)
	nPyr := buildPyramid(ng, levels, minSize)
	if len(nPyr) < len(pPyr) {
		pPyr = pPyr[:len(nPyr)]
	} else if len(pPyr) < len(nPyr) {
		nPyr = nPyr[:len(pPyr)]
	}
	nl := len(pPyr)

	// Flow at the current (coarsest first) level, stored as two planes.
	var curU, curV []float64
	var curRows, curCols int

	for lvl := nl - 1; lvl >= 0; lvl-- {
		p := pPyr[lvl]
		n := nPyr[lvl]
		rows, cols := p.Rows, p.Cols

		u := make([]float64, rows*cols)
		v := make([]float64, rows*cols)
		if curU == nil {
			// Coarsest level starts from zero motion.
		} else {
			// Upsample the coarser flow into this level and scale by the size
			// ratio so displacements are expressed in this level's pixels.
			sx := float64(cols) / float64(curCols)
			sy := float64(rows) / float64(curRows)
			for y := 0; y < rows; y++ {
				fy := (float64(y)+0.5)/sy - 0.5
				for x := 0; x < cols; x++ {
					fx := (float64(x)+0.5)/sx - 0.5
					u[y*cols+x] = bilerpPlane(curU, curRows, curCols, fx, fy) * sx
					v[y*cols+x] = bilerpPlane(curV, curRows, curCols, fx, fy) * sy
				}
			}
		}

		refineLevel(p, n, u, v, patchRadius, searchRadius)
		smoothPlanes(u, v, rows, cols)

		curU, curV = u, v
		curRows, curCols = rows, cols
	}

	flow := NewFlowField(curRows, curCols)
	for i := 0; i < curRows*curCols; i++ {
		flow.Data[i*2] = curU[i]
		flow.Data[i*2+1] = curV[i]
	}
	return flow
}

// refineLevel performs a local integer search around each pixel's current
// (u, v) estimate, updating u and v in place with the offset that minimises the
// patch SSD between p and the bilinearly-sampled n.
func refineLevel(p, n *grid, u, v []float64, patchRadius, searchRadius int) {
	rows, cols := p.Rows, p.Cols
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			baseU := u[i]
			baseV := v[i]
			bestSSD := math.Inf(1)
			bestU := baseU
			bestV := baseV
			bestMag := math.Inf(1)
			for dy := -searchRadius; dy <= searchRadius; dy++ {
				for dx := -searchRadius; dx <= searchRadius; dx++ {
					cu := baseU + float64(dx)
					cv := baseV + float64(dy)
					var ssd float64
					for wy := -patchRadius; wy <= patchRadius; wy++ {
						for wx := -patchRadius; wx <= patchRadius; wx++ {
							a := p.atClamp(x+wx, y+wy)
							b := n.bilinear(float64(x+wx)+cu, float64(y+wy)+cv)
							d := a - b
							ssd += d * d
						}
					}
					mag := cu*cu + cv*cv
					if ssd < bestSSD || (ssd == bestSSD && mag < bestMag) {
						bestSSD = ssd
						bestU = cu
						bestV = cv
						bestMag = mag
					}
				}
			}
			u[i] = bestU
			v[i] = bestV
		}
	}
}

// smoothPlanes applies a 3x3 box blur to both flow planes in place, damping the
// per-pixel noise that independent local searches introduce before the field is
// propagated to a finer level. Border pixels are replicated.
func smoothPlanes(u, v []float64, rows, cols int) {
	su := boxBlur3(u, rows, cols)
	sv := boxBlur3(v, rows, cols)
	copy(u, su)
	copy(v, sv)
}

// boxBlur3 returns a 3x3 mean-filtered copy of a single plane with border
// replication.
func boxBlur3(src []float64, rows, cols int) []float64 {
	out := make([]float64, rows*cols)
	clampX := func(x int) int { return clampInt(x, 0, cols-1) }
	clampY := func(y int) int { return clampInt(y, 0, rows-1) }
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					s += src[clampY(y+dy)*cols+clampX(x+dx)]
				}
			}
			out[y*cols+x] = s / 9.0
		}
	}
	return out
}

// bilerpPlane bilinearly samples a single row-major plane at fractional
// coordinates (x, y) with border replication.
func bilerpPlane(plane []float64, rows, cols int, x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	dx := x - float64(x0)
	dy := y - float64(y0)
	at := func(xx, yy int) float64 {
		return plane[clampInt(yy, 0, rows-1)*cols+clampInt(xx, 0, cols-1)]
	}
	v00 := at(x0, y0)
	v01 := at(x0+1, y0)
	v10 := at(x0, y0+1)
	v11 := at(x0+1, y0+1)
	top := v00*(1-dx) + v01*dx
	bot := v10*(1-dx) + v11*dx
	return top*(1-dy) + bot*dy
}

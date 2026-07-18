package segment2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SLICParams configures the [SLICWithParams] superpixel segmenter.
type SLICParams struct {
	// RegionSize is the grid spacing of the initial cluster centres in pixels;
	// it sets the approximate superpixel diameter. Must be >= 2.
	RegionSize int
	// Compactness trades colour adherence for spatial regularity: larger values
	// yield squarer superpixels. Must be > 0. Ignored when Adaptive is true
	// (SLICO adapts it per cluster).
	Compactness float64
	// Iterations is the number of assignment/update rounds.
	Iterations int
	// Adaptive selects the SLICO variant, which adapts the compactness of each
	// cluster to the maximum observed colour distance, removing the need to tune
	// Compactness.
	Adaptive bool
}

// segment2slic is the shared SLIC/SLICO core.
func segment2slic(img *cv.Mat, p SLICParams) *LabelMap {
	segment2requireNonEmpty(img, "SLIC")
	if p.RegionSize < 2 {
		panic("segment2: SLIC RegionSize must be >= 2")
	}
	if !p.Adaptive && p.Compactness <= 0 {
		panic("segment2: SLIC Compactness must be positive")
	}
	if p.Iterations < 1 {
		p.Iterations = 1
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	n := rows * cols

	gray := segment2gray(img)
	mag := segment2sobelMag(gray, rows, cols)

	// Seed centres on a regular grid, nudged to the lowest-gradient pixel in a
	// 3x3 neighbourhood.
	type center struct {
		x, y float64
		col  []float64
	}
	var centers []center
	for gy := p.RegionSize / 2; gy < rows; gy += p.RegionSize {
		for gx := p.RegionSize / 2; gx < cols; gx += p.RegionSize {
			bx, by := gx, gy
			bm := math.Inf(1)
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := gx+dx, gy+dy
					if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
						continue
					}
					if mag[ny*cols+nx] < bm {
						bm = mag[ny*cols+nx]
						bx, by = nx, ny
					}
				}
			}
			centers = append(centers, center{
				x: float64(bx), y: float64(by), col: segment2colorAt(img, bx, by),
			})
		}
	}
	k := len(centers)
	labels := make([]int, n)
	for i := range labels {
		labels[i] = -1
	}
	dist := make([]float64, n)

	invS := 1.0 / float64(p.RegionSize)
	maxColor := make([]float64, k)
	for i := range maxColor {
		maxColor[i] = 100 // avoids division by zero on first pass (SLICO)
	}
	m2Fixed := p.Compactness * p.Compactness

	cur := make([]float64, ch)
	for it := 0; it < p.Iterations; it++ {
		for i := range dist {
			dist[i] = math.Inf(1)
		}
		for ci, c := range centers {
			x0 := int(c.x) - p.RegionSize
			x1 := int(c.x) + p.RegionSize
			y0 := int(c.y) - p.RegionSize
			y1 := int(c.y) + p.RegionSize
			for yy := y0; yy <= y1; yy++ {
				if yy < 0 || yy >= rows {
					continue
				}
				for xx := x0; xx <= x1; xx++ {
					if xx < 0 || xx >= cols {
						continue
					}
					idx := yy*cols + xx
					segment2colorInto(img, xx, yy, cur)
					dc2 := segment2colorDist2(cur, c.col)
					dsx := float64(xx) - c.x
					dsy := float64(yy) - c.y
					ds2 := dsx*dsx + dsy*dsy
					var m2 float64
					if p.Adaptive {
						m2 = maxColor[ci]
					} else {
						m2 = m2Fixed
					}
					d := dc2 + (ds2*invS*invS)*m2
					if d < dist[idx] {
						dist[idx] = d
						labels[idx] = ci
					}
				}
			}
		}
		// Update centres and, for SLICO, the per-cluster max colour distance.
		sx := make([]float64, k)
		sy := make([]float64, k)
		sc := make([][]float64, k)
		cnt := make([]int, k)
		newMax := make([]float64, k)
		for i := range sc {
			sc[i] = make([]float64, ch)
		}
		for idx := 0; idx < n; idx++ {
			l := labels[idx]
			if l < 0 {
				continue
			}
			xx := idx % cols
			yy := idx / cols
			sx[l] += float64(xx)
			sy[l] += float64(yy)
			b := idx * ch
			for c := 0; c < ch; c++ {
				sc[l][c] += float64(img.Data[b+c])
			}
			cnt[l]++
			segment2colorInto(img, xx, yy, cur)
			dc2 := segment2colorDist2(cur, centers[l].col)
			if dc2 > newMax[l] {
				newMax[l] = dc2
			}
		}
		for ci := 0; ci < k; ci++ {
			if cnt[ci] == 0 {
				continue
			}
			centers[ci].x = sx[ci] / float64(cnt[ci])
			centers[ci].y = sy[ci] / float64(cnt[ci])
			for c := 0; c < ch; c++ {
				centers[ci].col[c] = sc[ci][c] / float64(cnt[ci])
			}
			if newMax[ci] > 0 {
				maxColor[ci] = newMax[ci]
			}
		}
	}

	// Assign any pixel that was never claimed to the nearest centre.
	for idx := 0; idx < n; idx++ {
		if labels[idx] >= 0 {
			continue
		}
		xx := idx % cols
		yy := idx / cols
		best, bestD := 0, math.Inf(1)
		for ci, c := range centers {
			dsx := float64(xx) - c.x
			dsy := float64(yy) - c.y
			d := dsx*dsx + dsy*dsy
			if d < bestD {
				bestD, best = d, ci
			}
		}
		labels[idx] = best
	}

	lm := &LabelMap{Rows: rows, Cols: cols, Labels: labels, NumLabels: k}
	segment2enforceConnectivity(lm, p.RegionSize)
	lm.Compact()
	return lm
}

// segment2enforceConnectivity relabels stray fragments so each superpixel is a
// single connected component, absorbing fragments smaller than a quarter of the
// nominal area into an adjacent label.
func segment2enforceConnectivity(lm *LabelMap, regionSize int) {
	rows, cols := lm.Rows, lm.Cols
	minArea := regionSize * regionSize / 4
	if minArea < 1 {
		minArea = 1
	}
	newLabels := make([]int, rows*cols)
	for i := range newLabels {
		newLabels[i] = -1
	}
	next := 0
	stack := make([]int, 0, 64)
	for start := 0; start < rows*cols; start++ {
		if newLabels[start] != -1 {
			continue
		}
		orig := lm.Labels[start]
		stack = stack[:0]
		stack = append(stack, start)
		newLabels[start] = next
		comp := []int{start}
		var adj int = -1
		for len(stack) > 0 {
			cur := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			cx := cur % cols
			cy := cur / cols
			for _, o := range segment2neighbors4 {
				nx, ny := cx+o.dx, cy+o.dy
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					continue
				}
				ni := ny*cols + nx
				if lm.Labels[ni] == orig && newLabels[ni] == -1 {
					newLabels[ni] = next
					stack = append(stack, ni)
					comp = append(comp, ni)
				} else if lm.Labels[ni] != orig && newLabels[ni] != -1 {
					adj = newLabels[ni]
				}
			}
		}
		if len(comp) < minArea && adj != -1 {
			for _, ci := range comp {
				newLabels[ci] = adj
			}
		} else {
			next++
		}
	}
	lm.Labels = newLabels
	lm.NumLabels = next
}

// SLICWithParams computes superpixels with Simple Linear Iterative Clustering
// (Achanta et al., 2012) using the supplied [SLICParams], returning a [LabelMap]
// that partitions img into compact, roughly uniform regions adhering to colour
// boundaries. A final connectivity pass guarantees every superpixel is one
// connected component.
//
// It panics if img is empty or the parameters are invalid.
func SLICWithParams(img *cv.Mat, p SLICParams) *LabelMap {
	return segment2slic(img, p)
}

// SLIC is [SLICWithParams] with the fixed-compactness variant selected. img must
// be non-empty, regionSize >= 2 and compactness > 0.
func SLIC(img *cv.Mat, regionSize int, compactness float64, iterations int) *LabelMap {
	return segment2slic(img, SLICParams{
		RegionSize:  regionSize,
		Compactness: compactness,
		Iterations:  iterations,
	})
}

// SLICO is the zero-parameter SLIC variant that adapts each cluster's
// compactness automatically, so only the region size need be chosen. img must be
// non-empty and regionSize >= 2.
func SLICO(img *cv.Mat, regionSize, iterations int) *LabelMap {
	return segment2slic(img, SLICParams{
		RegionSize: regionSize,
		Iterations: iterations,
		Adaptive:   true,
	})
}

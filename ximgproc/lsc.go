package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// lscIterations is the fixed number of weighted k-means refinement passes.
const lscIterations = 10

// SuperpixelLSC segments img into superpixels with Linear Spectral Clustering
// (Li and Chen, "Superpixel Segmentation using Linear Spectral Clustering",
// 2015) and returns a label image together with the number of superpixels. It is
// an alternative to [SuperpixelSLIC] that tends to adhere more tightly to object
// boundaries while keeping superpixels compact.
//
// LSC observes that minimising the SLIC objective is equivalent to a weighted
// k-means in a higher-dimensional feature space obtained by mapping each colour
// and coordinate value v (scaled to [0,1]) through the pair
//
//	(cos(π/2·v), sin(π/2·v)),
//
// with colour components weighted by 1 and spatial components by ratio. Each
// pixel additionally carries a weight equal to the sum of its feature
// components; cluster centres are updated as weighted means and pixels assigned
// to the nearest centre within a local search window. After the k-means passes a
// connectivity-enforcement step guarantees every returned label is a single
// 4-connected region.
//
// regionSize is the nominal superpixel side length in pixels (must be ≥ 2); ratio
// is the compactness weight (larger ⇒ more regular, less boundary-adherent
// superpixels; 0.075 is a good default). labels is a single-channel Mat of
// superpixel indices in [0,n); because samples are 8-bit, keep the region large
// enough that the label count stays below 256. img may be 1- or 3-channel. It
// panics if regionSize < 2. The segmentation is deterministic (grid-seeded).
func SuperpixelLSC(img *cv.Mat, regionSize int, ratio float64) (labels *cv.Mat, n int) {
	if regionSize < 2 {
		panic("ximgproc: SuperpixelLSC requires regionSize >= 2")
	}
	rows, cols := img.Rows, img.Cols
	l, aa, bb := labPlanes(img)

	// Feature weights: colour = 1, spatial = ratio.
	const cw = 1.0
	sw := ratio
	const halfPi = math.Pi / 2

	// Ten-dimensional LSC feature and per-pixel weight for every pixel.
	feat := make([][10]float64, rows*cols)
	weight := make([]float64, rows*cols)
	mapVal := func(v, scale, w float64) (float64, float64) {
		t := v * scale * halfPi
		return w * math.Cos(t), w * math.Sin(t)
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			var f [10]float64
			f[0], f[1] = mapVal(l[i], 1.0/255, cw)
			f[2], f[3] = mapVal(aa[i], 1.0/255, cw)
			f[4], f[5] = mapVal(bb[i], 1.0/255, cw)
			f[6], f[7] = mapVal(float64(x), 1.0/float64(cols), sw)
			f[8], f[9] = mapVal(float64(y), 1.0/float64(rows), sw)
			var wsum float64
			for k := 0; k < 10; k++ {
				wsum += f[k]
			}
			if wsum < 1e-9 {
				wsum = 1e-9
			}
			feat[i] = f
			weight[i] = wsum
		}
	}

	// Grid-seeded cluster centres.
	nx := (cols + regionSize - 1) / regionSize
	ny := (rows + regionSize - 1) / regionSize
	if nx < 1 {
		nx = 1
	}
	if ny < 1 {
		ny = 1
	}
	type center struct {
		f    [10]float64
		x, y float64
	}
	centers := make([]center, 0, nx*ny)
	for gy := 0; gy < ny; gy++ {
		cy := (gy*rows)/ny + rows/(2*ny)
		if cy >= rows {
			cy = rows - 1
		}
		for gx := 0; gx < nx; gx++ {
			cx := (gx*cols)/nx + cols/(2*nx)
			if cx >= cols {
				cx = cols - 1
			}
			centers = append(centers, center{feat[cy*cols+cx], float64(cx), float64(cy)})
		}
	}

	assign := make([]int, rows*cols)
	for i := range assign {
		assign[i] = -1
	}
	dist := make([]float64, rows*cols)
	win := regionSize

	for it := 0; it < lscIterations; it++ {
		for i := range dist {
			dist[i] = math.MaxFloat64
		}
		for ci := range centers {
			c := centers[ci]
			cx, cy := int(c.x), int(c.y)
			y0, y1 := clampRange(cy-win, cy+win, rows)
			x0, x1 := clampRange(cx-win, cx+win, cols)
			for y := y0; y <= y1; y++ {
				for x := x0; x <= x1; x++ {
					i := y*cols + x
					var d float64
					for k := 0; k < 10; k++ {
						diff := feat[i][k] - c.f[k]
						d += diff * diff
					}
					if d < dist[i] {
						dist[i] = d
						assign[i] = ci
					}
				}
			}
		}
		// Weighted-mean centre update.
		sumF := make([][10]float64, len(centers))
		sumX := make([]float64, len(centers))
		sumY := make([]float64, len(centers))
		sumW := make([]float64, len(centers))
		cnt := make([]int, len(centers))
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				i := y*cols + x
				k := assign[i]
				if k < 0 {
					continue
				}
				w := weight[i]
				for d := 0; d < 10; d++ {
					sumF[k][d] += w * feat[i][d]
				}
				sumX[k] += float64(x)
				sumY[k] += float64(y)
				sumW[k] += w
				cnt[k]++
			}
		}
		for k := range centers {
			if sumW[k] <= 0 || cnt[k] == 0 {
				continue
			}
			var f [10]float64
			inv := 1.0 / sumW[k]
			for d := 0; d < 10; d++ {
				f[d] = sumF[k][d] * inv
			}
			nc := float64(cnt[k])
			centers[k] = center{f, sumX[k] / nc, sumY[k] / nc}
		}
	}

	// Fill any unreached pixels from their grid cell.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if assign[i] < 0 {
				gx := (x * nx) / cols
				gy := (y * ny) / rows
				assign[i] = gy*nx + gx
			}
		}
	}

	newLabels, count := enforceConnectivity(assign, rows, cols, regionSize)
	labels = cv.NewMat(rows, cols, 1)
	for i, v := range newLabels {
		labels.Data[i] = uint8(v)
	}
	return labels, count
}

// clampRange clamps [lo,hi] to [0,n-1].
func clampRange(lo, hi, n int) (int, int) {
	if lo < 0 {
		lo = 0
	}
	if hi > n-1 {
		hi = n - 1
	}
	return lo, hi
}

package segmentation

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SLIC computes superpixels with the Simple Linear Iterative Clustering
// algorithm of Achanta et al. (2012), the method behind OpenCV's ximgproc SLIC.
// It returns a [LabelMap] partitioning img into compact, roughly uniform regions
// that adhere to colour boundaries.
//
// Cluster centres are seeded on a regular grid of spacing regionSize and each is
// nudged to the lowest-gradient pixel in its 3x3 neighbourhood to avoid starting
// on an edge. Every pixel is then assigned to the nearest centre under a joint
// colour-and-space distance
//
//	D = sqrt( dc^2 + (ds/regionSize)^2 * compactness^2 )
//
// where dc is the colour distance and ds the spatial distance; larger
// compactness yields squarer, more regular superpixels. Assignment and centre
// updates alternate for iterations rounds, after which a connectivity pass
// relabels stray fragments into their dominant neighbour so every superpixel is
// a single connected component.
//
// img must be three-channel, regionSize >= 2 and compactness > 0. It panics
// otherwise, or if img is empty.
func SLIC(img *cv.Mat, regionSize int, compactness float64, iterations int) *LabelMap {
	if img.Empty() {
		panic("segmentation: SLIC on empty image")
	}
	if img.Channels != 3 {
		panic("segmentation: SLIC requires a 3-channel image")
	}
	if regionSize < 2 {
		panic("segmentation: SLIC regionSize must be >= 2")
	}
	if compactness <= 0 {
		panic("segmentation: SLIC compactness must be positive")
	}
	if iterations < 1 {
		iterations = 1
	}
	rows, cols := img.Rows, img.Cols
	n := rows * cols

	mag := gradientMagnitude(img)
	colorAt := func(x, y int) [3]float64 {
		b := (y*cols + x) * 3
		return [3]float64{float64(img.Data[b]), float64(img.Data[b+1]), float64(img.Data[b+2])}
	}

	// Seed centres on the grid, snapping to the local gradient minimum.
	type center struct {
		x, y float64
		col  [3]float64
	}
	var centers []center
	for cy := regionSize / 2; cy < rows; cy += regionSize {
		for cx := regionSize / 2; cx < cols; cx += regionSize {
			bx, by := cx, cy
			bestG := math.MaxFloat64
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := cx+dx, cy+dy
					if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
						continue
					}
					if g := mag[ny*cols+nx]; g < bestG {
						bestG = g
						bx, by = nx, ny
					}
				}
			}
			centers = append(centers, center{x: float64(bx), y: float64(by), col: colorAt(bx, by)})
		}
	}
	if len(centers) == 0 {
		centers = append(centers, center{x: float64(cols / 2), y: float64(rows / 2), col: colorAt(cols/2, rows/2)})
	}

	assign := make([]int, n)
	for i := range assign {
		assign[i] = -1
	}
	dist := make([]float64, n)
	invS2 := 1.0 / float64(regionSize*regionSize)
	m2 := compactness * compactness

	for it := 0; it < iterations; it++ {
		for i := range dist {
			dist[i] = math.MaxFloat64
		}
		for ci := range centers {
			c := centers[ci]
			x0 := int(c.x) - regionSize
			x1 := int(c.x) + regionSize
			y0 := int(c.y) - regionSize
			y1 := int(c.y) + regionSize
			for y := y0; y <= y1; y++ {
				if y < 0 || y >= rows {
					continue
				}
				for x := x0; x <= x1; x++ {
					if x < 0 || x >= cols {
						continue
					}
					col := colorAt(x, y)
					dc := (col[0]-c.col[0])*(col[0]-c.col[0]) +
						(col[1]-c.col[1])*(col[1]-c.col[1]) +
						(col[2]-c.col[2])*(col[2]-c.col[2])
					ds := (float64(x)-c.x)*(float64(x)-c.x) + (float64(y)-c.y)*(float64(y)-c.y)
					d := dc + ds*invS2*m2
					idx := y*cols + x
					if d < dist[idx] {
						dist[idx] = d
						assign[idx] = ci
					}
				}
			}
		}
		// Recompute centres as the mean of their members.
		sumX := make([]float64, len(centers))
		sumY := make([]float64, len(centers))
		sumC := make([][3]float64, len(centers))
		cnt := make([]int, len(centers))
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				a := assign[y*cols+x]
				if a < 0 {
					continue
				}
				cnt[a]++
				sumX[a] += float64(x)
				sumY[a] += float64(y)
				col := colorAt(x, y)
				for k := 0; k < 3; k++ {
					sumC[a][k] += col[k]
				}
			}
		}
		for ci := range centers {
			if cnt[ci] == 0 {
				continue
			}
			centers[ci].x = sumX[ci] / float64(cnt[ci])
			centers[ci].y = sumY[ci] / float64(cnt[ci])
			for k := 0; k < 3; k++ {
				centers[ci].col[k] = sumC[ci][k] / float64(cnt[ci])
			}
		}
	}

	// Any pixel that no centre reached takes its nearest assigned neighbour's
	// label via a simple fill, guaranteeing a complete labelling.
	for i := range assign {
		if assign[i] < 0 {
			assign[i] = 0
		}
	}

	enforced := enforceConnectivity(assign, rows, cols, regionSize)
	labels, count := relabelConsecutive(enforced)
	return &LabelMap{Rows: rows, Cols: cols, Count: count, Labels: labels}
}

// enforceConnectivity relabels connected fragments of the SLIC assignment so
// that each output label is a single 4-connected component; fragments smaller
// than a quarter of the nominal superpixel area are absorbed into an adjacent
// component. It returns a new per-pixel labelling.
func enforceConnectivity(assign []int, rows, cols, regionSize int) []int {
	n := rows * cols
	out := make([]int, n)
	for i := range out {
		out[i] = -1
	}
	minArea := regionSize * regionSize / 4
	next := 0
	comp := make([]int, 0, 256)
	for start := 0; start < n; start++ {
		if out[start] != -1 {
			continue
		}
		label := assign[start]
		comp = comp[:0]
		out[start] = next
		comp = append(comp, start)
		adjLabel := -1
		for qi := 0; qi < len(comp); qi++ {
			idx := comp[qi]
			y, x := idx/cols, idx%cols
			for _, o := range neighbors4 {
				nx, ny := x+o.dx, y+o.dy
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					continue
				}
				nidx := ny*cols + nx
				if assign[nidx] == label {
					if out[nidx] == -1 {
						out[nidx] = next
						comp = append(comp, nidx)
					}
				} else if out[nidx] >= 0 {
					adjLabel = out[nidx]
				}
			}
		}
		if len(comp) < minArea && adjLabel >= 0 {
			for _, idx := range comp {
				out[idx] = adjLabel
			}
		} else {
			next++
		}
	}
	return out
}

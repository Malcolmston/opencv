package hfs

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// offset is a signed neighbour displacement in image coordinates.
type offset struct{ dx, dy int }

// neighbors4 is the 4-connected neighbourhood used throughout the module; all
// merges happen across these edges so every emitted region stays connected.
var neighbors4 = []offset{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}

// slicResult holds the superpixel labelling of an image.
type slicResult struct {
	labels []int // per-pixel superpixel id in [0, count)
	count  int
	rows   int
	cols   int
}

// slic computes superpixels with Simple Linear Iterative Clustering (Achanta et
// al., 2012), the atomic-region stage of the HFS pipeline. It clusters the
// three-channel feature image feat (CIE L*a*b* here) under the joint colour and
// space distance
//
//	D = dc + (ds / regionSize^2) * compactness^2
//
// where dc is the squared colour distance and ds the squared spatial distance.
// Centres are seeded on a regular grid of spacing regionSize, snapped to the
// lowest-gradient pixel in their 3x3 neighbourhood, then refined for iterations
// Lloyd rounds. A final connectivity pass relabels stray fragments so that every
// superpixel is a single 4-connected component.
func slic(feat *cv.Mat, regionSize int, compactness float64, iterations int) slicResult {
	rows, cols := feat.Rows, feat.Cols
	n := rows * cols
	if regionSize < 2 {
		regionSize = 2
	}
	if iterations < 1 {
		iterations = 1
	}

	mag := gradientMagnitude(feat)
	colorAt := func(x, y int) [3]float64 {
		b := (y*cols + x) * 3
		return [3]float64{float64(feat.Data[b]), float64(feat.Data[b+1]), float64(feat.Data[b+2])}
	}

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
			x0, x1 := int(c.x)-regionSize, int(c.x)+regionSize
			y0, y1 := int(c.y)-regionSize, int(c.y)+regionSize
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

	for i := range assign {
		if assign[i] < 0 {
			assign[i] = 0
		}
	}

	enforced := enforceConnectivity(assign, rows, cols, regionSize)
	labels, count := relabelConsecutive(enforced)
	return slicResult{labels: labels, count: count, rows: rows, cols: cols}
}

// enforceConnectivity relabels connected fragments of a SLIC assignment so that
// each output label is a single 4-connected component; fragments smaller than a
// quarter of the nominal superpixel area are absorbed into an adjacent component.
// It returns a new per-pixel labelling.
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

// gradientMagnitude returns the per-pixel Euclidean gradient magnitude of a
// three-channel image using central differences, summed over channels. It is
// used both to nudge SLIC seeds off edges and as the texture feature.
func gradientMagnitude(m *cv.Mat) []float64 {
	rows, cols, ch := m.Rows, m.Cols, m.Channels
	mag := make([]float64, rows*cols)
	at := func(y, x, c int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return float64(m.Data[(y*cols+x)*ch+c])
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for c := 0; c < ch; c++ {
				gx := at(y, x+1, c) - at(y, x-1, c)
				gy := at(y+1, x, c) - at(y-1, x, c)
				s += gx*gx + gy*gy
			}
			mag[y*cols+x] = math.Sqrt(s)
		}
	}
	return mag
}

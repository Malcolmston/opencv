package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// slicIterations is the fixed number of k-means refinement passes. Ten is the
// value recommended by Achanta et al. and is enough for convergence in
// practice.
const slicIterations = 10

// SuperpixelSLIC segments img into compact, roughly equally sized superpixels
// using Simple Linear Iterative Clustering (Achanta et al., 2012) and returns a
// label image together with the number of superpixels produced.
//
// SLIC runs a constrained k-means in the 5-D space of CIE Lab colour plus (x,y)
// position. Cluster centres start on a regular grid spaced regionSize pixels
// apart; each is then refined by assigning nearby pixels (within a 2·regionSize
// window) to the nearest centre under the distance
//
//	D = √( d_lab² + (d_xy / regionSize)² · ruler² ),
//
// and recomputing centres as the mean of their members. regionSize sets the
// nominal superpixel side length in pixels; ruler is the compactness weight —
// larger values favour square, spatially regular superpixels over colour
// adherence. After the k-means passes a connectivity-enforcement step relabels
// the image so that every returned label is a single 4-connected region, with
// stray fragments merged into an adjacent superpixel.
//
// labels is a single-channel Mat whose sample at (y,x) is that pixel's
// superpixel index in [0, n). Because Mat samples are 8-bit, at most 256
// distinct labels can be represented; keep regionSize large enough that
// rows/regionSize · cols/regionSize < 256 (larger label counts are deferred —
// see the package documentation). img may be 1- or 3-channel; a grayscale image
// is treated as Lab with zero chroma. It panics if regionSize < 2.
//
// The segmentation is deterministic: centres are seeded from the fixed grid, so
// repeated calls on identical input yield identical labels.
func SuperpixelSLIC(img *cv.Mat, regionSize int, ruler float64) (labels *cv.Mat, n int) {
	if regionSize < 2 {
		panic("ximgproc: SuperpixelSLIC requires regionSize >= 2")
	}
	rows, cols := img.Rows, img.Cols

	// Lab colour planes as floats.
	l, aa, bb := labPlanes(img)

	// Seed grid of cluster centres. Offset by half a cell so centres sit inside
	// the image, then snap the count to cover all rows/cols.
	nx := (cols + regionSize - 1) / regionSize
	ny := (rows + regionSize - 1) / regionSize
	if nx < 1 {
		nx = 1
	}
	if ny < 1 {
		ny = 1
	}
	type center struct {
		l, a, b, x, y float64
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
			i := cy*cols + cx
			centers = append(centers, center{l[i], aa[i], bb[i], float64(cx), float64(cy)})
		}
	}

	assign := make([]int, rows*cols)
	for i := range assign {
		assign[i] = -1
	}
	dist := make([]float64, rows*cols)

	invRegion2 := 1.0 / float64(regionSize*regionSize)
	m2 := ruler * ruler
	win := regionSize // half-window is regionSize -> 2*regionSize side

	for it := 0; it < slicIterations; it++ {
		for i := range dist {
			dist[i] = math.MaxFloat64
		}
		for ci := range centers {
			c := centers[ci]
			cx, cy := int(c.x), int(c.y)
			y0, y1 := cy-win, cy+win
			x0, x1 := cx-win, cx+win
			if y0 < 0 {
				y0 = 0
			}
			if y1 > rows-1 {
				y1 = rows - 1
			}
			if x0 < 0 {
				x0 = 0
			}
			if x1 > cols-1 {
				x1 = cols - 1
			}
			for y := y0; y <= y1; y++ {
				for x := x0; x <= x1; x++ {
					i := y*cols + x
					dl := l[i] - c.l
					da := aa[i] - c.a
					db := bb[i] - c.b
					dc := dl*dl + da*da + db*db
					dx := float64(x) - c.x
					dy := float64(y) - c.y
					ds := (dx*dx + dy*dy) * invRegion2
					d := dc + ds*m2
					if d < dist[i] {
						dist[i] = d
						assign[i] = ci
					}
				}
			}
		}
		// Recompute centres as the mean of assigned pixels.
		sumL := make([]float64, len(centers))
		sumA := make([]float64, len(centers))
		sumB := make([]float64, len(centers))
		sumX := make([]float64, len(centers))
		sumY := make([]float64, len(centers))
		cnt := make([]int, len(centers))
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				i := y*cols + x
				k := assign[i]
				if k < 0 {
					continue
				}
				sumL[k] += l[i]
				sumA[k] += aa[i]
				sumB[k] += bb[i]
				sumX[k] += float64(x)
				sumY[k] += float64(y)
				cnt[k]++
			}
		}
		for k := range centers {
			if cnt[k] == 0 {
				continue
			}
			f := 1.0 / float64(cnt[k])
			centers[k] = center{sumL[k] * f, sumA[k] * f, sumB[k] * f, sumX[k] * f, sumY[k] * f}
		}
	}

	// Any pixel never reached by a search window (possible for tiny images) is
	// assigned to its grid cell.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y * cols
			if assign[i+x] < 0 {
				gx := (x * nx) / cols
				gy := (y * ny) / rows
				assign[i+x] = gy*nx + gx
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

// labPlanes returns the L, a, b float planes of img on the OpenCV 8-bit Lab
// scale. A grayscale image maps to L = gray with zero chroma.
func labPlanes(img *cv.Mat) (l, a, b []float64) {
	rows, cols := img.Rows, img.Cols
	l = make([]float64, rows*cols)
	a = make([]float64, rows*cols)
	b = make([]float64, rows*cols)
	switch img.Channels {
	case 1:
		for i, v := range img.Data {
			l[i] = float64(v)
		}
	case 3:
		lab := cv.CvtColor(img, cv.ColorRGB2Lab)
		for i := 0; i < rows*cols; i++ {
			l[i] = float64(lab.Data[i*3+0])
			a[i] = float64(lab.Data[i*3+1])
			b[i] = float64(lab.Data[i*3+2])
		}
	default:
		panic("ximgproc: SuperpixelSLIC expects a 1- or 3-channel image")
	}
	return l, a, b
}

// enforceConnectivity relabels a k-means assignment so that each output label
// is a single 4-connected component. Fragments smaller than a quarter of the
// nominal superpixel area are merged into an adjacent (already relabelled)
// superpixel. It returns the new labels (0..count-1) and the count.
func enforceConnectivity(assign []int, rows, cols, regionSize int) ([]int, int) {
	const (
		nDir = 4
	)
	dx := [nDir]int{-1, 1, 0, 0}
	dy := [nDir]int{0, 0, -1, 1}

	nl := make([]int, rows*cols)
	for i := range nl {
		nl[i] = -1
	}
	minSize := regionSize * regionSize / 4
	label := 0
	queue := make([]int, 0, rows*cols)

	for start := 0; start < rows*cols; start++ {
		if nl[start] >= 0 {
			continue
		}
		nl[start] = label
		oldLabel := assign[start]

		// Find an adjacent already-labelled superpixel to absorb this segment
		// into if it turns out to be too small.
		adj := 0
		sy, sx := start/cols, start%cols
		for d := 0; d < nDir; d++ {
			ny, nx := sy+dy[d], sx+dx[d]
			if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
				continue
			}
			if nl[ny*cols+nx] >= 0 {
				adj = nl[ny*cols+nx]
			}
		}

		queue = queue[:0]
		queue = append(queue, start)
		for head := 0; head < len(queue); head++ {
			p := queue[head]
			py, px := p/cols, p%cols
			for d := 0; d < nDir; d++ {
				ny, nx := py+dy[d], px+dx[d]
				if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
					continue
				}
				q := ny*cols + nx
				if nl[q] < 0 && assign[q] == oldLabel {
					nl[q] = label
					queue = append(queue, q)
				}
			}
		}

		if len(queue) <= minSize {
			for _, p := range queue {
				nl[p] = adj
			}
			// Reuse this label number for the next segment.
			continue
		}
		label++
	}
	if label == 0 {
		label = 1 // single-segment image
	}
	return nl, label
}

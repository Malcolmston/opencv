package segment2

import (
	cv "github.com/malcolmston/opencv"
)

// ComponentStat summarises one connected component of a [LabelMap].
type ComponentStat struct {
	// Label is the component's label value.
	Label int
	// Area is the component's pixel count.
	Area int
	// Rect is the component's axis-aligned bounding box.
	Rect cv.Rect
	// Centroid is the component's rounded centre of mass.
	Centroid cv.Point
}

// OtsuThreshold computes the global intensity threshold that maximises
// between-class variance (Otsu's method) over the luminance histogram of img.
// The returned value lies in [0, 255]; pixels with intensity greater than the
// threshold are the bright class.
//
// It panics if img is empty.
func OtsuThreshold(img *cv.Mat) float64 {
	segment2requireNonEmpty(img, "OtsuThreshold")
	gray := segment2gray(img)
	var hist [256]float64
	for _, v := range gray {
		iv := int(v + 0.5)
		if iv < 0 {
			iv = 0
		} else if iv > 255 {
			iv = 255
		}
		hist[iv]++
	}
	total := float64(len(gray))
	var sum float64
	for i := 0; i < 256; i++ {
		sum += float64(i) * hist[i]
	}
	var sumB, wB float64
	var maxVar float64
	best := 0
	for t := 0; t < 256; t++ {
		wB += hist[t]
		if wB == 0 {
			continue
		}
		wF := total - wB
		if wF == 0 {
			break
		}
		sumB += float64(t) * hist[t]
		mB := sumB / wB
		mF := (sum - sumB) / wF
		between := wB * wF * (mB - mF) * (mB - mF)
		if between > maxVar {
			maxVar = between
			best = t
		}
	}
	return float64(best)
}

// ConnectedComponents labels the connected non-zero (foreground) regions of a
// binary single-channel image with a two-pass union-find algorithm. In the
// returned [LabelMap], label 0 is the background and labels 1..k are the
// components. conn selects 4- or 8-connectivity.
//
// It panics if binary is empty, has more than one channel, or conn is invalid.
func ConnectedComponents(binary *cv.Mat, conn Connectivity) *LabelMap {
	segment2requireNonEmpty(binary, "ConnectedComponents")
	if binary.Channels != 1 {
		panic("segment2: ConnectedComponents requires a single-channel image")
	}
	if conn != Conn4 && conn != Conn8 {
		panic("segment2: ConnectedComponents connectivity must be Conn4 or Conn8")
	}
	rows, cols := binary.Rows, binary.Cols
	prov := make([]int, rows*cols)
	uf := segment2newUF(1) // index 0 reserved for background
	next := 1

	// Neighbours already visited in a raster scan.
	var scan []segment2offset
	if conn == Conn8 {
		scan = []segment2offset{{-1, 0}, {0, -1}, {-1, -1}, {1, -1}}
	} else {
		scan = []segment2offset{{-1, 0}, {0, -1}}
	}

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if binary.Data[i] == 0 {
				continue
			}
			best := 0
			for _, o := range scan {
				nx, ny := x+o.dx, y+o.dy
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					continue
				}
				nl := prov[ny*cols+nx]
				if nl == 0 {
					continue
				}
				if best == 0 {
					best = nl
				} else {
					uf.union(best, nl)
					best = uf.find(best)
				}
			}
			if best == 0 {
				uf.parent = append(uf.parent, next)
				best = next
				next++
			}
			prov[i] = best
		}
	}

	// Second pass: resolve to root labels and compact to 1..k.
	remap := make(map[int]int)
	labels := make([]int, rows*cols)
	count := 1
	for i, p := range prov {
		if p == 0 {
			continue
		}
		r := uf.find(p)
		nl, ok := remap[r]
		if !ok {
			nl = count
			remap[r] = nl
			count++
		}
		labels[i] = nl
	}
	return &LabelMap{Rows: rows, Cols: cols, Labels: labels, NumLabels: count}
}

// ThresholdComponents thresholds img at the given intensity (pixels with
// luminance greater than thresh become foreground) and returns the connected
// components of the resulting binary image as a [LabelMap] with background
// label 0. If thresh is negative, the [OtsuThreshold] of img is used.
//
// It panics if img is empty or conn is invalid.
func ThresholdComponents(img *cv.Mat, thresh float64, conn Connectivity) *LabelMap {
	segment2requireNonEmpty(img, "ThresholdComponents")
	if thresh < 0 {
		thresh = OtsuThreshold(img)
	}
	gray := segment2gray(img)
	bin := cv.NewMat(img.Rows, img.Cols, 1)
	for i, v := range gray {
		if v > thresh {
			bin.Data[i] = 255
		}
	}
	return ConnectedComponents(bin, conn)
}

// ComponentStats measures every non-background label (1..NumLabels-1) of lm and
// returns their [ComponentStat] records ordered by label. Empty labels are
// skipped.
func ComponentStats(lm *LabelMap) []ComponentStat {
	sizes := lm.RegionSizes()
	rects := lm.BoundingRects()
	cents := lm.RegionCentroids()
	var out []ComponentStat
	for l := 1; l < lm.NumLabels; l++ {
		if sizes[l] == 0 {
			continue
		}
		out = append(out, ComponentStat{
			Label:    l,
			Area:     sizes[l],
			Rect:     rects[l],
			Centroid: cents[l],
		})
	}
	return out
}

// LargestComponent returns the label of the largest non-background component of
// lm, or 0 if there are no foreground pixels.
func LargestComponent(lm *LabelMap) int {
	sizes := lm.RegionSizes()
	best, bestArea := 0, 0
	for l := 1; l < len(sizes); l++ {
		if sizes[l] > bestArea {
			bestArea = sizes[l]
			best = l
		}
	}
	return best
}

// FilterComponentsBySize returns a copy of lm in which every non-background
// component smaller than minArea pixels is reset to the background label 0. The
// surviving labels are then compacted to 1..k. lm itself is not modified.
func FilterComponentsBySize(lm *LabelMap, minArea int) *LabelMap {
	sizes := lm.RegionSizes()
	out := lm.Clone()
	for i, l := range out.Labels {
		if l > 0 && l < len(sizes) && sizes[l] < minArea {
			out.Labels[i] = 0
		}
	}
	// Compact non-zero labels while keeping background at 0.
	remap := map[int]int{0: 0}
	next := 1
	for i, l := range out.Labels {
		if l == 0 {
			continue
		}
		nl, ok := remap[l]
		if !ok {
			nl = next
			remap[l] = nl
			next++
		}
		out.Labels[i] = nl
	}
	out.NumLabels = next
	return out
}

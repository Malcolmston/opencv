package connected

import cv "github.com/malcolmston/opencv"

// Component holds summary statistics for one connected component, mirroring the
// per-label output of OpenCV's connectedComponentsWithStats.
type Component struct {
	// Label is the component's label in its source [Labels] map (>= 1).
	Label int
	// Area is the number of foreground pixels in the component.
	Area int
	// BBox is the tightest upright bounding box around the component.
	BBox cv.Rect
	// CentroidX is the mean column coordinate of the component's pixels.
	CentroidX float64
	// CentroidY is the mean row coordinate of the component's pixels.
	CentroidY float64
}

// Width returns the width in pixels of the component's bounding box.
func (c Component) Width() int { return c.BBox.Width }

// Height returns the height in pixels of the component's bounding box.
func (c Component) Height() int { return c.BBox.Height }

// Centroid returns the component's centre of mass as (x, y).
func (c Component) Centroid() (x, y float64) { return c.CentroidX, c.CentroidY }

// AspectRatio returns the bounding box aspect ratio (width divided by height).
// It returns 0 for a degenerate zero-height box.
func (c Component) AspectRatio() float64 {
	if c.BBox.Height == 0 {
		return 0
	}
	return float64(c.BBox.Width) / float64(c.BBox.Height)
}

// Extent returns the ratio of the component's area to its bounding-box area, a
// value in (0, 1] measuring how completely the component fills its box. It
// returns 0 for a degenerate box.
func (c Component) Extent() float64 {
	boxArea := c.BBox.Width * c.BBox.Height
	if boxArea == 0 {
		return 0
	}
	return float64(c.Area) / float64(boxArea)
}

// Contains reports whether pixel (x, y) lies inside the component's bounding
// box. It is a cheap coarse test; use a [Labels] map for exact membership.
func (c Component) Contains(x, y int) bool {
	return x >= c.BBox.X && x < c.BBox.X+c.BBox.Width &&
		y >= c.BBox.Y && y < c.BBox.Y+c.BBox.Height
}

// ComponentStats labels src and returns the statistics of every foreground
// component, ordered by label. It is shorthand for LabelWithStats followed by
// discarding the label map.
func ComponentStats(src *cv.Mat, conn Connectivity) []Component {
	return Label(src, conn).Components()
}

// CountComponents returns the number of connected foreground components in a
// binary image.
func CountComponents(src *cv.Mat, conn Connectivity) int {
	return Label(src, conn).Count
}

// Areas returns the pixel area of every foreground component, ordered by label.
func Areas(src *cv.Mat, conn Connectivity) []int {
	comps := ComponentStats(src, conn)
	out := make([]int, len(comps))
	for i, c := range comps {
		out[i] = c.Area
	}
	return out
}

// BoundingBoxes returns the bounding box of every foreground component, ordered
// by label.
func BoundingBoxes(src *cv.Mat, conn Connectivity) []cv.Rect {
	comps := ComponentStats(src, conn)
	out := make([]cv.Rect, len(comps))
	for i, c := range comps {
		out[i] = c.BBox
	}
	return out
}

// LargestComponent returns the component with the greatest area and true, or a
// zero Component and false when the image has no foreground. Ties are broken in
// favour of the lower label.
func LargestComponent(src *cv.Mat, conn Connectivity) (Component, bool) {
	comps := ComponentStats(src, conn)
	if len(comps) == 0 {
		return Component{}, false
	}
	best := comps[0]
	for _, c := range comps[1:] {
		if c.Area > best.Area {
			best = c
		}
	}
	return best, true
}

// SmallestComponent returns the component with the least area and true, or a
// zero Component and false when the image has no foreground. Ties are broken in
// favour of the lower label.
func SmallestComponent(src *cv.Mat, conn Connectivity) (Component, bool) {
	comps := ComponentStats(src, conn)
	if len(comps) == 0 {
		return Component{}, false
	}
	best := comps[0]
	for _, c := range comps[1:] {
		if c.Area < best.Area {
			best = c
		}
	}
	return best, true
}

// LargestComponentMask returns a binary image containing only the largest
// component (255 on that component, 0 elsewhere). An all-background input
// yields an all-zero image.
func LargestComponentMask(src *cv.Mat, conn Connectivity) *cv.Mat {
	lbl := Label(src, conn)
	comps := lbl.Components()
	out := cv.NewMat(src.Rows, src.Cols, 1)
	if len(comps) == 0 {
		return out
	}
	best := comps[0]
	for _, c := range comps[1:] {
		if c.Area > best.Area {
			best = c
		}
	}
	for i, v := range lbl.Data {
		if v == best.Label {
			out.Data[i] = 255
		}
	}
	return out
}

// KeepLargest is an alias for [LargestComponentMask]: it removes every
// component except the single largest one.
func KeepLargest(src *cv.Mat, conn Connectivity) *cv.Mat {
	return LargestComponentMask(src, conn)
}

// KeepLargestN returns a binary image retaining only the n largest components
// by area (255 on kept components, 0 elsewhere). If n is zero or negative the
// result is empty; if n exceeds the component count all components are kept.
// Ties are broken in favour of the lower label.
func KeepLargestN(src *cv.Mat, conn Connectivity, n int) *cv.Mat {
	lbl := Label(src, conn)
	out := cv.NewMat(src.Rows, src.Cols, 1)
	if n <= 0 {
		return out
	}
	comps := lbl.Components()
	// Stable selection of the n largest by area (label breaks ties).
	order := make([]int, len(comps))
	for i := range comps {
		order[i] = i
	}
	connectedSortByAreaDesc(comps, order)
	keep := make(map[int]bool)
	for i := 0; i < n && i < len(order); i++ {
		keep[comps[order[i]].Label] = true
	}
	for i, v := range lbl.Data {
		if keep[v] {
			out.Data[i] = 255
		}
	}
	return out
}

// FilterByArea returns a binary image keeping only components whose area is in
// the inclusive range [minArea, maxArea]. A negative maxArea means "no upper
// bound". Kept components are 255, everything else 0.
func FilterByArea(src *cv.Mat, conn Connectivity, minArea, maxArea int) *cv.Mat {
	lbl := Label(src, conn)
	areas := lbl.Areas()
	out := cv.NewMat(src.Rows, src.Cols, 1)
	for i, v := range lbl.Data {
		if v == 0 {
			continue
		}
		a := areas[v]
		if a < minArea {
			continue
		}
		if maxArea >= 0 && a > maxArea {
			continue
		}
		out.Data[i] = 255
	}
	return out
}

// RemoveSmallComponents returns a binary image with every component smaller
// than minArea pixels deleted; the survivors are rendered at 255. It is the
// classic "despeckle" operation.
func RemoveSmallComponents(src *cv.Mat, conn Connectivity, minArea int) *cv.Mat {
	return FilterByArea(src, conn, minArea, -1)
}

// connectedSortByAreaDesc sorts order (indices into comps) by descending area,
// breaking ties by ascending label, using an insertion sort so the result is
// deterministic and allocation-free.
func connectedSortByAreaDesc(comps []Component, order []int) {
	for i := 1; i < len(order); i++ {
		j := i
		for j > 0 {
			a, b := comps[order[j-1]], comps[order[j]]
			if a.Area > b.Area || (a.Area == b.Area && a.Label <= b.Label) {
				break
			}
			order[j-1], order[j] = order[j], order[j-1]
			j--
		}
	}
}

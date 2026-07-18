package connected

import cv "github.com/malcolmston/opencv"

// Labels is the result of connected-component labelling: a dense, row-major map
// assigning each pixel an integer label. Label 0 is the background; labels
// 1..Count identify the foreground components. Width and Height give the map
// dimensions and Data has length Width*Height.
type Labels struct {
	// Width is the image width in pixels (number of columns).
	Width int
	// Height is the image height in pixels (number of rows).
	Height int
	// Count is the number of foreground components (labels 1..Count).
	Count int
	// Data holds the per-pixel labels in row-major order.
	Data []int
}

// Label assigns a unique label to every connected foreground region of a binary
// single-channel image using a two-pass union-find algorithm. Any non-zero
// sample is foreground. conn selects 4- or 8-connectivity. It panics if src is
// not single-channel or conn is invalid.
func Label(src *cv.Mat, conn Connectivity) *Labels {
	connectedRequireBinary(src, "Label")
	connectedCheckConn(conn, "Label")
	data, count := connectedLabelMask(connectedForeground(src), src.Cols, src.Rows, conn)
	return &Labels{Width: src.Cols, Height: src.Rows, Count: count, Data: data}
}

// LabelWithStats labels src like [Label] and additionally returns per-component
// statistics (area, bounding box and centroid), mirroring OpenCV's
// connectedComponentsWithStats. The stats slice has one [Component] per
// foreground label, ordered by label (element 0 is label 1).
func LabelWithStats(src *cv.Mat, conn Connectivity) (*Labels, []Component) {
	lbl := Label(src, conn)
	return lbl, lbl.Components()
}

// At returns the label of pixel (x, y). It returns 0 for coordinates outside
// the image so callers can probe freely.
func (l *Labels) At(x, y int) int {
	if x < 0 || y < 0 || x >= l.Width || y >= l.Height {
		return 0
	}
	return l.Data[y*l.Width+x]
}

// Size returns the map dimensions as (width, height).
func (l *Labels) Size() (w, h int) { return l.Width, l.Height }

// NumComponents returns the number of foreground components, excluding the
// background. It is equal to the Count field.
func (l *Labels) NumComponents() int { return l.Count }

// Areas returns a slice of length Count+1 giving the pixel area of each label;
// element 0 is the background area and element k the area of label k.
func (l *Labels) Areas() []int {
	areas := make([]int, l.Count+1)
	for _, v := range l.Data {
		areas[v]++
	}
	return areas
}

// Mask returns a binary single-channel image in which pixels carrying the given
// label are 255 and all others 0. It panics if label is out of range (0 selects
// the background).
func (l *Labels) Mask(label int) *cv.Mat {
	if label < 0 || label > l.Count {
		panic("connected: Labels.Mask: label out of range")
	}
	out := cv.NewMat(l.Height, l.Width, 1)
	for i, v := range l.Data {
		if v == label {
			out.Data[i] = 255
		}
	}
	return out
}

// ToMat renders the label map as a single-channel image, storing each label
// modulo 256. It is primarily useful for debugging small images; use
// [Labels.ColorImage] for a visualisation that distinguishes many components.
func (l *Labels) ToMat() *cv.Mat {
	out := cv.NewMat(l.Height, l.Width, 1)
	for i, v := range l.Data {
		out.Data[i] = uint8(v & 0xff)
	}
	return out
}

// ColorImage renders the label map as a 3-channel RGB image, assigning each
// component a distinct, deterministic colour and painting the background black.
// It is intended for visual inspection of labelling results.
func (l *Labels) ColorImage() *cv.Mat {
	out := cv.NewMat(l.Height, l.Width, 3)
	palette := make([][3]uint8, l.Count+1)
	for k := 1; k <= l.Count; k++ {
		palette[k] = connectedLabelColor(k)
	}
	for i, v := range l.Data {
		c := palette[v]
		out.Data[i*3] = c[0]
		out.Data[i*3+1] = c[1]
		out.Data[i*3+2] = c[2]
	}
	return out
}

// BoundingBox returns the tightest upright rectangle enclosing the pixels of
// the given label. It panics if label is not in 1..Count.
func (l *Labels) BoundingBox(label int) cv.Rect {
	l.checkFg(label, "BoundingBox")
	minX, minY := l.Width, l.Height
	maxX, maxY := -1, -1
	for y := 0; y < l.Height; y++ {
		for x := 0; x < l.Width; x++ {
			if l.Data[y*l.Width+x] != label {
				continue
			}
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}
	}
	return cv.Rect{X: minX, Y: minY, Width: maxX - minX + 1, Height: maxY - minY + 1}
}

// Centroid returns the area-weighted centre of mass (mean x, mean y) of the
// given label's pixels. It panics if label is not in 1..Count.
func (l *Labels) Centroid(label int) (cx, cy float64) {
	l.checkFg(label, "Centroid")
	var sx, sy, n int
	for y := 0; y < l.Height; y++ {
		for x := 0; x < l.Width; x++ {
			if l.Data[y*l.Width+x] == label {
				sx += x
				sy += y
				n++
			}
		}
	}
	return float64(sx) / float64(n), float64(sy) / float64(n)
}

// Component computes the [Component] statistics for a single label. It panics
// if label is not in 1..Count.
func (l *Labels) Component(label int) Component {
	l.checkFg(label, "Component")
	var sx, sy, area int
	minX, minY := l.Width, l.Height
	maxX, maxY := -1, -1
	for y := 0; y < l.Height; y++ {
		for x := 0; x < l.Width; x++ {
			if l.Data[y*l.Width+x] != label {
				continue
			}
			area++
			sx += x
			sy += y
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}
	}
	return Component{
		Label:     label,
		Area:      area,
		BBox:      cv.Rect{X: minX, Y: minY, Width: maxX - minX + 1, Height: maxY - minY + 1},
		CentroidX: float64(sx) / float64(area),
		CentroidY: float64(sy) / float64(area),
	}
}

// Components computes [Component] statistics for every foreground label in a
// single pass and returns them ordered by label (element 0 is label 1). The
// slice is empty when there are no components.
func (l *Labels) Components() []Component {
	if l.Count == 0 {
		return nil
	}
	area := make([]int, l.Count+1)
	sx := make([]int, l.Count+1)
	sy := make([]int, l.Count+1)
	minX := make([]int, l.Count+1)
	minY := make([]int, l.Count+1)
	maxX := make([]int, l.Count+1)
	maxY := make([]int, l.Count+1)
	for k := 1; k <= l.Count; k++ {
		minX[k], minY[k] = l.Width, l.Height
		maxX[k], maxY[k] = -1, -1
	}
	for y := 0; y < l.Height; y++ {
		for x := 0; x < l.Width; x++ {
			v := l.Data[y*l.Width+x]
			if v == 0 {
				continue
			}
			area[v]++
			sx[v] += x
			sy[v] += y
			if x < minX[v] {
				minX[v] = x
			}
			if x > maxX[v] {
				maxX[v] = x
			}
			if y < minY[v] {
				minY[v] = y
			}
			if y > maxY[v] {
				maxY[v] = y
			}
		}
	}
	out := make([]Component, l.Count)
	for k := 1; k <= l.Count; k++ {
		out[k-1] = Component{
			Label:     k,
			Area:      area[k],
			BBox:      cv.Rect{X: minX[k], Y: minY[k], Width: maxX[k] - minX[k] + 1, Height: maxY[k] - minY[k] + 1},
			CentroidX: float64(sx[k]) / float64(area[k]),
			CentroidY: float64(sy[k]) / float64(area[k]),
		}
	}
	return out
}

// checkFg panics unless label identifies a foreground component (1..Count).
func (l *Labels) checkFg(label int, who string) {
	if label < 1 || label > l.Count {
		panic("connected: Labels." + who + ": label out of range")
	}
}

// connectedLabelColor maps a label index to a deterministic, bright RGB colour
// using a simple fixed-point hash so adjacent labels look distinct.
func connectedLabelColor(k int) [3]uint8 {
	h := uint32(k) * 2654435761
	r := uint8(64 + (h>>0)%192)
	g := uint8(64 + (h>>8)%192)
	b := uint8(64 + (h>>16)%192)
	return [3]uint8{r, g, b}
}

package connected

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Blob is a richer descriptor for one connected component, adding shape
// measurements to the basic statistics of a [Component]. All measurements are
// derived from the discrete pixel set and its inner boundary.
type Blob struct {
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
	// Perimeter is the number of inner boundary pixels of the component.
	Perimeter int
	// Holes is the number of enclosed holes within the component.
	Holes int
}

// AspectRatio returns the bounding box aspect ratio (width divided by height),
// or 0 for a degenerate zero-height box.
func (b Blob) AspectRatio() float64 {
	if b.BBox.Height == 0 {
		return 0
	}
	return float64(b.BBox.Width) / float64(b.BBox.Height)
}

// Extent returns the ratio of blob area to bounding-box area, in (0, 1]. It
// returns 0 for a degenerate box.
func (b Blob) Extent() float64 {
	boxArea := b.BBox.Width * b.BBox.Height
	if boxArea == 0 {
		return 0
	}
	return float64(b.Area) / float64(boxArea)
}

// EquivalentDiameter returns the diameter of the circle whose area equals the
// blob's area, i.e. sqrt(4*Area/pi).
func (b Blob) EquivalentDiameter() float64 {
	return math.Sqrt(4 * float64(b.Area) / math.Pi)
}

// Circularity returns the isoperimetric shape factor 4*pi*Area/Perimeter^2, a
// value approaching 1 for a filled disc and tending to 0 for elongated or ragged
// shapes. It returns 0 when the perimeter is 0.
func (b Blob) Circularity() float64 {
	if b.Perimeter == 0 {
		return 0
	}
	return 4 * math.Pi * float64(b.Area) / (float64(b.Perimeter) * float64(b.Perimeter))
}

// AnalyzeBlobs labels src and returns a [Blob] descriptor for every foreground
// component, ordered by label. Perimeters are measured with the inner boundary
// under conn and hole counts under the dual connectivity. It is the most
// complete single-call analysis in the package.
func AnalyzeBlobs(src *cv.Mat, conn Connectivity) []Blob {
	connectedRequireBinary(src, "AnalyzeBlobs")
	connectedCheckConn(conn, "AnalyzeBlobs")
	lbl := Label(src, conn)
	comps := lbl.Components()
	if len(comps) == 0 {
		return nil
	}

	// Per-label perimeter: count boundary pixels once over the whole image.
	w, h := lbl.Width, lbl.Height
	off := connectedOffsets(conn)
	perim := make([]int, lbl.Count+1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := lbl.Data[y*w+x]
			if v == 0 {
				continue
			}
			for _, d := range off {
				ny, nx := y+d[0], x+d[1]
				if nx < 0 || ny < 0 || nx >= w || ny >= h || lbl.Data[ny*w+nx] == 0 {
					perim[v]++
					break
				}
			}
		}
	}

	// Per-label hole count: enclosed background pixels labelled under the dual
	// connectivity, then attributed to the enclosing component.
	holes := connectedHolesPerLabel(src, lbl, conn)

	out := make([]Blob, len(comps))
	for i, c := range comps {
		out[i] = Blob{
			Label:     c.Label,
			Area:      c.Area,
			BBox:      c.BBox,
			CentroidX: c.CentroidX,
			CentroidY: c.CentroidY,
			Perimeter: perim[c.Label],
			Holes:     holes[c.Label],
		}
	}
	return out
}

// connectedHolesPerLabel returns, indexed by foreground label, the number of
// enclosed holes belonging to each component. A hole is a background component
// (under the dual connectivity) that does not touch the image border; it is
// attributed to the single foreground label that surrounds it.
func connectedHolesPerLabel(src *cv.Mat, lbl *Labels, conn Connectivity) []int {
	bgConn := connectedDual(conn)
	reachable := connectedBorderReachable(src, bgConn)
	w, h := lbl.Width, lbl.Height
	enclosed := make([]bool, w*h)
	for i := range enclosed {
		enclosed[i] = src.Data[i] == 0 && !reachable[i]
	}
	holeLabels, holeCount := connectedLabelMask(enclosed, w, h, bgConn)
	if holeCount == 0 {
		return make([]int, lbl.Count+1)
	}
	// Map each hole to the foreground label of an adjacent pixel.
	holeToFg := make([]int, holeCount+1)
	off := connectedOffsets(conn)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			hl := holeLabels[y*w+x]
			if hl == 0 || holeToFg[hl] != 0 {
				continue
			}
			for _, d := range off {
				ny, nx := y+d[0], x+d[1]
				if nx < 0 || ny < 0 || nx >= w || ny >= h {
					continue
				}
				if fg := lbl.Data[ny*w+nx]; fg != 0 {
					holeToFg[hl] = fg
					break
				}
			}
		}
	}
	result := make([]int, lbl.Count+1)
	for hl := 1; hl <= holeCount; hl++ {
		if fg := holeToFg[hl]; fg != 0 {
			result[fg]++
		}
	}
	return result
}

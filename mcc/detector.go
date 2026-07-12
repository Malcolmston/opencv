package mcc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// canonS is the side, in canonical pixels, of one grid cell in the rectified
// coordinate frame used for perspective sampling. Its exact value does not
// affect results (the homography is scale-free); a round number keeps the
// integer corner coordinates readable.
const canonS = 100

// CCheckerDetector locates a color chart of a fixed [CheckerType] in an image
// and samples its patches. Construct one with [NewCCheckerDetector]. The
// exported fields tune the classical detection pipeline and may be adjusted
// before calling [CCheckerDetector.Detect]; their defaults suit charts that
// occupy a good fraction of a reasonably-exposed frame.
type CCheckerDetector struct {
	// Type is the chart this detector looks for.
	Type CheckerType
	// MinPatchAreaFrac and MaxPatchAreaFrac bound a patch-candidate contour's
	// area as a fraction of the whole image, discarding specks and
	// full-background blobs respectively.
	MinPatchAreaFrac float64
	MaxPatchAreaFrac float64
	// ApproxEpsilonFrac is the Douglas-Peucker tolerance for reducing a contour
	// to a quadrilateral, as a fraction of the contour perimeter.
	ApproxEpsilonFrac float64
}

// NewCCheckerDetector returns a detector for the given chart with default
// parameters.
func NewCCheckerDetector(t CheckerType) *CCheckerDetector {
	return &CCheckerDetector{
		Type:              t,
		MinPatchAreaFrac:  0.0005,
		MaxPatchAreaFrac:  0.2,
		ApproxEpsilonFrac: 0.08,
	}
}

// CChecker is a detected chart: the four image points that frame it and the
// measured sRGB color of every patch in grid order. Measured colors are kept as
// float64 triples in the 0..255 range (channel-averaged over each patch's
// centre).
type CChecker struct {
	// Type is the chart that was matched.
	Type CheckerType
	// Corners are the four image points the chart was sampled from, ordered
	// top-left, top-right, bottom-right, bottom-left. For an automatic detection
	// they are the centres of the four corner patches; for a hinted detection
	// they are the outer corners supplied by the caller.
	Corners [4]cv.Point
	// Measured holds the measured sRGB color of each patch in grid order (patch
	// index = row*Cols + col).
	Measured [][3]float64
}

// MeasuredRGB returns a copy of the measured per-patch colors in grid order.
func (c *CChecker) MeasuredRGB() [][3]float64 {
	out := make([][3]float64, len(c.Measured))
	copy(out, c.Measured)
	return out
}

// Reference returns the reference patches for this checker's chart type.
func (c *CChecker) Reference() []Patch { return ReferenceChart(c.Type) }

// PatchErrors returns the CIE76 Delta E between each measured patch color and
// its reference, in grid order. Large values indicate a mis-sampled patch or a
// device with strong color error.
func (c *CChecker) PatchErrors() []float64 {
	ref := charts[c.Type]
	out := make([]float64, len(c.Measured))
	for i, m := range c.Measured {
		out[i] = DeltaE76(rgbToLabF(m[0], m[1], m[2]), ref[i].Lab)
	}
	return out
}

// MeanError returns the mean of [CChecker.PatchErrors].
func (c *CChecker) MeanError() float64 { return mean(c.PatchErrors()) }

// MaxError returns the largest of [CChecker.PatchErrors].
func (c *CChecker) MaxError() float64 {
	m := 0.0
	for _, e := range c.PatchErrors() {
		if e > m {
			m = e
		}
	}
	return m
}

// DetectWithHint samples the chart directly from the four outer corners of its
// patch array, supplied in top-left, top-right, bottom-right, bottom-left order
// (the outer corners of the corner patches). This skips the search for the chart
// and is the most robust path when the caller already knows where the chart is —
// for example from manual annotation or an earlier detection. img must be a
// three-channel RGB Mat. It returns false only if img is unusable.
func (d *CCheckerDetector) DetectWithHint(img *cv.Mat, outerQuad [4]cv.Point) (*CChecker, bool) {
	if img == nil || img.Empty() {
		return nil, false
	}
	rgb := toRGB(img)
	rows, cols := d.Type.Rows(), d.Type.Cols()
	src := [4]cv.Point{
		{X: 0, Y: 0},
		{X: cols * canonS, Y: 0},
		{X: cols * canonS, Y: rows * canonS},
		{X: 0, Y: rows * canonS},
	}
	h := cv.GetPerspectiveTransform(src, outerQuad)
	measured := sampleChart(rgb, h, rows, cols)
	return &CChecker{Type: d.Type, Corners: outerQuad, Measured: measured}, true
}

// Detect locates the chart automatically and samples its patches. The pipeline
// thresholds the image so the dark inter-patch gaps become background, traces
// external contours, keeps convex four-vertex quadrilaterals of plausible size
// as patch candidates, takes the convex hull of their centres and picks the four
// extreme corners as the chart frame, then tries all eight corner orderings —
// building a perspective transform and sampling for each — and keeps the reading
// whose colors best match the reference (lowest total CIE76 Delta E).
//
// img must be a three-channel RGB Mat and is not modified. Detect returns false
// when no plausible chart is found.
func (d *CCheckerDetector) Detect(img *cv.Mat) (*CChecker, bool) {
	if img == nil || img.Empty() {
		return nil, false
	}
	rgb := toRGB(img)
	gray := cv.CvtColor(rgb, cv.ColorRGB2Gray)
	bin := thresholdGaps(gray)

	centers := d.patchCenters(bin)
	if len(centers) < 4 {
		return nil, false
	}
	corners, ok := chartCorners(centers)
	if !ok {
		return nil, false
	}

	rows, cols := d.Type.Rows(), d.Type.Cols()
	// Canonical corner-patch centres, matching the sampled grid frame.
	src := [4]cv.Point{
		{X: canonS / 2, Y: canonS / 2},
		{X: cols*canonS - canonS/2, Y: canonS / 2},
		{X: cols*canonS - canonS/2, Y: rows*canonS - canonS/2},
		{X: canonS / 2, Y: rows*canonS - canonS/2},
	}
	ref := charts[d.Type]

	best := math.Inf(1)
	var bestMeasured [][3]float64
	var bestCorners [4]cv.Point
	for _, ord := range cornerOrderings(corners) {
		h := cv.GetPerspectiveTransform(src, ord)
		measured := sampleChart(rgb, h, rows, cols)
		score := 0.0
		for i, m := range measured {
			score += DeltaE76(rgbToLabF(m[0], m[1], m[2]), ref[i].Lab)
		}
		if score < best {
			best = score
			bestMeasured = measured
			bestCorners = ord
		}
	}
	if bestMeasured == nil {
		return nil, false
	}
	return &CChecker{Type: d.Type, Corners: bestCorners, Measured: bestMeasured}, true
}

// thresholdGaps binarises a grayscale chart so the dark gaps between patches (and
// any dark surround) become background 0 and the patches become foreground 255,
// which is what cv.FindContours expects. The level is a small fraction of the
// brightest sample, chosen so that even the chart's darkest neutral patch stays
// above the near-black gaps.
func thresholdGaps(gray *cv.Mat) *cv.Mat {
	maxV := uint8(0)
	for _, v := range gray.Data {
		if v > maxV {
			maxV = v
		}
	}
	level := float64(maxV) * 0.1
	if level < 10 {
		level = 10
	}
	if level > 60 {
		level = 60
	}
	bin, _ := cv.Threshold(gray, level, 255, cv.ThreshBinary)
	return bin
}

// patchCenters returns the centres of the contour quadrilaterals that pass the
// detector's size and squareness filters, then keeps only those whose area is
// near the median (dropping stray blobs of a different scale).
func (d *CCheckerDetector) patchCenters(bin *cv.Mat) []cv.Point {
	contours, _ := cv.FindContours(bin, cv.RetrExternal, cv.ChainApproxNone)
	imgArea := float64(bin.Rows * bin.Cols)
	var centers []cv.Point
	var areas []float64
	for _, c := range contours {
		pts := []cv.Point(c)
		if len(pts) < 4 {
			continue
		}
		peri := cv.ArcLength(pts, true)
		approx := cv.ApproxPolyDP(pts, d.ApproxEpsilonFrac*peri, true)
		if len(approx) != 4 || !isConvex(approx) {
			continue
		}
		a := polygonArea(approx)
		if a < imgArea*d.MinPatchAreaFrac || a > imgArea*d.MaxPatchAreaFrac {
			continue
		}
		br := cv.BoundingRect(approx)
		if br.Height == 0 {
			continue
		}
		ar := float64(br.Width) / float64(br.Height)
		if ar < 0.5 || ar > 2.0 {
			continue
		}
		centers = append(centers, polygonCentroid(approx))
		areas = append(areas, a)
	}
	if len(centers) < 4 {
		return centers
	}
	med := median(areas)
	var kept []cv.Point
	for i, a := range areas {
		if a >= 0.4*med && a <= 2.5*med {
			kept = append(kept, centers[i])
		}
	}
	return kept
}

// chartCorners finds the four extreme corner points among the patch centres: it
// takes their convex hull and, from the hull, the four vertices spanning the
// greatest quadrilateral area. Those are the four corner patches of the grid.
func chartCorners(centers []cv.Point) ([4]cv.Point, bool) {
	hull := cv.ConvexHull(centers)
	if len(hull) < 4 {
		return [4]cv.Point{}, false
	}
	return maxAreaQuad(hull)
}

// maxAreaQuad returns the four hull vertices (in their hull order) that enclose
// the largest area. hull must have at least four points.
func maxAreaQuad(hull []cv.Point) ([4]cv.Point, bool) {
	n := len(hull)
	best := -1.0
	var out [4]cv.Point
	found := false
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			for k := j + 1; k < n; k++ {
				for l := k + 1; l < n; l++ {
					q := [4]cv.Point{hull[i], hull[j], hull[k], hull[l]}
					a := polygonArea(q[:])
					if a > best {
						best = a
						out = q
						found = true
					}
				}
			}
		}
	}
	return out, found
}

// cornerOrderings returns the eight ways to assign four convex-hull corners to
// the canonical (top-left, top-right, bottom-right, bottom-left) slots: the four
// rotations of the cycle in both winding directions. The detector scores each
// and keeps the best, which removes any ambiguity about the chart's rotation or
// mirroring.
func cornerOrderings(q [4]cv.Point) [][4]cv.Point {
	rev := [4]cv.Point{q[0], q[3], q[2], q[1]}
	var out [][4]cv.Point
	for _, base := range [][4]cv.Point{q, rev} {
		for s := 0; s < 4; s++ {
			out = append(out, [4]cv.Point{
				base[s], base[(s+1)%4], base[(s+2)%4], base[(s+3)%4],
			})
		}
	}
	return out
}

// sampleChart reads the mean color of every patch. The homography h maps the
// canonical grid frame (0..cols*canonS by 0..rows*canonS) into the image; for
// each cell the central 40% is sampled on a 5x5 lattice of canonical points,
// each mapped into the image and read with nearest-neighbour lookup, and the
// results are averaged. Sampling the source directly (rather than warping the
// whole image) keeps the measurement perspective-correct and cheap.
func sampleChart(img *cv.Mat, h cv.PerspectiveMatrix, rows, cols int) [][3]float64 {
	out := make([][3]float64, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			var sr, sg, sb, n float64
			for iy := 0; iy < 5; iy++ {
				for ix := 0; ix < 5; ix++ {
					fx := (float64(c) + 0.3 + 0.1*float64(ix)) * canonS
					fy := (float64(r) + 0.3 + 0.1*float64(iy)) * canonS
					px, py := applyPerspective(h, fx, fy)
					rgb, ok := sampleNearest(img, px, py)
					if ok {
						sr += rgb[0]
						sg += rgb[1]
						sb += rgb[2]
						n++
					}
				}
			}
			if n > 0 {
				out[r*cols+c] = [3]float64{sr / n, sg / n, sb / n}
			}
		}
	}
	return out
}

// applyPerspective maps a point through a 3x3 projective transform stored in the
// row-major layout documented on cv.PerspectiveMatrix.
func applyPerspective(m cv.PerspectiveMatrix, x, y float64) (float64, float64) {
	w := m[6]*x + m[7]*y + m[8]
	if w == 0 {
		return 0, 0
	}
	return (m[0]*x + m[1]*y + m[2]) / w, (m[3]*x + m[4]*y + m[5]) / w
}

// sampleNearest reads the nearest-pixel RGB of img at floating-point coordinates,
// reporting ok=false when the point lies outside the image.
func sampleNearest(img *cv.Mat, x, y float64) ([3]float64, bool) {
	xi := int(math.Round(x))
	yi := int(math.Round(y))
	if xi < 0 || xi >= img.Cols || yi < 0 || yi >= img.Rows {
		return [3]float64{}, false
	}
	p := img.AtPixel(yi, xi)
	return [3]float64{float64(p[0]), float64(p[1]), float64(p[2])}, true
}

// toRGB returns a three-channel RGB view of img, converting a single-channel
// image by replicating its gray value across the channels and copying an image
// that is already RGB.
func toRGB(img *cv.Mat) *cv.Mat {
	if img.Channels == 3 {
		return img
	}
	if img.Channels == 1 {
		out := cv.NewMat(img.Rows, img.Cols, 3)
		for i := 0; i < img.Total(); i++ {
			v := img.Data[i]
			out.Data[i*3+0] = v
			out.Data[i*3+1] = v
			out.Data[i*3+2] = v
		}
		return out
	}
	// Fall back to the root package's own conversion for other channel counts.
	return cv.FromImage(img.ToImage())
}

// isConvex reports whether a polygon's vertices all turn the same way.
func isConvex(p []cv.Point) bool {
	n := len(p)
	if n < 3 {
		return false
	}
	sign := 0
	for i := 0; i < n; i++ {
		a := p[i]
		b := p[(i+1)%n]
		c := p[(i+2)%n]
		cross := (b.X-a.X)*(c.Y-b.Y) - (b.Y-a.Y)*(c.X-b.X)
		if cross == 0 {
			continue
		}
		s := 1
		if cross < 0 {
			s = -1
		}
		if sign == 0 {
			sign = s
		} else if s != sign {
			return false
		}
	}
	return sign != 0
}

// polygonArea returns the absolute shoelace area of a polygon.
func polygonArea(p []cv.Point) float64 {
	n := len(p)
	var a float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += float64(p[i].X)*float64(p[j].Y) - float64(p[j].X)*float64(p[i].Y)
	}
	return math.Abs(a) / 2
}

// polygonCentroid returns the mean of a polygon's vertices, rounded to the
// nearest pixel.
func polygonCentroid(p []cv.Point) cv.Point {
	var sx, sy int
	for _, q := range p {
		sx += q.X
		sy += q.Y
	}
	return cv.Point{X: int(math.Round(float64(sx) / float64(len(p)))), Y: int(math.Round(float64(sy) / float64(len(p))))}
}

// mean returns the arithmetic mean of a slice, or 0 when it is empty.
func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// median returns the median of a slice of areas without mutating the input.
func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	c := make([]float64, len(v))
	copy(c, v)
	// Simple insertion sort; candidate counts are small.
	for i := 1; i < len(c); i++ {
		for j := i; j > 0 && c[j-1] > c[j]; j-- {
			c[j-1], c[j] = c[j], c[j-1]
		}
	}
	n := len(c)
	if n%2 == 1 {
		return c[n/2]
	}
	return (c[n/2-1] + c[n/2]) / 2
}

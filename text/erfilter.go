package text

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ERFeatures is the geometric and topological descriptor OpenCV's Neumann–Matas
// Extremal Region classifier is trained on, computed here for a single [Region].
// The trained AdaBoost cascade of the original is replaced by the documented
// threshold heuristics in [ERFilterNM1] and [ERFilterNM2], but the features
// themselves are the same quantities the paper describes.
type ERFeatures struct {
	// Area is the region pixel count (len(Region.Points)).
	Area int
	// Width and Height are the bounding-box dimensions in pixels.
	Width  int
	Height int
	// Perimeter is the number of region pixels having at least one 4-connected
	// neighbour outside the region (the region's outer + inner boundary length).
	Perimeter int
	// AspectRatio is Width/Height. Characters are rarely far from 1 in either
	// direction.
	AspectRatio float64
	// Compactness is sqrt(Area)/Perimeter, the first-stage NM1 feature. A filled
	// convex blob peaks near 0.25; thin strokes and noise fall well below.
	Compactness float64
	// FillRatio is Area/(Width*Height), the fraction of the bounding box inked.
	FillRatio float64
	// Holes is the number of background connected components fully enclosed by the
	// region — the topological hole count (Euler number 1-Holes). Letters such as
	// O and B have one or two; most noise has none.
	Holes int
	// HoleAreaRatio is the summed hole area divided by Area, an NM2 feature.
	HoleAreaRatio float64
	// Convexity is Area divided by the region convex-hull area (from
	// [cv.ConvexHull]); solid characters approach or exceed 1, ragged noise is
	// lower. Values above 1 are possible because the hull area is computed from
	// pixel-centre polygon vertices.
	Convexity float64
	// StrokeWidthMean and StrokeWidthStd summarise the per-pixel stroke thickness,
	// approximated as twice the 4-connected distance transform to the region
	// boundary. Text has a nearly constant stroke width, so a small
	// StrokeWidthStd/StrokeWidthMean ratio is a strong character cue (NM2).
	StrokeWidthMean float64
	StrokeWidthStd  float64
}

// StrokeWidthVariation returns StrokeWidthStd/StrokeWidthMean, the coefficient of
// variation of stroke thickness. It is 0 for a region with no interior pixels.
func (f ERFeatures) StrokeWidthVariation() float64 {
	if f.StrokeWidthMean <= 0 {
		return 0
	}
	return f.StrokeWidthStd / f.StrokeWidthMean
}

// ComputeERFeatures measures the full [ERFeatures] descriptor of a region. It
// works purely from the region's pixel set and bounding box, so it composes with
// any region source (MSER output, connected components, a hand-built mask).
func ComputeERFeatures(r Region) ERFeatures {
	f := ERFeatures{
		Area:   r.Area,
		Width:  r.Rect.Width,
		Height: r.Rect.Height,
	}
	if r.Rect.Width <= 0 || r.Rect.Height <= 0 || len(r.Points) == 0 {
		return f
	}
	if f.Area == 0 {
		f.Area = len(r.Points)
	}

	w, h := r.Rect.Width, r.Rect.Height
	ox, oy := r.Rect.X, r.Rect.Y
	// mask[y*w+x] marks a region pixel inside the local bounding-box grid.
	mask := make([]bool, w*h)
	for _, p := range r.Points {
		lx, ly := p.X-ox, p.Y-oy
		if lx >= 0 && lx < w && ly >= 0 && ly < h {
			mask[ly*w+lx] = true
		}
	}

	f.AspectRatio = float64(w) / float64(h)
	f.FillRatio = float64(f.Area) / float64(w*h)
	f.Perimeter = maskPerimeter(mask, w, h)
	if f.Perimeter > 0 {
		f.Compactness = math.Sqrt(float64(f.Area)) / float64(f.Perimeter)
	}
	f.Holes, f.HoleAreaRatio = maskHoles(mask, w, h, f.Area)

	// Convex hull area from the region's pixel coordinates.
	hull := cv.ConvexHull(r.Points)
	hullArea := cv.ContourArea(cv.Contour(hull))
	if hullArea > 0 {
		f.Convexity = float64(f.Area) / hullArea
	}

	f.StrokeWidthMean, f.StrokeWidthStd = maskStrokeStats(mask, w, h)
	return f
}

// maskPerimeter counts region pixels with a 4-connected neighbour outside the
// region (out-of-grid counts as outside).
func maskPerimeter(mask []bool, w, h int) int {
	per := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if !mask[y*w+x] {
				continue
			}
			if x == 0 || !mask[y*w+x-1] ||
				x == w-1 || !mask[y*w+x+1] ||
				y == 0 || !mask[(y-1)*w+x] ||
				y == h-1 || !mask[(y+1)*w+x] {
				per++
			}
		}
	}
	return per
}

// maskHoles counts background components fully enclosed by the region and their
// total area. The bounding-box border is flooded first; any unreached background
// pixel belongs to an enclosed hole.
func maskHoles(mask []bool, w, h, area int) (holes int, holeAreaRatio float64) {
	// outside[i] true once reached from the grid border through background pixels.
	outside := make([]bool, w*h)
	var stack []int
	push := func(x, y int) {
		if x < 0 || x >= w || y < 0 || y >= h {
			return
		}
		i := y*w + x
		if mask[i] || outside[i] {
			return
		}
		outside[i] = true
		stack = append(stack, i)
	}
	for x := 0; x < w; x++ {
		push(x, 0)
		push(x, h-1)
	}
	for y := 0; y < h; y++ {
		push(0, y)
		push(w-1, y)
	}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x, y := i%w, i/w
		push(x-1, y)
		push(x+1, y)
		push(x, y-1)
		push(x, y+1)
	}

	// Label the remaining (enclosed) background pixels into components.
	visited := make([]bool, w*h)
	holeArea := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if mask[i] || outside[i] || visited[i] {
				continue
			}
			holes++
			comp := []int{i}
			visited[i] = true
			for len(comp) > 0 {
				j := comp[len(comp)-1]
				comp = comp[:len(comp)-1]
				holeArea++
				jx, jy := j%w, j/w
				for _, d := range [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} {
					nx, ny := jx+d[0], jy+d[1]
					if nx < 0 || nx >= w || ny < 0 || ny >= h {
						continue
					}
					k := ny*w + nx
					if mask[k] || outside[k] || visited[k] {
						continue
					}
					visited[k] = true
					comp = append(comp, k)
				}
			}
		}
	}
	if area > 0 {
		holeAreaRatio = float64(holeArea) / float64(area)
	}
	return holes, holeAreaRatio
}

// maskStrokeStats approximates per-pixel stroke thickness as twice the
// 4-connected distance transform of each region pixel to the nearest boundary,
// and returns the mean and population standard deviation over all region pixels.
func maskStrokeStats(mask []bool, w, h int) (mean, std float64) {
	const inf = 1 << 30
	dt := make([]int, w*h)
	for i := range dt {
		if mask[i] {
			dt[i] = inf
		}
	}
	// Forward pass (top-left to bottom-right).
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if !mask[i] {
				continue
			}
			best := inf
			if x == 0 || y == 0 || x == w-1 || y == h-1 {
				best = 1 // border pixel: 1 step from outside the grid
			}
			if x > 0 {
				best = minInt(best, dt[i-1]+1)
			}
			if y > 0 {
				best = minInt(best, dt[i-w]+1)
			}
			dt[i] = best
		}
	}
	// Backward pass (bottom-right to top-left).
	for y := h - 1; y >= 0; y-- {
		for x := w - 1; x >= 0; x-- {
			i := y*w + x
			if !mask[i] {
				continue
			}
			best := dt[i]
			if x < w-1 {
				best = minInt(best, dt[i+1]+1)
			}
			if y < h-1 {
				best = minInt(best, dt[i+w]+1)
			}
			dt[i] = best
		}
	}

	var sum, sumSq float64
	count := 0
	for i := range dt {
		if !mask[i] {
			continue
		}
		v := 2 * float64(dt[i]) // full thickness ≈ twice the distance to the edge
		sum += v
		sumSq += v * v
		count++
	}
	if count == 0 {
		return 0, 0
	}
	mean = sum / float64(count)
	variance := sumSq/float64(count) - mean*mean
	if variance < 0 {
		variance = 0
	}
	return mean, math.Sqrt(variance)
}

// ERFilterNM1 is the first, high-recall stage of the two-stage Neumann–Matas
// Extremal Region classifier. OpenCV trains a real-valued AdaBoost classifier on
// the incrementally computable features (aspect ratio, compactness and hole
// count); here that learned decision is replaced by documented thresholds on the
// same [ERFeatures]. The stage is deliberately permissive: it should reject
// obvious non-characters (thin lines, huge blobs) while keeping essentially every
// true character for the more selective [ERFilterNM2].
type ERFilterNM1 struct {
	// MinArea and MaxArea bound the region area in pixels (MaxArea <= 0 disables
	// the upper bound).
	MinArea int
	MaxArea int
	// MinAspect and MaxAspect bound the bounding-box aspect ratio (Width/Height).
	MinAspect float64
	MaxAspect float64
	// MinCompactness and MaxCompactness bound sqrt(Area)/Perimeter.
	MinCompactness float64
	MaxCompactness float64
	// MaxHoles bounds the enclosed-hole count; characters have at most two.
	MaxHoles int
	// MinFillRatio drops near-empty bounding boxes.
	MinFillRatio float64
}

// DefaultERFilterNM1 returns permissive first-stage thresholds tuned to keep
// printed and hand-drawn characters while discarding degenerate shapes.
func DefaultERFilterNM1() ERFilterNM1 {
	return ERFilterNM1{
		MinArea:        8,
		MaxArea:        0,
		MinAspect:      0.1,
		MaxAspect:      2.5,
		MinCompactness: 0.03,
		MaxCompactness: 0.6,
		MaxHoles:       3,
		MinFillRatio:   0.15,
	}
}

// NewERFilterNM1 returns first-stage thresholds with the given area bounds and
// the default shape limits.
func NewERFilterNM1(minArea, maxArea int) ERFilterNM1 {
	f := DefaultERFilterNM1()
	f.MinArea = minArea
	f.MaxArea = maxArea
	return f
}

// KeepFeatures reports whether a precomputed feature vector passes stage one.
func (f ERFilterNM1) KeepFeatures(x ERFeatures) bool {
	if x.Area < f.MinArea {
		return false
	}
	if f.MaxArea > 0 && x.Area > f.MaxArea {
		return false
	}
	if x.Width <= 0 || x.Height <= 0 {
		return false
	}
	if x.AspectRatio < f.MinAspect || x.AspectRatio > f.MaxAspect {
		return false
	}
	if x.Compactness < f.MinCompactness || x.Compactness > f.MaxCompactness {
		return false
	}
	if x.Holes > f.MaxHoles {
		return false
	}
	return x.FillRatio >= f.MinFillRatio
}

// Keep reports whether a region passes stage one, computing its features first.
func (f ERFilterNM1) Keep(r Region) bool {
	return f.KeepFeatures(ComputeERFeatures(r))
}

// Filter returns the subset of regions that pass stage one, preserving order.
func (f ERFilterNM1) Filter(regions []Region) []Region {
	var out []Region
	for _, r := range regions {
		if f.Keep(r) {
			out = append(out, r)
		}
	}
	return out
}

// ERFilterNM2 is the second, high-precision stage of the Neumann–Matas Extremal
// Region classifier. It reapplies the stage-one gates and adds the more
// expensive holistic features — hole-area ratio, convexity and stroke-width
// constancy — that separate characters from character-like clutter. Feed it the
// output of [ERFilterNM1].
type ERFilterNM2 struct {
	// Stage1 holds the first-stage gates, re-evaluated here so NM2 can be used
	// stand-alone.
	Stage1 ERFilterNM1
	// MaxHoleAreaRatio bounds the summed hole area relative to region area.
	MaxHoleAreaRatio float64
	// MinConvexity bounds Area/hull-area; ragged, concave clutter falls below.
	MinConvexity float64
	// MaxStrokeVariation bounds StrokeWidthStd/StrokeWidthMean; text strokes are
	// near-constant in width, most clutter is not.
	MaxStrokeVariation float64
}

// DefaultERFilterNM2 returns second-stage thresholds tuned to admit printed and
// hand-drawn characters while rejecting concave or variable-stroke clutter.
func DefaultERFilterNM2() ERFilterNM2 {
	return ERFilterNM2{
		Stage1:             DefaultERFilterNM1(),
		MaxHoleAreaRatio:   1.6,
		MinConvexity:       0.38,
		MaxStrokeVariation: 0.75,
	}
}

// NewERFilterNM2 returns second-stage thresholds built on the given first stage
// and the default holistic limits.
func NewERFilterNM2(stage1 ERFilterNM1) ERFilterNM2 {
	f := DefaultERFilterNM2()
	f.Stage1 = stage1
	return f
}

// KeepFeatures reports whether a precomputed feature vector passes both stages.
func (f ERFilterNM2) KeepFeatures(x ERFeatures) bool {
	if !f.Stage1.KeepFeatures(x) {
		return false
	}
	if x.HoleAreaRatio > f.MaxHoleAreaRatio {
		return false
	}
	if x.Convexity < f.MinConvexity {
		return false
	}
	return x.StrokeWidthVariation() <= f.MaxStrokeVariation
}

// Keep reports whether a region passes both stages, computing its features first.
func (f ERFilterNM2) Keep(r Region) bool {
	return f.KeepFeatures(ComputeERFeatures(r))
}

// Filter returns the subset of regions that pass both stages, preserving order.
func (f ERFilterNM2) Filter(regions []Region) []Region {
	var out []Region
	for _, r := range regions {
		if f.Keep(r) {
			out = append(out, r)
		}
	}
	return out
}

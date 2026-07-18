package textdet

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// MSERPolarity selects whether Maximally Stable Extremal Regions are grown as
// dark regions on a light background (MSER+) or bright regions on a dark
// background (MSER-).
type MSERPolarity int

const (
	// MSERDark detects dark regions: the extremal sets are pixels with grey
	// level at or below a rising threshold. This is the common case for dark
	// text.
	MSERDark MSERPolarity = iota
	// MSERBright detects bright regions by running the same analysis on the
	// inverted image.
	MSERBright
)

// MSEROptions configures [DetectMSER].
type MSEROptions struct {
	// Delta is the grey-level step over which region-area stability is
	// measured. Larger values demand stability across a wider intensity band.
	Delta int
	// MinArea is the smallest region area, in pixels, that is reported.
	MinArea int
	// MaxArea is the largest region area, in pixels, that is reported. A value
	// <= 0 defaults to half the image area, which suppresses the trivial
	// whole-image region.
	MaxArea int
	// MaxVariation is the largest area-growth ratio a region may have at its
	// most stable level and still be reported; smaller values keep only very
	// stable regions.
	MaxVariation float64
	// MinDiversity rejects a region that is nested inside an already-accepted
	// region and differs from it in area by less than this fraction.
	MinDiversity float64
	// Connectivity selects 4- or 8-connectivity for the extremal regions.
	Connectivity Connectivity
	// Polarity chooses dark (MSER+) or bright (MSER-) regions.
	Polarity MSERPolarity
}

// DefaultMSEROptions returns options suitable for detecting dark glyph-sized
// regions: Delta 5, MinArea 10, MaxArea 0 (half the image), MaxVariation 0.25,
// MinDiversity 0.2, 8-connectivity, dark polarity.
func DefaultMSEROptions() MSEROptions {
	return MSEROptions{
		Delta:        5,
		MinArea:      10,
		MaxArea:      0,
		MaxVariation: 0.25,
		MinDiversity: 0.2,
		Connectivity: Conn8,
		Polarity:     MSERDark,
	}
}

// MSERRegion is one maximally stable extremal region.
type MSERRegion struct {
	// Bounds is the region's upright bounding box.
	Bounds cv.Rect
	// Area is the region's pixel count at its stable level.
	Area int
	// Level is the grey level (on the possibly inverted image) at which the
	// region is most stable, in [0,255].
	Level int
	// Variation is the area-growth ratio at the stable level; smaller is more
	// stable.
	Variation float64
	// Pixels lists every pixel of the region at its stable level.
	Pixels []cv.Point
}

// AspectRatio returns the region's bounding-box width divided by its height, or
// 0 for a degenerate zero-height box.
func (r MSERRegion) AspectRatio() float64 {
	if r.Bounds.Height == 0 {
		return 0
	}
	return float64(r.Bounds.Width) / float64(r.Bounds.Height)
}

// textdetSeedStats computes, for the threshold {gray <= level}, a mapping from
// each component's seed index (the minimum-intensity, minimum-linear-index
// pixel) to that component's area.
func textdetSeedStats(gray []uint8, rows, cols, level int, conn Connectivity) map[int]int {
	fg := make([]bool, rows*cols)
	for i, v := range gray {
		if int(v) <= level {
			fg[i] = true
		}
	}
	labels, count := textdetLabelMask(fg, rows, cols, conn)
	area := make([]int, count+1)
	minVal := make([]int, count+1)
	seedIdx := make([]int, count+1)
	for l := 1; l <= count; l++ {
		minVal[l] = 256
		seedIdx[l] = -1
	}
	for i, l := range labels {
		if l == 0 {
			continue
		}
		area[l]++
		v := int(gray[i])
		if v < minVal[l] {
			minVal[l] = v
			seedIdx[l] = i
		}
	}
	out := make(map[int]int, count)
	for l := 1; l <= count; l++ {
		if seedIdx[l] >= 0 {
			out[seedIdx[l]] = area[l]
		}
	}
	return out
}

// textdetSeedRegion recovers the pixels and bounding box of the component that
// contains seedIdx at threshold {gray <= level}.
func textdetSeedRegion(gray []uint8, rows, cols, level, seedIdx int, conn Connectivity) (pixels []cv.Point, bounds cv.Rect) {
	fg := make([]bool, rows*cols)
	for i, v := range gray {
		if int(v) <= level {
			fg[i] = true
		}
	}
	labels, _ := textdetLabelMask(fg, rows, cols, conn)
	target := labels[seedIdx]
	if target == 0 {
		return nil, cv.Rect{}
	}
	minX, minY, maxX, maxY := cols, rows, -1, -1
	for i, l := range labels {
		if l != target {
			continue
		}
		x := i % cols
		y := i / cols
		pixels = append(pixels, cv.Point{X: x, Y: y})
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
	return pixels, cv.Rect{X: minX, Y: minY, Width: maxX - minX + 1, Height: maxY - minY + 1}
}

// DetectMSER detects Maximally Stable Extremal Regions in src. It sweeps the
// grey-level threshold from 0 to 255, tracks the area of every extremal region
// (identified by its darkest pixel) as a function of level, and reports each
// region at the level where its area grows least — the definition of maximal
// stability. Colour input is reduced to luma; for [MSERBright] polarity the
// image is inverted first. Regions are returned ordered top-to-bottom then
// left-to-right. It returns [ErrEmpty] for an empty image and
// [ErrInvalidArgument] for a non-positive Delta or invalid connectivity.
func DetectMSER(src *cv.Mat, opts MSEROptions) ([]MSERRegion, error) {
	if opts.Delta <= 0 {
		return nil, ErrInvalidArgument
	}
	if opts.Connectivity != Conn4 && opts.Connectivity != Conn8 {
		return nil, ErrInvalidArgument
	}
	gray, rows, cols, err := textdetGray(src)
	if err != nil {
		return nil, err
	}
	work := gray
	if opts.Polarity == MSERBright {
		work = make([]uint8, len(gray))
		for i, v := range gray {
			work[i] = 255 - v
		}
	}
	total := rows * cols
	maxArea := opts.MaxArea
	if maxArea <= 0 {
		maxArea = total / 2
	}

	// areaHist[seed][level] = area of the region seeded at seed when the
	// threshold is level, or 0 if the seed does not head a component there.
	areaHist := make(map[int][]int)
	for level := 0; level < 256; level++ {
		stats := textdetSeedStats(work, rows, cols, level, opts.Connectivity)
		for seed, a := range stats {
			h := areaHist[seed]
			if h == nil {
				h = make([]int, 256)
				areaHist[seed] = h
			}
			h[level] = a
		}
	}

	d := opts.Delta
	var regions []MSERRegion
	for seed, h := range areaHist {
		bestLevel := -1
		bestVar := opts.MaxVariation
		for level := d; level < 256-d; level++ {
			a := h[level]
			if a == 0 {
				continue
			}
			aLo := h[level-d]
			aHi := h[level+d]
			if aLo == 0 || aHi == 0 {
				continue
			}
			variation := float64(aHi-aLo) / float64(a)
			if variation < 0 {
				variation = 0
			}
			if variation <= bestVar {
				bestVar = variation
				bestLevel = level
			}
		}
		if bestLevel < 0 {
			continue
		}
		area := h[bestLevel]
		if area < opts.MinArea || area > maxArea {
			continue
		}
		pixels, bounds := textdetSeedRegion(work, rows, cols, bestLevel, seed, opts.Connectivity)
		if len(pixels) == 0 {
			continue
		}
		regions = append(regions, MSERRegion{
			Bounds:    bounds,
			Area:      len(pixels),
			Level:     bestLevel,
			Variation: bestVar,
			Pixels:    pixels,
		})
	}

	regions = textdetDiversityFilter(regions, opts.MinDiversity)
	sort.SliceStable(regions, func(i, j int) bool {
		if regions[i].Bounds.Y != regions[j].Bounds.Y {
			return regions[i].Bounds.Y < regions[j].Bounds.Y
		}
		return regions[i].Bounds.X < regions[j].Bounds.X
	})
	return regions, nil
}

// textdetDiversityFilter removes a region that is spatially nested inside a
// larger accepted region and differs from it in area by less than minDiversity
// (relative to the larger area).
func textdetDiversityFilter(regions []MSERRegion, minDiversity float64) []MSERRegion {
	if minDiversity <= 0 {
		return regions
	}
	sort.SliceStable(regions, func(i, j int) bool { return regions[i].Area > regions[j].Area })
	var kept []MSERRegion
	for _, r := range regions {
		redundant := false
		for _, k := range kept {
			if textdetContains(k.Bounds, r.Bounds) {
				diff := float64(k.Area-r.Area) / float64(k.Area)
				if diff < minDiversity {
					redundant = true
					break
				}
			}
		}
		if !redundant {
			kept = append(kept, r)
		}
	}
	return kept
}

// textdetContains reports whether outer fully contains inner.
func textdetContains(outer, inner cv.Rect) bool {
	return inner.X >= outer.X && inner.Y >= outer.Y &&
		inner.X+inner.Width <= outer.X+outer.Width &&
		inner.Y+inner.Height <= outer.Y+outer.Height
}

// FilterTextRegions keeps the MSER regions whose shape is plausible for a text
// glyph: bounding-box aspect ratio in [minAspect, maxAspect], height in
// [minHeight, maxHeight] pixels (a maxHeight <= 0 disables the upper bound), and
// fill ratio (area over bounding-box area) at least minFill. The regions are
// returned in their input order.
func FilterTextRegions(regions []MSERRegion, minAspect, maxAspect float64, minHeight, maxHeight int, minFill float64) []MSERRegion {
	out := make([]MSERRegion, 0, len(regions))
	for _, r := range regions {
		if r.Bounds.Height < minHeight {
			continue
		}
		if maxHeight > 0 && r.Bounds.Height > maxHeight {
			continue
		}
		ar := r.AspectRatio()
		if ar < minAspect || ar > maxAspect {
			continue
		}
		boxArea := r.Bounds.Width * r.Bounds.Height
		if boxArea == 0 {
			continue
		}
		if float64(r.Area)/float64(boxArea) < minFill {
			continue
		}
		out = append(out, r)
	}
	return out
}

// RegionsToMask renders a set of MSER regions into a single fresh
// single-channel 0/255 [cv.Mat] of the given size, painting every region pixel
// 255. It returns [ErrInvalidArgument] for non-positive dimensions.
func RegionsToMask(regions []MSERRegion, rows, cols int) (*cv.Mat, error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrInvalidArgument
	}
	dst := cv.NewMat(rows, cols, 1)
	for _, r := range regions {
		for _, p := range r.Pixels {
			if p.X >= 0 && p.X < cols && p.Y >= 0 && p.Y < rows {
				dst.Data[p.Y*cols+p.X] = 255
			}
		}
	}
	return dst, nil
}

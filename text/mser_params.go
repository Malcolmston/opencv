package text

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// MSERParams collects the tunable parameters of the MSER region detector into one
// struct, mirroring the arguments of OpenCV's cv::MSER, including the min-diversity
// control that the positional [MSERRegions] entry point omits.
type MSERParams struct {
	// Delta is the intensity step over which stability is measured; larger values
	// demand stability across a wider threshold band. Values below 1 are treated
	// as 1.
	Delta int
	// MinArea and MaxArea bound the region pixel area. MaxArea <= 0 means the whole
	// image.
	MinArea int
	MaxArea int
	// MaxVariation rejects regions whose relative area growth over the delta band
	// exceeds it — the primary stability gate. Smaller keeps only very stable
	// regions.
	MaxVariation float64
	// MinDiversity prunes nested near-duplicates: when a region's bounding box is
	// contained in a more stable kept region and its area is at least
	// (1-MinDiversity) of the container's, the less stable region is dropped. 0
	// disables diversity pruning; OpenCV's default is 0.2.
	MinDiversity float64
	// Polarity selects which polarities to search.
	Polarity MSERPolarity
}

// MSERPolarity selects the intensity polarity searched by MSER.
type MSERPolarity int

const (
	// MSERBoth searches both dark-on-light (MSER+) and bright-on-dark (MSER-).
	MSERBoth MSERPolarity = iota
	// MSERDark searches only dark regions on a lighter background (MSER+).
	MSERDark
	// MSERBright searches only bright regions on a darker background (MSER-).
	MSERBright
)

// DefaultMSERParams returns parameters matching OpenCV's cv::MSER defaults scaled
// for small test images: delta 5, area in [10, 14400], max variation 0.25 and
// min diversity 0.2, searching both polarities.
func DefaultMSERParams() MSERParams {
	return MSERParams{
		Delta:        5,
		MinArea:      10,
		MaxArea:      0,
		MaxVariation: 0.25,
		MinDiversity: 0.2,
		Polarity:     MSERBoth,
	}
}

// MSERRegionsWithParams extracts MSERs under the given [MSERParams]. It differs
// from [MSERRegions] in exposing polarity selection and the min-diversity pruning
// that removes nested near-duplicate regions. Results are sorted top-to-bottom
// then left-to-right and are deterministic.
func MSERRegionsWithParams(img *cv.Mat, p MSERParams) []Region {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	n := rows * cols

	delta := p.Delta
	if delta < 1 {
		delta = 1
	}
	minArea := p.MinArea
	if minArea < 1 {
		minArea = 1
	}
	maxArea := p.MaxArea
	if maxArea <= 0 || maxArea > n {
		maxArea = n
	}

	var out []Region
	if p.Polarity == MSERBoth || p.Polarity == MSERDark {
		out = append(out, extractMSER(gray.Data, rows, cols, delta, minArea, maxArea, p.MaxVariation, false)...)
	}
	if p.Polarity == MSERBoth || p.Polarity == MSERBright {
		inv := make([]uint8, n)
		for i, v := range gray.Data {
			inv[i] = 255 - v
		}
		out = append(out, extractMSER(inv, rows, cols, delta, minArea, maxArea, p.MaxVariation, true)...)
	}

	deduped := dedupeRegions(out)
	if p.MinDiversity > 0 {
		deduped = pruneByDiversity(deduped, p.MinDiversity)
	}
	return deduped
}

// DetectRegionsMSERParams is the boxes-only counterpart of
// [MSERRegionsWithParams], returning just the bounding boxes.
func DetectRegionsMSERParams(img *cv.Mat, p MSERParams) []cv.Rect {
	regions := MSERRegionsWithParams(img, p)
	boxes := make([]cv.Rect, len(regions))
	for i, r := range regions {
		boxes[i] = r.Rect
	}
	return boxes
}

// pruneByDiversity removes a region when a more stable region contains its
// bounding box and their areas are within (1-minDiversity) of each other,
// suppressing the redundant levels of a nested MSER stack. The more stable
// region (lower variation) is the one retained.
func pruneByDiversity(in []Region, minDiversity float64) []Region {
	order := make([]int, len(in))
	for i := range order {
		order[i] = i
	}
	// Consider the most stable (then largest) regions first as containers.
	sort.SliceStable(order, func(a, b int) bool {
		ra, rb := in[order[a]], in[order[b]]
		if ra.Variation != rb.Variation {
			return ra.Variation < rb.Variation
		}
		return ra.Area > rb.Area
	})

	var kept []Region
	for _, idx := range order {
		r := in[idx]
		redundant := false
		for _, k := range kept {
			if !rectContains(k.Rect, r.Rect) && !rectContains(r.Rect, k.Rect) {
				continue
			}
			hi, lo := r.Area, k.Area
			if hi < lo {
				hi, lo = lo, hi
			}
			if hi > 0 && float64(lo)/float64(hi) >= 1-minDiversity {
				redundant = true
				break
			}
		}
		if !redundant {
			kept = append(kept, r)
		}
	}

	sort.SliceStable(kept, func(a, b int) bool {
		ra, rb := kept[a].Rect, kept[b].Rect
		if ra.Y != rb.Y {
			return ra.Y < rb.Y
		}
		if ra.X != rb.X {
			return ra.X < rb.X
		}
		return rectArea(ra) < rectArea(rb)
	})
	return kept
}

// rectContains reports whether outer fully contains inner.
func rectContains(outer, inner cv.Rect) bool {
	return inner.X >= outer.X && inner.Y >= outer.Y &&
		inner.X+inner.Width <= outer.X+outer.Width &&
		inner.Y+inner.Height <= outer.Y+outer.Height
}

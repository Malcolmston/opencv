package colorspaces2

import (
	"image/color"
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Palette is an ordered set of representative colours, typically extracted from
// an image by [MedianCut], [KMeansQuantize] or [ExtractPalette].
type Palette []RGB

// Nearest returns the index of the palette entry closest to c in squared
// Euclidean RGB distance, or -1 if the palette is empty.
func (p Palette) Nearest(c RGB) int {
	best := -1
	bestD := math.MaxFloat64
	for i, e := range p {
		dr := e.R - c.R
		dg := e.G - c.G
		db := e.B - c.B
		d := dr*dr + dg*dg + db*db
		if d < bestD {
			bestD = d
			best = i
		}
	}
	return best
}

// ToRGBA converts the palette to a slice of standard-library [color.RGBA]
// values, each fully opaque.
func (p Palette) ToRGBA() []color.RGBA {
	out := make([]color.RGBA, len(p))
	for i, c := range p {
		out[i] = c.ToRGBA()
	}
	return out
}

// NearestColorIndex returns the index of the palette entry closest to c in
// squared Euclidean RGB distance, or -1 for an empty palette. It is the
// package-level form of [Palette.Nearest].
func NearestColorIndex(p Palette, c RGB) int {
	return p.Nearest(c)
}

// QuantizeUniform returns a new Mat in which each channel of src is quantised to
// the given number of evenly spaced levels. The reconstructed value for a bin
// is the centre of that bin's input range, so the output spans the full [0,255]
// scale. It panics if levels is less than 2.
func QuantizeUniform(src *cv.Mat, levels int) *cv.Mat {
	if src == nil || src.Empty() {
		panic("colorspaces2: QuantizeUniform: empty Mat")
	}
	if levels < 2 {
		panic("colorspaces2: QuantizeUniform requires levels >= 2")
	}
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		bin := i * levels / 256
		if bin >= levels {
			bin = levels - 1
		}
		// Centre of the bin mapped back onto [0,255].
		lut[i] = uint8(math.Round((float64(bin) + 0.5) / float64(levels) * 255))
	}
	return ApplyLUTMat(src, lut)
}

// Posterize returns a new Mat in which each channel of src is reduced to the
// given number of levels, snapping to the nearest of levels values spread
// exactly across [0,255] (so 0 and 255 are preserved). It panics if levels is
// less than 2.
func Posterize(src *cv.Mat, levels int) *cv.Mat {
	if src == nil || src.Empty() {
		panic("colorspaces2: Posterize: empty Mat")
	}
	if levels < 2 {
		panic("colorspaces2: Posterize requires levels >= 2")
	}
	step := 255.0 / float64(levels-1)
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		q := math.Round(float64(i)/step) * step
		lut[i] = uint8(math.Round(q))
	}
	return ApplyLUTMat(src, lut)
}

// ApplyPalette returns a new Mat in which every pixel of src is replaced by its
// nearest colour in p. It panics if src is not a three-channel RGB Mat or if p
// is empty.
func ApplyPalette(src *cv.Mat, p Palette) *cv.Mat {
	colorspaces2RequireRGB(src, "ApplyPalette")
	if len(p) == 0 {
		panic("colorspaces2: ApplyPalette: empty palette")
	}
	dst := cv.NewMat(src.Rows, src.Cols, 3)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			in := colorspaces2ReadRGB(src, y, x)
			colorspaces2WriteRGB(dst, y, x, p[p.Nearest(in)])
		}
	}
	return dst
}

// DominantColor returns the single most representative colour of src as the
// mean of its pixels in linear proportion (equivalent to a one-colour
// gray-world summary). It panics if src is not a three-channel RGB Mat.
func DominantColor(src *cv.Mat) RGB {
	means := ChannelMeans(src)
	return RGB{R: means[0] / 255, G: means[1] / 255, B: means[2] / 255}
}

// colorspaces2Box is an axis-aligned RGB bounding box over a set of pixels,
// used by the median-cut algorithm.
type colorspaces2Box struct {
	pixels []RGB
}

// longestAxis returns the channel index (0=R,1=G,2=B) with the greatest spread
// and that spread's magnitude.
func (b colorspaces2Box) longestAxis() (axis int, span float64) {
	var min, max [3]float64
	for c := 0; c < 3; c++ {
		min[c] = math.MaxFloat64
		max[c] = -math.MaxFloat64
	}
	for _, p := range b.pixels {
		vals := [3]float64{p.R, p.G, p.B}
		for c := 0; c < 3; c++ {
			if vals[c] < min[c] {
				min[c] = vals[c]
			}
			if vals[c] > max[c] {
				max[c] = vals[c]
			}
		}
	}
	for c := 0; c < 3; c++ {
		if s := max[c] - min[c]; s > span {
			span = s
			axis = c
		}
	}
	return axis, span
}

// mean returns the average colour of the box.
func (b colorspaces2Box) mean() RGB {
	var sr, sg, sb float64
	for _, p := range b.pixels {
		sr += p.R
		sg += p.G
		sb += p.B
	}
	n := float64(len(b.pixels))
	return RGB{R: sr / n, G: sg / n, B: sb / n}
}

// MedianCut extracts a palette of up to n colours from src using the classic
// median-cut algorithm: the colour box with the longest axis is repeatedly
// split at the median of that axis until n boxes exist, and each box's mean
// colour becomes a palette entry. The result is deterministic and ordered by
// the order in which boxes were produced. It panics if src is not a
// three-channel RGB Mat or if n is less than 1.
func MedianCut(src *cv.Mat, n int) Palette {
	colorspaces2RequireRGB(src, "MedianCut")
	if n < 1 {
		panic("colorspaces2: MedianCut requires n >= 1")
	}
	total := src.Rows * src.Cols
	pixels := make([]RGB, total)
	for i := 0; i < total; i++ {
		base := i * 3
		pixels[i] = RGB{
			R: float64(src.Data[base]) / 255,
			G: float64(src.Data[base+1]) / 255,
			B: float64(src.Data[base+2]) / 255,
		}
	}
	boxes := []colorspaces2Box{{pixels: pixels}}
	for len(boxes) < n {
		// Pick the box with the largest spread to split.
		bestIdx := -1
		bestSpan := 0.0
		for i, b := range boxes {
			if len(b.pixels) < 2 {
				continue
			}
			_, span := b.longestAxis()
			if span > bestSpan {
				bestSpan = span
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break // no box can be split further
		}
		box := boxes[bestIdx]
		axis, _ := box.longestAxis()
		sort.Slice(box.pixels, func(a, b int) bool {
			return axisValue(box.pixels[a], axis) < axisValue(box.pixels[b], axis)
		})
		mid := len(box.pixels) / 2
		left := colorspaces2Box{pixels: box.pixels[:mid]}
		right := colorspaces2Box{pixels: box.pixels[mid:]}
		boxes[bestIdx] = left
		boxes = append(boxes, right)
	}
	palette := make(Palette, len(boxes))
	for i, b := range boxes {
		palette[i] = b.mean()
	}
	return palette
}

// axisValue returns the component of c on the given axis (0=R,1=G,2=B).
func axisValue(c RGB, axis int) float64 {
	switch axis {
	case 0:
		return c.R
	case 1:
		return c.G
	default:
		return c.B
	}
}

// KMeansQuantize clusters the pixels of src into k colours with Lloyd's
// algorithm and returns the cluster-centre palette together with a per-pixel
// label slice (row-major, length Rows*Cols) giving each pixel's cluster index.
//
// Initial centres are chosen deterministically with a k-means++ style seeding
// driven by seed, so the result is reproducible for a given seed. iterations
// bounds the number of Lloyd refinement passes; the loop also stops early once
// assignments stop changing. It panics if src is not a three-channel RGB Mat,
// if k is less than 1, or if iterations is negative.
func KMeansQuantize(src *cv.Mat, k, iterations int, seed uint64) (Palette, []int) {
	colorspaces2RequireRGB(src, "KMeansQuantize")
	if k < 1 {
		panic("colorspaces2: KMeansQuantize requires k >= 1")
	}
	if iterations < 0 {
		panic("colorspaces2: KMeansQuantize requires iterations >= 0")
	}
	total := src.Rows * src.Cols
	pixels := make([]RGB, total)
	for i := 0; i < total; i++ {
		base := i * 3
		pixels[i] = RGB{
			R: float64(src.Data[base]) / 255,
			G: float64(src.Data[base+1]) / 255,
			B: float64(src.Data[base+2]) / 255,
		}
	}
	if k > total {
		k = total
	}
	centres := colorspaces2SeedCentres(pixels, k, seed)
	labels := make([]int, total)
	for it := 0; it <= iterations; it++ {
		changed := false
		for i, p := range pixels {
			best := 0
			bestD := math.MaxFloat64
			for c, ctr := range centres {
				d := sqDist(p, ctr)
				if d < bestD {
					bestD = d
					best = c
				}
			}
			if labels[i] != best {
				labels[i] = best
				changed = true
			}
		}
		if it == iterations {
			break
		}
		// Recompute centres.
		sums := make([]RGB, k)
		counts := make([]int, k)
		for i, p := range pixels {
			l := labels[i]
			sums[l].R += p.R
			sums[l].G += p.G
			sums[l].B += p.B
			counts[l]++
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				continue // keep the previous centre for empty clusters
			}
			n := float64(counts[c])
			centres[c] = RGB{R: sums[c].R / n, G: sums[c].G / n, B: sums[c].B / n}
		}
		if !changed {
			break
		}
	}
	return Palette(centres), labels
}

// sqDist returns the squared Euclidean distance between two RGB colours.
func sqDist(a, b RGB) float64 {
	dr := a.R - b.R
	dg := a.G - b.G
	db := a.B - b.B
	return dr*dr + dg*dg + db*db
}

// colorspaces2SeedCentres selects k initial cluster centres using k-means++
// weighting with a deterministic SplitMix64 generator seeded by seed.
func colorspaces2SeedCentres(pixels []RGB, k int, seed uint64) []RGB {
	rng := seed
	next := func() float64 {
		// SplitMix64.
		rng += 0x9E3779B97F4A7C15
		z := rng
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		z ^= z >> 31
		return float64(z>>11) / float64(1<<53)
	}
	centres := make([]RGB, 0, k)
	// First centre chosen uniformly.
	centres = append(centres, pixels[int(next()*float64(len(pixels)))%len(pixels)])
	dist := make([]float64, len(pixels))
	for len(centres) < k {
		var total float64
		for i, p := range pixels {
			best := math.MaxFloat64
			for _, c := range centres {
				if d := sqDist(p, c); d < best {
					best = d
				}
			}
			dist[i] = best
			total += best
		}
		if total == 0 {
			// All remaining pixels coincide with a chosen centre; pad by reuse.
			centres = append(centres, pixels[len(centres)%len(pixels)])
			continue
		}
		target := next() * total
		var acc float64
		chosen := len(pixels) - 1
		for i, d := range dist {
			acc += d
			if acc >= target {
				chosen = i
				break
			}
		}
		centres = append(centres, pixels[chosen])
	}
	return centres
}

// ExtractPalette returns the n most representative colours of src using
// [KMeansQuantize] with a fixed seed and a modest iteration budget, ordered by
// descending cluster population so the most common colours come first. It
// panics if src is not a three-channel RGB Mat or if n is less than 1.
func ExtractPalette(src *cv.Mat, n int) Palette {
	colorspaces2RequireRGB(src, "ExtractPalette")
	if n < 1 {
		panic("colorspaces2: ExtractPalette requires n >= 1")
	}
	centres, labels := KMeansQuantize(src, n, 16, 1)
	counts := make([]int, len(centres))
	for _, l := range labels {
		counts[l]++
	}
	idx := make([]int, len(centres))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return counts[idx[a]] > counts[idx[b]] })
	out := make(Palette, len(centres))
	for i, j := range idx {
		out[i] = centres[j]
	}
	return out
}

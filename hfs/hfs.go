package hfs

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// Default parameter values, matching the OpenCV hfs module.
const (
	// DefaultSegEgbThresholdI is the stage-I EGB merge threshold.
	DefaultSegEgbThresholdI = 0.08
	// DefaultMinRegionSizeI is the stage-I small-region absorption size (pixels).
	DefaultMinRegionSizeI = 100
	// DefaultSegEgbThresholdII is the stage-II EGB merge threshold.
	DefaultSegEgbThresholdII = 0.28
	// DefaultMinRegionSizeII is the stage-II small-region absorption size (pixels).
	DefaultMinRegionSizeII = 200
	// DefaultSpatialWeight is the SLIC colour-vs-space compactness weight.
	DefaultSpatialWeight = 0.6
	// DefaultSlicSpixelSize is the nominal SLIC superpixel edge length (pixels).
	DefaultSlicSpixelSize = 8
	// DefaultNumSlicIter is the number of SLIC Lloyd iterations.
	DefaultNumSlicIter = 5
)

// DrawMode selects how [HfsSegment.DrawSegmentation] colours each region.
type DrawMode int

const (
	// DrawAverageColor paints every region with the mean colour of the source
	// pixels it covers, giving a posterised view of the input.
	DrawAverageColor DrawMode = iota
	// DrawRandomColor paints every region with a distinct, deterministic
	// pseudo-random colour, making region boundaries easy to see.
	DrawRandomColor
)

// HfsSegment performs Hierarchical Feature Selection segmentation for a fixed
// image size. Construct it with [Create] or [CreateWithDefaults], adjust its
// parameters through the Set/Get accessors, then call
// [HfsSegment.PerformSegmentCpu] (or the compatibility alias
// [HfsSegment.PerformSegmentGpu]).
//
// A single HfsSegment may be reused across images of its configured size; each
// segmentation call overwrites the stored labelling used by [HfsSegment.Labels]
// and [HfsSegment.DrawSegmentation].
type HfsSegment struct {
	height int
	width  int

	segEgbThresholdI  float64
	minRegionSizeI    int
	segEgbThresholdII float64
	minRegionSizeII   int
	spatialWeight     float64
	slicSpixelSize    int
	numSlicIter       int

	// Result of the most recent segmentation.
	srcColor  *cv.Mat // 3-channel RGB copy of the last input
	labels    []int   // per-pixel region id in [0, numSegments)
	numSegs   int
	labelRows int
	labelCols int
}

// Create returns an [HfsSegment] configured for images of the given height and
// width with every parameter specified explicitly. It panics if height or width
// is not positive. See [CreateWithDefaults] for the OpenCV default parameters.
func Create(height, width int, segEgbThresholdI float64, minRegionSizeI int, segEgbThresholdII float64, minRegionSizeII int, spatialWeight float64, slicSpixelSize, numSlicIter int) *HfsSegment {
	if height <= 0 || width <= 0 {
		panic("hfs: Create requires positive height and width")
	}
	return &HfsSegment{
		height:            height,
		width:             width,
		segEgbThresholdI:  segEgbThresholdI,
		minRegionSizeI:    minRegionSizeI,
		segEgbThresholdII: segEgbThresholdII,
		minRegionSizeII:   minRegionSizeII,
		spatialWeight:     spatialWeight,
		slicSpixelSize:    slicSpixelSize,
		numSlicIter:       numSlicIter,
	}
}

// CreateWithDefaults returns an [HfsSegment] for images of the given height and
// width using the OpenCV default parameters (see the Default* constants).
func CreateWithDefaults(height, width int) *HfsSegment {
	return Create(height, width,
		DefaultSegEgbThresholdI, DefaultMinRegionSizeI,
		DefaultSegEgbThresholdII, DefaultMinRegionSizeII,
		DefaultSpatialWeight, DefaultSlicSpixelSize, DefaultNumSlicIter)
}

// Height returns the configured image height.
func (h *HfsSegment) Height() int { return h.height }

// Width returns the configured image width.
func (h *HfsSegment) Width() int { return h.width }

// SetSegEgbThresholdI sets the stage-I EGB merge threshold.
func (h *HfsSegment) SetSegEgbThresholdI(v float64) { h.segEgbThresholdI = v }

// GetSegEgbThresholdI returns the stage-I EGB merge threshold.
func (h *HfsSegment) GetSegEgbThresholdI() float64 { return h.segEgbThresholdI }

// SetMinRegionSizeI sets the stage-I minimum region size in pixels.
func (h *HfsSegment) SetMinRegionSizeI(v int) { h.minRegionSizeI = v }

// GetMinRegionSizeI returns the stage-I minimum region size in pixels.
func (h *HfsSegment) GetMinRegionSizeI() int { return h.minRegionSizeI }

// SetSegEgbThresholdII sets the stage-II EGB merge threshold.
func (h *HfsSegment) SetSegEgbThresholdII(v float64) { h.segEgbThresholdII = v }

// GetSegEgbThresholdII returns the stage-II EGB merge threshold.
func (h *HfsSegment) GetSegEgbThresholdII() float64 { return h.segEgbThresholdII }

// SetMinRegionSizeII sets the stage-II minimum region size in pixels.
func (h *HfsSegment) SetMinRegionSizeII(v int) { h.minRegionSizeII = v }

// GetMinRegionSizeII returns the stage-II minimum region size in pixels.
func (h *HfsSegment) GetMinRegionSizeII() int { return h.minRegionSizeII }

// SetSpatialWeight sets the SLIC colour-vs-space compactness weight.
func (h *HfsSegment) SetSpatialWeight(v float64) { h.spatialWeight = v }

// GetSpatialWeight returns the SLIC colour-vs-space compactness weight.
func (h *HfsSegment) GetSpatialWeight() float64 { return h.spatialWeight }

// SetSlicSpixelSize sets the nominal SLIC superpixel edge length in pixels.
func (h *HfsSegment) SetSlicSpixelSize(v int) { h.slicSpixelSize = v }

// GetSlicSpixelSize returns the nominal SLIC superpixel edge length in pixels.
func (h *HfsSegment) GetSlicSpixelSize() int { return h.slicSpixelSize }

// SetNumSlicIter sets the number of SLIC Lloyd iterations.
func (h *HfsSegment) SetNumSlicIter(v int) { h.numSlicIter = v }

// GetNumSlicIter returns the number of SLIC Lloyd iterations.
func (h *HfsSegment) GetNumSlicIter() int { return h.numSlicIter }

// PerformSegmentCpu segments src and stores the result. When ifDraw is true it
// returns the average-colour rendering of the segmentation (a three-channel
// image, as in OpenCV); when ifDraw is false it returns a single-channel label
// image whose samples are the region ids (clamped to 255 — use [HfsSegment.Labels]
// for the complete labelling).
//
// src must be non-empty and match the configured Height and Width. A grayscale
// input is promoted to RGB. It panics on a size mismatch or an empty image.
func (h *HfsSegment) PerformSegmentCpu(src *cv.Mat, ifDraw bool) *cv.Mat {
	if src.Empty() {
		panic("hfs: PerformSegmentCpu on empty image")
	}
	if src.Rows != h.height || src.Cols != h.width {
		panic("hfs: image size does not match the configured segmenter size")
	}
	h.segment(src)
	if ifDraw {
		return h.DrawSegmentation(DrawAverageColor)
	}
	return h.labelImage()
}

// PerformSegmentGpu is provided for API compatibility with OpenCV's HfsSegment.
// This port has no GPU backend, so it forwards to [HfsSegment.PerformSegmentCpu]
// and returns an identical result.
func (h *HfsSegment) PerformSegmentGpu(src *cv.Mat, ifDraw bool) *cv.Mat {
	return h.PerformSegmentCpu(src, ifDraw)
}

// segment runs the full SLIC -> EGB-I -> EGB-II -> absorption pipeline and stores
// the per-pixel labelling and the RGB source copy on h.
func (h *HfsSegment) segment(src *cv.Mat) {
	color := ensureColor(src)
	lab := cv.CvtColor(color, cv.ColorRGB2Lab)
	rows, cols := lab.Rows, lab.Cols
	mag := gradientMagnitude(lab)

	// Stage 0: SLIC superpixels.
	sp := slic(lab, h.slicSpixelSize, h.spatialWeight, h.numSlicIter)

	// Stage I: EGB merge over the superpixel RAG, then small-region absorption.
	stage1 := h.egbStage(sp.labels, sp.count, lab, mag, rows, cols, h.segEgbThresholdI, h.minRegionSizeI)

	// Stage II: repeat on the coarser stage-I regions.
	label1, count1 := relabelConsecutive(stage1)
	stage2 := h.egbStage(label1, count1, lab, mag, rows, cols, h.segEgbThresholdII, h.minRegionSizeII)

	labels, count := relabelConsecutive(stage2)
	h.srcColor = color
	h.labels = labels
	h.numSegs = count
	h.labelRows = rows
	h.labelCols = cols
}

// egbStage runs one EGB merge with small-region absorption over the region
// adjacency graph of a per-pixel labelling and returns the new per-pixel
// labelling (as raw union-find roots, not yet made consecutive).
func (h *HfsSegment) egbStage(pixLabels []int, count int, lab *cv.Mat, mag []float64, rows, cols int, threshold float64, minRegion int) []int {
	feats, sizes := regionFeatures(pixLabels, count, lab, mag)
	edges := buildRAG(pixLabels, count, feats, rows, cols)

	uf := egbMerge(count, edges, threshold)
	absorbSmall(uf, edges, sizes, minRegion)

	out := make([]int, len(pixLabels))
	for i, l := range pixLabels {
		out[i] = uf.find(l)
	}
	return out
}

// Labels returns a copy of the per-pixel region labelling from the most recent
// segmentation together with its dimensions. Labels are dense in
// [0, NumSegments). It returns (nil, 0, 0) before any segmentation has run.
func (h *HfsSegment) Labels() (labels []int, rows, cols int) {
	if h.labels == nil {
		return nil, 0, 0
	}
	out := make([]int, len(h.labels))
	copy(out, h.labels)
	return out, h.labelRows, h.labelCols
}

// NumSegments returns the number of regions produced by the most recent
// segmentation, or 0 before any segmentation has run.
func (h *HfsSegment) NumSegments() int { return h.numSegs }

// DrawSegmentation renders the most recent segmentation as a three-channel image
// using the given [DrawMode]. It panics if no segmentation has been performed.
func (h *HfsSegment) DrawSegmentation(mode DrawMode) *cv.Mat {
	if h.labels == nil {
		panic("hfs: DrawSegmentation before PerformSegmentCpu")
	}
	rows, cols := h.labelRows, h.labelCols
	out := cv.NewMat(rows, cols, 3)
	switch mode {
	case DrawRandomColor:
		palette := make([][3]uint8, h.numSegs)
		rng := rand.New(rand.NewSource(0x5f3759df))
		for i := range palette {
			palette[i] = [3]uint8{
				uint8(rng.Intn(256)),
				uint8(rng.Intn(256)),
				uint8(rng.Intn(256)),
			}
		}
		for i, l := range h.labels {
			c := palette[l]
			out.Data[i*3+0] = c[0]
			out.Data[i*3+1] = c[1]
			out.Data[i*3+2] = c[2]
		}
	default: // DrawAverageColor
		sum := make([][3]float64, h.numSegs)
		cnt := make([]int, h.numSegs)
		for i, l := range h.labels {
			b := i * 3
			sum[l][0] += float64(h.srcColor.Data[b+0])
			sum[l][1] += float64(h.srcColor.Data[b+1])
			sum[l][2] += float64(h.srcColor.Data[b+2])
			cnt[l]++
		}
		mean := make([][3]uint8, h.numSegs)
		for l := range mean {
			if cnt[l] == 0 {
				continue
			}
			for k := 0; k < 3; k++ {
				mean[l][k] = clampU8(sum[l][k] / float64(cnt[l]))
			}
		}
		for i, l := range h.labels {
			c := mean[l]
			out.Data[i*3+0] = c[0]
			out.Data[i*3+1] = c[1]
			out.Data[i*3+2] = c[2]
		}
	}
	return out
}

// labelImage renders the stored labelling as a single-channel image whose
// samples are the region ids clamped to 255.
func (h *HfsSegment) labelImage() *cv.Mat {
	out := cv.NewMat(h.labelRows, h.labelCols, 1)
	for i, l := range h.labels {
		if l > 255 {
			l = 255
		}
		out.Data[i] = uint8(l)
	}
	return out
}

// regionFeatures computes, for each of the count regions in a per-pixel
// labelling, a normalised feature vector [L, a, b, texture] (components roughly
// in [0, 1]) and the region's pixel count.
func regionFeatures(pixLabels []int, count int, lab *cv.Mat, mag []float64) (feats [][4]float64, sizes []int) {
	sums := make([][4]float64, count)
	sizes = make([]int, count)
	for i, l := range pixLabels {
		b := i * 3
		sums[l][0] += float64(lab.Data[b+0])
		sums[l][1] += float64(lab.Data[b+1])
		sums[l][2] += float64(lab.Data[b+2])
		sums[l][3] += mag[i]
		sizes[l]++
	}
	feats = make([][4]float64, count)
	for l := 0; l < count; l++ {
		if sizes[l] == 0 {
			continue
		}
		inv := 1.0 / float64(sizes[l])
		feats[l][0] = sums[l][0] * inv / 255.0
		feats[l][1] = sums[l][1] * inv / 255.0
		feats[l][2] = sums[l][2] * inv / 255.0
		t := sums[l][3] * inv / 255.0
		if t > 1 {
			t = 1
		}
		feats[l][3] = t
	}
	return feats, sizes
}

// buildRAG builds the region-adjacency graph of a per-pixel labelling: one edge
// per pair of 4-adjacent regions, weighted by the Euclidean distance between
// their feature vectors. Because merges only ever happen across these edges,
// every region produced downstream stays 4-connected.
func buildRAG(pixLabels []int, count int, feats [][4]float64, rows, cols int) []edge {
	seen := make(map[int64]struct{})
	var edges []edge
	addPair := func(a, b int) {
		if a == b {
			return
		}
		if a > b {
			a, b = b, a
		}
		key := int64(a)*int64(count) + int64(b)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		edges = append(edges, edge{a: a, b: b, w: featDist(feats[a], feats[b])})
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			l := pixLabels[y*cols+x]
			if x+1 < cols {
				addPair(l, pixLabels[y*cols+x+1])
			}
			if y+1 < rows {
				addPair(l, pixLabels[(y+1)*cols+x])
			}
		}
	}
	return edges
}

// featDist is the Euclidean distance between two region feature vectors.
func featDist(a, b [4]float64) float64 {
	var s float64
	for k := 0; k < 4; k++ {
		d := a[k] - b[k]
		s += d * d
	}
	return math.Sqrt(s)
}

// ensureColor returns a three-channel RGB copy of src, promoting grayscale input
// and treating the first three channels of a wider image as RGB.
func ensureColor(src *cv.Mat) *cv.Mat {
	switch src.Channels {
	case 3:
		return src.Clone()
	case 1:
		return cv.CvtColor(src, cv.ColorGray2RGB)
	default:
		out := cv.NewMat(src.Rows, src.Cols, 3)
		for i := 0; i < src.Total(); i++ {
			for k := 0; k < 3; k++ {
				if k < src.Channels {
					out.Data[i*3+k] = src.Data[i*src.Channels+k]
				}
			}
		}
		return out
	}
}

// clampU8 rounds v to the nearest integer and clamps it into [0, 255].
func clampU8(v float64) uint8 {
	v += 0.5
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

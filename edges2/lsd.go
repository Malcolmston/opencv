package edges2

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// LSDOptions configures the [LSD] line-segment detector.
type LSDOptions struct {
	// Sigma is the standard deviation of the Gaussian pre-smoothing.
	Sigma float64
	// AngleTolerance is the maximum gradient-orientation deviation, in
	// radians, allowed between a pixel and its growing region.
	AngleTolerance float64
	// GradientThreshold is the minimum gradient magnitude for a pixel to seed
	// or join a region. When non-positive an automatic threshold of one tenth
	// of the maximum magnitude is used.
	GradientThreshold float64
	// MinRegionSize is the smallest number of pixels a region must contain to
	// yield a segment.
	MinRegionSize int
	// MinLength is the smallest length, in pixels, of a reported segment.
	MinLength float64
}

// DefaultLSDOptions returns sensible defaults for [LSD]: light smoothing, a
// 22.5° orientation tolerance, an automatic gradient threshold and a minimum
// region of five pixels.
func DefaultLSDOptions() LSDOptions {
	return LSDOptions{
		Sigma:             0.8,
		AngleTolerance:    math.Pi / 8,
		GradientThreshold: 0,
		MinRegionSize:     5,
		MinLength:         3,
	}
}

// LSD detects line segments in a single-channel image with a region-growing
// line-segment detector. It smooths the image, computes Sobel gradients, and
// grows connected regions of pixels that share a common gradient orientation
// (within opts.AngleTolerance); each sufficiently large region is fitted with a
// segment along its principal axis. Segments are returned sorted by descending
// length.
//
// This is a simplified, deterministic variant of the LSD algorithm of von Gioi
// et al.: it performs the gradient-orientation region growing and rectangle
// approximation but omits the a-contrario NFA validation step, using region
// size and length thresholds instead. It panics on multi-channel input.
func LSD(src *cv.Mat, opts LSDOptions) []Segment {
	edges2RequireGray(src, "LSD")
	if opts.AngleTolerance <= 0 {
		opts.AngleTolerance = math.Pi / 8
	}
	if opts.MinRegionSize <= 0 {
		opts.MinRegionSize = 5
	}
	rows, cols := src.Rows, src.Cols
	blurred := edges2Blur(src, math.Max(opts.Sigma, 0.5))
	f := Sobel(blurred)
	mag := f.Magnitude()
	ang := f.Orientation()

	maxMag := 0.0
	for _, v := range mag.Data {
		if v > maxMag {
			maxMag = v
		}
	}
	thresh := opts.GradientThreshold
	if thresh <= 0 {
		thresh = 0.1 * maxMag
	}

	// Seed order: descending gradient magnitude for stability.
	type seed struct {
		i int
		m float64
	}
	seeds := make([]seed, 0, len(mag.Data))
	for i, m := range mag.Data {
		if m >= thresh {
			seeds = append(seeds, seed{i, m})
		}
	}
	sort.SliceStable(seeds, func(a, b int) bool { return seeds[a].m > seeds[b].m })

	used := make([]bool, rows*cols)
	var segs []Segment
	neigh := [8][2]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}

	for _, sd := range seeds {
		if used[sd.i] {
			continue
		}
		// Grow a region of coherent gradient orientation.
		region := []int{sd.i}
		used[sd.i] = true
		sumSin := math.Sin(ang.Data[sd.i])
		sumCos := math.Cos(ang.Data[sd.i])
		for qi := 0; qi < len(region); qi++ {
			p := region[qi]
			py := p / cols
			px := p % cols
			meanAng := math.Atan2(sumSin, sumCos)
			for _, d := range neigh {
				ny := py + d[0]
				nx := px + d[1]
				if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
					continue
				}
				ni := ny*cols + nx
				if used[ni] || mag.Data[ni] < thresh {
					continue
				}
				if edges2AngleDiff(ang.Data[ni], meanAng) <= opts.AngleTolerance {
					used[ni] = true
					region = append(region, ni)
					sumSin += math.Sin(ang.Data[ni])
					sumCos += math.Cos(ang.Data[ni])
				}
			}
		}
		if len(region) < opts.MinRegionSize {
			continue
		}
		if seg, ok := edges2FitSegment(region, cols, mag); ok && seg.Length() >= opts.MinLength {
			segs = append(segs, seg)
		}
	}
	sort.SliceStable(segs, func(i, j int) bool { return segs[i].Length() > segs[j].Length() })
	return segs
}

// edges2AngleDiff returns the absolute smallest difference between two angles
// in radians, in [0, pi].
func edges2AngleDiff(a, b float64) float64 {
	d := math.Abs(a - b)
	for d > math.Pi {
		d = math.Abs(d - 2*math.Pi)
	}
	return d
}

// edges2FitSegment fits a segment to a region of pixels using the
// magnitude-weighted principal axis and returns the extreme projections as
// endpoints.
func edges2FitSegment(region []int, cols int, mag *FloatGrid) (Segment, bool) {
	var wsum, mx, my float64
	for _, i := range region {
		w := mag.Data[i]
		y := float64(i / cols)
		x := float64(i % cols)
		wsum += w
		mx += w * x
		my += w * y
	}
	if wsum == 0 {
		return Segment{}, false
	}
	mx /= wsum
	my /= wsum
	var sxx, syy, sxy float64
	for _, i := range region {
		w := mag.Data[i]
		dx := float64(i%cols) - mx
		dy := float64(i/cols) - my
		sxx += w * dx * dx
		syy += w * dy * dy
		sxy += w * dx * dy
	}
	theta := 0.5 * math.Atan2(2*sxy, sxx-syy)
	dirX := math.Cos(theta)
	dirY := math.Sin(theta)
	tMin := math.Inf(1)
	tMax := math.Inf(-1)
	for _, i := range region {
		dx := float64(i%cols) - mx
		dy := float64(i/cols) - my
		t := dx*dirX + dy*dirY
		if t < tMin {
			tMin = t
		}
		if t > tMax {
			tMax = t
		}
	}
	return Segment{
		X1: mx + dirX*tMin, Y1: my + dirY*tMin,
		X2: mx + dirX*tMax, Y2: my + dirY*tMax,
	}, true
}

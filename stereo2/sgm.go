package stereo2

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// FourPaths returns the four axis-aligned scanline directions (left-to-right,
// right-to-left, top-to-bottom, bottom-to-top) used by [SemiGlobalAggregate].
func FourPaths() [][2]int {
	return [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
}

// EightPaths returns the eight directions (the four of [FourPaths] plus the four
// diagonals) used by [SemiGlobalAggregate] for the full SGM approximation.
func EightPaths() [][2]int {
	return [][2]int{
		{0, 1}, {0, -1}, {1, 0}, {-1, 0},
		{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
	}
}

// SemiGlobalAggregate performs Hirschmüller's semi-global cost aggregation. For
// each direction in dirs it accumulates the smoothed path cost
//
//	Lr(p,d) = C(p,d) + min( Lr(p-r,d),
//	                        Lr(p-r,d-1)+p1,
//	                        Lr(p-r,d+1)+p1,
//	                        min_k Lr(p-r,k)+p2 ) - min_k Lr(p-r,k)
//
// where C is the input volume, p1 penalises a one-step disparity change and p2
// (>= p1) penalises larger jumps. The returned volume holds, per pixel and
// disparity, the sum of Lr over all directions; extract disparities from it with
// [CostVolume.ToDisparity] or [RefineSubpixelParabola].
//
// It panics if p1 < 0, p2 < p1, or dirs is empty.
func SemiGlobalAggregate(vol *CostVolume, p1, p2 float32, dirs [][2]int) *CostVolume {
	if p1 < 0 || p2 < p1 {
		panic(fmt.Sprintf("stereo2: SemiGlobalAggregate requires 0 <= p1 <= p2, got p1=%g p2=%g", p1, p2))
	}
	if len(dirs) == 0 {
		panic("stereo2: SemiGlobalAggregate requires at least one direction")
	}
	rows, cols, D := vol.Rows, vol.Cols, vol.Disparities
	agg := NewCostVolume(rows, cols, D, vol.MinDisparity)
	lr := make([]float32, rows*cols*D)
	for _, dir := range dirs {
		dy, dx := dir[0], dir[1]
		if dy == 0 && dx == 0 {
			continue
		}
		ys, ye, ystep := 0, rows, 1
		if dy < 0 {
			ys, ye, ystep = rows-1, -1, -1
		}
		xs, xe, xstep := 0, cols, 1
		if dx < 0 {
			xs, xe, xstep = cols-1, -1, -1
		}
		for y := ys; y != ye; y += ystep {
			for x := xs; x != xe; x += xstep {
				p := (y*cols + x) * D
				py, px := y-dy, x-dx
				if py < 0 || py >= rows || px < 0 || px >= cols {
					// First pixel of this path: seed with the raw cost.
					for d := 0; d < D; d++ {
						v := vol.Data[p+d]
						lr[p+d] = v
						agg.Data[p+d] += v
					}
					continue
				}
				prev := (py*cols + px) * D
				prevMin := lr[prev]
				for d := 1; d < D; d++ {
					if lr[prev+d] < prevMin {
						prevMin = lr[prev+d]
					}
				}
				for d := 0; d < D; d++ {
					best := lr[prev+d]
					if d > 0 {
						if v := lr[prev+d-1] + p1; v < best {
							best = v
						}
					}
					if d < D-1 {
						if v := lr[prev+d+1] + p1; v < best {
							best = v
						}
					}
					if v := prevMin + p2; v < best {
						best = v
					}
					val := vol.Data[p+d] + best - prevMin
					lr[p+d] = val
					agg.Data[p+d] += val
				}
			}
		}
	}
	return agg
}

// SGMMatcher is a semi-global matcher. It builds a per-pixel census matching
// cost, aggregates it along several directions with [SemiGlobalAggregate], and
// extracts a (optionally sub-pixel) disparity map. SGM approaches the accuracy
// of global optimisation at a fraction of the cost and is the default choice for
// high-quality dense stereo in this package.
//
// The zero value is not useful; construct one with [NewSGMMatcher].
type SGMMatcher struct {
	// MinDisparity is the smallest disparity searched.
	MinDisparity int
	// NumDisparities is the width of the disparity search range in pixels.
	NumDisparities int
	// CensusW is the census window width (odd) used for the matching cost.
	CensusW int
	// CensusH is the census window height (odd) used for the matching cost.
	CensusH int
	// P1 penalises a one-pixel disparity change between adjacent pixels.
	P1 float32
	// P2 penalises larger disparity changes; must be >= P1.
	P2 float32
	// Paths is the aggregation directions; use [FourPaths] or [EightPaths]. When
	// nil, [EightPaths] is used.
	Paths [][2]int
	// Subpixel enables parabola sub-pixel interpolation of the winning disparity.
	Subpixel bool
	// UniquenessRatio, when positive, applies the uniqueness test.
	UniquenessRatio int
}

// NewSGMMatcher returns an SGMMatcher with sensible defaults: a 5x5 census
// window, eight aggregation paths, P1=10, P2=120 and sub-pixel enabled. It
// panics if numDisparities is not positive.
func NewSGMMatcher(minDisparity, numDisparities int) *SGMMatcher {
	if numDisparities <= 0 {
		panic(fmt.Sprintf("stereo2: NewSGMMatcher requires numDisparities > 0, got %d", numDisparities))
	}
	return &SGMMatcher{
		MinDisparity:   minDisparity,
		NumDisparities: numDisparities,
		CensusW:        5,
		CensusH:        5,
		P1:             10,
		P2:             120,
		Paths:          EightPaths(),
		Subpixel:       true,
	}
}

// ComputeCostVolume builds the census matching cost for the pair and returns the
// semi-globally aggregated cost volume, before disparity extraction. It is
// exposed so callers can run their own refinement on the volume. It panics if
// the images differ in size.
func (s *SGMMatcher) ComputeCostVolume(left, right *cv.Mat) *CostVolume {
	stereo2checkPair(left, right)
	lc := CensusTransform(left, s.CensusW, s.CensusH)
	rc := CensusTransform(right, s.CensusW, s.CensusH)
	vol := BuildCensusCostVolume(lc, rc, s.MinDisparity, s.NumDisparities)
	paths := s.Paths
	if paths == nil {
		paths = EightPaths()
	}
	return SemiGlobalAggregate(vol, s.P1, s.P2, paths)
}

// Compute matches left against right with semi-global aggregation and returns
// the left-referenced disparity map, sub-pixel refined when Subpixel is set.
// Census-border pixels are marked [InvalidDisparity]. It panics if the images
// differ in size.
func (s *SGMMatcher) Compute(left, right *cv.Mat) *DisparityMap {
	agg := s.ComputeCostVolume(left, right)
	var disp *DisparityMap
	if s.Subpixel {
		disp = RefineSubpixelParabola(agg, s.UniquenessRatio)
	} else {
		disp = agg.ToDisparity(s.UniquenessRatio)
	}
	rw, rh := s.CensusW/2, s.CensusH/2
	for y := 0; y < disp.Rows; y++ {
		for x := 0; x < disp.Cols; x++ {
			if y < rh || y >= disp.Rows-rh || x < rw || x >= disp.Cols-rw {
				disp.Data[y*disp.Cols+x] = InvalidDisparity
			}
		}
	}
	return disp
}

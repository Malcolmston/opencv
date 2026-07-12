package ximgproc

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// WeightedMedianFilter applies a guided weighted-median filter to src and
// returns a new Mat of the same shape. It is an edge-preserving, structure-aware
// smoother: like a median filter it removes impulsive noise and does not blur
// across edges, but each neighbour's vote is weighted by its guidance affinity
// so that only pixels similar to the centre in the guide image influence the
// result (Zhang, Xu and Jia, "100+ Times Faster Weighted Median Filter", 2014).
//
// Within the square window of the given radius, neighbour q contributes weight
//
//	w = exp(−‖guide_p − guide_q‖₁² / (2·sigma²))
//
// to the histogram of src(q); the output is the weighted median — the value at
// which the accumulated weight first reaches half of the total. Each src channel
// is filtered independently.
//
// radius is the window radius in pixels (window side = 2·radius+1) and must be
// positive; sigma is the guidance affinity scale on the native [0,255] range. If
// guide is nil, src guides itself (a plain weighted median). guide, when given,
// must share width and height with src and may be 1- or 3-channel. It panics on
// a non-positive radius or a size mismatch. The filter is deterministic.
func WeightedMedianFilter(src, guide *cv.Mat, radius int, sigma float64) *cv.Mat {
	if radius <= 0 {
		panic("ximgproc: WeightedMedianFilter requires a positive radius")
	}
	if guide == nil {
		guide = src
	}
	if src.Rows != guide.Rows || src.Cols != guide.Cols {
		panic("ximgproc: WeightedMedianFilter guide and src must share dimensions")
	}
	if sigma <= 0 {
		sigma = 1
	}
	rows, cols := src.Rows, src.Cols
	gch := guide.Channels
	sch := src.Channels

	inv := 1.0 / (2 * sigma * sigma)
	maxDiff := 255*gch + 1
	rangeLUT := make([]float64, maxDiff)
	for k := 0; k < maxDiff; k++ {
		rangeLUT[k] = math.Exp(-float64(k*k) * inv)
	}

	dst := cv.NewMat(rows, cols, sch)
	// Reusable (value,weight) scratch, one entry per window pixel.
	type vw struct {
		v int
		w float64
	}
	side := 2*radius + 1
	buf := make([]vw, 0, side*side)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gi := (y*cols + x) * gch
			for c := 0; c < sch; c++ {
				buf = buf[:0]
				var total float64
				for dy := -radius; dy <= radius; dy++ {
					yy := reflect(y+dy, rows)
					for dx := -radius; dx <= radius; dx++ {
						xx := reflect(x+dx, cols)
						gj := (yy*cols + xx) * gch
						diff := 0
						for gc := 0; gc < gch; gc++ {
							diff += absInt(int(guide.Data[gi+gc]) - int(guide.Data[gj+gc]))
						}
						w := rangeLUT[diff]
						buf = append(buf, vw{int(src.Data[(yy*cols+xx)*sch+c]), w})
						total += w
					}
				}
				sort.Slice(buf, func(i, j int) bool { return buf[i].v < buf[j].v })
				half := total / 2
				var acc float64
				out := buf[len(buf)-1].v
				for _, e := range buf {
					acc += e.w
					if acc >= half {
						out = e.v
						break
					}
				}
				dst.Data[(y*cols+x)*sch+c] = uint8(out)
			}
		}
	}
	return dst
}

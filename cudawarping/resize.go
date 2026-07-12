package cudawarping

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Resize scales the GpuMat to the destination size dsize (width dsize.X, height
// dsize.Y) using the chosen interpolation, mirroring cv::cuda::resize. If dsize
// is the zero value (both components 0) the output size is derived from the
// scale factors fx and fy applied to the current size, matching OpenCV's
// dsize == Size() convention; in that case fx and fy must both be positive.
//
// All four interpolation modes are supported. [InterNearest] and [InterLinear]
// delegate to [cv.Resize]; [InterCubic] and [InterArea] are computed locally
// (the root package implements only nearest and linear). The stream argument is
// accepted for API compatibility and ignored. It panics on an empty GpuMat or a
// non-positive resolved output size.
func (g *GpuMat) Resize(dsize image.Point, fx, fy float64, interp Interpolation, stream *Stream) *GpuMat {
	src := g.host("Resize")
	w, h := dsize.X, dsize.Y
	if w == 0 && h == 0 {
		if fx <= 0 || fy <= 0 {
			panic("cudawarping: Resize with zero dsize requires positive fx and fy")
		}
		w = int(math.Round(float64(src.Cols) * fx))
		h = int(math.Round(float64(src.Rows) * fy))
	}
	if w <= 0 || h <= 0 {
		panic("cudawarping: Resize requires a positive output size")
	}
	switch interp {
	case InterNearest, InterLinear:
		return &GpuMat{mat: cv.Resize(src, w, h, cvInterp(interp))}
	case InterArea:
		return &GpuMat{mat: resizeArea(src, w, h)}
	case InterCubic:
		return &GpuMat{mat: resizeCubic(src, w, h)}
	default:
		panic("cudawarping: Resize unknown interpolation")
	}
}

// ResizeScale scales the GpuMat by the independent factors fx (horizontal) and
// fy (vertical), a thin wrapper over [GpuMat.Resize] with a zero dsize. It
// mirrors the cv::cuda::resize call form where the destination size is left
// empty and derived from the scale factors. It panics unless fx and fy are both
// positive.
func (g *GpuMat) ResizeScale(fx, fy float64, interp Interpolation, stream *Stream) *GpuMat {
	if fx <= 0 || fy <= 0 {
		panic("cudawarping: ResizeScale requires positive fx and fy")
	}
	return g.Resize(image.Point{}, fx, fy, interp, stream)
}

// resizeCubic resizes src to w×h using bicubic interpolation with pixel-centre
// alignment, replicating the border for taps that fall outside the source.
func resizeCubic(src *cv.Mat, w, h int) *cv.Mat {
	dst := cv.NewMat(h, w, src.Channels)
	scaleX := float64(src.Cols) / float64(w)
	scaleY := float64(src.Rows) / float64(h)
	for y := 0; y < h; y++ {
		fy := (float64(y)+0.5)*scaleY - 0.5
		for x := 0; x < w; x++ {
			fx := (float64(x)+0.5)*scaleX - 0.5
			di := (y*w + x) * src.Channels
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = clampU8(sampleBorder(src, fx, fy, c, InterCubic, BorderReplicate, 0))
			}
		}
	}
	return dst
}

// resizeArea resizes src to w×h using pixel-area resampling. When shrinking it
// averages every source pixel overlapped by the destination pixel's footprint
// (weighted by the overlap area, matching cv::INTER_AREA); when enlarging along
// an axis it degrades to nearest-neighbour on that axis, as OpenCV does.
func resizeArea(src *cv.Mat, w, h int) *cv.Mat {
	dst := cv.NewMat(h, w, src.Channels)
	scaleX := float64(src.Cols) / float64(w)
	scaleY := float64(src.Rows) / float64(h)
	sums := make([]float64, src.Channels)
	for y := 0; y < h; y++ {
		sy0 := float64(y) * scaleY
		sy1 := sy0 + scaleY
		for x := 0; x < w; x++ {
			sx0 := float64(x) * scaleX
			sx1 := sx0 + scaleX
			for c := range sums {
				sums[c] = 0
			}
			area := areaAverage(src, sx0, sx1, sy0, sy1, sums)
			di := (y*w + x) * src.Channels
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = clampU8(sums[c] / area)
			}
		}
	}
	return dst
}

// areaAverage accumulates into sums the area-weighted sum of every source pixel
// overlapping the destination footprint [sx0,sx1)×[sy0,sy1) and returns the
// total covered area. Degenerate (enlarging) footprints collapse to the single
// nearest source pixel with unit area.
func areaAverage(src *cv.Mat, sx0, sx1, sy0, sy1 float64, sums []float64) float64 {
	ix0 := int(math.Floor(sx0))
	iy0 := int(math.Floor(sy0))
	ix1 := int(math.Ceil(sx1))
	iy1 := int(math.Ceil(sy1))
	var total float64
	for sy := iy0; sy < iy1; sy++ {
		cy := clampIndex(sy, src.Rows)
		wy := overlap(float64(sy), float64(sy+1), sy0, sy1)
		if wy <= 0 {
			continue
		}
		for sx := ix0; sx < ix1; sx++ {
			cx := clampIndex(sx, src.Cols)
			wx := overlap(float64(sx), float64(sx+1), sx0, sx1)
			if wx <= 0 {
				continue
			}
			wgt := wx * wy
			total += wgt
			base := (cy*src.Cols + cx) * src.Channels
			for c := 0; c < src.Channels; c++ {
				sums[c] += wgt * float64(src.Data[base+c])
			}
		}
	}
	if total == 0 {
		// Enlarging: sample the single nearest pixel.
		cx := clampIndex(int(math.Floor((sx0+sx1)/2)), src.Cols)
		cy := clampIndex(int(math.Floor((sy0+sy1)/2)), src.Rows)
		base := (cy*src.Cols + cx) * src.Channels
		for c := range sums {
			sums[c] = float64(src.Data[base+c])
		}
		return 1
	}
	return total
}

// overlap returns the length of the intersection of the intervals [a0,a1) and
// [b0,b1), clamped at zero.
func overlap(a0, a1, b0, b1 float64) float64 {
	lo := math.Max(a0, b0)
	hi := math.Min(a1, b1)
	if hi <= lo {
		return 0
	}
	return hi - lo
}

// clampIndex clamps p into [0, length).
func clampIndex(p, length int) int {
	if p < 0 {
		return 0
	}
	if p >= length {
		return length - 1
	}
	return p
}

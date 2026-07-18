package superres

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// ResampleKernel describes a separable 1-D reconstruction filter used by the
// resize routines. The same kernel is applied independently along the
// horizontal and vertical axes.
type ResampleKernel struct {
	// Name identifies the kernel for diagnostics.
	Name string
	// Radius is the support radius of the kernel in source pixels: Weight is
	// assumed to be zero for |x| >= Radius.
	Radius float64
	// Weight returns the filter weight for a sample at signed distance x from
	// the reconstruction point.
	Weight func(x float64) float64
}

// Sinc returns the normalised sinc function sin(pi*x)/(pi*x), with Sinc(0)=1.
// It is the ideal (but infinite-support) reconstruction filter and the basis
// of the Lanczos kernels.
func Sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

// NearestKernel returns the nearest-neighbour kernel: a unit box of radius
// one half. Resampling with it selects the closest source sample.
func NearestKernel() ResampleKernel {
	return ResampleKernel{
		Name:   "nearest",
		Radius: 0.5,
		Weight: func(x float64) float64 {
			if x > -0.5 && x <= 0.5 {
				return 1
			}
			return 0
		},
	}
}

// LinearKernel returns the linear (triangle / tent) kernel of radius one,
// producing bilinear interpolation when applied separably.
func LinearKernel() ResampleKernel {
	return ResampleKernel{
		Name:   "linear",
		Radius: 1,
		Weight: func(x float64) float64 {
			x = math.Abs(x)
			if x < 1 {
				return 1 - x
			}
			return 0
		},
	}
}

// CubicKernel returns the parametric cubic convolution kernel of Keys with
// free parameter a (radius two). a = -0.5 gives the interpolating
// Catmull-Rom spline (OpenCV's INTER_CUBIC); a = -0.75 matches some other
// libraries. More negative a increases edge overshoot.
func CubicKernel(a float64) ResampleKernel {
	return ResampleKernel{
		Name:   fmt.Sprintf("cubic(a=%g)", a),
		Radius: 2,
		Weight: func(x float64) float64 {
			x = math.Abs(x)
			switch {
			case x < 1:
				return ((a+2)*x-(a+3))*x*x + 1
			case x < 2:
				return ((a*x-5*a)*x+8*a)*x - 4*a
			default:
				return 0
			}
		},
	}
}

// CatmullRomKernel returns the Catmull-Rom cubic (CubicKernel with a = -0.5),
// the standard interpolating bicubic used for photographic upscaling.
func CatmullRomKernel() ResampleKernel {
	k := CubicKernel(-0.5)
	k.Name = "catmull-rom"
	return k
}

// MitchellKernel returns the Mitchell-Netravali cubic with parameters b and c
// (radius two). The classic Mitchell filter uses b = c = 1/3, trading a little
// blur for greatly reduced ringing versus Catmull-Rom.
func MitchellKernel(b, c float64) ResampleKernel {
	return ResampleKernel{
		Name:   fmt.Sprintf("mitchell(b=%g,c=%g)", b, c),
		Radius: 2,
		Weight: func(x float64) float64 {
			x = math.Abs(x)
			switch {
			case x < 1:
				return ((12-9*b-6*c)*x*x*x + (-18+12*b+6*c)*x*x + (6 - 2*b)) / 6
			case x < 2:
				return ((-b-6*c)*x*x*x + (6*b+30*c)*x*x + (-12*b-48*c)*x + (8*b + 24*c)) / 6
			default:
				return 0
			}
		},
	}
}

// BSplineKernel returns the cubic B-spline kernel (radius two). Unlike the
// interpolating cubics it approximates rather than passes through the source
// samples, giving a very smooth, ring-free result — useful as a prefilter or
// when a soft enlargement is wanted.
func BSplineKernel() ResampleKernel {
	return ResampleKernel{
		Name:   "b-spline",
		Radius: 2,
		Weight: func(x float64) float64 {
			x = math.Abs(x)
			switch {
			case x < 1:
				return (3*x*x*x - 6*x*x + 4) / 6
			case x < 2:
				t := 2 - x
				return t * t * t / 6
			default:
				return 0
			}
		},
	}
}

// LanczosKernel returns the Lanczos windowed-sinc kernel with lobe count a
// (radius a). a must be a positive integer; a = 2 and a = 3 are the common
// choices, with a = 3 the sharpest general-purpose interpolator here.
func LanczosKernel(a int) ResampleKernel {
	if a < 1 {
		panic("superres: LanczosKernel requires a >= 1")
	}
	af := float64(a)
	return ResampleKernel{
		Name:   fmt.Sprintf("lanczos%d", a),
		Radius: af,
		Weight: func(x float64) float64 {
			if x <= -af || x >= af {
				return 0
			}
			return Sinc(x) * Sinc(x/af)
		},
	}
}

// Interpolation names a resize method for the convenience [Resize] entry
// point.
type Interpolation int

const (
	// InterpNearest selects nearest-neighbour resampling.
	InterpNearest Interpolation = iota
	// InterpBilinear selects bilinear (separable linear) resampling.
	InterpBilinear
	// InterpBicubic selects Catmull-Rom bicubic resampling.
	InterpBicubic
	// InterpMitchell selects Mitchell-Netravali (b=c=1/3) resampling.
	InterpMitchell
	// InterpBSpline selects cubic B-spline resampling.
	InterpBSpline
	// InterpLanczos2 selects 2-lobe Lanczos resampling.
	InterpLanczos2
	// InterpLanczos3 selects 3-lobe Lanczos resampling.
	InterpLanczos3
)

// kernelFor maps an Interpolation constant to its ResampleKernel.
func kernelFor(interp Interpolation) ResampleKernel {
	switch interp {
	case InterpNearest:
		return NearestKernel()
	case InterpBilinear:
		return LinearKernel()
	case InterpBicubic:
		return CatmullRomKernel()
	case InterpMitchell:
		return MitchellKernel(1.0/3.0, 1.0/3.0)
	case InterpBSpline:
		return BSplineKernel()
	case InterpLanczos2:
		return LanczosKernel(2)
	case InterpLanczos3:
		return LanczosKernel(3)
	default:
		panic(fmt.Sprintf("superres: unknown interpolation %d", interp))
	}
}

// superresTap is one source-sample contribution to a destination sample.
type superresTap struct {
	index  int
	weight float64
}

// superresAxisWeights precomputes, for every destination index along one axis,
// the list of source taps and weights for resampling srcSize samples to
// dstSize samples with kernel k. When downscaling (dstSize < srcSize) the
// kernel is widened by the scale factor so it acts as an anti-aliasing filter;
// weights are always normalised to sum to one, which both handles the image
// borders (by replication) and guarantees a constant image is reproduced
// exactly.
func superresAxisWeights(dstSize, srcSize int, k ResampleKernel) [][]superresTap {
	scale := float64(srcSize) / float64(dstSize)
	filterScale := scale
	if filterScale < 1 {
		filterScale = 1
	}
	support := k.Radius * filterScale
	out := make([][]superresTap, dstSize)
	for d := 0; d < dstSize; d++ {
		center := (float64(d)+0.5)*scale - 0.5
		left := int(math.Ceil(center - support))
		right := int(math.Floor(center + support))
		taps := make([]superresTap, 0, right-left+1)
		var sum float64
		for i := left; i <= right; i++ {
			w := k.Weight((float64(i) - center) / filterScale)
			if w == 0 {
				continue
			}
			idx := superresClampInt(i, 0, srcSize-1)
			taps = append(taps, superresTap{index: idx, weight: w})
			sum += w
		}
		if sum == 0 {
			// Degenerate (should not happen): fall back to nearest sample.
			taps = []superresTap{{index: superresClampInt(int(math.Round(center)), 0, srcSize-1), weight: 1}}
			sum = 1
		}
		for i := range taps {
			taps[i].weight /= sum
		}
		out[d] = taps
	}
	return out
}

// ResizeKernel resizes src to width×height using the given separable kernel.
// Each channel is resampled independently in full precision (horizontal pass
// then vertical pass) and only the final result is rounded to 8-bit. It
// panics if width or height is not positive or src is empty.
func ResizeKernel(src *cv.Mat, width, height int, k ResampleKernel) *cv.Mat {
	if src.Empty() {
		panic("superres: ResizeKernel on empty Mat")
	}
	if width <= 0 || height <= 0 {
		panic("superres: ResizeKernel requires positive width and height")
	}
	ch := src.Channels
	// Horizontal pass: src.Cols -> width, into a float intermediate of shape
	// src.Rows × width × ch.
	xw := superresAxisWeights(width, src.Cols, k)
	tmp := make([]float64, src.Rows*width*ch)
	for y := 0; y < src.Rows; y++ {
		rowBase := y * src.Cols
		for dx := 0; dx < width; dx++ {
			taps := xw[dx]
			for c := 0; c < ch; c++ {
				var acc float64
				for _, t := range taps {
					acc += t.weight * float64(src.Data[(rowBase+t.index)*ch+c])
				}
				tmp[(y*width+dx)*ch+c] = acc
			}
		}
	}
	// Vertical pass: src.Rows -> height.
	yw := superresAxisWeights(height, src.Rows, k)
	dst := cv.NewMat(height, width, ch)
	for dy := 0; dy < height; dy++ {
		taps := yw[dy]
		for x := 0; x < width; x++ {
			for c := 0; c < ch; c++ {
				var acc float64
				for _, t := range taps {
					acc += t.weight * tmp[(t.index*width+x)*ch+c]
				}
				dst.Data[(dy*width+x)*ch+c] = superresClamp8(acc)
			}
		}
	}
	return dst
}

// Resize resizes src to width×height using the named interpolation. It is the
// high-level entry point; the specific kernel wrappers below and
// [ResizeKernel] offer finer control.
func Resize(src *cv.Mat, width, height int, interp Interpolation) *cv.Mat {
	return ResizeKernel(src, width, height, kernelFor(interp))
}

// ResizeScale resizes src by an isotropic floating-point factor using the
// named interpolation. A factor above one enlarges, below one shrinks. The
// output dimensions are rounded to the nearest integer and clamped to at
// least one. It panics if factor is not positive.
func ResizeScale(src *cv.Mat, factor float64, interp Interpolation) *cv.Mat {
	if factor <= 0 {
		panic("superres: ResizeScale requires a positive factor")
	}
	w := int(math.Round(float64(src.Cols) * factor))
	h := int(math.Round(float64(src.Rows) * factor))
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	return Resize(src, w, h, interp)
}

// NearestResize resizes src to width×height by nearest-neighbour sampling.
func NearestResize(src *cv.Mat, width, height int) *cv.Mat {
	return ResizeKernel(src, width, height, NearestKernel())
}

// BilinearResize resizes src to width×height by bilinear interpolation.
func BilinearResize(src *cv.Mat, width, height int) *cv.Mat {
	return ResizeKernel(src, width, height, LinearKernel())
}

// BicubicResize resizes src to width×height using the Catmull-Rom bicubic
// kernel, the recommended general-purpose interpolator for photographic
// enlargement.
func BicubicResize(src *cv.Mat, width, height int) *cv.Mat {
	return ResizeKernel(src, width, height, CatmullRomKernel())
}

// LanczosResize resizes src to width×height using an a-lobe Lanczos kernel.
// a = 3 is the sharpest general choice; a = 2 rings less.
func LanczosResize(src *cv.Mat, width, height, a int) *cv.Mat {
	return ResizeKernel(src, width, height, LanczosKernel(a))
}

// SplineResize resizes src to width×height using the smooth cubic B-spline
// kernel, giving a ring-free (slightly soft) result.
func SplineResize(src *cv.Mat, width, height int) *cv.Mat {
	return ResizeKernel(src, width, height, BSplineKernel())
}

// MitchellResize resizes src to width×height using the Mitchell-Netravali
// kernel (b = c = 1/3), a balanced compromise between sharpness and ringing.
func MitchellResize(src *cv.Mat, width, height int) *cv.Mat {
	return ResizeKernel(src, width, height, MitchellKernel(1.0/3.0, 1.0/3.0))
}

package cudawarping

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// WarpPolar remaps the GpuMat between Cartesian and polar coordinates about
// center, mirroring cv::warpPolar / cv::cuda's polar helpers. dsize gives the
// output size (width dsize.X, height dsize.Y).
//
// flags is an [Interpolation] optionally OR-ed with [WarpPolarLog] and/or
// [WarpInverseMap]:
//
//   - Forward (WarpInverseMap clear): the source is Cartesian and the result is
//     polar. Columns index radius and rows index angle over [0, 2π); with
//     [WarpPolarLinear] the radial axis is linear (ρ = maxRadius·col/width) and
//     with [WarpPolarLog] it is logarithmic (ρ = maxRadius^(col/width), so col 0
//     samples radius 1 and the last column samples maxRadius).
//   - Inverse (WarpInverseMap set): the source is a polar image of that layout
//     and the result is the reconstructed Cartesian image.
//
// Because rotation and scaling about center become translations in the polar
// domain, the forward transform is a classic registration pre-processing step.
// Out-of-image samples are treated as zero and the radial edge is replicated on
// the inverse pass. The stream argument is ignored. It panics on an empty
// GpuMat, a non-positive dsize, or maxRadius ≤ 0 (≤ 1 for the log mode).
func (g *GpuMat) WarpPolar(dsize image.Point, center Point2f, maxRadius float64, flags int, stream *Stream) *GpuMat {
	src := g.host("WarpPolar")
	checkSize(dsize, "WarpPolar")
	interp := Interpolation(flags & interMask)
	logMode := flags&WarpPolarLog != 0
	inverse := flags&WarpInverseMap != 0
	if logMode {
		if maxRadius <= 1 {
			panic("cudawarping: WarpPolar log mode requires maxRadius > 1")
		}
	} else if maxRadius <= 0 {
		panic("cudawarping: WarpPolar requires maxRadius > 0")
	}
	if inverse {
		return &GpuMat{mat: polarInverse(src, dsize, center, maxRadius, logMode, interp)}
	}
	return &GpuMat{mat: polarForward(src, dsize, center, maxRadius, logMode, interp)}
}

// LinearPolar remaps the GpuMat into (radius, angle) space about center with a
// linear radial axis, the convenience form of [GpuMat.WarpPolar] with
// [WarpPolarLinear]. It mirrors cv::linearPolar. flags may add [WarpInverseMap]
// (to reconstruct the Cartesian image) and an [Interpolation]; the log bit, if
// present, is ignored. It panics on an empty GpuMat, a non-positive dsize or
// maxRadius ≤ 0.
func (g *GpuMat) LinearPolar(dsize image.Point, center Point2f, maxRadius float64, flags int, stream *Stream) *GpuMat {
	return g.WarpPolar(dsize, center, maxRadius, (flags &^ WarpPolarLog), stream)
}

// LogPolar remaps the GpuMat into (log-radius, angle) space about center, the
// convenience form of [GpuMat.WarpPolar] with [WarpPolarLog]. It mirrors
// cv::logPolar. In the log-polar domain both rotation and uniform scaling about
// center become translations, which is why it underpins scale/rotation-invariant
// matching. flags may add [WarpInverseMap] and an [Interpolation]. It panics on
// an empty GpuMat, a non-positive dsize or maxRadius ≤ 1.
func (g *GpuMat) LogPolar(dsize image.Point, center Point2f, maxRadius float64, flags int, stream *Stream) *GpuMat {
	return g.WarpPolar(dsize, center, maxRadius, flags|WarpPolarLog, stream)
}

// radiusForColumn returns the source radius sampled by output column col of a
// width-wide polar image, for either the linear or logarithmic radial axis.
func radiusForColumn(col, width int, maxRadius float64, logMode bool) float64 {
	if logMode {
		return math.Pow(maxRadius, float64(col)/float64(width))
	}
	return maxRadius * float64(col) / float64(width)
}

// polarForward builds the polar image (columns index radius, rows index angle)
// by sampling the Cartesian source at the mapped coordinate of each output pixel.
func polarForward(src *cv.Mat, dsize image.Point, center Point2f, maxRadius float64, logMode bool, interp Interpolation) *cv.Mat {
	dst := cv.NewMat(dsize.Y, dsize.X, src.Channels)
	for row := 0; row < dsize.Y; row++ {
		angle := 2 * math.Pi * float64(row) / float64(dsize.Y)
		ca := math.Cos(angle)
		sa := math.Sin(angle)
		for col := 0; col < dsize.X; col++ {
			rho := radiusForColumn(col, dsize.X, maxRadius, logMode)
			fx := center.X + rho*ca
			fy := center.Y + rho*sa
			di := (row*dst.Cols + col) * dst.Channels
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = clampU8(sampleBorder(src, fx, fy, c, interp, BorderConstant, 0))
			}
		}
	}
	return dst
}

// polarInverse reconstructs a Cartesian image of size dsize from the polar
// source, the inverse of polarForward. For each Cartesian pixel it computes the
// radius and angle relative to center and samples the polar image at the
// corresponding (radius column, angle row), replicating the border.
func polarInverse(src *cv.Mat, dsize image.Point, center Point2f, maxRadius float64, logMode bool, interp Interpolation) *cv.Mat {
	dst := cv.NewMat(dsize.Y, dsize.X, src.Channels)
	width := float64(src.Cols)
	height := float64(src.Rows)
	for y := 0; y < dsize.Y; y++ {
		dy := float64(y) - center.Y
		for x := 0; x < dsize.X; x++ {
			dx := float64(x) - center.X
			magnitude := math.Hypot(dx, dy)
			angle := math.Atan2(dy, dx)
			if angle < 0 {
				angle += 2 * math.Pi
			}
			var col float64
			if logMode {
				if magnitude < 1 {
					magnitude = 1
				}
				col = math.Log(magnitude) / math.Log(maxRadius) * width
			} else {
				col = magnitude / maxRadius * width
			}
			rowf := angle / (2 * math.Pi) * height
			di := (y*dst.Cols + x) * dst.Channels
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = clampU8(sampleBorder(src, col, rowf, c, interp, BorderReplicate, 0))
			}
		}
	}
	return dst
}

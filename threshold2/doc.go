// Package threshold2 implements image thresholding and binarization
// algorithms on top of the parent package's [cv.Mat] image type.
//
// The package is a pure-Go, standard-library-only companion to the core
// opencv module. It never allocates its own image type: every routine that
// produces an image returns a [cv.Mat], and every routine that consumes one
// accepts a [cv.Mat], so results interoperate directly with the rest of the
// library (drawing, morphology, I/O, and so on).
//
// The functionality is organised in a few groups:
//
//   - Histogram tools: [ComputeHistogram] and the [Histogram] type.
//   - Global threshold estimators that return a single grey level in [0,255]:
//     [OtsuThreshold], [TriangleThreshold], [MeanThreshold], [MedianThreshold],
//     [IsoDataThreshold], [LiThreshold], [KapurThreshold], [YenThreshold],
//     [PercentileThreshold], [MomentsThreshold], [MinimumThreshold],
//     [IntermodesThreshold] and [KittlerThreshold].
//   - Binarizers that pair with each estimator and return a black/white mask:
//     [Otsu], [Triangle], [Mean], [Median], [IsoData], [Li], [Kapur], [Yen],
//     [Percentile], [Moments], [Minimum], [Intermodes] and [Kittler].
//   - Multi-level and two-dimensional variants: [MultiOtsu],
//     [MultiOtsuQuantize], [Otsu2DThreshold], [Otsu2D], [MultiKapur] and
//     [MultiKapurQuantize].
//   - Local (adaptive) methods: [AdaptiveMean], [AdaptiveGaussian],
//     [AdaptiveMedian], [Sauvola], [Niblack], [Bernsen], [Wolf], [NICK] and
//     [Phansalkar].
//   - Utilities: [Hysteresis], [PerChannelOtsu], [PerChannelThreshold],
//     [InRange], [Binarize], [ToGray] and [ForegroundRatio].
//   - A dispatcher, [AutoThreshold]/[Auto], selecting any global estimator by
//     [Method].
//
// All routines are deterministic and CPU-only. Grey levels are 8-bit; colour
// input is reduced to luma with the Rec. 601 weights unless a routine states
// otherwise.
package threshold2

import (
	"errors"
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// ErrEmpty is returned by routines given a nil or zero-sized [cv.Mat].
var ErrEmpty = errors.New("threshold2: empty image")

// Polarity selects which side of a threshold is treated as foreground
// (rendered as 255) when a routine binarizes an image.
type Polarity int

const (
	// ObjectBright treats samples strictly greater than the threshold as
	// foreground. Use it when the object of interest is brighter than the
	// background.
	ObjectBright Polarity = iota
	// ObjectDark treats samples less than or equal to the threshold as
	// foreground. Use it when the object of interest is darker than the
	// background, as with dark text on a light page.
	ObjectDark
)

// threshold2gray reduces src to a single 8-bit sample per pixel. A
// single-channel Mat is returned as its underlying data; a three-channel Mat
// is converted to luma with the Rec. 601 weights; any other channel count
// falls back to the first channel.
func threshold2gray(src *cv.Mat) ([]uint8, int, int, error) {
	if src.Empty() {
		return nil, 0, 0, ErrEmpty
	}
	rows, cols := src.Rows, src.Cols
	if src.Channels == 1 {
		out := make([]uint8, rows*cols)
		copy(out, src.Data)
		return out, rows, cols, nil
	}
	out := make([]uint8, rows*cols)
	ch := src.Channels
	if ch == 3 {
		for p := 0; p < rows*cols; p++ {
			i := p * 3
			r := int(src.Data[i])
			g := int(src.Data[i+1])
			b := int(src.Data[i+2])
			out[p] = uint8((77*r + 150*g + 29*b) >> 8)
		}
		return out, rows, cols, nil
	}
	for p := 0; p < rows*cols; p++ {
		out[p] = src.Data[p*ch]
	}
	return out, rows, cols, nil
}

// ToGray returns a fresh single-channel [cv.Mat] holding the luma of src.
// A single-channel input is copied unchanged; a three-channel input is
// converted with the Rec. 601 weights (0.299R + 0.587G + 0.114B). It returns
// [ErrEmpty] if src has no samples.
func ToGray(src *cv.Mat) (*cv.Mat, error) {
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	dst := cv.NewMat(rows, cols, 1)
	copy(dst.Data, gray)
	return dst, nil
}

// threshold2binarize maps grey samples to a single-channel mask (0 or 255)
// according to the threshold and polarity.
func threshold2binarize(gray []uint8, rows, cols, thresh int, p Polarity) *cv.Mat {
	dst := cv.NewMat(rows, cols, 1)
	for i, v := range gray {
		fg := int(v) > thresh
		if p == ObjectDark {
			fg = int(v) <= thresh
		}
		if fg {
			dst.Data[i] = 255
		}
	}
	return dst
}

// Binarize thresholds src at the given grey level and returns a
// single-channel mask in which foreground pixels (selected by p) are 255 and
// the rest are 0. Colour input is reduced to luma first. The threshold is
// clamped to [0,255].
func Binarize(src *cv.Mat, thresh int, p Polarity) (*cv.Mat, error) {
	gray, rows, cols, err := threshold2gray(src)
	if err != nil {
		return nil, err
	}
	if thresh < 0 {
		thresh = 0
	} else if thresh > 255 {
		thresh = 255
	}
	return threshold2binarize(gray, rows, cols, thresh, p), nil
}

// ForegroundRatio reports the fraction of pixels of src that would be labelled
// foreground when thresholded at thresh with polarity p. The result lies in
// [0,1]. Colour input is reduced to luma first.
func ForegroundRatio(src *cv.Mat, thresh int, p Polarity) (float64, error) {
	gray, _, _, err := threshold2gray(src)
	if err != nil {
		return 0, err
	}
	fg := 0
	for _, v := range gray {
		isFg := int(v) > thresh
		if p == ObjectDark {
			isFg = int(v) <= thresh
		}
		if isFg {
			fg++
		}
	}
	return float64(fg) / float64(len(gray)), nil
}

// Method identifies a global threshold-estimation algorithm for the
// [AutoThreshold] and [Auto] dispatchers.
type Method int

const (
	// MethodOtsu selects [OtsuThreshold].
	MethodOtsu Method = iota
	// MethodTriangle selects [TriangleThreshold].
	MethodTriangle
	// MethodMean selects [MeanThreshold].
	MethodMean
	// MethodMedian selects [MedianThreshold].
	MethodMedian
	// MethodIsoData selects [IsoDataThreshold].
	MethodIsoData
	// MethodLi selects [LiThreshold].
	MethodLi
	// MethodKapur selects [KapurThreshold].
	MethodKapur
	// MethodYen selects [YenThreshold].
	MethodYen
	// MethodPercentile selects [PercentileThreshold] with fraction 0.5.
	MethodPercentile
	// MethodMoments selects [MomentsThreshold].
	MethodMoments
	// MethodMinimum selects [MinimumThreshold].
	MethodMinimum
	// MethodIntermodes selects [IntermodesThreshold].
	MethodIntermodes
	// MethodKittler selects [KittlerThreshold].
	MethodKittler
)

// String returns the lowercase name of the method.
func (m Method) String() string {
	switch m {
	case MethodOtsu:
		return "otsu"
	case MethodTriangle:
		return "triangle"
	case MethodMean:
		return "mean"
	case MethodMedian:
		return "median"
	case MethodIsoData:
		return "isodata"
	case MethodLi:
		return "li"
	case MethodKapur:
		return "kapur"
	case MethodYen:
		return "yen"
	case MethodPercentile:
		return "percentile"
	case MethodMoments:
		return "moments"
	case MethodMinimum:
		return "minimum"
	case MethodIntermodes:
		return "intermodes"
	case MethodKittler:
		return "kittler"
	default:
		return fmt.Sprintf("method(%d)", int(m))
	}
}

// AutoThreshold computes a global threshold for src using the estimator named
// by m and returns the grey level in [0,255]. It reduces colour input to luma.
func AutoThreshold(src *cv.Mat, m Method) (int, error) {
	switch m {
	case MethodOtsu:
		return OtsuThreshold(src)
	case MethodTriangle:
		return TriangleThreshold(src)
	case MethodMean:
		return MeanThreshold(src)
	case MethodMedian:
		return MedianThreshold(src)
	case MethodIsoData:
		return IsoDataThreshold(src)
	case MethodLi:
		return LiThreshold(src)
	case MethodKapur:
		return KapurThreshold(src)
	case MethodYen:
		return YenThreshold(src)
	case MethodPercentile:
		return PercentileThreshold(src, 0.5)
	case MethodMoments:
		return MomentsThreshold(src)
	case MethodMinimum:
		return MinimumThreshold(src)
	case MethodIntermodes:
		return IntermodesThreshold(src)
	case MethodKittler:
		return KittlerThreshold(src)
	default:
		return 0, fmt.Errorf("threshold2: unknown method %d", int(m))
	}
}

// Auto computes a global threshold for src with the estimator named by m and
// returns the binarized mask (foreground selected by p) together with the grey
// level that was used.
func Auto(src *cv.Mat, m Method, p Polarity) (*cv.Mat, int, error) {
	t, err := AutoThreshold(src, m)
	if err != nil {
		return nil, 0, err
	}
	dst, err := Binarize(src, t, p)
	if err != nil {
		return nil, 0, err
	}
	return dst, t, nil
}

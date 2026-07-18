// Package textdet implements classical (non-neural) text-detection and
// document-analysis primitives on top of the parent module's [cv.Mat] image
// type.
//
// The package is a pure-Go, standard-library-only companion to the core opencv
// module. It never defines its own image type: every routine that produces an
// image returns a [cv.Mat], every routine that consumes one accepts a
// [cv.Mat], and geometry is expressed with the parent package's [cv.Rect] and
// [cv.Point], so results interoperate directly with the rest of the library
// (drawing, morphology, thresholding, I/O and so on).
//
// The functionality is organised in a few groups:
//
//   - Binarization for OCR (binarize.go): [Otsu], [OtsuThreshold], [Sauvola],
//     [Niblack], [Wolf], [Bernsen], [AdaptiveMean], [Binarize],
//     [ForegroundRatio] and the [IntegralImage] summed-area helper.
//   - Stroke-Width Transform (swt.go): [StrokeWidthTransform], the [SWTResult]
//     map, letter-candidate extraction with [SWTLetters], and the [SWTOptions]
//     and [SWTPolarity] configuration.
//   - MSER text regions (mser.go): [DetectMSER], the [MSERRegion] type,
//     text-shape filtering with [FilterTextRegions], and [MSEROptions].
//   - Connected-component analysis and grouping (components.go):
//     [LabelComponents], the [ComponentSet] and [Component] types, size and
//     shape filters, edge-density text localization with [EdgeDensityMap] and
//     [LocalizeByEdgeDensity], and line grouping with [GroupTextLines].
//   - Projection profiles, segmentation and skew (projection.go):
//     [HorizontalProjection], [VerticalProjection], [SegmentBands],
//     [SegmentTextLines], [SegmentWords], [EstimateSkew] and [CorrectSkew].
//
// All routines are deterministic and CPU-only. Grey levels are 8-bit; colour
// input is reduced to luma with the Rec. 601 weights (0.299R + 0.587G +
// 0.114B) unless a routine states otherwise.
package textdet

import (
	"errors"

	cv "github.com/malcolmston/opencv"
)

// ErrEmpty is returned by routines given a nil or zero-sized [cv.Mat].
var ErrEmpty = errors.New("textdet: empty image")

// ErrInvalidArgument is returned when a caller passes an out-of-range or
// otherwise inconsistent parameter, such as a non-positive window size.
var ErrInvalidArgument = errors.New("textdet: invalid argument")

// textdetGray reduces src to one 8-bit sample per pixel in row-major order.
// A single-channel Mat is copied; a three-channel Mat is converted to luma
// with the Rec. 601 weights; any other channel count falls back to channel 0.
func textdetGray(src *cv.Mat) (gray []uint8, rows, cols int, err error) {
	if src.Empty() {
		return nil, 0, 0, ErrEmpty
	}
	rows, cols = src.Rows, src.Cols
	out := make([]uint8, rows*cols)
	ch := src.Channels
	switch ch {
	case 1:
		copy(out, src.Data)
	case 3:
		for p := 0; p < rows*cols; p++ {
			i := p * 3
			r := int(src.Data[i])
			g := int(src.Data[i+1])
			b := int(src.Data[i+2])
			out[p] = uint8((77*r + 150*g + 29*b) >> 8)
		}
	default:
		for p := 0; p < rows*cols; p++ {
			out[p] = src.Data[p*ch]
		}
	}
	return out, rows, cols, nil
}

// Gray returns a fresh single-channel [cv.Mat] holding the luma of src. A
// single-channel input is copied unchanged; a three-channel input is converted
// with the Rec. 601 weights. It returns [ErrEmpty] if src has no samples.
func Gray(src *cv.Mat) (*cv.Mat, error) {
	g, rows, cols, err := textdetGray(src)
	if err != nil {
		return nil, err
	}
	dst := cv.NewMat(rows, cols, 1)
	copy(dst.Data, g)
	return dst, nil
}

// textdetForeground returns a boolean foreground mask from a single-channel
// binary Mat: any non-zero sample is foreground. rows and cols are returned for
// convenience.
func textdetForeground(binary *cv.Mat) (fg []bool, rows, cols int, err error) {
	if binary.Empty() {
		return nil, 0, 0, ErrEmpty
	}
	rows, cols = binary.Rows, binary.Cols
	ch := binary.Channels
	fg = make([]bool, rows*cols)
	for p := 0; p < rows*cols; p++ {
		if binary.Data[p*ch] != 0 {
			fg[p] = true
		}
	}
	return fg, rows, cols, nil
}

// textdetMaskFromBool builds a single-channel 0/255 mask from a boolean slice.
func textdetMaskFromBool(fg []bool, rows, cols int) *cv.Mat {
	dst := cv.NewMat(rows, cols, 1)
	for i, v := range fg {
		if v {
			dst.Data[i] = 255
		}
	}
	return dst
}

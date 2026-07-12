package gapi

import (
	cv "github.com/malcolmston/opencv"
)

// Re-exported enum types from the root package, so callers building graphs need
// not import cv directly for the common parameter types.
type (
	// ThresholdType selects the behaviour of [Threshold]; see [cv.ThresholdType].
	ThresholdType = cv.ThresholdType
	// ColorConversionCode selects a conversion for [CvtColor]; see
	// [cv.ColorConversionCode].
	ColorConversionCode = cv.ColorConversionCode
	// InterpolationFlag selects the resampling used by [Resize]; see
	// [cv.InterpolationFlag].
	InterpolationFlag = cv.InterpolationFlag
	// FlipCode selects the axis for [Flip]; see [cv.FlipCode].
	FlipCode = cv.FlipCode
	// MorphShape selects a structuring-element shape for [Dilate] and [Erode];
	// see [cv.MorphShape].
	MorphShape = cv.MorphShape
)

// Re-exported enum constants for the parameter types above.
const (
	ThreshBinary    = cv.ThreshBinary
	ThreshBinaryInv = cv.ThreshBinaryInv
	ThreshTrunc     = cv.ThreshTrunc
	ThreshToZero    = cv.ThreshToZero
	ThreshToZeroInv = cv.ThreshToZeroInv
	ThreshOtsu      = cv.ThreshOtsu

	ColorRGB2Gray = cv.ColorRGB2Gray
	ColorBGR2Gray = cv.ColorBGR2Gray
	ColorGray2RGB = cv.ColorGray2RGB
	ColorRGB2BGR  = cv.ColorRGB2BGR
	ColorBGR2RGB  = cv.ColorBGR2RGB
	ColorRGB2HSV  = cv.ColorRGB2HSV
	ColorHSV2RGB  = cv.ColorHSV2RGB

	InterNearest = cv.InterNearest
	InterLinear  = cv.InterLinear

	FlipVertical   = cv.FlipVertical
	FlipHorizontal = cv.FlipHorizontal
	FlipBoth       = cv.FlipBoth

	MorphRect    = cv.MorphRect
	MorphCross   = cv.MorphCross
	MorphEllipse = cv.MorphEllipse
)

// Operation names assigned to the built-in graph operations. Use these as the
// [GKernel].Op value to override an operation's implementation at compile time.
const (
	OpAdd          = "add"
	OpSub          = "sub"
	OpMul          = "mul"
	OpDiv          = "div"
	OpAddWeighted  = "addWeighted"
	OpAddC         = "addC"
	OpMulC         = "mulC"
	OpMask         = "mask"
	OpMerge3       = "merge3"
	OpSplit3       = "split3"
	OpBitwiseAnd   = "bitwiseAnd"
	OpBitwiseOr    = "bitwiseOr"
	OpBitwiseXor   = "bitwiseXor"
	OpBitwiseNot   = "bitwiseNot"
	OpCmp          = "cmp"
	OpThreshold    = "threshold"
	OpResize       = "resize"
	OpFlip         = "flip"
	OpTranspose    = "transpose"
	OpNormalize    = "normalize"
	OpCvtColor     = "cvtColor"
	OpBlur         = "blur"
	OpGaussianBlur = "gaussianBlur"
	OpMedianBlur   = "medianBlur"
	OpSobel        = "sobel"
	OpLaplacian    = "laplacian"
	OpCanny        = "canny"
	OpDilate       = "dilate"
	OpErode        = "erode"
	OpEqualizeHist = "equalizeHist"
)

// CmpOp selects the per-sample comparison performed by [Cmp].
type CmpOp int

const (
	// CmpOpEQ marks samples where a == b.
	CmpOpEQ CmpOp = iota
	// CmpOpGT marks samples where a > b.
	CmpOpGT
	// CmpOpGE marks samples where a >= b.
	CmpOpGE
	// CmpOpLT marks samples where a < b.
	CmpOpLT
	// CmpOpLE marks samples where a <= b.
	CmpOpLE
	// CmpOpNE marks samples where a != b.
	CmpOpNE
)

// Add returns a lazy node computing the saturating per-sample sum of a and b.
func Add(a, b GMat) GMat {
	return newOp(OpAdd, []GMat{a, b}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Add(ctx.Mats[0], ctx.Mats[1])
	})
}

// Sub returns a lazy node computing the saturating per-sample difference a-b.
func Sub(a, b GMat) GMat {
	return newOp(OpSub, []GMat{a, b}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Subtract(ctx.Mats[0], ctx.Mats[1])
	})
}

// Mul returns a lazy node computing the per-sample product a*b*scale.
func Mul(a, b GMat, scale float64) GMat {
	return newOp(OpMul, []GMat{a, b}, nil, nil, []float64{scale}, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Multiply(ctx.Mats[0], ctx.Mats[1], ctx.Floats[0])
	})
}

// Div returns a lazy node computing the per-sample quotient scale*a/b.
func Div(a, b GMat, scale float64) GMat {
	return newOp(OpDiv, []GMat{a, b}, nil, nil, []float64{scale}, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Divide(ctx.Mats[0], ctx.Mats[1], ctx.Floats[0])
	})
}

// AddWeighted returns a lazy node computing alpha*a + beta*b + gamma.
func AddWeighted(a GMat, alpha float64, b GMat, beta, gamma float64) GMat {
	return newOp(OpAddWeighted, []GMat{a, b}, nil, nil, []float64{alpha, beta, gamma}, nil, func(ctx KernelContext) *cv.Mat {
		return cv.AddWeighted(ctx.Mats[0], ctx.Floats[0], ctx.Mats[1], ctx.Floats[1], ctx.Floats[2])
	})
}

// AddC returns a lazy node that adds a run-time scalar to every sample of a,
// saturating to [0,255].
func AddC(a GMat, s GScalar) GMat {
	return newOp(OpAddC, []GMat{a}, []GScalar{s}, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		src := ctx.Mats[0]
		v := ctx.Scalars[0]
		out := cv.NewMat(src.Rows, src.Cols, src.Channels)
		for i, s := range src.Data {
			out.Data[i] = clampUint8(float64(s) + v + 0.5)
		}
		return out
	})
}

// MulC returns a lazy node that multiplies every sample of a by a run-time
// scalar, saturating to [0,255].
func MulC(a GMat, s GScalar) GMat {
	return newOp(OpMulC, []GMat{a}, []GScalar{s}, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		src := ctx.Mats[0]
		v := ctx.Scalars[0]
		out := cv.NewMat(src.Rows, src.Cols, src.Channels)
		for i, s := range src.Data {
			out.Data[i] = clampUint8(float64(s)*v + 0.5)
		}
		return out
	})
}

// Mask returns a lazy node that keeps samples of src where the single-channel
// mask is non-zero and zeroes them elsewhere. src and mask must share
// dimensions; the result has src's channel count.
func Mask(src, mask GMat) GMat {
	return newOp(OpMask, []GMat{src, mask}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		s := ctx.Mats[0]
		m := ctx.Mats[1]
		if m.Channels != 1 || s.Rows != m.Rows || s.Cols != m.Cols {
			panic("gapi: Mask requires a single-channel mask matching src dimensions")
		}
		out := cv.NewMat(s.Rows, s.Cols, s.Channels)
		for p := 0; p < s.Total(); p++ {
			if m.Data[p] == 0 {
				continue
			}
			base := p * s.Channels
			copy(out.Data[base:base+s.Channels], s.Data[base:base+s.Channels])
		}
		return out
	})
}

// Merge3 returns a lazy node interleaving three single-channel images into one
// three-channel image.
func Merge3(a, b, c GMat) GMat {
	return newOp(OpMerge3, []GMat{a, b, c}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Merge([]*cv.Mat{ctx.Mats[0], ctx.Mats[1], ctx.Mats[2]})
	})
}

// Split3 returns three lazy nodes, one per channel, extracting the planes of a
// three-channel image. It panics at run time if the input is not three-channel.
func Split3(in GMat) (GMat, GMat, GMat) {
	plane := func(idx int) GMat {
		return newOp(OpSplit3, []GMat{in}, nil, []int{idx}, nil, nil, func(ctx KernelContext) *cv.Mat {
			src := ctx.Mats[0]
			if src.Channels != 3 {
				panic("gapi: Split3 requires a three-channel input")
			}
			return src.Split()[ctx.Ints[0]]
		})
	}
	return plane(0), plane(1), plane(2)
}

// BitwiseAnd returns a lazy node computing the per-sample bitwise AND of a and b.
func BitwiseAnd(a, b GMat) GMat {
	return newOp(OpBitwiseAnd, []GMat{a, b}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.BitwiseAnd(ctx.Mats[0], ctx.Mats[1])
	})
}

// BitwiseOr returns a lazy node computing the per-sample bitwise OR of a and b.
func BitwiseOr(a, b GMat) GMat {
	return newOp(OpBitwiseOr, []GMat{a, b}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.BitwiseOr(ctx.Mats[0], ctx.Mats[1])
	})
}

// BitwiseXor returns a lazy node computing the per-sample bitwise XOR of a and b.
func BitwiseXor(a, b GMat) GMat {
	return newOp(OpBitwiseXor, []GMat{a, b}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.BitwiseXor(ctx.Mats[0], ctx.Mats[1])
	})
}

// BitwiseNot returns a lazy node computing the per-sample complement of src.
func BitwiseNot(src GMat) GMat {
	return newOp(OpBitwiseNot, []GMat{src}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.BitwiseNot(ctx.Mats[0])
	})
}

// Cmp returns a lazy node comparing a and b per sample under op, producing 255
// where the relation holds and 0 elsewhere, matching OpenCV's cv::compare. The
// inputs must have identical dimensions and channel counts.
func Cmp(a, b GMat, op CmpOp) GMat {
	return newOp(OpCmp, []GMat{a, b}, nil, []int{int(op)}, nil, nil, func(ctx KernelContext) *cv.Mat {
		x := ctx.Mats[0]
		y := ctx.Mats[1]
		if x.Rows != y.Rows || x.Cols != y.Cols || x.Channels != y.Channels {
			panic("gapi: Cmp shape mismatch")
		}
		out := cv.NewMat(x.Rows, x.Cols, x.Channels)
		cmp := CmpOp(ctx.Ints[0])
		for i := range x.Data {
			if cmpHolds(cmp, x.Data[i], y.Data[i]) {
				out.Data[i] = 255
			}
		}
		return out
	})
}

// CmpGT is Cmp with [CmpOpGT].
func CmpGT(a, b GMat) GMat { return Cmp(a, b, CmpOpGT) }

// CmpGE is Cmp with [CmpOpGE].
func CmpGE(a, b GMat) GMat { return Cmp(a, b, CmpOpGE) }

// CmpLT is Cmp with [CmpOpLT].
func CmpLT(a, b GMat) GMat { return Cmp(a, b, CmpOpLT) }

// CmpLE is Cmp with [CmpOpLE].
func CmpLE(a, b GMat) GMat { return Cmp(a, b, CmpOpLE) }

// CmpEQ is Cmp with [CmpOpEQ].
func CmpEQ(a, b GMat) GMat { return Cmp(a, b, CmpOpEQ) }

// CmpNE is Cmp with [CmpOpNE].
func CmpNE(a, b GMat) GMat { return Cmp(a, b, CmpOpNE) }

// Threshold returns a lazy node applying a fixed-level threshold to a
// single-channel image. See [cv.Threshold]; the auto-computed level (when
// [ThreshOtsu] is OR-ed into typ) is not surfaced by the graph.
func Threshold(src GMat, thresh, maxval float64, typ ThresholdType) GMat {
	return newOp(OpThreshold, []GMat{src}, nil, []int{int(typ)}, []float64{thresh, maxval}, nil, func(ctx KernelContext) *cv.Mat {
		out, _ := cv.Threshold(ctx.Mats[0], ctx.Floats[0], ctx.Floats[1], ThresholdType(ctx.Ints[0]))
		return out
	})
}

// Resize returns a lazy node scaling src to width×height using interp.
func Resize(src GMat, width, height int, interp InterpolationFlag) GMat {
	return newOp(OpResize, []GMat{src}, nil, []int{width, height, int(interp)}, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Resize(ctx.Mats[0], ctx.Ints[0], ctx.Ints[1], InterpolationFlag(ctx.Ints[2]))
	})
}

// Flip returns a lazy node mirroring src along the axis chosen by code.
func Flip(src GMat, code FlipCode) GMat {
	return newOp(OpFlip, []GMat{src}, nil, []int{int(code)}, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Flip(ctx.Mats[0], FlipCode(ctx.Ints[0]))
	})
}

// Transpose returns a lazy node swapping the rows and columns of src.
func Transpose(src GMat) GMat {
	return newOp(OpTranspose, []GMat{src}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Transpose(ctx.Mats[0])
	})
}

// Normalize returns a lazy node linearly rescaling src so its minimum maps to
// alpha and its maximum to beta (NORM_MINMAX).
func Normalize(src GMat, alpha, beta float64) GMat {
	return newOp(OpNormalize, []GMat{src}, nil, nil, []float64{alpha, beta}, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Normalize(ctx.Mats[0], ctx.Floats[0], ctx.Floats[1])
	})
}

// cmpHolds reports whether the comparison op holds for samples x and y.
func cmpHolds(op CmpOp, x, y uint8) bool {
	switch op {
	case CmpOpEQ:
		return x == y
	case CmpOpGT:
		return x > y
	case CmpOpGE:
		return x >= y
	case CmpOpLT:
		return x < y
	case CmpOpLE:
		return x <= y
	case CmpOpNE:
		return x != y
	default:
		panic("gapi: unknown CmpOp")
	}
}

// clampUint8 rounds toward zero after the caller adds any rounding bias and
// clamps into [0,255], matching the root package's saturation behaviour.
func clampUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

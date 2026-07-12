package cudaarithm

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// wrap builds a GpuMat around an already-owned *cv.Mat without copying.
func wrap(m *cv.Mat) *GpuMat {
	return &GpuMat{mat: m}
}

// Add returns the per-sample saturating sum a+b as a new GpuMat. It delegates to
// the root package's [cv.Add]. The optional stream is accepted for API
// compatibility and ignored (work is synchronous). Operands must share a shape.
func Add(a, b *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "Add")
	return wrap(cv.Add(a.mat, b.mat))
}

// Subtract returns the per-sample saturating difference a-b (floored at 0),
// delegating to [cv.Subtract]. Operands must share a shape.
func Subtract(a, b *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "Subtract")
	return wrap(cv.Subtract(a.mat, b.mat))
}

// Multiply returns the per-sample product a*b*scale, rounded and saturated,
// delegating to [cv.Multiply]. Operands must share a shape.
func Multiply(a, b *GpuMat, scale float64, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "Multiply")
	return wrap(cv.Multiply(a.mat, b.mat, scale))
}

// Divide returns the per-sample quotient scale*a/b (0 where b is 0), rounded and
// saturated, delegating to [cv.Divide]. Operands must share a shape.
func Divide(a, b *GpuMat, scale float64, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "Divide")
	return wrap(cv.Divide(a.mat, b.mat, scale))
}

// AbsDiff returns the per-sample absolute difference |a-b|, delegating to
// [cv.AbsDiff]. Operands must share a shape.
func AbsDiff(a, b *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "AbsDiff")
	return wrap(cv.AbsDiff(a.mat, b.mat))
}

// AddWeighted returns the per-sample weighted sum alpha*a + beta*b + gamma,
// rounded and saturated, delegating to [cv.AddWeighted]. Operands must share a
// shape.
func AddWeighted(a *GpuMat, alpha float64, b *GpuMat, beta, gamma float64, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "AddWeighted")
	return wrap(cv.AddWeighted(a.mat, alpha, b.mat, beta, gamma))
}

// BitwiseAnd returns the per-sample bitwise AND of a and b, delegating to
// [cv.BitwiseAnd]. Operands must share a shape.
func BitwiseAnd(a, b *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "BitwiseAnd")
	return wrap(cv.BitwiseAnd(a.mat, b.mat))
}

// BitwiseOr returns the per-sample bitwise OR of a and b, delegating to
// [cv.BitwiseOr]. Operands must share a shape.
func BitwiseOr(a, b *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "BitwiseOr")
	return wrap(cv.BitwiseOr(a.mat, b.mat))
}

// BitwiseXor returns the per-sample bitwise XOR of a and b, delegating to
// [cv.BitwiseXor]. Operands must share a shape.
func BitwiseXor(a, b *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "BitwiseXor")
	return wrap(cv.BitwiseXor(a.mat, b.mat))
}

// BitwiseNot returns the per-sample bitwise complement (255-value) of src,
// delegating to [cv.BitwiseNot].
func BitwiseNot(src *GpuMat, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "BitwiseNot")
	return wrap(cv.BitwiseNot(src.mat))
}

// Min returns the per-sample minimum of a and b, delegating to [cv.Min].
// Operands must share a shape.
func Min(a, b *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "Min")
	return wrap(cv.Min(a.mat, b.mat))
}

// Max returns the per-sample maximum of a and b, delegating to [cv.Max].
// Operands must share a shape.
func Max(a, b *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "Max")
	return wrap(cv.Max(a.mat, b.mat))
}

// Abs returns |src| per sample. Because [cv.Mat] samples are unsigned, every
// value is already non-negative, so this returns a copy of src. It exists for
// API parity with cv::cuda::abs.
func Abs(src *GpuMat, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "Abs")
	return src.Clone()
}

// CmpOp selects the relation applied by [Compare].
type CmpOp int

const (
	// CmpEQ marks samples where a == b.
	CmpEQ CmpOp = iota
	// CmpGT marks samples where a > b.
	CmpGT
	// CmpGE marks samples where a >= b.
	CmpGE
	// CmpLT marks samples where a < b.
	CmpLT
	// CmpLE marks samples where a <= b.
	CmpLE
	// CmpNE marks samples where a != b.
	CmpNE
)

// Compare compares a and b per sample and returns a mask that is 255 where the
// relation op holds and 0 elsewhere, matching cv::cuda::compare. Operands must
// share a shape.
func Compare(a, b *GpuMat, op CmpOp, _ ...*Stream) *GpuMat {
	requireSameShape(a, b, "Compare")
	out := cv.NewMat(a.mat.Rows, a.mat.Cols, a.mat.Channels)
	for i := range a.mat.Data {
		av, bv := a.mat.Data[i], b.mat.Data[i]
		var hit bool
		switch op {
		case CmpEQ:
			hit = av == bv
		case CmpGT:
			hit = av > bv
		case CmpGE:
			hit = av >= bv
		case CmpLT:
			hit = av < bv
		case CmpLE:
			hit = av <= bv
		case CmpNE:
			hit = av != bv
		default:
			panic(fmt.Sprintf("cudaarithm: Compare unknown op %d", op))
		}
		if hit {
			out.Data[i] = 255
		}
	}
	return wrap(out)
}

// Threshold applies a fixed-level threshold to a single-channel GpuMat and
// returns the thresholded result together with the threshold used, delegating to
// [cv.Threshold]. typ is a [cv.ThresholdType] (optionally OR-ed with
// [cv.ThreshOtsu]).
func Threshold(src *GpuMat, thresh, maxval float64, typ cv.ThresholdType, _ ...*Stream) (*GpuMat, float64) {
	requireChannels(src, 1, "Threshold")
	dst, used := cv.Threshold(src.mat, thresh, maxval, typ)
	return wrap(dst), used
}

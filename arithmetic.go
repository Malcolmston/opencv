package cv

import "fmt"

// requireSameShape panics unless a and b have identical dimensions and channel
// counts.
func requireSameShape(a, b *Mat, name string) {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic(fmt.Sprintf("cv: %s shape mismatch %dx%dx%d vs %dx%dx%d",
			name, a.Rows, a.Cols, a.Channels, b.Rows, b.Cols, b.Channels))
	}
}

// Add returns the per-sample sum of a and b, saturating at 255. The inputs must
// have identical dimensions and channel counts.
func Add(a, b *Mat) *Mat {
	requireSameShape(a, b, "Add")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		s := int(a.Data[i]) + int(b.Data[i])
		if s > 255 {
			s = 255
		}
		out.Data[i] = uint8(s)
	}
	return out
}

// Subtract returns the per-sample difference a-b, saturating at 0. The inputs
// must have identical dimensions and channel counts.
func Subtract(a, b *Mat) *Mat {
	requireSameShape(a, b, "Subtract")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		s := int(a.Data[i]) - int(b.Data[i])
		if s < 0 {
			s = 0
		}
		out.Data[i] = uint8(s)
	}
	return out
}

// AbsDiff returns the per-sample absolute difference |a-b|. The inputs must have
// identical dimensions and channel counts.
func AbsDiff(a, b *Mat) *Mat {
	requireSameShape(a, b, "AbsDiff")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		d := int(a.Data[i]) - int(b.Data[i])
		if d < 0 {
			d = -d
		}
		out.Data[i] = uint8(d)
	}
	return out
}

// AddWeighted computes the per-sample weighted sum alpha*a + beta*b + gamma,
// rounding and saturating to [0,255]. It is the classic image-blending
// operation. The inputs must have identical dimensions and channel counts.
func AddWeighted(a *Mat, alpha float64, b *Mat, beta, gamma float64) *Mat {
	requireSameShape(a, b, "AddWeighted")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		v := alpha*float64(a.Data[i]) + beta*float64(b.Data[i]) + gamma
		out.Data[i] = clampToUint8(v + 0.5)
	}
	return out
}

// Multiply returns the per-sample product a*b*scale, rounding and saturating to
// [0,255]. The inputs must have identical dimensions and channel counts.
func Multiply(a, b *Mat, scale float64) *Mat {
	requireSameShape(a, b, "Multiply")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		v := float64(a.Data[i]) * float64(b.Data[i]) * scale
		out.Data[i] = clampToUint8(v + 0.5)
	}
	return out
}

// Divide returns the per-sample quotient scale*a/b, rounding and saturating to
// [0,255]. Division by zero yields 0 (matching OpenCV). The inputs must have
// identical dimensions and channel counts.
func Divide(a, b *Mat, scale float64) *Mat {
	requireSameShape(a, b, "Divide")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		if b.Data[i] == 0 {
			out.Data[i] = 0
			continue
		}
		v := scale * float64(a.Data[i]) / float64(b.Data[i])
		out.Data[i] = clampToUint8(v + 0.5)
	}
	return out
}

// BitwiseAnd returns the per-sample bitwise AND of a and b. The inputs must have
// identical dimensions and channel counts.
func BitwiseAnd(a, b *Mat) *Mat {
	requireSameShape(a, b, "BitwiseAnd")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		out.Data[i] = a.Data[i] & b.Data[i]
	}
	return out
}

// BitwiseOr returns the per-sample bitwise OR of a and b. The inputs must have
// identical dimensions and channel counts.
func BitwiseOr(a, b *Mat) *Mat {
	requireSameShape(a, b, "BitwiseOr")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		out.Data[i] = a.Data[i] | b.Data[i]
	}
	return out
}

// BitwiseXor returns the per-sample bitwise XOR of a and b. The inputs must have
// identical dimensions and channel counts.
func BitwiseXor(a, b *Mat) *Mat {
	requireSameShape(a, b, "BitwiseXor")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		out.Data[i] = a.Data[i] ^ b.Data[i]
	}
	return out
}

// BitwiseNot returns the per-sample bitwise complement (255-value) of src.
func BitwiseNot(src *Mat) *Mat {
	out := NewMat(src.Rows, src.Cols, src.Channels)
	for i := range src.Data {
		out.Data[i] = ^src.Data[i]
	}
	return out
}

// Min returns the per-sample minimum of a and b. The inputs must have identical
// dimensions and channel counts.
func Min(a, b *Mat) *Mat {
	requireSameShape(a, b, "Min")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		if a.Data[i] < b.Data[i] {
			out.Data[i] = a.Data[i]
		} else {
			out.Data[i] = b.Data[i]
		}
	}
	return out
}

// Max returns the per-sample maximum of a and b. The inputs must have identical
// dimensions and channel counts.
func Max(a, b *Mat) *Mat {
	requireSameShape(a, b, "Max")
	out := NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		if a.Data[i] > b.Data[i] {
			out.Data[i] = a.Data[i]
		} else {
			out.Data[i] = b.Data[i]
		}
	}
	return out
}

// ConvertScaleAbs computes |src*alpha + beta| per sample, rounds, and saturates
// to [0,255]. It is OpenCV's convertScaleAbs and is handy for turning a signed
// gradient into a displayable magnitude.
func ConvertScaleAbs(src *Mat, alpha, beta float64) *Mat {
	out := NewMat(src.Rows, src.Cols, src.Channels)
	for i := range src.Data {
		v := float64(src.Data[i])*alpha + beta
		if v < 0 {
			v = -v
		}
		out.Data[i] = clampToUint8(v + 0.5)
	}
	return out
}

// Normalize linearly rescales src so that its minimum sample maps to alpha and
// its maximum to beta, writing the rounded, clamped result to a new Mat. This is
// OpenCV's NORM_MINMAX normalisation. A constant image maps every sample to
// alpha.
func Normalize(src *Mat, alpha, beta float64) *Mat {
	out := NewMat(src.Rows, src.Cols, src.Channels)
	if len(src.Data) == 0 {
		return out
	}
	lo, hi := src.Data[0], src.Data[0]
	for _, v := range src.Data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if hi == lo {
		v := clampToUint8(alpha + 0.5)
		for i := range out.Data {
			out.Data[i] = v
		}
		return out
	}
	scale := (beta - alpha) / float64(hi-lo)
	for i, s := range src.Data {
		v := alpha + float64(int(s)-int(lo))*scale
		out.Data[i] = clampToUint8(v + 0.5)
	}
	return out
}

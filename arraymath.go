package cv

import (
	"fmt"
	"math"
	"sort"
)

// ReduceOp selects the accumulation performed by [Reduce].
type ReduceOp int

const (
	// ReduceSum sums the elements.
	ReduceSum ReduceOp = iota
	// ReduceAvg averages the elements.
	ReduceAvg
	// ReduceMax keeps the maximum element.
	ReduceMax
	// ReduceMin keeps the minimum element.
	ReduceMin
)

// Reduce collapses a FloatMat to a single row or single column by combining
// elements with op. When toRow is true every column is reduced across its rows,
// producing a 1×Cols result; otherwise every row is reduced across its columns,
// producing a Rows×1 result. It panics on an empty matrix.
func Reduce(src *FloatMat, toRow bool, op ReduceOp) *FloatMat {
	if src.Rows == 0 || src.Cols == 0 {
		panic("cv: Reduce on empty matrix")
	}
	if toRow {
		out := NewFloatMat(1, src.Cols)
		for x := 0; x < src.Cols; x++ {
			acc := src.Data[x]
			for y := 1; y < src.Rows; y++ {
				acc = reduceStep(acc, src.Data[y*src.Cols+x], op)
			}
			if op == ReduceAvg {
				acc /= float64(src.Rows)
			}
			out.Data[x] = acc
		}
		return out
	}
	out := NewFloatMat(src.Rows, 1)
	for y := 0; y < src.Rows; y++ {
		acc := src.Data[y*src.Cols]
		for x := 1; x < src.Cols; x++ {
			acc = reduceStep(acc, src.Data[y*src.Cols+x], op)
		}
		if op == ReduceAvg {
			acc /= float64(src.Cols)
		}
		out.Data[y] = acc
	}
	return out
}

// reduceStep folds b into the accumulator acc according to op (except averaging,
// which is finished by the caller after summation).
func reduceStep(acc, b float64, op ReduceOp) float64 {
	switch op {
	case ReduceSum, ReduceAvg:
		return acc + b
	case ReduceMax:
		if b > acc {
			return b
		}
		return acc
	case ReduceMin:
		if b < acc {
			return b
		}
		return acc
	default:
		return acc
	}
}

// Repeat tiles src ny times vertically and nx times horizontally, returning a
// FloatMat of size (Rows*ny)×(Cols*nx). It panics unless ny and nx are
// positive.
func Repeat(src *FloatMat, ny, nx int) *FloatMat {
	if ny <= 0 || nx <= 0 {
		panic(fmt.Sprintf("cv: Repeat requires positive counts, got ny=%d nx=%d", ny, nx))
	}
	out := NewFloatMat(src.Rows*ny, src.Cols*nx)
	for y := 0; y < out.Rows; y++ {
		sy := y % src.Rows
		for x := 0; x < out.Cols; x++ {
			sx := x % src.Cols
			out.Data[y*out.Cols+x] = src.Data[sy*src.Cols+sx]
		}
	}
	return out
}

// Exp returns the element-wise natural exponential of src.
func Exp(src *FloatMat) *FloatMat {
	out := NewFloatMat(src.Rows, src.Cols)
	for i, v := range src.Data {
		out.Data[i] = math.Exp(v)
	}
	return out
}

// Log returns the element-wise natural logarithm of src. Non-positive inputs
// map to negative infinity, following math.Log.
func Log(src *FloatMat) *FloatMat {
	out := NewFloatMat(src.Rows, src.Cols)
	for i, v := range src.Data {
		out.Data[i] = math.Log(v)
	}
	return out
}

// Pow raises every element of src to the given power. Fractional powers of
// negative numbers follow OpenCV by using the absolute value of the base.
func Pow(src *FloatMat, power float64) *FloatMat {
	out := NewFloatMat(src.Rows, src.Cols)
	_, frac := math.Modf(power)
	useAbs := frac != 0
	for i, v := range src.Data {
		if useAbs && v < 0 {
			out.Data[i] = math.Pow(-v, power)
		} else {
			out.Data[i] = math.Pow(v, power)
		}
	}
	return out
}

// Sqrt returns the element-wise square root of src. Negative inputs yield NaN,
// following math.Sqrt.
func Sqrt(src *FloatMat) *FloatMat {
	out := NewFloatMat(src.Rows, src.Cols)
	for i, v := range src.Data {
		out.Data[i] = math.Sqrt(v)
	}
	return out
}

// Magnitude returns the element-wise Euclidean magnitude sqrt(x²+y²) of two
// matrices of matching size. It panics on a size mismatch.
func Magnitude(x, y *FloatMat) *FloatMat {
	requireSameFloatShape(x, y, "Magnitude")
	out := NewFloatMat(x.Rows, x.Cols)
	for i := range x.Data {
		out.Data[i] = math.Hypot(x.Data[i], y.Data[i])
	}
	return out
}

// Phase returns the element-wise orientation atan2(y, x) of two matrices of
// matching size. When angleInDegrees is true the result is in degrees in
// [0, 360); otherwise it is in radians in [0, 2π), matching OpenCV. It panics on
// a size mismatch.
func Phase(x, y *FloatMat, angleInDegrees bool) *FloatMat {
	requireSameFloatShape(x, y, "Phase")
	out := NewFloatMat(x.Rows, x.Cols)
	for i := range x.Data {
		a := math.Atan2(y.Data[i], x.Data[i])
		if a < 0 {
			a += 2 * math.Pi
		}
		if angleInDegrees {
			a *= 180 / math.Pi
		}
		out.Data[i] = a
	}
	return out
}

// CartToPolar converts Cartesian coordinates (x, y) to polar magnitude and
// angle. The angle range and unit follow [Phase]. It panics on a size mismatch.
func CartToPolar(x, y *FloatMat, angleInDegrees bool) (magnitude, angle *FloatMat) {
	return Magnitude(x, y), Phase(x, y, angleInDegrees)
}

// PolarToCart converts polar magnitude and angle back to Cartesian x and y
// components. When angleInDegrees is true the angle is interpreted in degrees.
// It panics on a size mismatch.
func PolarToCart(magnitude, angle *FloatMat, angleInDegrees bool) (x, y *FloatMat) {
	requireSameFloatShape(magnitude, angle, "PolarToCart")
	x = NewFloatMat(magnitude.Rows, magnitude.Cols)
	y = NewFloatMat(magnitude.Rows, magnitude.Cols)
	for i := range magnitude.Data {
		a := angle.Data[i]
		if angleInDegrees {
			a *= math.Pi / 180
		}
		m := magnitude.Data[i]
		x.Data[i] = m * math.Cos(a)
		y.Data[i] = m * math.Sin(a)
	}
	return x, y
}

// requireSameFloatShape panics unless a and b have identical dimensions.
func requireSameFloatShape(a, b *FloatMat, name string) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("cv: %s shape mismatch %dx%d vs %dx%d",
			name, a.Rows, a.Cols, b.Rows, b.Cols))
	}
}

// ScaleAdd computes the element-wise alpha*a + b for two matrices of matching
// size (OpenCV's scaleAdd / axpy). It panics on a size mismatch.
func ScaleAdd(a *FloatMat, alpha float64, b *FloatMat) *FloatMat {
	requireSameFloatShape(a, b, "ScaleAdd")
	out := NewFloatMat(a.Rows, a.Cols)
	for i := range a.Data {
		out.Data[i] = alpha*a.Data[i] + b.Data[i]
	}
	return out
}

// Sort sorts a FloatMat either row-by-row (byRow true) or column-by-column
// (byRow false), in ascending order unless descending is set. It returns a new
// matrix, matching OpenCV's cv::sort with SORT_EVERY_ROW / SORT_EVERY_COLUMN.
func Sort(src *FloatMat, byRow, descending bool) *FloatMat {
	out := fclone(src)
	if byRow {
		for y := 0; y < out.Rows; y++ {
			row := out.Data[y*out.Cols : (y+1)*out.Cols]
			sortSlice(row, descending)
		}
		return out
	}
	col := make([]float64, out.Rows)
	for x := 0; x < out.Cols; x++ {
		for y := 0; y < out.Rows; y++ {
			col[y] = src.Data[y*src.Cols+x]
		}
		sortSlice(col, descending)
		for y := 0; y < out.Rows; y++ {
			out.Data[y*out.Cols+x] = col[y]
		}
	}
	return out
}

// SortIdx returns, for each row (byRow true) or column (byRow false), the
// indices that would sort the elements in ascending order unless descending is
// set. The result has the same shape as src, matching OpenCV's cv::sortIdx.
func SortIdx(src *FloatMat, byRow, descending bool) [][]int {
	if byRow {
		out := make([][]int, src.Rows)
		for y := 0; y < src.Rows; y++ {
			row := src.Data[y*src.Cols : (y+1)*src.Cols]
			out[y] = argsort(row, descending)
		}
		return out
	}
	out := make([][]int, src.Cols)
	col := make([]float64, src.Rows)
	for x := 0; x < src.Cols; x++ {
		for y := 0; y < src.Rows; y++ {
			col[y] = src.Data[y*src.Cols+x]
		}
		out[x] = argsort(col, descending)
	}
	return out
}

// sortSlice sorts s ascending, or descending when desc is set.
func sortSlice(s []float64, desc bool) {
	if desc {
		sort.Slice(s, func(i, j int) bool { return s[i] > s[j] })
		return
	}
	sort.Float64s(s)
}

// argsort returns the indices that sort s ascending (or descending when desc).
func argsort(s []float64, desc bool) []int {
	idx := make([]int, len(s))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool {
		if desc {
			return s[idx[i]] > s[idx[j]]
		}
		return s[idx[i]] < s[idx[j]]
	})
	return idx
}

// MinMaxIdx scans a FloatMat and returns the minimum and maximum values along
// with their flat indices (row-major). This mirrors OpenCV's cv::minMaxIdx for
// a 2-D array. It panics on an empty matrix.
func MinMaxIdx(src *FloatMat) (minVal, maxVal float64, minIdx, maxIdx int) {
	if len(src.Data) == 0 {
		panic("cv: MinMaxIdx on empty matrix")
	}
	minVal, maxVal = src.Data[0], src.Data[0]
	for i, v := range src.Data {
		if v < minVal {
			minVal = v
			minIdx = i
		}
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return
}

// FindNonZero returns the coordinates of every non-zero sample in a
// single-channel Mat, in row-major order. It panics if src is not
// single-channel.
func FindNonZero(src *Mat) []Point {
	requireChannels(src, 1, "FindNonZero")
	var out []Point
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			if src.Data[y*src.Cols+x] != 0 {
				out = append(out, Point{X: x, Y: y})
			}
		}
	}
	return out
}

// CountNonZero returns the number of non-zero samples across all channels of
// src, matching OpenCV's cv::countNonZero (which requires a single channel but
// is generalised here to any channel count).
func CountNonZero(src *Mat) int {
	n := 0
	for _, v := range src.Data {
		if v != 0 {
			n++
		}
	}
	return n
}

// Transform applies a linear transform to the channel vector of every pixel of
// src. The transform matrix m has one row per output channel and either
// Channels or Channels+1 columns; the extra column, when present, is a constant
// (bias) term. The result is a Mat with m.Rows channels, with values rounded
// and saturated to [0,255]. It panics on a dimension mismatch.
func Transform(src *Mat, m *FloatMat) *Mat {
	inCh := src.Channels
	if m.Cols != inCh && m.Cols != inCh+1 {
		panic(fmt.Sprintf("cv: Transform matrix has %d cols, want %d or %d", m.Cols, inCh, inCh+1))
	}
	outCh := m.Rows
	out := NewMat(src.Rows, src.Cols, outCh)
	hasBias := m.Cols == inCh+1
	px := make([]float64, inCh)
	for p := 0; p < src.Total(); p++ {
		base := p * inCh
		for c := 0; c < inCh; c++ {
			px[c] = float64(src.Data[base+c])
		}
		ob := p * outCh
		for r := 0; r < outCh; r++ {
			v := 0.0
			for c := 0; c < inCh; c++ {
				v += m.Data[r*m.Cols+c] * px[c]
			}
			if hasBias {
				v += m.Data[r*m.Cols+inCh]
			}
			out.Data[ob+r] = clampToUint8(v + 0.5)
		}
	}
	return out
}

// ExtractChannel returns a single-channel Mat holding channel coi of src. It
// panics if coi is out of range.
func ExtractChannel(src *Mat, coi int) *Mat {
	if coi < 0 || coi >= src.Channels {
		panic(fmt.Sprintf("cv: ExtractChannel coi=%d out of range for %d channels", coi, src.Channels))
	}
	out := NewMat(src.Rows, src.Cols, 1)
	for p := 0; p < src.Total(); p++ {
		out.Data[p] = src.Data[p*src.Channels+coi]
	}
	return out
}

// InsertChannel writes the single-channel src into channel coi of dst in place.
// The two matrices must share dimensions and src must be single-channel. It
// panics on a mismatch or an out-of-range coi.
func InsertChannel(src, dst *Mat, coi int) {
	if src.Channels != 1 {
		panic("cv: InsertChannel source must be single-channel")
	}
	if src.Rows != dst.Rows || src.Cols != dst.Cols {
		panic("cv: InsertChannel size mismatch")
	}
	if coi < 0 || coi >= dst.Channels {
		panic(fmt.Sprintf("cv: InsertChannel coi=%d out of range for %d channels", coi, dst.Channels))
	}
	for p := 0; p < src.Total(); p++ {
		dst.Data[p*dst.Channels+coi] = src.Data[p]
	}
}

// MixChannels copies specified channels from a set of source Mats into a set of
// destination Mats. Each entry of fromTo is a pair (srcIndex, dstIndex) into the
// flattened channel lists of srcs and dsts respectively. All matrices must share
// the same width and height. This mirrors OpenCV's cv::mixChannels. It panics on
// a dimension mismatch or an out-of-range index.
func MixChannels(srcs, dsts []*Mat, fromTo [][2]int) {
	rows, cols := srcs[0].Rows, srcs[0].Cols
	srcCh := channelOffsets(srcs, rows, cols, "MixChannels src")
	dstCh := channelOffsets(dsts, rows, cols, "MixChannels dst")
	total := rows * cols
	for _, ft := range fromTo {
		sm, sc := locateChannel(srcs, srcCh, ft[0], "MixChannels from")
		dm, dc := locateChannel(dsts, dstCh, ft[1], "MixChannels to")
		for p := 0; p < total; p++ {
			dm.Data[p*dm.Channels+dc] = sm.Data[p*sm.Channels+sc]
		}
	}
}

// channelOffsets validates that every Mat shares the given size and returns the
// cumulative channel count so a flat channel index can be resolved.
func channelOffsets(mats []*Mat, rows, cols int, name string) []int {
	offs := make([]int, len(mats)+1)
	for i, m := range mats {
		if m.Rows != rows || m.Cols != cols {
			panic(fmt.Sprintf("cv: %s size mismatch", name))
		}
		offs[i+1] = offs[i] + m.Channels
	}
	return offs
}

// locateChannel maps a flat channel index into the owning Mat and its local
// channel offset.
func locateChannel(mats []*Mat, offs []int, flat int, name string) (*Mat, int) {
	for i := range mats {
		if flat >= offs[i] && flat < offs[i+1] {
			return mats[i], flat - offs[i]
		}
	}
	panic(fmt.Sprintf("cv: %s channel index %d out of range", name, flat))
}

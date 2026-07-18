package morph2

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Erode performs a single grey-scale erosion: each output sample is the minimum
// input sample over the set cells of the structuring element. Samples that fall
// outside the image are treated as the maximum value (255), matching OpenCV's
// default morphology border so that erosion never eats into objects along the
// border artificially. It panics on multi-channel input.
func Erode(src *cv.Mat, e *Element) *cv.Mat { return ErodeN(src, e, 1) }

// Dilate performs a single grey-scale dilation: each output sample is the
// maximum input sample over the reflected structuring element. Samples outside
// the image are treated as the minimum value (0). It panics on multi-channel
// input.
func Dilate(src *cv.Mat, e *Element) *cv.Mat { return DilateN(src, e, 1) }

// ErodeN applies [Erode] the given number of times (values < 1 are treated as 1).
func ErodeN(src *cv.Mat, e *Element, iterations int) *cv.Mat {
	requireGray(src)
	if iterations < 1 {
		iterations = 1
	}
	offs := e.offsets()
	cur := src
	for it := 0; it < iterations; it++ {
		cur = flatMorph(cur, offs, false)
	}
	return cur
}

// DilateN applies [Dilate] the given number of times (values < 1 are treated as 1).
func DilateN(src *cv.Mat, e *Element, iterations int) *cv.Mat {
	requireGray(src)
	if iterations < 1 {
		iterations = 1
	}
	offs := e.offsets()
	cur := src
	for it := 0; it < iterations; it++ {
		cur = flatMorph(cur, offs, true)
	}
	return cur
}

// flatMorph computes one flat erosion or dilation. offs are (dy, dx) set-cell
// displacements relative to the anchor. For dilation the element is implicitly
// reflected by negating the displacements, which yields the adjunct dilation.
func flatMorph(src *cv.Mat, offs [][2]int, dilate bool) *cv.Mat {
	rows, cols := src.Rows, src.Cols
	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc uint8
			if dilate {
				acc = 0
			} else {
				acc = 255
			}
			for _, o := range offs {
				var yy, xx int
				if dilate {
					yy, xx = y-o[0], x-o[1]
				} else {
					yy, xx = y+o[0], x+o[1]
				}
				var v uint8
				if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
					if dilate {
						v = 0
					} else {
						v = 255
					}
				} else {
					v = src.Data[idx(yy, xx, cols)]
				}
				if dilate {
					if v > acc {
						acc = v
					}
				} else if v < acc {
					acc = v
				}
			}
			dst.Data[idx(y, x, cols)] = acc
		}
	}
	return dst
}

// Open performs a morphological opening (erosion followed by dilation). It
// removes small bright details while preserving the overall shape and size of
// larger objects.
func Open(src *cv.Mat, e *Element) *cv.Mat {
	return Dilate(Erode(src, e), e)
}

// Close performs a morphological closing (dilation followed by erosion). It
// fills small dark holes and gaps while preserving larger structures.
func Close(src *cv.Mat, e *Element) *cv.Mat {
	return Erode(Dilate(src, e), e)
}

// Gradient returns the morphological gradient, the dilation minus the erosion,
// which highlights object boundaries. Subtraction saturates at zero.
func Gradient(src *cv.Mat, e *Element) *cv.Mat {
	return Subtract(Dilate(src, e), Erode(src, e))
}

// InternalGradient returns the internal morphological gradient, the source
// minus its erosion, marking the inner boundary of bright objects.
func InternalGradient(src *cv.Mat, e *Element) *cv.Mat {
	return Subtract(src, Erode(src, e))
}

// ExternalGradient returns the external morphological gradient, the dilation
// minus the source, marking the outer boundary of bright objects.
func ExternalGradient(src *cv.Mat, e *Element) *cv.Mat {
	return Subtract(Dilate(src, e), src)
}

// TopHat returns the white top-hat, the source minus its opening, which
// isolates bright details smaller than the structuring element.
func TopHat(src *cv.Mat, e *Element) *cv.Mat {
	return Subtract(src, Open(src, e))
}

// BlackHat returns the black top-hat, the closing minus the source, which
// isolates dark details smaller than the structuring element.
func BlackHat(src *cv.Mat, e *Element) *cv.Mat {
	return Subtract(Close(src, e), src)
}

// Op selects the compound operation performed by [MorphologyEx].
type Op int

const (
	// OpErode is a plain erosion.
	OpErode Op = iota
	// OpDilate is a plain dilation.
	OpDilate
	// OpOpen is an opening (erode then dilate).
	OpOpen
	// OpClose is a closing (dilate then erode).
	OpClose
	// OpGradient is the morphological gradient (dilate minus erode).
	OpGradient
	// OpTopHat is the white top-hat (source minus opening).
	OpTopHat
	// OpBlackHat is the black top-hat (closing minus source).
	OpBlackHat
)

// MorphologyEx dispatches to the compound operator named by op, applying the
// given number of iterations to the primitive erosion/dilation stages (values
// < 1 are treated as 1). It panics on an unknown op.
func MorphologyEx(src *cv.Mat, e *Element, op Op, iterations int) *cv.Mat {
	if iterations < 1 {
		iterations = 1
	}
	switch op {
	case OpErode:
		return ErodeN(src, e, iterations)
	case OpDilate:
		return DilateN(src, e, iterations)
	case OpOpen:
		return DilateN(ErodeN(src, e, iterations), e, iterations)
	case OpClose:
		return ErodeN(DilateN(src, e, iterations), e, iterations)
	case OpGradient:
		return Subtract(DilateN(src, e, iterations), ErodeN(src, e, iterations))
	case OpTopHat:
		return Subtract(src, DilateN(ErodeN(src, e, iterations), e, iterations))
	case OpBlackHat:
		return Subtract(ErodeN(DilateN(src, e, iterations), e, iterations), src)
	default:
		panic(fmt.Sprintf("morph2: unknown op %d", op))
	}
}

// Subtract returns the per-sample saturating difference a - b, clamped at zero.
// It panics unless a and b are single-channel and identically sized.
func Subtract(a, b *cv.Mat) *cv.Mat {
	requireSameSize(a, b)
	out := newLike(a)
	for i := range a.Data {
		if a.Data[i] > b.Data[i] {
			out.Data[i] = a.Data[i] - b.Data[i]
		}
	}
	return out
}

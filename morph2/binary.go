package morph2

import cv "github.com/malcolmston/opencv"

// Binarize returns a binary image in which every input sample strictly greater
// than thresh becomes foreground (255) and every other sample becomes
// background (0). It panics on multi-channel input.
func Binarize(src *cv.Mat, thresh uint8) *cv.Mat {
	requireGray(src)
	out := newLike(src)
	for i, v := range src.Data {
		if v > thresh {
			out.Data[i] = 255
		}
	}
	return out
}

// Complement returns the pointwise complement 255 - src of a grey-scale image;
// for a binary image this swaps foreground and background. It panics on
// multi-channel input.
func Complement(src *cv.Mat) *cv.Mat {
	requireGray(src)
	out := newLike(src)
	for i, v := range src.Data {
		out.Data[i] = 255 - v
	}
	return out
}

// Union returns the pointwise maximum of two identically sized images; for
// binary images this is the set union. It panics on a size or channel mismatch.
func Union(a, b *cv.Mat) *cv.Mat {
	requireSameSize(a, b)
	out := newLike(a)
	for i := range a.Data {
		out.Data[i] = maxU8(a.Data[i], b.Data[i])
	}
	return out
}

// Intersection returns the pointwise minimum of two identically sized images;
// for binary images this is the set intersection. It panics on a size or
// channel mismatch.
func Intersection(a, b *cv.Mat) *cv.Mat {
	requireSameSize(a, b)
	out := newLike(a)
	for i := range a.Data {
		out.Data[i] = minU8(a.Data[i], b.Data[i])
	}
	return out
}

// Difference returns the binary set difference a \ b: foreground where a is
// foreground and b is background. It panics on a size or channel mismatch.
func Difference(a, b *cv.Mat) *cv.Mat {
	requireSameSize(a, b)
	out := newLike(a)
	for i := range a.Data {
		if a.Data[i] != 0 && b.Data[i] == 0 {
			out.Data[i] = 255
		}
	}
	return out
}

// Equal reports whether two images have identical dimensions and identical
// samples.
func Equal(a, b *cv.Mat) bool {
	if a == nil || b == nil || a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		return false
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}

// CountForeground returns the number of non-zero (foreground) samples.
func CountForeground(src *cv.Mat) int {
	requireGray(src)
	n := 0
	for _, v := range src.Data {
		if v != 0 {
			n++
		}
	}
	return n
}

// binaryErode returns a binary erosion: a pixel is foreground iff every set
// cell of the element maps onto a foreground input pixel. Displacements that
// leave the image are treated as background so objects erode at the border.
func binaryErode(src *cv.Mat, offs [][2]int) *cv.Mat {
	rows, cols := src.Rows, src.Cols
	out := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			fit := true
			for _, o := range offs {
				yy, xx := y+o[0], x+o[1]
				if yy < 0 || yy >= rows || xx < 0 || xx >= cols || src.Data[idx(yy, xx, cols)] == 0 {
					fit = false
					break
				}
			}
			if fit {
				out.Data[idx(y, x, cols)] = 255
			}
		}
	}
	return out
}

// HitOrMiss computes the binary hit-or-miss transform of src using two disjoint
// structuring elements: hit specifies cells that must lie on foreground and
// miss specifies cells that must lie on background. A pixel is set in the
// result iff both conditions hold at that position, i.e.
// (src eroded by hit) intersected with (complement of src eroded by miss). It
// panics on multi-channel input.
func HitOrMiss(src *cv.Mat, hit, miss *Element) *cv.Mat {
	requireGray(src)
	bin := Binarize(src, 0)
	comp := Complement(bin)
	fg := binaryErode(bin, hit.offsets())
	bg := binaryErode(comp, miss.offsets())
	return Intersection(fg, bg)
}

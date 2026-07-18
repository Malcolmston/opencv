package connected

import cv "github.com/malcolmston/opencv"

// Binarize returns a binary single-channel image in which every input sample
// strictly greater than thresh becomes foreground (255) and every other sample
// becomes background (0). It is a convenient way to prepare an image for the
// labelling routines. src is not modified.
func Binarize(src *cv.Mat, thresh uint8) *cv.Mat {
	connectedRequireBinary(src, "Binarize")
	out := connectedNewMask(src)
	for i, v := range src.Data {
		if v > thresh {
			out.Data[i] = 255
		}
	}
	return out
}

// Invert returns the complement of a binary image: foreground pixels (non-zero)
// become 0 and background pixels become 255. src is not modified.
func Invert(src *cv.Mat) *cv.Mat {
	connectedRequireBinary(src, "Invert")
	out := connectedNewMask(src)
	for i, v := range src.Data {
		if v == 0 {
			out.Data[i] = 255
		}
	}
	return out
}

// IsBinary reports whether every sample of src is either 0 or 255.
func IsBinary(src *cv.Mat) bool {
	connectedRequireBinary(src, "IsBinary")
	for _, v := range src.Data {
		if v != 0 && v != 255 {
			return false
		}
	}
	return true
}

// CountForeground returns the number of non-zero (foreground) samples in src.
func CountForeground(src *cv.Mat) int {
	connectedRequireBinary(src, "CountForeground")
	n := 0
	for _, v := range src.Data {
		if v != 0 {
			n++
		}
	}
	return n
}

// CountBackground returns the number of zero (background) samples in src.
func CountBackground(src *cv.Mat) int {
	connectedRequireBinary(src, "CountBackground")
	return len(src.Data) - CountForeground(src)
}

// connectedRequireSameSize panics unless a and b are single-channel matrices of
// identical dimensions.
func connectedRequireSameSize(a, b *cv.Mat, who string) {
	connectedRequireBinary(a, who)
	connectedRequireBinary(b, who)
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("connected: " + who + ": matrices must have identical dimensions")
	}
}

// And returns the pixelwise intersection of two binary masks: a pixel is 255
// only where both a and b are non-zero. a and b must have identical dimensions.
func And(a, b *cv.Mat) *cv.Mat {
	connectedRequireSameSize(a, b, "And")
	out := connectedNewMask(a)
	for i := range out.Data {
		if a.Data[i] != 0 && b.Data[i] != 0 {
			out.Data[i] = 255
		}
	}
	return out
}

// Or returns the pixelwise union of two binary masks: a pixel is 255 wherever
// either a or b is non-zero. a and b must have identical dimensions.
func Or(a, b *cv.Mat) *cv.Mat {
	connectedRequireSameSize(a, b, "Or")
	out := connectedNewMask(a)
	for i := range out.Data {
		if a.Data[i] != 0 || b.Data[i] != 0 {
			out.Data[i] = 255
		}
	}
	return out
}

// Xor returns the pixelwise symmetric difference of two binary masks: a pixel
// is 255 where exactly one of a and b is non-zero. a and b must have identical
// dimensions.
func Xor(a, b *cv.Mat) *cv.Mat {
	connectedRequireSameSize(a, b, "Xor")
	out := connectedNewMask(a)
	for i := range out.Data {
		if (a.Data[i] != 0) != (b.Data[i] != 0) {
			out.Data[i] = 255
		}
	}
	return out
}

// Subtract returns the set difference a minus b: a pixel is 255 where a is
// non-zero and b is zero. a and b must have identical dimensions.
func Subtract(a, b *cv.Mat) *cv.Mat {
	connectedRequireSameSize(a, b, "Subtract")
	out := connectedNewMask(a)
	for i := range out.Data {
		if a.Data[i] != 0 && b.Data[i] == 0 {
			out.Data[i] = 255
		}
	}
	return out
}

// Equal reports whether two binary masks agree at every pixel, comparing by
// foreground/background rather than exact sample value. a and b must have
// identical dimensions.
func Equal(a, b *cv.Mat) bool {
	connectedRequireSameSize(a, b, "Equal")
	for i := range a.Data {
		if (a.Data[i] != 0) != (b.Data[i] != 0) {
			return false
		}
	}
	return true
}

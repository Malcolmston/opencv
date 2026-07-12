package bgsegm

import cv "github.com/malcolmston/opencv"

// CloseMask returns a morphologically closed copy of mask: a dilation followed
// by an erosion over a ksize×ksize rectangular structuring element, performed
// with [cv.MorphologyEx] and [cv.MorphClose]. Closing fills dark holes and gaps
// smaller than the kernel inside a foreground blob while preserving its outer
// shape — the natural complement of the opening done by [CleanupMask]. If ksize
// is even it is rounded up to the next odd value; ksize <= 1 returns mask
// unchanged. The input mask is not modified.
func CloseMask(mask *cv.Mat, ksize int) *cv.Mat {
	if ksize <= 1 {
		return mask
	}
	if ksize%2 == 0 {
		ksize++
	}
	kernel := cv.GetStructuringElement(cv.MorphRect, ksize, ksize)
	return cv.MorphologyEx(mask, kernel, cv.MorphClose, 1)
}

// RefineMask returns a cleaned-up copy of a raw foreground mask by first opening
// it with [CleanupMask] at openKsize (removing isolated speckle) and then
// closing it with [CloseMask] at closeKsize (filling interior holes and welding
// nearby fragments). Either size may be <= 1 to skip that stage, so RefineMask
// with both sizes <= 1 returns mask unchanged. The input mask is not modified.
//
// This is the standard morphological post-processing applied to background-
// subtraction output before connected-component analysis. It is also available
// per model through the OpenKernel field, which performs only the opening stage.
func RefineMask(mask *cv.Mat, openKsize, closeKsize int) *cv.Mat {
	out := CleanupMask(mask, openKsize)
	return CloseMask(out, closeKsize)
}

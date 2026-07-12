package linedescriptor

import cv "github.com/malcolmston/opencv"

// DetectWithMask detects line segments in img but keeps only those whose
// midpoint falls on a non-zero pixel of mask, mirroring the mask parameter of
// cv::line_descriptor::LSDDetector::detect. This restricts detection to a
// region of interest without having to crop the image (which would shift
// coordinates). mask must be single-channel and the same size as img; a nil
// mask disables filtering and behaves exactly like [LSDDetector.Detect].
//
// The midpoint test matches the upstream behaviour of masking on the segment
// centre: a segment survives when the mask admits the pixel halfway between its
// endpoints.
func (d *LSDDetector) DetectWithMask(img *cv.Mat, mask *cv.Mat) []KeyLine {
	lines := d.Detect(img)
	if mask == nil {
		return lines
	}
	if mask.Channels != 1 {
		panic("linedescriptor: DetectWithMask requires a single-channel mask")
	}
	if mask.Rows != img.Rows || mask.Cols != img.Cols {
		panic("linedescriptor: DetectWithMask mask size must match the image")
	}
	return filterByMask(lines, mask)
}

// filterByMask returns the subset of lines whose midpoint lies on a non-zero
// mask pixel, preserving order.
func filterByMask(lines []KeyLine, mask *cv.Mat) []KeyLine {
	out := lines[:0:0]
	for _, kl := range lines {
		mx := (kl.StartPoint.X + kl.EndPoint.X) / 2
		my := (kl.StartPoint.Y + kl.EndPoint.Y) / 2
		if mx < 0 || my < 0 || mx >= mask.Cols || my >= mask.Rows {
			continue
		}
		if mask.Data[my*mask.Cols+mx] != 0 {
			out = append(out, kl)
		}
	}
	return out
}

// RectMask builds a single-channel mask of the given size in which the
// axis-aligned rectangle [x, x+width) × [y, y+height) is set to 255 and every
// other pixel is 0. It is a convenience for constructing a region-of-interest
// mask to pass to [LSDDetector.DetectWithMask].
func RectMask(rows, cols, y, x, height, width int) *cv.Mat {
	mask := cv.NewMat(rows, cols, 1)
	for ry := y; ry < y+height; ry++ {
		if ry < 0 || ry >= rows {
			continue
		}
		for rx := x; rx < x+width; rx++ {
			if rx < 0 || rx >= cols {
				continue
			}
			mask.Data[ry*cols+rx] = 255
		}
	}
	return mask
}

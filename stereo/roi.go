package stereo

// GetValidDisparityROI computes the rectangle of a rectified image within which
// disparities can be estimated reliably, mirroring OpenCV's
// cv::getValidDisparityROI. Given the valid rectification ROIs of the two views
// (roi1 for the left/reference image, roi2 for the right image), the disparity
// search range and the block size, it trims the overlap by the search range on
// the left and by half the block window on every side:
//
//	minD = minDisparity,  maxD = minDisparity + numberOfDisparities - 1
//	xmin = max(roi1.x,               roi2.x + maxD) + SW2
//	xmax = min(roi1.x + roi1.width,  roi2.x + roi2.width - minD) - SW2
//	ymin = max(roi1.y,               roi2.y) + SW2
//	ymax = min(roi1.y + roi1.height, roi2.y + roi2.height) - SW2
//
// where SW2 = blockSize/2. When the resulting rectangle is degenerate it returns
// the empty [Rect]. Disparities outside this ROI should be treated as invalid.
func GetValidDisparityROI(roi1, roi2 Rect, minDisparity, numberOfDisparities, blockSize int) Rect {
	sw2 := blockSize / 2
	minD := minDisparity
	maxD := minDisparity + numberOfDisparities - 1

	xmin := maxInt(roi1.X, roi2.X+maxD) + sw2
	xmax := minInt(roi1.X+roi1.Width, roi2.X+roi2.Width-minD) - sw2
	ymin := maxInt(roi1.Y, roi2.Y) + sw2
	ymax := minInt(roi1.Y+roi1.Height, roi2.Y+roi2.Height) - sw2

	r := Rect{X: xmin, Y: ymin, Width: xmax - xmin, Height: ymax - ymin}
	if r.Width > 0 && r.Height > 0 {
		return r
	}
	return Rect{}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

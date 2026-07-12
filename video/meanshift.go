package video

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// clampRectToImage confines a window to the [0,cols) x [0,rows) image extent,
// keeping a strictly positive size. It returns the clipped rectangle.
func clampRectToImage(w cv.Rect, rows, cols int) cv.Rect {
	if w.Width < 1 {
		w.Width = 1
	}
	if w.Height < 1 {
		w.Height = 1
	}
	if w.X < 0 {
		w.X = 0
	}
	if w.Y < 0 {
		w.Y = 0
	}
	if w.X > cols-1 {
		w.X = cols - 1
	}
	if w.Y > rows-1 {
		w.Y = rows - 1
	}
	if w.X+w.Width > cols {
		w.Width = cols - w.X
	}
	if w.Y+w.Height > rows {
		w.Height = rows - w.Y
	}
	return w
}

// windowMoments accumulates the zeroth and first spatial moments of the weight
// image over the rectangle w (already clipped to the image).
func windowMoments(prob *cv.FloatMat, w cv.Rect) (m00, m10, m01 float64) {
	for y := w.Y; y < w.Y+w.Height; y++ {
		for x := w.X; x < w.X+w.Width; x++ {
			p := prob.At(y, x)
			if p <= 0 {
				continue
			}
			m00 += p
			m10 += p * float64(x)
			m01 += p * float64(y)
		}
	}
	return m00, m10, m01
}

// MeanShift finds a local mode of the weight image prob (typically a colour
// back-projection) by iteratively recentring the search window on the centroid
// of the weights inside it, mirroring cv::meanShift. It returns the number of
// iterations performed and the converged window (its size is unchanged; only
// the position moves).
//
// Iteration stops when TermCriteria is satisfied: either MaxCount iterations
// have run or the centre shifts by no more than Epsilon pixels. prob must be a
// non-empty [cv.FloatMat] with non-negative weights and window must have a
// positive size.
func MeanShift(prob *cv.FloatMat, window cv.Rect, crit TermCriteria) (iterations int, result cv.Rect) {
	if prob == nil || len(prob.Data) == 0 {
		panic("video: MeanShift requires a non-empty probability image")
	}
	if window.Width < 1 || window.Height < 1 {
		panic("video: MeanShift requires a window with positive size")
	}
	w := clampRectToImage(window, prob.Rows, prob.Cols)
	maxIter := crit.iterCap(100)
	for iter := 0; iter < maxIter; iter++ {
		iterations = iter + 1
		m00, m10, m01 := windowMoments(prob, w)
		if m00 <= 0 {
			// No mass in the window: nothing to track, stop where we are.
			break
		}
		cx := m10 / m00
		cy := m01 / m00
		newX := int(math.Round(cx - float64(w.Width)/2))
		newY := int(math.Round(cy - float64(w.Height)/2))
		moved := clampRectToImage(cv.Rect{X: newX, Y: newY, Width: w.Width, Height: w.Height}, prob.Rows, prob.Cols)
		shift := math.Hypot(float64(moved.X-w.X), float64(moved.Y-w.Y))
		w = moved
		if crit.reached(iter, shift) {
			break
		}
	}
	return iterations, w
}

// CamShift runs the Continuously Adaptive Mean Shift tracker on the weight image
// prob, mirroring cv::CamShift. It first calls [MeanShift] to locate the mode,
// then adapts the window size to the zeroth moment (total weight) and estimates
// the object's orientation and extent from the second-order central moments,
// returning them as a [cv.RotatedRect]. The second result is the upright search
// window CamShift would carry into the next frame.
//
// prob must be a non-empty [cv.FloatMat]; window must have a positive size.
func CamShift(prob *cv.FloatMat, window cv.Rect, crit TermCriteria) (box cv.RotatedRect, result cv.Rect) {
	if prob == nil || len(prob.Data) == 0 {
		panic("video: CamShift requires a non-empty probability image")
	}
	_, w := MeanShift(prob, window, crit)

	// Zeroth and first moments in the converged window.
	m00, m10, m01 := windowMoments(prob, w)
	if m00 <= 0 {
		return cv.RotatedRect{
			CenterX: float64(w.X) + float64(w.Width)/2,
			CenterY: float64(w.Y) + float64(w.Height)/2,
			Width:   float64(w.Width),
			Height:  float64(w.Height),
		}, w
	}
	xc := m10 / m00
	yc := m01 / m00

	// Second-order central moments.
	var mu20, mu02, mu11 float64
	for y := w.Y; y < w.Y+w.Height; y++ {
		for x := w.X; x < w.X+w.Width; x++ {
			p := prob.At(y, x)
			if p <= 0 {
				continue
			}
			dx := float64(x) - xc
			dy := float64(y) - yc
			mu20 += p * dx * dx
			mu02 += p * dy * dy
			mu11 += p * dx * dy
		}
	}
	a := mu20 / m00
	b := 2 * (mu11 / m00)
	c := mu02 / m00

	// Orientation of the principal axis and the two axis lengths, following the
	// closed form used by OpenCV's CamShift.
	angle := 0.5 * math.Atan2(b, a-c)
	common := math.Sqrt(b*b + (a-c)*(a-c))
	l1 := 0.5 * (a + c + common)
	l2 := 0.5 * (a + c - common)
	if l2 < 0 {
		l2 = 0
	}
	length := 2 * math.Sqrt(l1) * 2 // 4*sqrt(eigenvalue): 2 sigma on each side
	width := 2 * math.Sqrt(l2) * 2
	if length < 1 {
		length = 1
	}
	if width < 1 {
		width = 1
	}

	// Adapt the upright window to the new extent, centred on the mode.
	newW := int(math.Round(math.Max(length, width)))
	if newW < 1 {
		newW = 1
	}
	next := clampRectToImage(cv.Rect{
		X:      int(math.Round(xc)) - newW/2,
		Y:      int(math.Round(yc)) - newW/2,
		Width:  newW,
		Height: newW,
	}, prob.Rows, prob.Cols)

	box = cv.RotatedRect{
		CenterX: xc,
		CenterY: yc,
		Width:   width,
		Height:  length,
		Angle:   angle * 180 / math.Pi,
	}
	return box, next
}

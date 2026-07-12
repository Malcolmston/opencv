package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MeanShift moves a fixed-size window to the local mode of a back-projection
// probability image and returns the converged window. probImage must be a
// single-channel "probability" map such as the output of [cv.CalcBackProject],
// where brighter pixels are more likely to belong to the object.
//
// Each iteration computes the intensity-weighted centroid of probImage inside
// the window and recentres the window on it, stopping after maxIter iterations
// or once the window stops moving. The window keeps its input size. It panics if
// probImage is not single-channel or maxIter < 1.
func MeanShift(probImage *cv.Mat, window cv.Rect, maxIter int) cv.Rect {
	if probImage.Channels != 1 {
		panic("tracking: MeanShift requires a single-channel probability image")
	}
	if maxIter < 1 {
		panic("tracking: MeanShift requires maxIter >= 1")
	}
	win := clampRect(window, probImage.Rows, probImage.Cols)
	cols := probImage.Cols
	for i := 0; i < maxIter; i++ {
		var m00, m10, m01 float64
		for y := win.Y; y < win.Y+win.Height; y++ {
			row := y * cols
			for x := win.X; x < win.X+win.Width; x++ {
				v := float64(probImage.Data[row+x])
				m00 += v
				m10 += v * float64(x)
				m01 += v * float64(y)
			}
		}
		if m00 == 0 {
			break
		}
		cx := m10 / m00
		cy := m01 / m00
		newX := int(math.Round(cx)) - win.Width/2
		newY := int(math.Round(cy)) - win.Height/2
		next := clampWindowPos(cv.Rect{X: newX, Y: newY, Width: win.Width, Height: win.Height}, probImage.Rows, probImage.Cols)
		if next.X == win.X && next.Y == win.Y {
			win = next
			break
		}
		win = next
	}
	return win
}

// CamShift ("Continuously Adaptive Mean-Shift") runs [MeanShift] to locate the
// object and then adapts the window's size and orientation to the probability
// distribution. It returns the fitted [cv.RotatedRect] (centre, side lengths and
// angle in degrees, derived from the second-order image moments) and the upright
// window (the axis-aligned bounding box of that rotated rectangle) to seed the
// next frame. It panics if probImage is not single-channel or maxIter < 1.
func CamShift(probImage *cv.Mat, window cv.Rect, maxIter int) (cv.RotatedRect, cv.Rect) {
	win := MeanShift(probImage, window, maxIter)
	cols := probImage.Cols
	var m00, m10, m01, m20, m11, m02 float64
	for y := win.Y; y < win.Y+win.Height; y++ {
		row := y * cols
		fy := float64(y)
		for x := win.X; x < win.X+win.Width; x++ {
			v := float64(probImage.Data[row+x])
			fx := float64(x)
			m00 += v
			m10 += v * fx
			m01 += v * fy
			m20 += v * fx * fx
			m11 += v * fx * fy
			m02 += v * fy * fy
		}
	}
	if m00 == 0 {
		cx, cy := rectCenter(win)
		rr := cv.RotatedRect{CenterX: cx, CenterY: cy, Width: float64(win.Width), Height: float64(win.Height)}
		return rr, win
	}
	cx := m10 / m00
	cy := m01 / m00
	// Normalised central second moments (the covariance of the distribution).
	a := m20/m00 - cx*cx
	b := m11/m00 - cx*cy
	c := m02/m00 - cy*cy
	common := math.Sqrt((a-c)*(a-c) + 4*b*b)
	l1 := (a + c + common) / 2 // major-axis variance
	l2 := (a + c - common) / 2 // minor-axis variance
	if l1 < 0 {
		l1 = 0
	}
	if l2 < 0 {
		l2 = 0
	}
	// Full extents are ~4 standard deviations across (covers most of the mass).
	length := 4 * math.Sqrt(l1)
	width := 4 * math.Sqrt(l2)
	if length < 1 {
		length = 1
	}
	if width < 1 {
		width = 1
	}
	angle := 0.5 * math.Atan2(2*b, a-c) * 180 / math.Pi
	// Width lies along the rectangle's own x-axis (the Angle direction in
	// cv.RotatedRect.Points), so the major axis maps to Width.
	rr := cv.RotatedRect{CenterX: cx, CenterY: cy, Width: length, Height: width, Angle: angle}

	// Axis-aligned bounding box of the rotated rectangle for the next window.
	corners := rr.Points()
	minX, minY := corners[0].X, corners[0].Y
	maxX, maxY := corners[0].X, corners[0].Y
	for _, p := range corners[1:] {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	next := clampRect(cv.Rect{X: minX, Y: minY, Width: maxX - minX + 1, Height: maxY - minY + 1}, probImage.Rows, probImage.Cols)
	return rr, next
}

// clampWindowPos shifts a fixed-size window so it stays inside a rows×cols image
// without changing its dimensions.
func clampWindowPos(win cv.Rect, rows, cols int) cv.Rect {
	if win.X < 0 {
		win.X = 0
	}
	if win.Y < 0 {
		win.Y = 0
	}
	if win.X+win.Width > cols {
		win.X = cols - win.Width
	}
	if win.Y+win.Height > rows {
		win.Y = rows - win.Height
	}
	if win.X < 0 {
		win.X = 0
	}
	if win.Y < 0 {
		win.Y = 0
	}
	return win
}

// windowMass returns the total probability inside win of a single-channel image.
func windowMass(probImage *cv.Mat, win cv.Rect) float64 {
	cols := probImage.Cols
	var sum float64
	for y := win.Y; y < win.Y+win.Height; y++ {
		row := y * cols
		for x := win.X; x < win.X+win.Width; x++ {
			sum += float64(probImage.Data[row+x])
		}
	}
	return sum
}

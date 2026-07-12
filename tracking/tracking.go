package tracking

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Tracker is the common interface implemented by every tracker in this package,
// mirroring OpenCV's cv::Tracker. A tracker is stateful: call [Tracker.Init]
// exactly once with the first frame and the object's bounding box, then call
// [Tracker.Update] on each following frame to obtain the new box.
type Tracker interface {
	// Init primes the tracker with the object's appearance inside bbox on the
	// given frame. bbox is clamped to the image. It must be called before Update.
	Init(frame *cv.Mat, bbox cv.Rect)
	// Update locates the object in a new frame and returns its bounding box and a
	// confidence flag. A false flag means the tracker judged the match unreliable
	// (low correlation, too few surviving points, or no probability mass); the
	// returned box is still the tracker's best estimate.
	Update(frame *cv.Mat) (cv.Rect, bool)
}

// Compile-time checks that every tracker satisfies Tracker.
var (
	_ Tracker = (*TrackerTemplate)(nil)
	_ Tracker = (*TrackerKCF)(nil)
	_ Tracker = (*TrackerMedianFlow)(nil)
	_ Tracker = (*MeanShiftTracker)(nil)
	_ Tracker = (*CamShiftTracker)(nil)
)

// toGray returns a single-channel luma view of img. A 1-channel Mat is cloned; a
// 3-channel Mat is converted with the BT.601 weights via cv.CvtColor. It panics
// for other channel counts.
func toGray(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return img.Clone()
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		panic("tracking: expected a 1- or 3-channel image")
	}
}

// toHSV returns the HSV form of a 3-channel RGB image; hue is channel 0 in
// [0,179]. It panics unless img has three channels, because hue tracking is
// undefined for grayscale input.
func toHSV(img *cv.Mat) *cv.Mat {
	if img.Channels != 3 {
		panic("tracking: hue tracking requires a 3-channel RGB image")
	}
	return cv.CvtColor(img, cv.ColorRGB2HSV)
}

// clampRect clamps r to lie inside a rows×cols image, shrinking an oversized
// rectangle and forcing a minimum size of 1×1.
func clampRect(r cv.Rect, rows, cols int) cv.Rect {
	if r.Width < 1 {
		r.Width = 1
	}
	if r.Height < 1 {
		r.Height = 1
	}
	if r.Width > cols {
		r.Width = cols
	}
	if r.Height > rows {
		r.Height = rows
	}
	if r.X < 0 {
		r.X = 0
	}
	if r.Y < 0 {
		r.Y = 0
	}
	if r.X+r.Width > cols {
		r.X = cols - r.Width
	}
	if r.Y+r.Height > rows {
		r.Y = rows - r.Height
	}
	if r.X < 0 {
		r.X = 0
	}
	if r.Y < 0 {
		r.Y = 0
	}
	return r
}

// rectCenter returns the fractional centre (x, y) of r.
func rectCenter(r cv.Rect) (float64, float64) {
	return float64(r.X) + float64(r.Width)/2, float64(r.Y) + float64(r.Height)/2
}

// searchWindow returns the region of a rows×cols image to scan for a template of
// size tw×th, centred on box and grown by margin on every side. The window is
// never smaller than the template nor larger than the image, and is shifted so
// it stays fully inside the image.
func searchWindow(box cv.Rect, tw, th, margin, rows, cols int) cv.Rect {
	win := cv.Rect{
		X:      box.X - margin,
		Y:      box.Y - margin,
		Width:  box.Width + 2*margin,
		Height: box.Height + 2*margin,
	}
	if win.Width < tw {
		win.Width = tw
	}
	if win.Height < th {
		win.Height = th
	}
	if win.Width > cols {
		win.Width = cols
	}
	if win.Height > rows {
		win.Height = rows
	}
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

// median returns the median of vals (the mean of the two central elements for an
// even count). It returns 0 for an empty slice and does not modify vals.
func median(vals []float64) float64 {
	n := len(vals)
	if n == 0 {
		return 0
	}
	s := make([]float64, n)
	copy(s, vals)
	sort.Float64s(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// sampleBilinear returns the bilinearly interpolated sample of a single-channel
// Mat at fractional coordinates (x, y), clamping to the image border. It panics
// if m is not single-channel.
func sampleBilinear(m *cv.Mat, x, y float64) float64 {
	if m.Channels != 1 {
		panic("tracking: sampleBilinear requires a single-channel image")
	}
	maxX := float64(m.Cols - 1)
	maxY := float64(m.Rows - 1)
	if x < 0 {
		x = 0
	} else if x > maxX {
		x = maxX
	}
	if y < 0 {
		y = 0
	} else if y > maxY {
		y = maxY
	}
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 > m.Cols-1 {
		x1 = m.Cols - 1
	}
	if y1 > m.Rows-1 {
		y1 = m.Rows - 1
	}
	fx := x - float64(x0)
	fy := y - float64(y0)
	v00 := float64(m.Data[y0*m.Cols+x0])
	v01 := float64(m.Data[y0*m.Cols+x1])
	v10 := float64(m.Data[y1*m.Cols+x0])
	v11 := float64(m.Data[y1*m.Cols+x1])
	top := v00*(1-fx) + v01*fx
	bot := v10*(1-fx) + v11*fx
	return top*(1-fy) + bot*fy
}

package stitching

import (
	"image"

	cv "github.com/malcolmston/opencv"
)

// Timelapser composites the warped panorama images onto a single fixed-size
// canvas, one at a time, so that a sequence of frames can be saved as a
// time-lapse of the panorama being built up. Because every frame shares the same
// canvas and every image keeps its absolute position, the output frames register
// exactly on top of one another — the moving subject appears against a stationary
// stitched background.
//
// Process accumulates an image onto the shared canvas (returned by Canvas);
// Frame instead returns a standalone frame showing a single image on an otherwise
// blank canvas, which is the usual per-image time-lapse output.
type Timelapser struct {
	canvas *cv.Mat
	origin image.Point
}

// NewTimelapser allocates a Timelapser with a blank rows×cols canvas of the given
// channel count, whose top-left corner is the coordinate origin.
func NewTimelapser(rows, cols, channels int) *Timelapser {
	return &Timelapser{canvas: cv.NewMat(rows, cols, channels)}
}

// NewTimelapserForCorners allocates a Timelapser whose canvas is exactly the
// bounding box of the given image corners and sizes (sizes[i] is (X:width,
// Y:height)). The canvas origin is set so that a global corner maps correctly
// onto the canvas.
func NewTimelapserForCorners(corners, sizes []image.Point, channels int) *Timelapser {
	if len(corners) == 0 {
		return NewTimelapser(1, 1, channels)
	}
	minX, minY := corners[0].X, corners[0].Y
	maxX, maxY := corners[0].X+sizes[0].X, corners[0].Y+sizes[0].Y
	for i := range corners {
		minX = minInt(minX, corners[i].X)
		minY = minInt(minY, corners[i].Y)
		maxX = maxInt(maxX, corners[i].X+sizes[i].X)
		maxY = maxInt(maxY, corners[i].Y+sizes[i].Y)
	}
	t := NewTimelapser(maxY-minY, maxX-minX, channels)
	t.origin = image.Point{X: minX, Y: minY}
	return t
}

// Canvas returns the shared accumulation canvas.
func (t *Timelapser) Canvas() *cv.Mat { return t.canvas }

// Size returns the canvas dimensions as (rows, cols).
func (t *Timelapser) Size() (rows, cols int) { return t.canvas.Rows, t.canvas.Cols }

// Process draws img onto the shared canvas at the given global corner. When mask
// is non-nil, only pixels where the mask is positive are drawn, so the black
// border of a warped image does not erase earlier content; a nil mask draws every
// pixel.
func (t *Timelapser) Process(img *cv.Mat, mask *cv.FloatMat, corner image.Point) {
	place(t.canvas, img, mask, corner.X-t.origin.X, corner.Y-t.origin.Y)
}

// Frame returns a fresh canvas-sized image containing only img, drawn at the
// given global corner, with every other pixel left blank. This is the standard
// per-image time-lapse frame.
func (t *Timelapser) Frame(img *cv.Mat, mask *cv.FloatMat, corner image.Point) *cv.Mat {
	frame := cv.NewMat(t.canvas.Rows, t.canvas.Cols, t.canvas.Channels)
	place(frame, img, mask, corner.X-t.origin.X, corner.Y-t.origin.Y)
	return frame
}

// place copies img onto dst so its top-left sample lands at (dstX, dstY),
// clipping at the canvas edges and honouring an optional coverage mask.
func place(dst, img *cv.Mat, mask *cv.FloatMat, dstX, dstY int) {
	if img.Channels != dst.Channels {
		return
	}
	for ry := 0; ry < img.Rows; ry++ {
		dy := dstY + ry
		if dy < 0 || dy >= dst.Rows {
			continue
		}
		for rx := 0; rx < img.Cols; rx++ {
			dx := dstX + rx
			if dx < 0 || dx >= dst.Cols {
				continue
			}
			sp := ry*img.Cols + rx
			if mask != nil && mask.Data[sp] <= 0 {
				continue
			}
			si := sp * img.Channels
			di := (dy*dst.Cols + dx) * dst.Channels
			copy(dst.Data[di:di+img.Channels], img.Data[si:si+img.Channels])
		}
	}
}

package saliency2

import (
	"image"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliency is implemented by the single-image saliency detectors. It
// mirrors OpenCV's cv::saliency::StaticSaliency: a detector consumes one image
// and produces a single-channel saliency map in which brighter samples mark
// more visually salient locations.
type StaticSaliency interface {
	// ComputeSaliency returns a single-channel saliency map the same size as
	// img, min-max normalised to the 8-bit range.
	ComputeSaliency(img *cv.Mat) *cv.Mat
}

// MotionSaliency is implemented by the streaming (video) saliency detectors,
// mirroring OpenCV's cv::saliency::MotionSaliency. Each frame passed to
// ComputeSaliency updates the detector's internal state, so frames must be fed
// in temporal order; Reset clears that state.
type MotionSaliency interface {
	// ComputeSaliency ingests the next frame and returns its motion saliency
	// map as a single-channel [cv.Mat].
	ComputeSaliency(frame *cv.Mat) *cv.Mat
	// Reset discards accumulated temporal state so the next frame is treated as
	// the start of a new sequence.
	Reset()
}

// Objectness is implemented by generic object-proposal generators, mirroring
// OpenCV's cv::saliency::Objectness. It scores and ranks candidate windows that
// are likely to contain an object of any class.
type Objectness interface {
	// ObjectnessBoundingBoxes returns candidate object windows ranked by
	// decreasing objectness score.
	ObjectnessBoundingBoxes(img *cv.Mat) []Box
}

// Box is a scored candidate window produced by an [Objectness] detector. Higher
// Score means the window is more likely to tightly bound an object.
type Box struct {
	// Rect is the window in image coordinates (Min inclusive, Max exclusive).
	Rect image.Rectangle
	// Score is the objectness score; larger is more object-like.
	Score float64
}

// Center returns the centre point of the box as floating-point (x, y)
// coordinates.
func (b Box) Center() (x, y float64) {
	return float64(b.Rect.Min.X+b.Rect.Max.X) / 2, float64(b.Rect.Min.Y+b.Rect.Max.Y) / 2
}

// Area returns the area of the box in pixels.
func (b Box) Area() int {
	d := b.Rect.Size()
	if d.X <= 0 || d.Y <= 0 {
		return 0
	}
	return d.X * d.Y
}

// IoU returns the intersection-over-union overlap of b and other, a value in
// [0,1] where 1 means the two windows coincide exactly and 0 means they are
// disjoint.
func (b Box) IoU(other Box) float64 {
	inter := b.Rect.Intersect(other.Rect)
	if inter.Empty() {
		return 0
	}
	ia := inter.Dx() * inter.Dy()
	ua := b.Area() + other.Area() - ia
	if ua <= 0 {
		return 0
	}
	return float64(ia) / float64(ua)
}

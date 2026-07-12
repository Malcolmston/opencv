package stitching

import (
	"math"
	"testing"
)

// rotZ returns a rotation of theta radians about the Z axis (roll).
func rotZ(theta float64) mat3 {
	c, s := math.Cos(theta), math.Sin(theta)
	return mat3{c, -s, 0, s, c, 0, 0, 0, 1}
}

// waviness measures how far the cameras' x-axes tilt out of the horizontal
// plane: the sum of squared y-components of each rotation's first column.
func waviness(cams []CameraParams) float64 {
	var s float64
	for _, c := range cams {
		y := c.R[3] // column 0, row 1
		s += y * y
	}
	return s
}

func TestWaveCorrectFlattens(t *testing.T) {
	// Build cameras with a yaw sweep plus a growing roll — the classic "wave".
	angles := []float64{-0.3, -0.1, 0.1, 0.3}
	rolls := []float64{-0.15, -0.05, 0.05, 0.15}
	cams := make([]CameraParams, len(angles))
	for i := range angles {
		r := rotZ(rolls[i]).mul(rotY(angles[i]))
		cams[i] = CameraParams{Focal: 250, Aspect: 1, R: [9]float64(r)}
	}
	before := waviness(cams)

	WaveCorrect(cams, WaveCorrectHoriz)

	after := waviness(cams)
	if after > before+1e-9 {
		t.Errorf("wave correction increased waviness: before=%.5f after=%.5f", before, after)
	}
	// Rotations must remain proper after correction.
	for i, c := range cams {
		if d := c.rot().det(); math.Abs(d-1) > 1e-6 {
			t.Errorf("camera %d rotation det = %.6f, want 1", i, d)
		}
	}
}

func TestWaveCorrectVertical(t *testing.T) {
	angles := []float64{-0.2, 0, 0.2}
	cams := make([]CameraParams, len(angles))
	for i := range angles {
		cams[i] = CameraParams{Focal: 200, Aspect: 1, R: [9]float64(rotY(angles[i]))}
	}
	WaveCorrect(cams, WaveCorrectVert)
	for i, c := range cams {
		if d := c.rot().det(); math.Abs(d-1) > 1e-6 {
			t.Errorf("camera %d rotation det = %.6f, want 1", i, d)
		}
	}
}

func TestWaveCorrectEmpty(t *testing.T) {
	// Must not panic on an empty slice.
	WaveCorrect(nil, WaveCorrectHoriz)
}

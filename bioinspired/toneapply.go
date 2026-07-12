package bioinspired

import cv "github.com/malcolmston/opencv"

// ApplyFastToneMapping is a one-shot, allocation-free convenience wrapper around
// [RetinaFastToneMapping]: it builds a tone-mapping operator sized to the input,
// with default neighbourhood radii, and returns the tone-mapped image. Use it
// when a single high-dynamic-range-ish frame needs its shadows lifted and its
// range compressed without keeping an operator around. It panics on an empty
// input.
func ApplyFastToneMapping(input *cv.Mat) *cv.Mat {
	if input.Empty() {
		panic("bioinspired: ApplyFastToneMapping given an empty Mat")
	}
	return NewRetinaFastToneMapping(input.Rows, input.Cols).ProcessFrame(input)
}

// ApplyFastToneMapping tone-maps a single frame using an operator derived from
// this retina's photoreceptor and ganglion-cell spatial constants, mirroring
// OpenCV's Retina::applyFastToneMapping. It carries no temporal state and does
// not disturb the retina's running buffers, so it can be interleaved with
// [Retina.Run]. The input must match the retina's configured size. It panics on
// a size or emptiness mismatch.
func (r *Retina) ApplyFastToneMapping(input *cv.Mat) *cv.Mat {
	if input.Empty() {
		panic("bioinspired: ApplyFastToneMapping given an empty Mat")
	}
	if input.Rows != r.rows || input.Cols != r.cols {
		panic("bioinspired: ApplyFastToneMapping size mismatch with the retina")
	}
	op := r.params.OPLandIplParvo
	tm := NewRetinaFastToneMapping(r.rows, r.cols)
	// Drive the tone-mapper from the retina's own adaptation scales.
	tm.Setup(op.PhotoreceptorsSpatialConstant*4, op.HCellsSpatialConstant, op.GanglionCellsSensitivity)
	return tm.ProcessFrame(input)
}

package bioinspired

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// OnOffChannels holds the two rectified halves of a centre-surround ganglion-cell
// response. Both Mats are single-channel and the same size as the source.
type OnOffChannels struct {
	// On is the ON pathway: the positive half of (centre - surround), strong
	// where a pixel is brighter than its local neighbourhood.
	On *cv.Mat
	// Off is the OFF pathway: the positive half of (surround - centre), strong
	// where a pixel is darker than its local neighbourhood.
	Off *cv.Mat
}

// SplitOnOffChannels models the ON/OFF ganglion-cell split of the parvocellular
// pathway. It band-passes the luminance of the input against a spatial surround
// (a low-pass with the given spatialConstant, in pixels) and rectifies the
// centre-minus-surround difference into two non-negative channels: an ON channel
// that fires where the centre is brighter than its surround, and an OFF channel
// that fires where it is darker. On a flat field both channels are zero; at a
// bright feature on a dark background the ON channel lights up inside the
// feature and the OFF channel in the surrounding dip.
//
// The result is scaled by gain before being quantised to [0,255]. It panics on
// an empty input or a negative spatialConstant.
func SplitOnOffChannels(input *cv.Mat, spatialConstant, gain float64) OnOffChannels {
	if input.Empty() {
		panic("bioinspired: SplitOnOffChannels given an empty Mat")
	}
	if spatialConstant < 0 {
		panic(fmt.Sprintf("bioinspired: SplitOnOffChannels spatialConstant must be >= 0, got %g", spatialConstant))
	}
	lum := luminance(matToFrames(input))
	surround := spatialLowPass(lum, spatialConstantToCoeff(spatialConstant))

	on := newFrame(input.Rows, input.Cols)
	off := newFrame(input.Rows, input.Cols)
	for i := range lum.data {
		d := lum.data[i] - surround.data[i]
		if d > 0 {
			on.data[i] = gain * d
		} else {
			off.data[i] = gain * (-d)
		}
	}
	return OnOffChannels{
		On:  framesToMat([]*frame{on}),
		Off: framesToMat([]*frame{off}),
	}
}

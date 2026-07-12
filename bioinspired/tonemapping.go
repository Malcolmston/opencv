package bioinspired

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// RetinaFastToneMapping is a stateless high-dynamic-range tone-mapping operator
// built from the retina's photoreceptor and ganglion-cell local-adaptation
// stages. It compresses the dynamic range of an input image so that detail in
// both very dark and very bright regions becomes visible in the displayable
// [0,255] range, using two cascaded Naka-Rushton compressions whose reference
// luminance is a spatial low-pass of the signal.
//
// Unlike [Retina], RetinaFastToneMapping keeps no temporal state — each call to
// [RetinaFastToneMapping.ProcessFrame] is independent — so it can be applied to
// single images as well as sequences.
type RetinaFastToneMapping struct {
	rows, cols int

	photoreceptorsNeighborhoodRadius float64
	ganglioncellsNeighborhoodRadius  float64
	meanLuminanceModulatorK          float64
}

// NewRetinaFastToneMapping creates a tone-mapping operator for images of the
// given size with sensible default neighbourhood radii. rows and cols must be
// positive.
func NewRetinaFastToneMapping(rows, cols int) *RetinaFastToneMapping {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("bioinspired: NewRetinaFastToneMapping requires positive size, got %dx%d", rows, cols))
	}
	t := &RetinaFastToneMapping{rows: rows, cols: cols}
	t.Setup(3.0, 1.5, 1.0)
	return t
}

// Setup configures the operator. photoreceptorsNeighborhoodRadius is the
// spatial constant of the photoreceptor local-luminance low-pass (larger =
// broader adaptation), ganglioncellsNeighborhoodRadius is the spatial constant
// of the ganglion-cell stage, and meanLuminanceModulatorK in [0,1] controls how
// strongly each stage adapts to local rather than global luminance (higher =
// stronger local adaptation and more aggressive range compression). Values are
// clamped to valid ranges.
func (t *RetinaFastToneMapping) Setup(photoreceptorsNeighborhoodRadius, ganglioncellsNeighborhoodRadius, meanLuminanceModulatorK float64) {
	if photoreceptorsNeighborhoodRadius < 0 {
		photoreceptorsNeighborhoodRadius = 0
	}
	if ganglioncellsNeighborhoodRadius < 0 {
		ganglioncellsNeighborhoodRadius = 0
	}
	if meanLuminanceModulatorK < 0 {
		meanLuminanceModulatorK = 0
	}
	if meanLuminanceModulatorK > 1 {
		meanLuminanceModulatorK = 1
	}
	t.photoreceptorsNeighborhoodRadius = photoreceptorsNeighborhoodRadius
	t.ganglioncellsNeighborhoodRadius = ganglioncellsNeighborhoodRadius
	t.meanLuminanceModulatorK = meanLuminanceModulatorK
}

// ProcessFrame tone-maps a single image and returns a new [cv.Mat] of the same
// channel count with values compressed into [0,255]. The input is treated as a
// high-dynamic-range-ish signal (its samples may be concentrated in a narrow
// sub-range, e.g. deep shadows); the output redistributes that range so low-end
// detail gains contrast while highlights are held below saturation.
//
// The pipeline runs two cascaded local-adaptation stages driven by the image
// luminance: a photoreceptor stage with a broad spatial reference and a
// ganglion-cell stage with a narrower reference. Colour images are processed by
// applying the luminance-derived gain to each channel, which preserves hue.
func (t *RetinaFastToneMapping) ProcessFrame(input *cv.Mat) *cv.Mat {
	if input.Empty() {
		panic("bioinspired: ProcessFrame given an empty Mat")
	}
	if input.Rows != t.rows || input.Cols != t.cols {
		panic(fmt.Sprintf("bioinspired: ProcessFrame size mismatch: got %dx%d want %dx%d",
			input.Rows, input.Cols, t.rows, t.cols))
	}

	chans := matToFrames(input)
	lum := luminance(chans)

	// Stage 1: photoreceptor local adaptation on luminance.
	aPhoto := spatialConstantToCoeff(t.photoreceptorsNeighborhoodRadius)
	localPhoto := spatialLowPass(lum, aPhoto)
	adapted := nakaRushton(lum, localPhoto, t.meanLuminanceModulatorK, maxSample)

	// Stage 2: ganglion-cell local adaptation on the photoreceptor output.
	aGang := spatialConstantToCoeff(t.ganglioncellsNeighborhoodRadius)
	localGang := spatialLowPass(adapted, aGang)
	toneLum := nakaRushton(adapted, localGang, t.meanLuminanceModulatorK, maxSample)

	// Apply the luminance gain to each channel to preserve colour.
	out := make([]*frame, len(chans))
	for c := range chans {
		out[c] = newFrame(t.rows, t.cols)
		for i := range out[c].data {
			l := lum.data[i]
			if l < eps {
				out[c].data[i] = toneLum.data[i]
				continue
			}
			gain := toneLum.data[i] / l
			out[c].data[i] = chans[c].data[i] * gain
		}
	}
	return framesToMat(out)
}

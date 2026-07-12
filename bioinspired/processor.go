package bioinspired

import cv "github.com/malcolmston/opencv"

// RetinaProcessor wraps a [Retina] and adds OpenCV's channel-activation toggles.
// The parvocellular (contours) and magnocellular (moving contours) channels can
// each be switched off; a deactivated channel still runs internally to keep the
// temporal state consistent, but its accessor returns an all-zero image, exactly
// as OpenCV's Retina::activateContoursProcessing /
// activateMovingContoursProcessing behave.
//
// A RetinaProcessor is stateful and not safe for concurrent use. Both channels
// are active on construction.
type RetinaProcessor struct {
	retina         *Retina
	contours       bool
	movingContours bool
	hasRun         bool
}

// NewRetinaProcessor creates a processor for frames of the given size using
// [DefaultRetinaParameters], with both channels active. rows and cols must be
// positive.
func NewRetinaProcessor(rows, cols int) *RetinaProcessor {
	return NewRetinaProcessorWithParams(rows, cols, DefaultRetinaParameters())
}

// NewRetinaProcessorWithParams is like [NewRetinaProcessor] but uses the
// supplied parameters.
func NewRetinaProcessorWithParams(rows, cols int, p RetinaParameters) *RetinaProcessor {
	return &RetinaProcessor{
		retina:         NewRetinaWithParams(rows, cols, p),
		contours:       true,
		movingContours: true,
	}
}

// Retina returns the underlying [Retina], for access to parameters or the raw
// (unquantised) responses.
func (rp *RetinaProcessor) Retina() *Retina { return rp.retina }

// GetInputSize returns the configured frame size, as (rows, cols).
func (rp *RetinaProcessor) GetInputSize() (rows, cols int) { return rp.retina.GetInputSize() }

// GetOutputSize returns the output image size, as (rows, cols).
func (rp *RetinaProcessor) GetOutputSize() (rows, cols int) { return rp.retina.GetOutputSize() }

// ActivateContoursProcessing enables or disables the parvocellular (contours /
// detail) channel. When disabled, [RetinaProcessor.GetParvo] returns a zero
// image. Mirrors OpenCV's Retina::activateContoursProcessing.
func (rp *RetinaProcessor) ActivateContoursProcessing(activate bool) {
	rp.contours = activate
}

// ActivateMovingContoursProcessing enables or disables the magnocellular
// (moving contours / motion) channel. When disabled, [RetinaProcessor.GetMagno]
// returns a zero image. Mirrors OpenCV's Retina::activateMovingContoursProcessing.
func (rp *RetinaProcessor) ActivateMovingContoursProcessing(activate bool) {
	rp.movingContours = activate
}

// ContoursProcessingActive reports whether the parvo channel is enabled.
func (rp *RetinaProcessor) ContoursProcessingActive() bool { return rp.contours }

// MovingContoursProcessingActive reports whether the magno channel is enabled.
func (rp *RetinaProcessor) MovingContoursProcessingActive() bool { return rp.movingContours }

// Run processes one frame through the underlying retina. See [Retina.Run].
func (rp *RetinaProcessor) Run(input *cv.Mat) {
	rp.retina.Run(input)
	rp.hasRun = true
}

// ClearBuffers resets the retina's temporal state. See [Retina.ClearBuffers].
func (rp *RetinaProcessor) ClearBuffers() {
	rp.retina.ClearBuffers()
	rp.hasRun = false
}

// GetParvo returns the parvo output, or an all-zero image of the input's shape
// when contour processing is disabled. It panics if [RetinaProcessor.Run] has
// not been called since the last [RetinaProcessor.ClearBuffers].
func (rp *RetinaProcessor) GetParvo() *cv.Mat {
	if !rp.hasRun {
		panic("bioinspired: RetinaProcessor.GetParvo called before Run")
	}
	if rp.contours {
		return rp.retina.GetParvo()
	}
	shape := rp.retina.GetParvo()
	return cv.NewMat(shape.Rows, shape.Cols, shape.Channels)
}

// GetMagno returns the single-channel magno output, or an all-zero image when
// moving-contour processing is disabled. It panics if [RetinaProcessor.Run] has
// not been called since the last [RetinaProcessor.ClearBuffers].
func (rp *RetinaProcessor) GetMagno() *cv.Mat {
	if !rp.hasRun {
		panic("bioinspired: RetinaProcessor.GetMagno called before Run")
	}
	if rp.movingContours {
		return rp.retina.GetMagno()
	}
	r, c := rp.retina.GetOutputSize()
	return cv.NewMat(r, c, 1)
}

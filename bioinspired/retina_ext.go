package bioinspired

// GetInputSize returns the frame size the retina expects, as (rows, cols),
// mirroring OpenCV's Retina::getInputSize. It is an alias for [Retina.InputSize]
// provided under the OpenCV name.
func (r *Retina) GetInputSize() (rows, cols int) {
	return r.rows, r.cols
}

// GetOutputSize returns the size of the retina's output images, as (rows, cols),
// mirroring OpenCV's Retina::getOutputSize. The parvo and magno outputs are the
// same spatial size as the input.
func (r *Retina) GetOutputSize() (rows, cols int) {
	return r.rows, r.cols
}

// Write serialises the retina's current parameters to the plain-text format of
// [WriteRetinaParameters], mirroring OpenCV's Retina::write. The text
// round-trips through [Retina.SetupFromText].
func (r *Retina) Write() string {
	return WriteRetinaParameters(r.params)
}

// SetupFromText parses a parameter text (as produced by [Retina.Write] or
// [WriteRetinaParameters]) and applies it, mirroring OpenCV's Retina::setup with
// a parameter file. Keys omitted from the text keep their default values. It
// returns an error (and leaves the retina unchanged) if the text is malformed or
// the resulting parameters fail validation. It does not clear the temporal
// state; call [Retina.ClearBuffers] for a clean restart.
func (r *Retina) SetupFromText(text string) error {
	p, err := ReadRetinaParameters(text)
	if err != nil {
		return err
	}
	r.params = p
	return nil
}

// PrintSetup returns a human-readable description of the retina's current
// parameters, mirroring OpenCV's Retina::printSetup.
func (r *Retina) PrintSetup() string {
	return "Retina setup:\n" + WriteRetinaParameters(r.params)
}

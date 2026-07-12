package bioinspired

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// SegmentationParameters groups the parameters of the
// [TransientAreasSegmentationModule]. Field names mirror OpenCV's
// bioinspired::SegmentationParameters. All spatial constants are in pixels and
// all temporal constants in frames; both must be non-negative. The two
// thresholds are expressed in the same units as the motion-energy signal.
type SegmentationParameters struct {
	// ThresholdON is the motion-contrast a pixel must exceed (local energy above
	// its neighbourhood) to be flagged as a moving area.
	ThresholdON float64
	// ThresholdOFF is a lower release threshold: once flagged, a pixel stays
	// flagged until its motion-contrast drops below this value (hysteresis).
	ThresholdOFF float64
	// LocalEnergyTemporalConstant is the temporal constant of the local
	// motion-energy integrator (small = fast response to new motion).
	LocalEnergyTemporalConstant float64
	// LocalEnergySpatialConstant is the spatial constant of the local
	// motion-energy low-pass (small = tight localisation of the moving region).
	LocalEnergySpatialConstant float64
	// NeighborhoodEnergyTemporalConstant is the temporal constant of the broad
	// surround ("neighbourhood") energy that the local energy is compared against.
	NeighborhoodEnergyTemporalConstant float64
	// NeighborhoodEnergySpatialConstant is the spatial constant of the surround
	// energy; it is normally much larger than the local one so that a compact
	// moving object stands out from its context.
	NeighborhoodEnergySpatialConstant float64
	// ContextEnergyTemporalConstant is the temporal constant of the adaptive
	// context energy used to modulate the decision threshold.
	ContextEnergyTemporalConstant float64
	// ContextEnergySpatialConstant is the spatial constant of the context energy.
	ContextEnergySpatialConstant float64
}

// DefaultSegmentationParameters returns parameters tuned so that a compact,
// persistently moving region is flagged while a static scene produces an empty
// segmentation.
func DefaultSegmentationParameters() SegmentationParameters {
	return SegmentationParameters{
		ThresholdON:                        1.0,
		ThresholdOFF:                       0.5,
		LocalEnergyTemporalConstant:        0.5,
		LocalEnergySpatialConstant:         1.0,
		NeighborhoodEnergyTemporalConstant: 1.0,
		NeighborhoodEnergySpatialConstant:  8.0,
		ContextEnergyTemporalConstant:      1.0,
		ContextEnergySpatialConstant:       12.0,
	}
}

// Validate reports whether the segmentation parameters are usable: constants
// must be non-negative and ThresholdOFF must not exceed ThresholdON (so the
// hysteresis band is well formed).
func (p SegmentationParameters) Validate() error {
	nonneg := []struct {
		name string
		v    float64
	}{
		{"LocalEnergyTemporalConstant", p.LocalEnergyTemporalConstant},
		{"LocalEnergySpatialConstant", p.LocalEnergySpatialConstant},
		{"NeighborhoodEnergyTemporalConstant", p.NeighborhoodEnergyTemporalConstant},
		{"NeighborhoodEnergySpatialConstant", p.NeighborhoodEnergySpatialConstant},
		{"ContextEnergyTemporalConstant", p.ContextEnergyTemporalConstant},
		{"ContextEnergySpatialConstant", p.ContextEnergySpatialConstant},
	}
	for _, f := range nonneg {
		if f.v < 0 {
			return fmt.Errorf("bioinspired: SegmentationParameters.%s must be >= 0, got %g", f.name, f.v)
		}
	}
	if p.ThresholdOFF > p.ThresholdON {
		return fmt.Errorf("bioinspired: SegmentationParameters ThresholdOFF (%g) must not exceed ThresholdON (%g)", p.ThresholdOFF, p.ThresholdON)
	}
	return nil
}

// TransientAreasSegmentationModule segments the moving areas of a scene from a
// transient (motion) signal, typically the magnocellular output of a [Retina].
// It follows OpenCV's bioinspired::TransientAreasSegmentationModule: the input's
// motion energy is integrated locally in space and time, compared against a much
// broader surround energy, and thresholded with hysteresis so that a compact
// moving object is flagged while its static context is not.
//
// The module is stateful: feed the transient of successive frames with
// [TransientAreasSegmentationModule.Run] (or
// [TransientAreasSegmentationModule.RunFloat]) and read the binary result with
// [TransientAreasSegmentationModule.GetSegmentationPicture]. Call
// [TransientAreasSegmentationModule.ClearAllBuffers] to reset the temporal state
// between independent sequences. It is not safe for concurrent use.
type TransientAreasSegmentationModule struct {
	rows, cols int
	params     SegmentationParameters

	localTemporal   *frame
	neighTemporal   *frame
	contextTemporal *frame

	segmented []bool
	hasOutput bool
}

// NewTransientAreasSegmentationModule creates a segmentation module for inputs
// of the given size using [DefaultSegmentationParameters]. rows and cols must be
// positive.
func NewTransientAreasSegmentationModule(rows, cols int) *TransientAreasSegmentationModule {
	return NewTransientAreasSegmentationModuleWithParams(rows, cols, DefaultSegmentationParameters())
}

// NewTransientAreasSegmentationModuleWithParams is like
// [NewTransientAreasSegmentationModule] but uses the supplied parameters. It
// panics if the size is not positive or the parameters do not [SegmentationParameters.Validate].
func NewTransientAreasSegmentationModuleWithParams(rows, cols int, p SegmentationParameters) *TransientAreasSegmentationModule {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("bioinspired: segmentation module requires positive size, got %dx%d", rows, cols))
	}
	if err := p.Validate(); err != nil {
		panic(err.Error())
	}
	m := &TransientAreasSegmentationModule{rows: rows, cols: cols}
	m.Setup(p)
	m.localTemporal = newFrame(rows, cols)
	m.neighTemporal = newFrame(rows, cols)
	m.contextTemporal = newFrame(rows, cols)
	m.segmented = make([]bool, rows*cols)
	return m
}

// Setup replaces the module parameters. It panics if the parameters do not
// [SegmentationParameters.Validate]. It does not reset the temporal state; call
// [TransientAreasSegmentationModule.ClearAllBuffers] for a clean restart.
func (m *TransientAreasSegmentationModule) Setup(p SegmentationParameters) {
	if err := p.Validate(); err != nil {
		panic(err.Error())
	}
	m.params = p
}

// GetParameters returns a copy of the current parameters.
func (m *TransientAreasSegmentationModule) GetParameters() SegmentationParameters {
	return m.params
}

// GetInputSize returns the frame size the module expects, as (rows, cols).
func (m *TransientAreasSegmentationModule) GetInputSize() (rows, cols int) {
	return m.rows, m.cols
}

// GetOutputSize returns the size of the segmentation picture, as (rows, cols).
// It equals the input size.
func (m *TransientAreasSegmentationModule) GetOutputSize() (rows, cols int) {
	return m.rows, m.cols
}

// ClearAllBuffers resets the temporal energy state and discards the last
// segmentation, so the next [TransientAreasSegmentationModule.Run] behaves as
// the first frame of a new sequence.
func (m *TransientAreasSegmentationModule) ClearAllBuffers() {
	m.localTemporal.zero()
	m.neighTemporal.zero()
	m.contextTemporal.zero()
	for i := range m.segmented {
		m.segmented[i] = false
	}
	m.hasOutput = false
}

// Run segments the moving areas of a single-channel transient image (for
// example the quantised magno output of a [Retina]). The Mat is interpreted as a
// motion magnitude; multi-channel inputs are reduced to luminance first. It
// panics on a size or emptiness mismatch.
func (m *TransientAreasSegmentationModule) Run(input *cv.Mat) {
	if input.Empty() {
		panic("bioinspired: segmentation Run given an empty Mat")
	}
	if input.Rows != m.rows || input.Cols != m.cols {
		panic(fmt.Sprintf("bioinspired: segmentation Run size mismatch: got %dx%d want %dx%d",
			input.Rows, input.Cols, m.rows, m.cols))
	}
	m.runFrame(luminance(matToFrames(input)))
}

// RunFloat is like [TransientAreasSegmentationModule.Run] but takes the raw,
// unquantised transient as a [cv.FloatMat] (for example [Retina.GetMagnoRAW]),
// preserving the small signed values of the motion response.
func (m *TransientAreasSegmentationModule) RunFloat(input *cv.FloatMat) {
	if input == nil || len(input.Data) == 0 {
		panic("bioinspired: segmentation RunFloat given an empty FloatMat")
	}
	if input.Rows != m.rows || input.Cols != m.cols {
		panic(fmt.Sprintf("bioinspired: segmentation RunFloat size mismatch: got %dx%d want %dx%d",
			input.Rows, input.Cols, m.rows, m.cols))
	}
	f := newFrame(m.rows, m.cols)
	copy(f.data, input.Data)
	m.runFrame(f)
}

// runFrame performs the segmentation on a single float frame of the transient.
func (m *TransientAreasSegmentationModule) runFrame(signal *frame) {
	p := m.params

	// Rectified instantaneous motion energy.
	energy := absFrame(signal)

	// Local motion energy: fast temporal integration + tight spatial low-pass.
	localE := spatioTemporalLowPass(m.localTemporal, energy,
		p.LocalEnergyTemporalConstant, p.LocalEnergySpatialConstant)

	// Surround ("neighbourhood") energy: broad, slow integration of the local
	// energy. A compact moving object dilutes into this wide average, so it
	// stays well below the local energy at the object.
	neighE := spatioTemporalLowPass(m.neighTemporal, localE,
		p.NeighborhoodEnergyTemporalConstant, p.NeighborhoodEnergySpatialConstant)

	// Motion contrast: how much the local energy stands out from its surround.
	contrast := newFrame(m.rows, m.cols)
	for i := range contrast.data {
		contrast.data[i] = localE.data[i] - neighE.data[i]
	}

	// Adaptive context energy modulates the effective threshold so that busy
	// regions require a stronger transient to be flagged.
	contextE := spatioTemporalLowPass(m.contextTemporal, absFrame(contrast),
		p.ContextEnergyTemporalConstant, p.ContextEnergySpatialConstant)

	for i := range m.segmented {
		onT := p.ThresholdON + contextE.data[i]
		offT := p.ThresholdOFF + contextE.data[i]
		if m.segmented[i] {
			// Hysteresis: stay on until the contrast drops below the release level.
			m.segmented[i] = contrast.data[i] > offT
		} else {
			m.segmented[i] = contrast.data[i] > onT
		}
	}
	m.hasOutput = true
}

// GetSegmentationMask returns the last segmentation as a boolean slice in
// row-major order (true = moving area). The returned slice is a copy. It panics
// if [TransientAreasSegmentationModule.Run] has not been called since the last
// [TransientAreasSegmentationModule.ClearAllBuffers].
func (m *TransientAreasSegmentationModule) GetSegmentationMask() []bool {
	if !m.hasOutput {
		panic("bioinspired: GetSegmentationMask called before Run")
	}
	out := make([]bool, len(m.segmented))
	copy(out, m.segmented)
	return out
}

// GetSegmentationPicture returns the last segmentation as a single-channel
// [cv.Mat] whose samples are 255 for moving areas and 0 elsewhere. It panics if
// [TransientAreasSegmentationModule.Run] has not been called since the last
// [TransientAreasSegmentationModule.ClearAllBuffers].
func (m *TransientAreasSegmentationModule) GetSegmentationPicture() *cv.Mat {
	if !m.hasOutput {
		panic("bioinspired: GetSegmentationPicture called before Run")
	}
	out := cv.NewMat(m.rows, m.cols, 1)
	for i, on := range m.segmented {
		if on {
			out.Data[i] = 255
		}
	}
	return out
}

// PrintSetup returns a human-readable multi-line description of the current
// segmentation parameters.
func (m *TransientAreasSegmentationModule) PrintSetup() string {
	p := m.params
	return fmt.Sprintf("TransientAreasSegmentationModule %dx%d\n"+
		"  ThresholdON = %g\n  ThresholdOFF = %g\n"+
		"  LocalEnergy: temporal = %g, spatial = %g\n"+
		"  NeighborhoodEnergy: temporal = %g, spatial = %g\n"+
		"  ContextEnergy: temporal = %g, spatial = %g\n",
		m.rows, m.cols, p.ThresholdON, p.ThresholdOFF,
		p.LocalEnergyTemporalConstant, p.LocalEnergySpatialConstant,
		p.NeighborhoodEnergyTemporalConstant, p.NeighborhoodEnergySpatialConstant,
		p.ContextEnergyTemporalConstant, p.ContextEnergySpatialConstant)
}

// spatioTemporalLowPass advances a temporal low-pass state in place (retention
// derived from temporalConstant) and returns its spatial low-pass (coefficient
// derived from spatialConstant). It is the common spatio-temporal integrator
// used by the cellular energy stages.
func spatioTemporalLowPass(state, in *frame, temporalConstant, spatialConstant float64) *frame {
	k := temporalConstantToRetention(temporalConstant)
	temporalUpdate(state, in, k)
	return spatialLowPass(state, spatialConstantToCoeff(spatialConstant))
}

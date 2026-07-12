package bioinspired

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// OPLandIplParvoParameters groups the parameters of the outer-plexiform-layer
// (OPL) and parvocellular inner-plexiform-layer (IPL) pathway — the channel
// that produces a denoised, detail- and colour-enhanced image. Field names
// mirror OpenCV's bioinspired::Retina parameters.
type OPLandIplParvoParameters struct {
	// ColorMode selects colour (true) or luminance-only (false) processing of
	// the parvo channel. It is advisory: the actual channel count follows the
	// input Mat.
	ColorMode bool
	// PhotoreceptorsLocalAdaptationSensitivity controls how strongly the
	// photoreceptor compression adapts to the local luminance (0..1). Higher
	// values give stronger local adaptation and dynamic-range compression.
	PhotoreceptorsLocalAdaptationSensitivity float64
	// PhotoreceptorsTemporalConstant is the temporal integration constant (in
	// frames) of the photoreceptor low-pass; larger values smooth more over time.
	PhotoreceptorsTemporalConstant float64
	// PhotoreceptorsSpatialConstant is the spatial integration constant (in
	// pixels) of the photoreceptor/local-luminance low-pass.
	PhotoreceptorsSpatialConstant float64
	// HorizontalCellsGain is the gain of the horizontal-cell feedback that is
	// subtracted from the photoreceptor signal to form the OPL band-pass. 0
	// keeps the low frequencies (no edge boost); values towards 1 remove more of
	// the local mean, sharpening edges.
	HorizontalCellsGain float64
	// HCellsTemporalConstant is the temporal constant of the horizontal cells.
	HCellsTemporalConstant float64
	// HCellsSpatialConstant is the spatial constant of the horizontal-cell
	// low-pass; it is normally larger than the photoreceptor constant.
	HCellsSpatialConstant float64
	// GanglionCellsSensitivity is the sensitivity of the parvo ganglion-cell
	// Naka-Rushton contrast normalisation (0..1).
	GanglionCellsSensitivity float64
}

// IplMagnoParameters groups the parameters of the magnocellular IPL pathway —
// the transient channel that responds to motion and moving contours. Field
// names mirror OpenCV's bioinspired::Retina parameters.
type IplMagnoParameters struct {
	// ParasolCellsBeta is a small leak added to the parasol-cell adaptation.
	ParasolCellsBeta float64
	// ParasolCellsTau is the temporal constant of the parasol cells.
	ParasolCellsTau float64
	// ParasolCellsK is the spatial constant of the parasol-cell low-pass.
	ParasolCellsK float64
	// AmacrinCellsTemporalCutFrequency is the temporal constant (in frames) of
	// the amacrine cells that build the transient response; it sets how quickly
	// a static stimulus fades from the magno channel.
	AmacrinCellsTemporalCutFrequency float64
	// V0CompressionParameter is the sensitivity of the magno output
	// compression (0..1).
	V0CompressionParameter float64
	// LocalAdaptintegrationTau is the temporal constant of the magno local
	// adaptation.
	LocalAdaptintegrationTau float64
	// LocalAdaptintegrationK is the spatial constant of the magno local
	// adaptation low-pass.
	LocalAdaptintegrationK float64
	// MagnoGain scales the rectified transient into the display range.
	MagnoGain float64
}

// RetinaParameters is the full parameter set of the [Retina] model, split into
// the parvo (OPL+IPL) and magno (IPL) pathways, mirroring OpenCV's
// bioinspired::Retina::RetinaParameters.
type RetinaParameters struct {
	OPLandIplParvo OPLandIplParvoParameters
	IplMagno       IplMagnoParameters
}

// DefaultRetinaParameters returns the model's default parameters. They are
// tuned so the parvo channel denoises while preserving edges and colour, and
// the magno channel yields a strong, quickly-adapting transient response.
func DefaultRetinaParameters() RetinaParameters {
	return RetinaParameters{
		OPLandIplParvo: OPLandIplParvoParameters{
			ColorMode:                                true,
			PhotoreceptorsLocalAdaptationSensitivity: 0.75,
			PhotoreceptorsTemporalConstant:           0.5,
			PhotoreceptorsSpatialConstant:            0.53,
			HorizontalCellsGain:                      0.5,
			HCellsTemporalConstant:                   1.0,
			HCellsSpatialConstant:                    3.0,
			GanglionCellsSensitivity:                 0.75,
		},
		IplMagno: IplMagnoParameters{
			ParasolCellsBeta:                 0.0,
			ParasolCellsTau:                  0.0,
			ParasolCellsK:                    2.0,
			AmacrinCellsTemporalCutFrequency: 2.0,
			V0CompressionParameter:           0.95,
			LocalAdaptintegrationTau:         0.0,
			LocalAdaptintegrationK:           1.0,
			MagnoGain:                        3.0,
		},
	}
}

// parvoDetailGain is the fraction of the OPL band-pass (detail) signal added
// back onto the denoised parvo image. It is kept small so detail enhancement
// does not reintroduce the noise removed by spatial smoothing.
const parvoDetailGain = 0.25

// Retina is a simplified Gipsa-lab retina model. It filters a stream of frames
// through two biologically-inspired pathways:
//
//   - the parvocellular (parvo) channel performs photoreceptor local luminance
//     adaptation, horizontal-cell band-pass filtering, spatial noise reduction
//     and ganglion-cell contrast normalisation, producing a denoised, detail-
//     and colour-enhanced image;
//   - the magnocellular (magno) channel performs temporal high-pass filtering
//     (via a temporal state carried across frames) followed by rectification
//     and spatial smoothing, producing a transient response that is strong at
//     moving edges and near zero on a static scene.
//
// A Retina is stateful: call [Retina.Run] once per frame, then read the last
// result with [Retina.GetParvo] and [Retina.GetMagno]. Use [Retina.ClearBuffers]
// to reset the temporal state between independent sequences. A Retina is not
// safe for concurrent use.
type Retina struct {
	rows, cols int
	nChannels  int
	params     RetinaParameters

	// Temporal state carried across frames.
	photoTemporal *frame   // photoreceptor local-luminance temporal low-pass
	hcellTemporal []*frame // horizontal-cell temporal low-pass, per channel
	magnoTemporal *frame   // amacrine-cell temporal low-pass (for the transient)

	// Last outputs.
	parvo     []*frame
	magno     *frame
	hasOutput bool
}

// NewRetina creates a Retina for frames of the given size using
// [DefaultRetinaParameters]. rows and cols must be positive. The channel count
// is taken from the first frame passed to [Retina.Run].
func NewRetina(rows, cols int) *Retina {
	return NewRetinaWithParams(rows, cols, DefaultRetinaParameters())
}

// NewRetinaWithParams is like [NewRetina] but uses the supplied parameters.
func NewRetinaWithParams(rows, cols int, p RetinaParameters) *Retina {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("bioinspired: NewRetina requires positive size, got %dx%d", rows, cols))
	}
	return &Retina{rows: rows, cols: cols, params: p}
}

// InputSize returns the frame size the Retina was constructed for.
func (r *Retina) InputSize() (rows, cols int) { return r.rows, r.cols }

// GetParameters returns a copy of the current parameters.
func (r *Retina) GetParameters() RetinaParameters { return r.params }

// SetParameters replaces the model parameters. It does not clear the temporal
// state; call [Retina.ClearBuffers] if a clean restart is wanted.
func (r *Retina) SetParameters(p RetinaParameters) { r.params = p }

// ClearBuffers resets all temporal state and discards the last output, so the
// next [Retina.Run] behaves as the first frame of a new sequence.
func (r *Retina) ClearBuffers() {
	if r.photoTemporal != nil {
		r.photoTemporal.zero()
	}
	for _, h := range r.hcellTemporal {
		h.zero()
	}
	if r.magnoTemporal != nil {
		r.magnoTemporal.zero()
	}
	r.hasOutput = false
}

// ensureState (re)allocates the temporal buffers for the given channel count.
func (r *Retina) ensureState(nChannels int) {
	if r.photoTemporal != nil && r.nChannels == nChannels {
		return
	}
	r.nChannels = nChannels
	r.photoTemporal = newFrame(r.rows, r.cols)
	r.magnoTemporal = newFrame(r.rows, r.cols)
	r.hcellTemporal = make([]*frame, nChannels)
	for c := range r.hcellTemporal {
		r.hcellTemporal[c] = newFrame(r.rows, r.cols)
	}
}

// Run processes one frame through both pathways and stores the results. input
// must match the size the Retina was constructed for; its channel count may be
// 1 (grayscale) or 3 (RGB) but must stay consistent between calls to a running
// sequence (changing it resets the temporal state). It panics on a size or
// emptiness mismatch.
func (r *Retina) Run(input *cv.Mat) {
	if input.Empty() {
		panic("bioinspired: Retina.Run given an empty Mat")
	}
	if input.Rows != r.rows || input.Cols != r.cols {
		panic(fmt.Sprintf("bioinspired: Retina.Run size mismatch: got %dx%d want %dx%d",
			input.Rows, input.Cols, r.rows, r.cols))
	}
	r.ensureState(input.Channels)

	chans := matToFrames(input)
	lum := luminance(chans)

	op := r.params.OPLandIplParvo

	// --- Photoreceptors: temporal + spatial local-luminance reference. ---
	kPhoto := temporalConstantToRetention(op.PhotoreceptorsTemporalConstant)
	temporalUpdate(r.photoTemporal, lum, kPhoto)
	aPhoto := spatialConstantToCoeff(op.PhotoreceptorsSpatialConstant)
	localLum := spatialLowPass(r.photoTemporal, aPhoto)

	// Photoreceptor compression per channel (Naka-Rushton local adaptation).
	photo := make([]*frame, len(chans))
	for c := range chans {
		photo[c] = nakaRushton(chans[c], localLum, op.PhotoreceptorsLocalAdaptationSensitivity, maxSample)
	}

	// --- Horizontal cells: spatiotemporal low-pass, per channel. ---
	kH := temporalConstantToRetention(op.HCellsTemporalConstant)
	aH := spatialConstantToCoeff(op.HCellsSpatialConstant)
	hcell := make([]*frame, len(chans))
	for c := range chans {
		temporalUpdate(r.hcellTemporal[c], photo[c], kH)
		hcell[c] = spatialLowPass(r.hcellTemporal[c], aH)
	}

	// --- Parvo IPL: denoise + OPL band-pass detail + ganglion normalisation. ---
	aParvo := spatialConstantToCoeff(op.PhotoreceptorsSpatialConstant + 0.5)
	parvo := make([]*frame, len(chans))
	for c := range chans {
		smoothed := spatialLowPass(photo[c], aParvo) // spatial noise reduction
		enhanced := newFrame(r.rows, r.cols)
		for i := range enhanced.data {
			band := photo[c].data[i] - op.HorizontalCellsGain*hcell[c].data[i] // OPL band-pass
			v := smoothed.data[i] + parvoDetailGain*band
			if v < 0 {
				v = 0
			}
			enhanced.data[i] = v
		}
		// Ganglion-cell contrast normalisation against the local luminance.
		parvo[c] = nakaRushton(enhanced, localLum, op.GanglionCellsSensitivity, maxSample)
	}
	r.parvo = parvo

	// --- Magno IPL: temporal high-pass (transient) on the OPL luminance. ---
	mp := r.params.IplMagno
	// Magno is driven by the photoreceptor luminance (post-compression).
	magnoInput := luminance(photo)
	// Transient = deviation of the instantaneous signal from its temporal mean.
	transient := newFrame(r.rows, r.cols)
	for i := range transient.data {
		transient.data[i] = magnoInput.data[i] - r.magnoTemporal.data[i]
	}
	kMagno := temporalConstantToRetention(mp.AmacrinCellsTemporalCutFrequency)
	temporalUpdate(r.magnoTemporal, magnoInput, kMagno)

	// Rectify, smooth spatially (parasol cells), and apply gain.
	rect := absFrame(transient)
	aMagno := spatialConstantToCoeff(mp.ParasolCellsK)
	magnoSmoothed := spatialLowPass(rect, aMagno)
	magno := newFrame(r.rows, r.cols)
	for i := range magno.data {
		magno.data[i] = mp.MagnoGain*magnoSmoothed.data[i] + mp.ParasolCellsBeta
	}
	r.magno = magno

	r.hasOutput = true
}

// GetParvo returns the last parvocellular output as a [cv.Mat] with the same
// channel count as the input (denoised, detail- and colour-enhanced). It panics
// if [Retina.Run] has not been called since the last [Retina.ClearBuffers].
func (r *Retina) GetParvo() *cv.Mat {
	r.requireOutput("GetParvo")
	return framesToMat(r.parvo)
}

// GetMagno returns the last magnocellular output as a single-channel [cv.Mat]
// (the transient/motion response). It panics if [Retina.Run] has not been
// called since the last [Retina.ClearBuffers].
func (r *Retina) GetMagno() *cv.Mat {
	r.requireOutput("GetMagno")
	return framesToMat([]*frame{r.magno})
}

// GetParvoRAW returns the last parvo luminance response as a [cv.FloatMat]
// without 8-bit quantisation, useful for measuring small signals in tests.
func (r *Retina) GetParvoRAW() *cv.FloatMat {
	r.requireOutput("GetParvoRAW")
	return frameToFloatMat(luminance(r.parvo))
}

// GetMagnoRAW returns the last magno response as a [cv.FloatMat] without 8-bit
// quantisation.
func (r *Retina) GetMagnoRAW() *cv.FloatMat {
	r.requireOutput("GetMagnoRAW")
	return frameToFloatMat(r.magno)
}

func (r *Retina) requireOutput(name string) {
	if !r.hasOutput {
		panic("bioinspired: Retina." + name + " called before Run")
	}
}

package mcc

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// CCMModelType selects the feature expansion of a [ColorCorrectionModel]. Beyond
// the simple linear and affine maps of the older [CCM], it offers full
// higher-degree polynomials and Finlayson root-polynomial models, the latter
// being invariant to exposure/scale — a valuable property for real cameras.
type CCMModelType int

const (
	// ModelLinear is a pure 3x3 linear map (3 terms per channel).
	ModelLinear CCMModelType = iota
	// ModelAffine is a 3x3 map plus a constant offset (4 terms).
	ModelAffine
	// ModelPoly2 is the full second-degree polynomial in (r,g,b) including the
	// constant term (10 terms).
	ModelPoly2
	// ModelPoly3 is the full third-degree polynomial in (r,g,b) including the
	// constant term (20 terms).
	ModelPoly3
	// ModelRootPoly2 is the Finlayson root-polynomial of degree 2 (6 terms:
	// r, g, b, √(rg), √(gb), √(rb)); it is invariant to exposure scaling.
	ModelRootPoly2
	// ModelRootPoly3 is the Finlayson root-polynomial of degree 3 (13 terms),
	// also exposure-invariant, with more freedom than [ModelRootPoly2].
	ModelRootPoly3
)

// numTerms returns the number of feature terms produced by the model type, or 0
// for an unknown type.
func (t CCMModelType) numTerms() int {
	switch t {
	case ModelLinear:
		return 3
	case ModelAffine:
		return 4
	case ModelPoly2:
		return 10
	case ModelPoly3:
		return 20
	case ModelRootPoly2:
		return 6
	case ModelRootPoly3:
		return 13
	default:
		return 0
	}
}

// Linearization selects how sRGB samples are decoded to a working space before
// the color matrix is fitted and applied. Fitting in linear light typically
// lowers residual error for real devices.
type Linearization int

const (
	// LinIdentity does no decoding: the model is fitted directly on the 0..1
	// gamma-encoded values.
	LinIdentity Linearization = iota
	// LinSRGB decodes with the standard sRGB transfer function.
	LinSRGB
	// LinGamma decodes with a pure power law using [ColorCorrectionConfig.Gamma].
	LinGamma
)

// ColorCorrectionConfig configures how a [ColorCorrectionModel] is fitted.
type ColorCorrectionConfig struct {
	// Model selects the feature expansion. The zero value is [ModelLinear].
	Model CCMModelType
	// Linearize selects the decoding applied before fitting. The zero value is
	// [LinIdentity].
	Linearize Linearization
	// Gamma is the exponent used when Linearize is [LinGamma] (for example 2.2).
	// Ignored otherwise.
	Gamma float64
	// WhiteBalance, when true, applies a per-channel diagonal gain before the
	// matrix so the measured white patch maps onto the reference white patch.
	// This "white balance first" step decorrelates exposure/illuminant from the
	// matrix fit and usually improves conditioning.
	WhiteBalance bool
	// WhitePatch is the index of the neutral white patch used for white balance.
	// A negative value selects the reference patch with the greatest luminance
	// automatically.
	WhitePatch int
	// Weights optionally gives a per-sample weight for the weighted least-squares
	// fit. When nil or all-equal the fit is ordinary least squares. Its length,
	// when non-nil, must equal the number of samples.
	Weights []float64
}

// ColorCorrectionModel is a fitted color-correction model mapping a device's
// measured colors toward a chart's reference colors. It generalises [CCM] with
// higher-degree and root-polynomial expansions, optional white balancing and
// weighted least squares. Build one with [TrainColorCorrection]; apply it with
// [ColorCorrectionModel.InferRGB] or [ColorCorrectionModel.Infer].
type ColorCorrectionModel struct {
	cfg     ColorCorrectionConfig
	gains   [3]float64   // white-balance gains (1,1,1 when disabled)
	weights [][3]float64 // numTerms x 3 coefficient matrix
}

// TrainColorCorrection fits a [ColorCorrectionModel] mapping the measured colors
// to the reference colors by (optionally weighted) linear least squares. Both
// slices must be the same non-empty length with at least as many samples as the
// model has terms; colors are sRGB triples in the 0..255 range. It returns an
// error on a length mismatch, too few samples, a bad weight vector, an unknown
// model type, or a singular system.
func TrainColorCorrection(measured, reference [][3]float64, cfg ColorCorrectionConfig) (*ColorCorrectionModel, error) {
	if len(measured) != len(reference) {
		return nil, errors.New("mcc: TrainColorCorrection measured and reference lengths differ")
	}
	m := len(measured)
	if m == 0 {
		return nil, errors.New("mcc: TrainColorCorrection needs at least one sample")
	}
	k := cfg.Model.numTerms()
	if k == 0 {
		return nil, errors.New("mcc: TrainColorCorrection unknown CCMModelType")
	}
	if m < k {
		return nil, errors.New("mcc: TrainColorCorrection needs at least as many samples as model terms")
	}
	if cfg.Weights != nil && len(cfg.Weights) != m {
		return nil, errors.New("mcc: TrainColorCorrection weights length must match samples")
	}

	gains := whiteBalanceGains(cfg, measured, reference)

	f := make([][]float64, m)
	y := make([][3]float64, m)
	for i := 0; i < m; i++ {
		f[i] = modelFeatures(cfg, applyGains(gains, measured[i]))
		y[i] = encodeModel(cfg, reference[i])
	}

	a := make([][]float64, k)
	b := make([][]float64, k)
	for r := 0; r < k; r++ {
		a[r] = make([]float64, k)
		b[r] = make([]float64, 3)
	}
	for i := 0; i < m; i++ {
		w := 1.0
		if cfg.Weights != nil {
			w = cfg.Weights[i]
		}
		for r := 0; r < k; r++ {
			wfr := w * f[i][r]
			for c := 0; c < k; c++ {
				a[r][c] += wfr * f[i][c]
			}
			for c := 0; c < 3; c++ {
				b[r][c] += wfr * y[i][c]
			}
		}
	}

	sol, ok := gaussSolve(a, b)
	if !ok {
		return nil, errors.New("mcc: TrainColorCorrection system is singular")
	}
	weights := make([][3]float64, k)
	for r := 0; r < k; r++ {
		weights[r] = [3]float64{sol[r][0], sol[r][1], sol[r][2]}
	}
	return &ColorCorrectionModel{cfg: cfg, gains: gains, weights: weights}, nil
}

// Type returns the model's feature expansion.
func (m *ColorCorrectionModel) Type() CCMModelType { return m.cfg.Model }

// WhiteBalanceGains returns the per-channel gains applied before the matrix
// (1,1,1 when white balancing is disabled).
func (m *ColorCorrectionModel) WhiteBalanceGains() [3]float64 { return m.gains }

// Matrix returns a copy of the fitted coefficient matrix, numTerms rows by three
// columns (one per output channel), in the term order documented for each model.
func (m *ColorCorrectionModel) Matrix() [][3]float64 {
	out := make([][3]float64, len(m.weights))
	copy(out, m.weights)
	return out
}

// InferRGB color-corrects a single sRGB color given in the 0..255 range and
// returns the corrected color in the same range, saturated to [0,255].
func (m *ColorCorrectionModel) InferRGB(rgb [3]float64) [3]float64 {
	f := modelFeatures(m.cfg, applyGains(m.gains, rgb))
	var out [3]float64
	for c := 0; c < 3; c++ {
		var s float64
		for r := range f {
			s += f[r] * m.weights[r][c]
		}
		out[c] = s
	}
	return decodeModel(m.cfg, out)
}

// Infer returns a color-corrected copy of a three-channel RGB image, running
// every pixel through the model with output clamped to the 0..255 range. It
// panics if img is not three-channel.
func (m *ColorCorrectionModel) Infer(img *cv.Mat) *cv.Mat {
	if img.Channels != 3 {
		panic("mcc: ColorCorrectionModel.Infer requires a 3-channel RGB image")
	}
	out := cv.NewMat(img.Rows, img.Cols, 3)
	for i := 0; i < img.Total(); i++ {
		base := i * 3
		c := m.InferRGB([3]float64{
			float64(img.Data[base+0]),
			float64(img.Data[base+1]),
			float64(img.Data[base+2]),
		})
		out.Data[base+0] = clampToUint8(c[0])
		out.Data[base+1] = clampToUint8(c[1])
		out.Data[base+2] = clampToUint8(c[2])
	}
	return out
}

// MeanDeltaE76 returns the mean CIE76 residual between the model's correction of
// the measured colors and the reference colors.
func (m *ColorCorrectionModel) MeanDeltaE76(measured, reference [][3]float64) float64 {
	return m.meanErr(measured, reference, DeltaE76)
}

// MeanDeltaE2000 returns the mean CIEDE2000 residual between the model's
// correction of the measured colors and the reference colors — the perceptually
// preferred way to judge a fitted model.
func (m *ColorCorrectionModel) MeanDeltaE2000(measured, reference [][3]float64) float64 {
	return m.meanErr(measured, reference, DeltaE2000)
}

// meanErr averages a Lab difference metric over the corrected measurements.
func (m *ColorCorrectionModel) meanErr(measured, reference [][3]float64, metric func(a, b [3]float64) float64) float64 {
	if len(measured) != len(reference) || len(measured) == 0 {
		return 0
	}
	errs := make([]float64, len(measured))
	for i := range measured {
		c := m.InferRGB(measured[i])
		errs[i] = metric(rgbToLabF(c[0], c[1], c[2]), rgbToLabF(reference[i][0], reference[i][1], reference[i][2]))
	}
	return mean(errs)
}

// PatchColorReport describes one patch after correction: the measured,
// corrected and reference colors in both sRGB (0..255) and CIE L*a*b* (D65),
// with the CIE76 and CIEDE2000 residuals of the corrected color against the
// reference.
type PatchColorReport struct {
	Index        int
	MeasuredRGB  [3]float64
	CorrectedRGB [3]float64
	ReferenceRGB [3]float64
	MeasuredLab  [3]float64
	CorrectedLab [3]float64
	ReferenceLab [3]float64
	DeltaE76     float64
	DeltaE2000   float64
}

// Report returns a per-patch [PatchColorReport] for every measured/reference
// pair, in input order — the detailed per-patch Lab and Delta E breakdown used
// to inspect where a correction succeeds or fails.
func (m *ColorCorrectionModel) Report(measured, reference [][3]float64) []PatchColorReport {
	n := len(measured)
	if len(reference) < n {
		n = len(reference)
	}
	out := make([]PatchColorReport, n)
	for i := 0; i < n; i++ {
		corr := m.InferRGB(measured[i])
		mLab := rgbToLabF(measured[i][0], measured[i][1], measured[i][2])
		cLab := rgbToLabF(corr[0], corr[1], corr[2])
		rLab := rgbToLabF(reference[i][0], reference[i][1], reference[i][2])
		out[i] = PatchColorReport{
			Index:        i,
			MeasuredRGB:  measured[i],
			CorrectedRGB: corr,
			ReferenceRGB: reference[i],
			MeasuredLab:  mLab,
			CorrectedLab: cLab,
			ReferenceLab: rLab,
			DeltaE76:     DeltaE76(cLab, rLab),
			DeltaE2000:   DeltaE2000(cLab, rLab),
		}
	}
	return out
}

// LuminanceWeights returns a per-sample weight vector proportional to each
// reference color's relative luminance raised to the given power, suitable for
// [ColorCorrectionConfig.Weights]. A power of 0 yields equal weights; positive
// powers bias the fit toward brighter patches (a common CCM heuristic).
func LuminanceWeights(reference [][3]float64, power float64) []float64 {
	w := make([]float64, len(reference))
	for i, c := range reference {
		y := RGBToXYZ(clampToUint8(c[0]), clampToUint8(c[1]), clampToUint8(c[2]))[1]
		w[i] = math.Pow(y, power)
	}
	return w
}

// whiteBalanceGains computes the diagonal pre-matrix gains, or (1,1,1) when
// white balancing is disabled or the chosen white patch is unusable.
func whiteBalanceGains(cfg ColorCorrectionConfig, measured, reference [][3]float64) [3]float64 {
	if !cfg.WhiteBalance {
		return [3]float64{1, 1, 1}
	}
	idx := cfg.WhitePatch
	if idx < 0 || idx >= len(measured) {
		idx = brightestPatch(reference)
	}
	meas := measured[idx]
	ref := reference[idx]
	var g [3]float64
	for c := 0; c < 3; c++ {
		if meas[c] <= 0 {
			g[c] = 1
		} else {
			g[c] = ref[c] / meas[c]
		}
	}
	return g
}

// brightestPatch returns the index of the color with the greatest channel sum.
func brightestPatch(colors [][3]float64) int {
	best := -1.0
	idx := 0
	for i, c := range colors {
		s := c[0] + c[1] + c[2]
		if s > best {
			best = s
			idx = i
		}
	}
	return idx
}

// applyGains scales a 0..255 color by the per-channel white-balance gains.
func applyGains(g, rgb [3]float64) [3]float64 {
	return [3]float64{rgb[0] * g[0], rgb[1] * g[1], rgb[2] * g[2]}
}

// encodeModelComponent maps a 0..255 sRGB component into the model's working
// space per the configured linearization.
func encodeModelComponent(cfg ColorCorrectionConfig, v float64) float64 {
	c := clamp01(v / 255)
	switch cfg.Linearize {
	case LinSRGB:
		return SRGBToLinear(c)
	case LinGamma:
		if cfg.Gamma > 0 {
			return math.Pow(c, cfg.Gamma)
		}
		return c
	default:
		return c
	}
}

// decodeModelComponent inverts encodeModelComponent back to the 0..255 range.
func decodeModelComponent(cfg ColorCorrectionConfig, c float64) float64 {
	c = clamp01(c)
	switch cfg.Linearize {
	case LinSRGB:
		c = LinearToSRGB(c)
	case LinGamma:
		if cfg.Gamma > 0 {
			c = math.Pow(c, 1/cfg.Gamma)
		}
	}
	return clamp01(c) * 255
}

// encodeModel maps a 0..255 triple into working space component-wise.
func encodeModel(cfg ColorCorrectionConfig, rgb [3]float64) [3]float64 {
	return [3]float64{
		encodeModelComponent(cfg, rgb[0]),
		encodeModelComponent(cfg, rgb[1]),
		encodeModelComponent(cfg, rgb[2]),
	}
}

// decodeModel maps a working-space triple back to 0..255.
func decodeModel(cfg ColorCorrectionConfig, c [3]float64) [3]float64 {
	return [3]float64{
		decodeModelComponent(cfg, c[0]),
		decodeModelComponent(cfg, c[1]),
		decodeModelComponent(cfg, c[2]),
	}
}

// rootTerm returns the p-th root of a nonnegative product, guarding against tiny
// negatives from the working-space clamp.
func rootTerm(v float64, p float64) float64 {
	if v < 0 {
		v = 0
	}
	return math.Pow(v, 1/p)
}

// modelFeatures builds the model's feature vector for a 0..255 sRGB color, first
// mapping it into the working space.
func modelFeatures(cfg ColorCorrectionConfig, rgb [3]float64) []float64 {
	e := encodeModel(cfg, rgb)
	r, g, b := e[0], e[1], e[2]
	switch cfg.Model {
	case ModelLinear:
		return []float64{r, g, b}
	case ModelAffine:
		return []float64{r, g, b, 1}
	case ModelPoly2:
		return polyFull(r, g, b, 2)
	case ModelPoly3:
		return polyFull(r, g, b, 3)
	case ModelRootPoly2:
		return []float64{
			r, g, b,
			rootTerm(r*g, 2), rootTerm(g*b, 2), rootTerm(r*b, 2),
		}
	case ModelRootPoly3:
		return []float64{
			r, g, b,
			rootTerm(r*g, 2), rootTerm(g*b, 2), rootTerm(r*b, 2),
			rootTerm(r*g*b, 3),
			rootTerm(r*r*g, 3), rootTerm(g*g*r, 3),
			rootTerm(g*g*b, 3), rootTerm(b*b*g, 3),
			rootTerm(r*r*b, 3), rootTerm(b*b*r, 3),
		}
	default:
		return nil
	}
}

// polyFull returns every monomial r^i g^j b^k with i+j+k <= degree, including
// the constant term, in a fixed deterministic order.
func polyFull(r, g, b float64, degree int) []float64 {
	var out []float64
	for total := 0; total <= degree; total++ {
		for i := total; i >= 0; i-- {
			for j := total - i; j >= 0; j-- {
				k := total - i - j
				out = append(out, math.Pow(r, float64(i))*math.Pow(g, float64(j))*math.Pow(b, float64(k)))
			}
		}
	}
	return out
}

package mcc

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// CCMType selects the algebraic shape of a [CCM] color-correction model.
type CCMType int

const (
	// CCMLinear3x3 is a pure 3x3 linear map (three coefficients per output
	// channel, no offset). It preserves black and is the classic color matrix.
	CCMLinear3x3 CCMType = iota
	// CCMAffine3x4 is a 3x3 map plus a constant offset per output channel (four
	// coefficients each), able to correct a black-level shift.
	CCMAffine3x4
	// CCMPolynomial expands each color into a fixed degree-2 polynomial
	// (r, g, b, rg, gb, br, r^2, g^2, b^2, 1) before the linear map, capturing
	// mild non-linearity at the cost of more coefficients.
	CCMPolynomial
)

// numTerms returns the number of polynomial/feature terms for the model type.
func (t CCMType) numTerms() int {
	switch t {
	case CCMLinear3x3:
		return 3
	case CCMAffine3x4:
		return 4
	case CCMPolynomial:
		return 10
	default:
		return 0
	}
}

// CCMConfig configures how a [CCM] is fitted and applied.
type CCMConfig struct {
	// Type selects the model shape. The zero value, CCMLinear3x3, is a sensible
	// default.
	Type CCMType
	// Linearize fits and applies the model in linear-light space: colors are
	// expanded out of their gamma encoding before the matrix and re-compressed
	// afterwards. This usually lowers residual error for real cameras.
	Linearize bool
	// Gamma sets the encoding used when Linearize is true. A value <= 0 uses the
	// standard sRGB transfer function; a positive value uses a pure power law
	// (component^Gamma to linearise, the inverse to re-encode), e.g. 2.2.
	Gamma float64
}

// CCM is a fitted color-correction model mapping a device's measured colors
// toward a chart's reference colors. Create one with [TrainCCM] and apply it
// with [CCM.Apply] or [CCM.ApplyRGB].
type CCM struct {
	cfg     CCMConfig
	weights [][3]float64 // numTerms x 3 coefficient matrix
}

// TrainCCM fits a color-correction model that maps the measured colors to the
// reference colors by linear least squares (solving the normal equations). Both
// slices must be the same non-empty length with at least as many samples as the
// model has terms, and colors are given as sRGB triples in the 0..255 range
// (measured typically from [CChecker.MeasuredRGB], reference from
// [ReferenceRGB]). It returns an error on a size mismatch, too few samples, or a
// singular (degenerate) system.
func TrainCCM(measured, reference [][3]float64, cfg CCMConfig) (*CCM, error) {
	if len(measured) != len(reference) {
		return nil, errors.New("mcc: TrainCCM measured and reference lengths differ")
	}
	m := len(measured)
	k := cfg.Type.numTerms()
	if k == 0 {
		return nil, errors.New("mcc: TrainCCM unknown CCMType")
	}
	if m < k {
		return nil, errors.New("mcc: TrainCCM needs at least as many samples as model terms")
	}

	// Build feature matrix F (m x k) and target matrix Y (m x 3).
	f := make([][]float64, m)
	y := make([][3]float64, m)
	for i := 0; i < m; i++ {
		f[i] = features(cfg, measured[i])
		y[i] = encode(cfg, reference[i])
	}

	// Normal equations: A = F^T F (k x k), B = F^T Y (k x 3).
	a := make([][]float64, k)
	b := make([][]float64, k)
	for r := 0; r < k; r++ {
		a[r] = make([]float64, k)
		b[r] = make([]float64, 3)
	}
	for i := 0; i < m; i++ {
		for r := 0; r < k; r++ {
			fr := f[i][r]
			for c := 0; c < k; c++ {
				a[r][c] += fr * f[i][c]
			}
			for c := 0; c < 3; c++ {
				b[r][c] += fr * y[i][c]
			}
		}
	}

	w, ok := gaussSolve(a, b)
	if !ok {
		return nil, errors.New("mcc: TrainCCM system is singular")
	}
	weights := make([][3]float64, k)
	for r := 0; r < k; r++ {
		weights[r] = [3]float64{w[r][0], w[r][1], w[r][2]}
	}
	return &CCM{cfg: cfg, weights: weights}, nil
}

// Type returns the model's shape.
func (m *CCM) Type() CCMType { return m.cfg.Type }

// Matrix returns a copy of the fitted coefficient matrix as numTerms rows of
// three columns (one column per output channel). For [CCMLinear3x3] this is the
// familiar 3x3 color matrix; for [CCMAffine3x4] the fourth row is the offset;
// for [CCMPolynomial] the rows follow the term order documented on the type.
func (m *CCM) Matrix() [][3]float64 {
	out := make([][3]float64, len(m.weights))
	copy(out, m.weights)
	return out
}

// ApplyRGB color-corrects a single sRGB color given in the 0..255 range and
// returns the corrected color in the same range (unclamped to integers but
// saturated to [0,255]).
func (m *CCM) ApplyRGB(rgb [3]float64) [3]float64 {
	f := features(m.cfg, rgb)
	var out [3]float64
	for c := 0; c < 3; c++ {
		var s float64
		for r := range f {
			s += f[r] * m.weights[r][c]
		}
		out[c] = s
	}
	return decode(m.cfg, out)
}

// Apply returns a color-corrected copy of a three-channel RGB image, running
// every pixel through the model. It panics if img is not three-channel.
func (m *CCM) Apply(img *cv.Mat) *cv.Mat {
	if img.Channels != 3 {
		panic("mcc: CCM.Apply requires a 3-channel RGB image")
	}
	out := cv.NewMat(img.Rows, img.Cols, 3)
	for i := 0; i < img.Total(); i++ {
		base := i * 3
		c := m.ApplyRGB([3]float64{
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

// MeanError returns the mean CIE76 Delta E between the model's correction of the
// measured colors and the reference colors — the residual error after
// correction. It is the natural way to check that a fitted model actually
// improves on the raw measurements.
func (m *CCM) MeanError(measured, reference [][3]float64) float64 {
	if len(measured) != len(reference) || len(measured) == 0 {
		return 0
	}
	errs := make([]float64, len(measured))
	for i := range measured {
		c := m.ApplyRGB(measured[i])
		errs[i] = DeltaE76(rgbToLabF(c[0], c[1], c[2]), rgbToLabF(reference[i][0], reference[i][1], reference[i][2]))
	}
	return mean(errs)
}

// MeanDeltaE returns the mean CIE76 Delta E between two equal-length lists of
// sRGB colors (0..255). It is handy for reporting error before correction, to
// compare against [CCM.MeanError] after it.
func MeanDeltaE(a, b [][3]float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	errs := make([]float64, len(a))
	for i := range a {
		errs[i] = DeltaE76(rgbToLabF(a[i][0], a[i][1], a[i][2]), rgbToLabF(b[i][0], b[i][1], b[i][2]))
	}
	return mean(errs)
}

// encodeComponent maps a 0..255 sRGB component into the fitting space: scaled to
// [0,1] and, when linearisation is enabled, expanded to linear light.
func encodeComponent(cfg CCMConfig, v float64) float64 {
	c := clamp01(v / 255)
	if !cfg.Linearize {
		return c
	}
	if cfg.Gamma > 0 {
		return math.Pow(c, cfg.Gamma)
	}
	return SRGBToLinear(c)
}

// decodeComponent inverts encodeComponent, mapping a fitting-space value back to
// the 0..255 sRGB range.
func decodeComponent(cfg CCMConfig, c float64) float64 {
	c = clamp01(c)
	if cfg.Linearize {
		if cfg.Gamma > 0 {
			c = math.Pow(c, 1/cfg.Gamma)
		} else {
			c = LinearToSRGB(c)
		}
	}
	return clamp01(c) * 255
}

// encode maps a 0..255 sRGB triple into the fitting space component-wise.
func encode(cfg CCMConfig, rgb [3]float64) [3]float64 {
	return [3]float64{
		encodeComponent(cfg, rgb[0]),
		encodeComponent(cfg, rgb[1]),
		encodeComponent(cfg, rgb[2]),
	}
}

// decode maps a fitting-space triple back to the 0..255 sRGB range.
func decode(cfg CCMConfig, c [3]float64) [3]float64 {
	return [3]float64{
		decodeComponent(cfg, c[0]),
		decodeComponent(cfg, c[1]),
		decodeComponent(cfg, c[2]),
	}
}

// features builds the model's feature vector for a 0..255 sRGB color, first
// mapping it into the fitting space.
func features(cfg CCMConfig, rgb [3]float64) []float64 {
	e := encode(cfg, rgb)
	r, g, b := e[0], e[1], e[2]
	switch cfg.Type {
	case CCMLinear3x3:
		return []float64{r, g, b}
	case CCMAffine3x4:
		return []float64{r, g, b, 1}
	case CCMPolynomial:
		return []float64{r, g, b, r * g, g * b, b * r, r * r, g * g, b * b, 1}
	default:
		return nil
	}
}

// gaussSolve solves the linear system a*x = b, where a is k x k and b is k x p,
// using Gauss-Jordan elimination with partial pivoting. It returns x (k x p) and
// reports whether a was non-singular. The inputs are not modified.
func gaussSolve(a [][]float64, b [][]float64) ([][]float64, bool) {
	k := len(a)
	p := len(b[0])
	// Work on copies.
	m := make([][]float64, k)
	x := make([][]float64, k)
	for i := 0; i < k; i++ {
		m[i] = make([]float64, k)
		copy(m[i], a[i])
		x[i] = make([]float64, p)
		copy(x[i], b[i])
	}
	for col := 0; col < k; col++ {
		piv := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < k; r++ {
			if math.Abs(m[r][col]) > best {
				best = math.Abs(m[r][col])
				piv = r
			}
		}
		if best < 1e-12 {
			return nil, false
		}
		m[col], m[piv] = m[piv], m[col]
		x[col], x[piv] = x[piv], x[col]
		pivVal := m[col][col]
		for c := col; c < k; c++ {
			m[col][c] /= pivVal
		}
		for c := 0; c < p; c++ {
			x[col][c] /= pivVal
		}
		for r := 0; r < k; r++ {
			if r == col {
				continue
			}
			factor := m[r][col]
			if factor == 0 {
				continue
			}
			for c := col; c < k; c++ {
				m[r][c] -= factor * m[col][c]
			}
			for c := 0; c < p; c++ {
				x[r][c] -= factor * x[col][c]
			}
		}
	}
	return x, true
}

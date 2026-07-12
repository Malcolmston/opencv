package bgsegm

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BackgroundSubtractorMOG is the original adaptive Gaussian-mixture background
// model of KaewTraKulPong and Bowden ("An Improved Adaptive Background Mixture
// Model for Real-time Tracking with Shadow Detection", 2001) — the predecessor
// of [BackgroundSubtractorMOG2]. Every pixel keeps up to NMixtures weighted
// Gaussians over its intensity. For each frame the model looks for the first
// Gaussian (ranked by weight ÷ standard deviation) that the observation falls
// within MatchSigma standard deviations of; that component is nudged toward the
// observation and its weight raised, while the others decay. When nothing
// matches, the least-confident component is reborn on the observation. A pixel
// is background when its matching Gaussian lies among the most confident ones
// that together hold BackgroundRatio of the mixture weight.
//
// Unlike MOG2 the number of active components is fixed and matching uses a plain
// standard-deviation gate rather than MOG2's variance-adaptive test, so MOG is
// lighter and reacts a little more abruptly. Construct one with
// [NewBackgroundSubtractorMOG]; the zero value is not usable.
type BackgroundSubtractorMOG struct {
	// ShadowParams supplies the embedded DetectShadows / ShadowValue /
	// ShadowThreshold configuration and their setters.
	ShadowParams

	// History is the nominal look-back length; it caps the steady-state learning
	// rate at 1/History.
	History int
	// NMixtures is the number of Gaussians kept per pixel.
	NMixtures int
	// BackgroundRatio is the fraction of total mixture weight, taken from the
	// most confident components, that constitutes the background.
	BackgroundRatio float64
	// MatchSigma is the matching gate in standard deviations: an observation
	// matches a component when (value-mean)² < (MatchSigma·σ)². The default 2.5
	// mirrors the original paper.
	MatchSigma float64
	// NoiseSigma is the standard deviation assigned to a freshly spawned
	// Gaussian.
	NoiseSigma float64
	// VarMin and VarMax bound the per-component variance, preventing a component
	// from collapsing to zero width on a perfectly static pixel.
	VarMin float64
	VarMax float64
	// OpenKernel, when greater than 1, morphologically opens the mask at that odd
	// size before Apply returns it (see [CleanupMask]).
	OpenKernel int

	rows, cols int
	frameCount int
	models     [][]gaussian
	inited     bool
}

// NewBackgroundSubtractorMOG creates an original-MOG subtractor. history and
// nmixtures fall back to the OpenCV defaults (200 and 5) when non-positive;
// detectShadows toggles shadow classification. The remaining tunables are set to
// sensible defaults on the returned value and may be overridden before the first
// Apply.
func NewBackgroundSubtractorMOG(history, nmixtures int, detectShadows bool) *BackgroundSubtractorMOG {
	if history <= 0 {
		history = 200
	}
	if nmixtures <= 0 {
		nmixtures = 5
	}
	sp := defaultShadowParams()
	sp.DetectShadows = detectShadows
	return &BackgroundSubtractorMOG{
		ShadowParams:    sp,
		History:         history,
		NMixtures:       nmixtures,
		BackgroundRatio: 0.7,
		MatchSigma:      2.5,
		NoiseSigma:      15,
		VarMin:          4,
		VarMax:          75,
	}
}

func (b *BackgroundSubtractorMOG) init(frame *cv.Mat) {
	b.rows, b.cols = frame.Rows, frame.Cols
	b.models = make([][]gaussian, frame.Total())
	for i := range b.models {
		b.models[i] = make([]gaussian, b.NMixtures)
	}
	b.inited = true
}

// Apply classifies frame, updates the mixture model and returns the foreground
// mask. See [BackgroundSubtractor].
func (b *BackgroundSubtractorMOG) Apply(frame *cv.Mat) *cv.Mat {
	intensity := toIntensity(frame)
	if !b.inited {
		b.init(frame)
	} else {
		checkFrame(b.rows, b.cols, frame)
	}
	b.frameCount++
	alpha := 1.0 / float64(min(b.frameCount, b.History))

	mask := newMask(b.rows, b.cols)
	for p := range b.models {
		mask.Data[p] = b.updatePixel(b.models[p], intensity[p], alpha)
	}
	return applyCleanup(mask, b.OpenKernel)
}

// updatePixel performs the per-pixel match, update and classification for a
// single observation and returns the mask sample.
func (b *BackgroundSubtractorMOG) updatePixel(g []gaussian, v, alpha float64) uint8 {
	gate := b.MatchSigma * b.MatchSigma
	matched := -1
	best := math.Inf(1)
	for k := range g {
		if g[k].weight <= 0 {
			continue
		}
		d := v - g[k].mean
		d2 := d * d
		if d2 < gate*g[k].variance && d2 < best {
			best = d2
			matched = k
		}
	}

	// Decay every weight toward the ownership indicator.
	for k := range g {
		if g[k].weight <= 0 {
			continue
		}
		own := 0.0
		if k == matched {
			own = 1.0
		}
		g[k].weight += alpha * (own - g[k].weight)
		if g[k].weight < 0 {
			g[k].weight = 0
		}
	}

	if matched >= 0 {
		rho := alpha / g[matched].weight
		if rho > 1 {
			rho = 1
		}
		d := v - g[matched].mean
		g[matched].mean += rho * d
		g[matched].variance += rho * (d*d - g[matched].variance)
		b.clampVar(&g[matched].variance)
	} else {
		slot := 0
		minKey := math.Inf(1)
		for k := range g {
			key := sortKey(g[k])
			if key < minKey {
				minKey = key
				slot = k
			}
		}
		g[slot].mean = v
		g[slot].variance = b.NoiseSigma * b.NoiseSigma
		b.clampVar(&g[slot].variance)
		g[slot].weight = alpha
	}

	normalizeWeights(g)
	return b.classify(g, v, matched)
}

func (b *BackgroundSubtractorMOG) clampVar(varp *float64) {
	if *varp < b.VarMin {
		*varp = b.VarMin
	}
	if *varp > b.VarMax {
		*varp = b.VarMax
	}
}

// classify decides the mask sample for a pixel given its updated mixture, the
// observation and the index of the matched Gaussian (-1 if none).
func (b *BackgroundSubtractorMOG) classify(g []gaussian, v float64, matched int) uint8 {
	bg := backgroundSet(g, b.BackgroundRatio)
	if matched >= 0 && bg[matched] {
		return BackgroundValue
	}
	if b.DetectShadows {
		for k := range g {
			if bg[k] && b.isShadowOf(v, g[k].mean) {
				return b.shadowSample()
			}
		}
	}
	return ForegroundValue
}

// GetBackgroundImage returns the per-pixel mean of the most confident background
// Gaussian as a single-channel image, or nil before the first Apply.
func (b *BackgroundSubtractorMOG) GetBackgroundImage() *cv.Mat {
	if !b.inited {
		return nil
	}
	out := cv.NewMat(b.rows, b.cols, 1)
	for p, g := range b.models {
		best := -1
		bestKey := 0.0
		for k := range g {
			if g[k].weight <= 0 {
				continue
			}
			key := sortKey(g[k])
			if best < 0 || key > bestKey {
				bestKey = key
				best = k
			}
		}
		if best >= 0 {
			out.Data[p] = clampUint8(g[best].mean)
		}
	}
	return out
}

var _ BackgroundSubtractor = (*BackgroundSubtractorMOG)(nil)

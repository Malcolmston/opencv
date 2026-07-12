package bgsegm

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// gaussian is one component of a per-pixel Gaussian mixture. Intensities are on
// the 0..255 scale; variance is stored (not standard deviation).
type gaussian struct {
	weight   float64
	mean     float64
	variance float64
}

// BackgroundSubtractorMOG2 is an adaptive Gaussian-mixture background model
// after Zivkovic ("Improved Adaptive Gaussian Mixture Model for Background
// Subtraction", 2004). Every pixel maintains up to NMixtures weighted Gaussians
// over its intensity. For each frame the closest matching Gaussian is updated —
// or, when nothing matches, the weakest component is replaced by a new Gaussian
// centred on the observation — and the mixture weights adapt with a learning
// rate of 1/min(frameCount, History). A pixel is background when its matching
// Gaussian belongs to the high-weight components that together account for
// BackgroundRatio of the mixture mass; otherwise it is foreground, optionally
// refined to a shadow classification.
//
// Construct one with [NewBackgroundSubtractorMOG2]. The exported fields may be
// tuned before the first call to Apply; the zero value is not usable.
type BackgroundSubtractorMOG2 struct {
	// History is the nominal number of frames the model looks back over; it
	// sets the steady-state learning rate to 1/History.
	History int
	// VarThreshold is the squared-Mahalanobis threshold for matching an
	// observation to a Gaussian: a component matches when (value-mean)² is below
	// VarThreshold·variance. The default of 16 corresponds to 4 standard
	// deviations.
	VarThreshold float64
	// DetectShadows enables classifying darkened background pixels as
	// [ShadowValue] instead of [ForegroundValue].
	DetectShadows bool
	// NMixtures is the maximum number of Gaussians kept per pixel.
	NMixtures int
	// BackgroundRatio is the fraction of total mixture weight, taken from the
	// most confident components, that defines the background model. A match
	// among those components is classified as background.
	BackgroundRatio float64
	// VarInit is the variance assigned to a freshly spawned Gaussian.
	VarInit float64
	// VarMin and VarMax bound the per-component variance.
	VarMin float64
	VarMax float64
	// ShadowThreshold is the darkest relative intensity (value/mean) still
	// considered a shadow rather than foreground, used only when DetectShadows
	// is set. The default 0.5 accepts pixels down to half the background
	// brightness.
	ShadowThreshold float64
	// OpenKernel, when greater than 1, runs a morphological opening of that odd
	// size on the mask before Apply returns it (see [CleanupMask]).
	OpenKernel int

	rows, cols int
	frameCount int
	models     [][]gaussian
	inited     bool
}

// NewBackgroundSubtractorMOG2 creates a MOG2 subtractor. history and
// varThreshold fall back to the OpenCV defaults (500 and 16) when non-positive;
// detectShadows toggles shadow classification. The remaining tunables are set
// to sensible defaults on the returned value and may be overridden before the
// first Apply.
func NewBackgroundSubtractorMOG2(history int, varThreshold float64, detectShadows bool) *BackgroundSubtractorMOG2 {
	if history <= 0 {
		history = 500
	}
	if varThreshold <= 0 {
		varThreshold = 16
	}
	return &BackgroundSubtractorMOG2{
		History:         history,
		VarThreshold:    varThreshold,
		DetectShadows:   detectShadows,
		NMixtures:       5,
		BackgroundRatio: 0.9,
		VarInit:         15,
		VarMin:          4,
		VarMax:          75,
		ShadowThreshold: 0.5,
	}
}

func (b *BackgroundSubtractorMOG2) init(frame *cv.Mat) {
	b.rows, b.cols = frame.Rows, frame.Cols
	b.models = make([][]gaussian, frame.Total())
	for i := range b.models {
		b.models[i] = make([]gaussian, b.NMixtures)
	}
	b.inited = true
}

// Apply classifies frame, updates the mixture model and returns the foreground
// mask. See [BackgroundSubtractor].
func (b *BackgroundSubtractorMOG2) Apply(frame *cv.Mat) *cv.Mat {
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

// updatePixel runs the per-pixel match, update and classification for one
// observation, returning the mask sample for that pixel.
func (b *BackgroundSubtractorMOG2) updatePixel(g []gaussian, v, alpha float64) uint8 {
	// Find the closest active Gaussian that matches the observation.
	matched := -1
	best := math.Inf(1)
	for k := range g {
		if g[k].weight <= 0 {
			continue
		}
		d := v - g[k].mean
		d2 := d * d
		if d2 < b.VarThreshold*g[k].variance && d2 < best {
			best = d2
			matched = k
		}
	}

	// Adapt all mixture weights toward the ownership indicator.
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
		// Move the matched Gaussian toward the observation.
		rho := alpha / g[matched].weight
		if rho > 1 {
			rho = 1
		}
		d := v - g[matched].mean
		g[matched].mean += rho * d
		g[matched].variance += rho * (d*d - g[matched].variance)
		if g[matched].variance < b.VarMin {
			g[matched].variance = b.VarMin
		}
		if g[matched].variance > b.VarMax {
			g[matched].variance = b.VarMax
		}
	} else {
		// Replace the weakest component with a new Gaussian on the observation.
		slot := 0
		minW := math.Inf(1)
		for k := range g {
			if g[k].weight < minW {
				minW = g[k].weight
				slot = k
			}
		}
		g[slot].mean = v
		g[slot].variance = b.VarInit
		g[slot].weight = alpha
	}

	normalizeWeights(g)
	return b.classify(g, v, matched)
}

// classify decides the mask value for a pixel given its updated mixture, the
// observation v and the index of the matched Gaussian (-1 if none matched).
func (b *BackgroundSubtractorMOG2) classify(g []gaussian, v float64, matched int) uint8 {
	bg := backgroundSet(g, b.BackgroundRatio)
	if matched >= 0 && bg[matched] {
		return BackgroundValue
	}
	if b.DetectShadows && b.isShadow(g, bg, v) {
		return ShadowValue
	}
	return ForegroundValue
}

// isShadow reports whether v looks like a darkened version of one of the
// background Gaussians: dimmer than the mean but no darker than
// ShadowThreshold·mean.
func (b *BackgroundSubtractorMOG2) isShadow(g []gaussian, bg []bool, v float64) bool {
	for k := range g {
		if !bg[k] || g[k].mean <= 0 {
			continue
		}
		if v <= g[k].mean && v >= b.ShadowThreshold*g[k].mean {
			return true
		}
	}
	return false
}

// GetBackgroundImage returns the per-pixel mean of the most confident
// background Gaussian as a single-channel image, or nil before the first Apply.
func (b *BackgroundSubtractorMOG2) GetBackgroundImage() *cv.Mat {
	if !b.inited {
		return nil
	}
	out := cv.NewMat(b.rows, b.cols, 1)
	for p, g := range b.models {
		best := -1
		bestW := 0.0
		for k := range g {
			if g[k].weight > bestW {
				bestW = g[k].weight
				best = k
			}
		}
		if best >= 0 {
			out.Data[p] = clampUint8(g[best].mean)
		}
	}
	return out
}

// normalizeWeights rescales the mixture weights to sum to one.
func normalizeWeights(g []gaussian) {
	sum := 0.0
	for k := range g {
		sum += g[k].weight
	}
	if sum <= 0 {
		return
	}
	for k := range g {
		g[k].weight /= sum
	}
}

// backgroundSet marks the smallest set of highest-confidence Gaussians whose
// cumulative weight reaches ratio. Components are ranked by weight/√variance so
// that tight, heavy Gaussians are preferred. The returned slice is indexed like
// g.
func backgroundSet(g []gaussian, ratio float64) []bool {
	order := make([]int, 0, len(g))
	for k := range g {
		if g[k].weight > 0 {
			order = append(order, k)
		}
	}
	sort.SliceStable(order, func(i, j int) bool {
		return sortKey(g[order[i]]) > sortKey(g[order[j]])
	})
	bg := make([]bool, len(g))
	cum := 0.0
	for _, k := range order {
		bg[k] = true
		cum += g[k].weight
		if cum >= ratio {
			break
		}
	}
	return bg
}

// sortKey ranks a Gaussian by confidence: high weight and low variance first.
func sortKey(gg gaussian) float64 {
	if gg.variance <= 0 {
		return gg.weight
	}
	return gg.weight / math.Sqrt(gg.variance)
}

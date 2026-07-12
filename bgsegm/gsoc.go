package bgsegm

import (
	"math"
	"math/rand"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// BackgroundSubtractorGSOC is the sample-consensus background model contributed
// to OpenCV's bgsegm module during Google Summer of Code. Each pixel keeps a
// bank of NSamples recent intensity samples; a pixel is background when at least
// MinMatches stored samples lie within DistThreshold of the observation. Matched
// background pixels refresh one of their own samples, and — modelling spatial
// coherence — one sample of a random neighbour, both with probability
// 1/ReplaceRate and 1/PropagationRate respectively.
//
// The distinguishing GSOC feature is static-foreground absorption: a pixel that
// remains foreground for StaticFrames consecutive frames (a parked car, a
// dropped bag) has its entire sample bank overwritten with the current value and
// is reclassified as background, so genuinely stationary objects melt into the
// scene while moving ones keep standing out. All pseudo-random choices are
// seeded from Seed, so the output is fully reproducible.
//
// Construct one with [NewBackgroundSubtractorGSOC]; the zero value is not usable.
type BackgroundSubtractorGSOC struct {
	// ShadowParams supplies the embedded DetectShadows / ShadowValue /
	// ShadowThreshold configuration and their setters.
	ShadowParams

	// NSamples is the number of intensity samples stored per pixel.
	NSamples int
	// MinMatches is the minimum number of neighbouring samples for a background
	// classification.
	MinMatches int
	// DistThreshold is the absolute intensity distance within which a stored
	// sample counts as a match.
	DistThreshold float64
	// ReplaceRate sets the probability 1/ReplaceRate that a matched pixel
	// refreshes one of its own samples with the observation.
	ReplaceRate int
	// PropagationRate sets the probability 1/PropagationRate that a matched pixel
	// also refreshes one sample of a random neighbour.
	PropagationRate int
	// StaticFrames is the number of consecutive foreground frames after which a
	// pixel is treated as a new static background object and absorbed. Zero or
	// negative disables absorption.
	StaticFrames int
	// Seed seeds the deterministic pseudo-random sample replacement.
	Seed int64
	// OpenKernel, when greater than 1, morphologically opens the mask at that odd
	// size before Apply returns it (see [CleanupMask]).
	OpenKernel int

	rows, cols int
	samples    [][]float64
	fgStreak   []int
	rng        *rand.Rand
	inited     bool
}

// NewBackgroundSubtractorGSOC creates a GSOC subtractor. nSamples and
// distThreshold fall back to sensible defaults (20 and 30) when non-positive;
// detectShadows toggles shadow classification. The remaining tunables are
// initialised to working defaults on the returned value and may be overridden
// before the first Apply.
func NewBackgroundSubtractorGSOC(nSamples int, distThreshold float64, detectShadows bool) *BackgroundSubtractorGSOC {
	if nSamples <= 0 {
		nSamples = 20
	}
	if distThreshold <= 0 {
		distThreshold = 30
	}
	sp := defaultShadowParams()
	sp.DetectShadows = detectShadows
	return &BackgroundSubtractorGSOC{
		ShadowParams:    sp,
		NSamples:        nSamples,
		MinMatches:      2,
		DistThreshold:   distThreshold,
		ReplaceRate:     8,
		PropagationRate: 16,
		StaticFrames:    150,
		Seed:            1,
	}
}

func (b *BackgroundSubtractorGSOC) init(frame *cv.Mat, intensity []float64) {
	b.rows, b.cols = frame.Rows, frame.Cols
	b.rng = rand.New(rand.NewSource(b.Seed))
	b.samples = make([][]float64, frame.Total())
	b.fgStreak = make([]int, frame.Total())
	for p := range b.samples {
		row := make([]float64, b.NSamples)
		for i := range row {
			row[i] = intensity[p]
		}
		b.samples[p] = row
	}
	b.inited = true
}

// Apply classifies frame, updates the sample banks (including static-foreground
// absorption) and returns the foreground mask. See [BackgroundSubtractor].
func (b *BackgroundSubtractorGSOC) Apply(frame *cv.Mat) *cv.Mat {
	intensity := toIntensity(frame)
	if !b.inited {
		b.init(frame, intensity)
	} else {
		checkFrame(b.rows, b.cols, frame)
	}

	mask := newMask(b.rows, b.cols)
	for p := range b.samples {
		mask.Data[p] = b.classify(b.samples[p], intensity[p])
	}
	// Update pass, ordered after classification so the current mask reflects the
	// pre-update model.
	for p := range b.samples {
		v := intensity[p]
		if mask.Data[p] == BackgroundValue {
			b.fgStreak[p] = 0
			if b.rng.Intn(b.ReplaceRate) == 0 {
				b.samples[p][b.rng.Intn(b.NSamples)] = v
			}
			if b.rng.Intn(b.PropagationRate) == 0 {
				np := b.randomNeighbour(p)
				b.samples[np][b.rng.Intn(b.NSamples)] = v
			}
			continue
		}
		// Foreground (or shadow): grow the streak and absorb if static too long.
		b.fgStreak[p]++
		if b.StaticFrames > 0 && b.fgStreak[p] >= b.StaticFrames {
			for i := range b.samples[p] {
				b.samples[p][i] = v
			}
			b.fgStreak[p] = 0
			mask.Data[p] = BackgroundValue
		}
	}
	return applyCleanup(mask, b.OpenKernel)
}

// classify returns the mask sample for a pixel given its sample bank and the
// current observation.
func (b *BackgroundSubtractorGSOC) classify(samples []float64, v float64) uint8 {
	matches := 0
	nearest := math.Inf(1)
	var nearestS float64
	for _, s := range samples {
		d := math.Abs(v - s)
		if d < nearest {
			nearest = d
			nearestS = s
		}
		if d < b.DistThreshold {
			matches++
			if matches >= b.MinMatches {
				return BackgroundValue
			}
		}
	}
	if b.isShadowOf(v, nearestS) {
		return b.shadowSample()
	}
	return ForegroundValue
}

// randomNeighbour returns a random 4-connected neighbour index of p, clamped to
// stay inside the image (falling back to p itself at a border where the drawn
// direction leaves the image).
func (b *BackgroundSubtractorGSOC) randomNeighbour(p int) int {
	y, x := p/b.cols, p%b.cols
	switch b.rng.Intn(4) {
	case 0:
		if y > 0 {
			return p - b.cols
		}
	case 1:
		if y < b.rows-1 {
			return p + b.cols
		}
	case 2:
		if x > 0 {
			return p - 1
		}
	default:
		if x < b.cols-1 {
			return p + 1
		}
	}
	return p
}

// GetBackgroundImage returns the per-pixel median stored sample as a
// single-channel image, or nil before the first Apply.
func (b *BackgroundSubtractorGSOC) GetBackgroundImage() *cv.Mat {
	if !b.inited {
		return nil
	}
	out := cv.NewMat(b.rows, b.cols, 1)
	buf := make([]float64, b.NSamples)
	for p, s := range b.samples {
		copy(buf, s)
		sort.Float64s(buf)
		out.Data[p] = clampUint8(buf[len(buf)/2])
	}
	return out
}

var _ BackgroundSubtractor = (*BackgroundSubtractorGSOC)(nil)

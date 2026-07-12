package tracking

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// TrackerBoosting is a lightweight online-boosting tracker in the spirit of
// Grabner et al.'s "Real-Time Tracking via On-line Boosting" (the classic
// OpenCV BOOSTING tracker, itself an online AdaBoost over Haar features). Each
// frame it draws one positive patch (the current location) and a ring of
// negative patches, updates a persistent pool of online generative weak learners
// ([weakClassifier]) over generalised Haar features, then runs several rounds of
// discrete AdaBoost over the freshly sampled patches to pick a set of weighted
// selectors. Detection scores the weighted weak-learner votes across a search
// grid and moves the box to the strongest, positively classified location.
//
// It is deterministic given its seed. Construct it with [NewTrackerBoosting].
type TrackerBoosting struct {
	// NumFeatures is the size of the Haar feature pool.
	NumFeatures int
	// NumSelectors is the number of AdaBoost rounds (selectors).
	NumSelectors int
	// NegSamples is the number of negative patches sampled per frame.
	NegSamples int
	// NegRadiusInner and NegRadiusOuter bound the negative sampling annulus.
	NegRadiusInner, NegRadiusOuter int
	// SearchRadius is the half-size (px) of the detection search grid.
	SearchRadius int
	// LearnRate is the online adaptation rate of the weak learners.
	LearnRate float64
	// Seed makes the feature pool and negative sampling reproducible.
	Seed int64

	rng    *rand.Rand
	pool   []haarFeature
	weak   []weakClassifier
	sel    []int
	alpha  []float64
	pw, ph int
	cx, cy int
	inited bool
}

// NewTrackerBoosting returns a TrackerBoosting with sensible defaults (120
// features, 30 selectors, 40 negatives in [8,28], SearchRadius 8, LearnRate
// 0.8, Seed 1).
func NewTrackerBoosting() *TrackerBoosting {
	return &TrackerBoosting{
		NumFeatures: 120, NumSelectors: 30, NegSamples: 40,
		NegRadiusInner: 8, NegRadiusOuter: 28,
		SearchRadius: 8, LearnRate: 0.8, Seed: 1,
	}
}

// Init builds the feature pool and trains the first strong classifier.
func (t *TrackerBoosting) Init(frame *cv.Mat, bbox cv.Rect) {
	gray := toGray(frame)
	b := clampRect(bbox, gray.Rows, gray.Cols)
	t.pw, t.ph = b.Width, b.Height
	t.cx = b.X + b.Width/2
	t.cy = b.Y + b.Height/2
	t.rng = rand.New(rand.NewSource(t.Seed))
	t.pool = generateHaarPool(t.rng, t.NumFeatures, t.pw, t.ph)
	t.weak = make([]weakClassifier, t.NumFeatures)
	ii := newIntegralImage(gray)
	t.trainAt(ii, t.cx, t.cy)
	t.inited = true
}

func (t *TrackerBoosting) topLeft(cx, cy int) (int, int) { return cx - t.pw/2, cy - t.ph/2 }

func (t *TrackerBoosting) featuresAt(ii *integralImage, cx, cy int) []float64 {
	ox, oy := t.topLeft(cx, cy)
	vals := make([]float64, len(t.pool))
	for i, f := range t.pool {
		vals[i] = f.eval(ii, ox, oy)
	}
	return vals
}

// negativeOffsets draws NegSamples deterministic offsets from the annulus.
func (t *TrackerBoosting) negativeOffsets() [][2]int {
	all := sampleOffsets(t.NegRadiusInner, t.NegRadiusOuter, 1)
	if len(all) <= t.NegSamples {
		return all
	}
	t.rng.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
	return all[:t.NegSamples]
}

// trainAt updates the weak learners with one positive and several negative
// samples, then re-fits the AdaBoost selectors over those samples.
func (t *TrackerBoosting) trainAt(ii *integralImage, cx, cy int) {
	posFeat := t.featuresAt(ii, cx, cy)
	negOff := t.negativeOffsets()
	negFeat := make([][]float64, len(negOff))
	for i, o := range negOff {
		negFeat[i] = t.featuresAt(ii, cx+o[0], cy+o[1])
	}

	for m := range t.weak {
		t.weak[m].updateClass(posFeat[m], true, t.LearnRate)
		for _, nf := range negFeat {
			t.weak[m].updateClass(nf[m], false, t.LearnRate)
		}
	}
	t.adaBoost(posFeat, negFeat)
}

// weakVote returns the ±1 vote of weak learner m on feature value v.
func (t *TrackerBoosting) weakVote(m int, v float64) float64 {
	if t.weak[m].logOdds(v) >= 0 {
		return 1
	}
	return -1
}

// adaBoost runs discrete AdaBoost over the current samples to select weighted
// weak learners.
func (t *TrackerBoosting) adaBoost(posFeat []float64, negFeat [][]float64) {
	type sample struct {
		feat []float64
		y    float64
	}
	samples := make([]sample, 0, 1+len(negFeat))
	samples = append(samples, sample{feat: posFeat, y: 1})
	for _, nf := range negFeat {
		samples = append(samples, sample{feat: nf, y: -1})
	}
	n := len(samples)
	w := make([]float64, n)
	for i := range w {
		w[i] = 1.0 / float64(n)
	}

	// Precompute votes: votes[m][i].
	votes := make([][]float64, len(t.weak))
	for m := range t.weak {
		v := make([]float64, n)
		for i, s := range samples {
			v[i] = t.weakVote(m, s.feat[m])
		}
		votes[m] = v
	}

	t.sel = t.sel[:0]
	t.alpha = t.alpha[:0]
	used := make([]bool, len(t.weak))
	for r := 0; r < t.NumSelectors; r++ {
		bestM := -1
		bestErr := math.Inf(1)
		for m := range t.weak {
			if used[m] {
				continue
			}
			var err float64
			for i, s := range samples {
				if votes[m][i] != s.y {
					err += w[i]
				}
			}
			if err < bestErr {
				bestErr = err
				bestM = m
			}
		}
		if bestM < 0 {
			break
		}
		e := bestErr
		if e <= 1e-6 {
			e = 1e-6
		}
		if e >= 0.5 {
			// No weak learner beats chance on this weighting; stop.
			break
		}
		alpha := 0.5 * math.Log((1-e)/e)
		used[bestM] = true
		t.sel = append(t.sel, bestM)
		t.alpha = append(t.alpha, alpha)
		var z float64
		for i, s := range samples {
			w[i] *= math.Exp(-alpha * s.y * votes[bestM][i])
			z += w[i]
		}
		if z > 0 {
			for i := range w {
				w[i] /= z
			}
		}
	}
	if len(t.sel) == 0 {
		// Degenerate frame: keep at least the single best weak learner.
		t.sel = append(t.sel, 0)
		t.alpha = append(t.alpha, 1)
	}
}

// strongScore returns the weighted-vote margin for the patch at (cx, cy).
func (t *TrackerBoosting) strongScore(ii *integralImage, cx, cy int) float64 {
	ox, oy := t.topLeft(cx, cy)
	var s float64
	for k, m := range t.sel {
		s += t.alpha[k] * t.weakVote(m, t.pool[m].eval(ii, ox, oy))
	}
	return s
}

// UpdateConfidence scans the search grid, moves the box to the strongest
// response, retrains, and returns the box with the weighted-vote margin as
// confidence. It panics before Init.
func (t *TrackerBoosting) UpdateConfidence(frame *cv.Mat) (cv.Rect, float64) {
	if !t.inited {
		panic("tracking: TrackerBoosting.Update called before Init")
	}
	gray := toGray(frame)
	ii := newIntegralImage(gray)

	bestVal := math.Inf(-1)
	bx, by := t.cx, t.cy
	r := t.SearchRadius
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			s := t.strongScore(ii, t.cx+dx, t.cy+dy)
			if s > bestVal {
				bestVal = s
				bx, by = t.cx+dx, t.cy+dy
			}
		}
	}
	t.cx, t.cy = bx, by
	t.trainAt(ii, t.cx, t.cy)
	return t.box(gray), bestVal
}

// Update satisfies [Tracker]; the flag is true when the weighted-vote margin is
// positive.
func (t *TrackerBoosting) Update(frame *cv.Mat) (cv.Rect, bool) {
	box, conf := t.UpdateConfidence(frame)
	return box, conf > 0
}

func (t *TrackerBoosting) box(gray *cv.Mat) cv.Rect {
	r := cv.Rect{X: t.cx - t.pw/2, Y: t.cy - t.ph/2, Width: t.pw, Height: t.ph}
	return clampRect(r, gray.Rows, gray.Cols)
}

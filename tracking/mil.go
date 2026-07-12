package tracking

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// TrackerMIL implements Multiple-Instance-Learning tracking (Babenko et al.,
// 2009). Rather than a single labelled patch, each frame provides a positive
// bag of patches near the current location and negative patches farther away;
// the tracker never has to decide which positive patch is the "true" object.
// A pool of online generative weak classifiers ([weakClassifier]) is maintained
// over generalised Haar features ([haarFeature]) computed from an integral
// image, and a strong classifier is assembled by greedy MILBoost selection: at
// each step the weak learner that most increases the noisy-OR bag log-likelihood
// is added. Detection evaluates the strong classifier over a search grid and
// moves the box to the maximum.
//
// It is deterministic given its seed. Construct it with [NewTrackerMIL].
type TrackerMIL struct {
	// NumFeatures is the size of the Haar feature pool.
	NumFeatures int
	// NumSelected is the number of weak classifiers in the strong classifier.
	NumSelected int
	// PosRadius is the radius (px) of the positive sampling region.
	PosRadius int
	// NegRadiusInner and NegRadiusOuter bound the negative sampling annulus.
	NegRadiusInner, NegRadiusOuter int
	// SearchRadius is the half-size (px) of the detection search grid.
	SearchRadius int
	// LearnRate is the online adaptation rate of the weak classifiers.
	LearnRate float64
	// Seed makes the feature pool reproducible.
	Seed int64

	rng     *rand.Rand
	pool    []haarFeature
	weak    []weakClassifier
	sel     []int // indices of selected weak classifiers
	pw, ph  int
	cx, cy  int
	lastVal float64
	inited  bool
}

// NewTrackerMIL returns a TrackerMIL with sensible defaults (150 features, 25
// selected, PosRadius 3, negatives in [8,28], SearchRadius 8, LearnRate 0.85,
// Seed 1).
func NewTrackerMIL() *TrackerMIL {
	return &TrackerMIL{
		NumFeatures: 150, NumSelected: 25,
		PosRadius: 3, NegRadiusInner: 8, NegRadiusOuter: 28,
		SearchRadius: 8, LearnRate: 0.85, Seed: 1,
	}
}

// sigmoid maps a log-odds score to a probability.
func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// Init builds the feature pool and trains the first strong classifier.
func (t *TrackerMIL) Init(frame *cv.Mat, bbox cv.Rect) {
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

// topLeft converts a patch centre to its integral-image top-left corner.
func (t *TrackerMIL) topLeft(cx, cy int) (int, int) {
	return cx - t.pw/2, cy - t.ph/2
}

// featuresAt evaluates every pooled feature for the patch centred at (cx, cy).
func (t *TrackerMIL) featuresAt(ii *integralImage, cx, cy int) []float64 {
	ox, oy := t.topLeft(cx, cy)
	vals := make([]float64, len(t.pool))
	for i, f := range t.pool {
		vals[i] = f.eval(ii, ox, oy)
	}
	return vals
}

// trainAt samples positive/negative bags around (cx, cy), updates every weak
// classifier, and re-selects the strong classifier by MILBoost.
func (t *TrackerMIL) trainAt(ii *integralImage, cx, cy int) {
	posOff := sampleOffsets(0, t.PosRadius, 1)
	negOff := sampleOffsets(t.NegRadiusInner, t.NegRadiusOuter, 4)

	posFeat := make([][]float64, 0, len(posOff))
	for _, o := range posOff {
		posFeat = append(posFeat, t.featuresAt(ii, cx+o[0], cy+o[1]))
	}
	negFeat := make([][]float64, 0, len(negOff))
	for _, o := range negOff {
		negFeat = append(negFeat, t.featuresAt(ii, cx+o[0], cy+o[1]))
	}

	// Online update of every weak classifier from both bags.
	for m := 0; m < len(t.weak); m++ {
		for _, pf := range posFeat {
			t.weak[m].updateClass(pf[m], true, t.LearnRate)
		}
		for _, nf := range negFeat {
			t.weak[m].updateClass(nf[m], false, t.LearnRate)
		}
	}

	t.sel = t.milBoostSelect(posFeat, negFeat)
}

// milBoostSelect greedily chooses NumSelected weak classifiers that maximise the
// noisy-OR positive-bag log-likelihood plus the negative-sample log-likelihood.
func (t *TrackerMIL) milBoostSelect(posFeat, negFeat [][]float64) []int {
	nPos := len(posFeat)
	nNeg := len(negFeat)
	// Running strong-classifier scores for each sample.
	posScore := make([]float64, nPos)
	negScore := make([]float64, nNeg)

	// Precompute each weak classifier's log-odds on every sample.
	posOdds := make([][]float64, len(t.weak))
	negOdds := make([][]float64, len(t.weak))
	for m := range t.weak {
		po := make([]float64, nPos)
		for i, pf := range posFeat {
			po[i] = t.weak[m].logOdds(pf[m])
		}
		no := make([]float64, nNeg)
		for i, nf := range negFeat {
			no[i] = t.weak[m].logOdds(nf[m])
		}
		posOdds[m] = po
		negOdds[m] = no
	}

	selected := make([]int, 0, t.NumSelected)
	used := make([]bool, len(t.weak))
	for k := 0; k < t.NumSelected && k < len(t.weak); k++ {
		bestM := -1
		bestL := math.Inf(-1)
		for m := range t.weak {
			if used[m] {
				continue
			}
			l := bagLogLikelihood(posScore, negScore, posOdds[m], negOdds[m])
			if l > bestL {
				bestL = l
				bestM = m
			}
		}
		if bestM < 0 {
			break
		}
		used[bestM] = true
		selected = append(selected, bestM)
		for i := range posScore {
			posScore[i] += posOdds[bestM][i]
		}
		for i := range negScore {
			negScore[i] += negOdds[bestM][i]
		}
	}
	return selected
}

// bagLogLikelihood evaluates the MILBoost objective if the candidate weak
// classifier (posAdd/negAdd) were added to the current scores: a single
// noisy-OR positive bag plus independent negative samples.
func bagLogLikelihood(posScore, negScore, posAdd, negAdd []float64) float64 {
	// Noisy-OR: p_bag = 1 - Π(1 - p_ij).
	logProdNeg := 0.0 // ln Π(1 - p_ij)
	for i := range posScore {
		p := sigmoid(posScore[i] + posAdd[i])
		logProdNeg += math.Log(1 - p + 1e-12)
	}
	pBag := 1 - math.Exp(logProdNeg)
	l := math.Log(pBag + 1e-12)
	for i := range negScore {
		p := sigmoid(negScore[i] + negAdd[i])
		l += math.Log(1 - p + 1e-12)
	}
	return l
}

// strongScore returns the strong-classifier log-odds for the patch at (cx, cy).
func (t *TrackerMIL) strongScore(ii *integralImage, cx, cy int) float64 {
	ox, oy := t.topLeft(cx, cy)
	var s float64
	for _, m := range t.sel {
		s += t.weak[m].logOdds(t.pool[m].eval(ii, ox, oy))
	}
	return s
}

// UpdateConfidence scans the search grid, moves the box to the strongest
// response, retrains, and returns the box with the strong-classifier score
// (a log-odds margin) as confidence. It panics before Init.
func (t *TrackerMIL) UpdateConfidence(frame *cv.Mat) (cv.Rect, float64) {
	if !t.inited {
		panic("tracking: TrackerMIL.Update called before Init")
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
	t.lastVal = bestVal
	t.trainAt(ii, t.cx, t.cy)
	return t.box(gray), bestVal
}

// Update satisfies [Tracker]; the flag is true when the strong-classifier margin
// is positive (object more likely than background).
func (t *TrackerMIL) Update(frame *cv.Mat) (cv.Rect, bool) {
	box, conf := t.UpdateConfidence(frame)
	return box, conf > 0
}

func (t *TrackerMIL) box(gray *cv.Mat) cv.Rect {
	r := cv.Rect{X: t.cx - t.pw/2, Y: t.cy - t.ph/2, Width: t.pw, Height: t.ph}
	return clampRect(r, gray.Rows, gray.Cols)
}

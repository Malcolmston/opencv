package xobjdetect

import (
	"encoding/gob"
	"errors"
	"io"
	"math"
	"math/rand"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Default detector parameters.
const (
	defaultWinW        = 24
	defaultWinH        = 24
	defaultRounds      = 64
	defaultNumFeatures = 256
	defaultScaleFactor = 1.2
	defaultStep        = 4
	defaultNMSOverlap  = 0.3
)

// WBDetector is a WaldBoost object detector, the pure-Go counterpart of OpenCV's
// cv::xobjdetect::WBDetector. It learns a boosted integral-channel-feature
// classifier from positive and negative sample patches and detects the learned
// object across a downscaling image pyramid with a fast SPRT early-exit cascade.
//
// Construct one with [NewWBDetector] (or [NewWBDetectorSize] to choose the
// detection window), adjust the exported tuning fields if desired, then call
// [WBDetector.Train]. A trained detector can be serialised with
// [WBDetector.Write] and restored with [WBDetector.Read].
type WBDetector struct {
	// WinW, WinH are the detection-window dimensions in pixels. Training patches
	// are resized to this size.
	WinW, WinH int
	// Rounds is the maximum number of boosting stumps to fit.
	Rounds int
	// NumFeatures is the size of the random integral-channel feature pool.
	NumFeatures int
	// Seed makes feature-pool sampling and training reproducible.
	Seed int64
	// ScaleFactor is the ratio between successive pyramid levels (> 1). Values
	// <= 1 default to 1.2.
	ScaleFactor float64
	// Step is the sliding-window stride in pixels (>= 1).
	Step int
	// NMSOverlap is the intersection-over-union above which overlapping
	// detections are merged by non-maximum suppression, in [0,1].
	NMSOverlap float64
	// DetectThreshold is the minimum confidence score a window must reach to be
	// reported. The natural boundary is 0.
	DetectThreshold float64

	pool    *FeaturePool
	boost   *WaldBoost
	trained bool
}

// NewWBDetector returns a detector with the default 24x24 window and default
// tuning.
func NewWBDetector() *WBDetector {
	return NewWBDetectorSize(defaultWinW, defaultWinH)
}

// NewWBDetectorSize returns a detector whose detection window is winW x winH. It
// panics if either dimension is not positive.
func NewWBDetectorSize(winW, winH int) *WBDetector {
	if winW <= 0 || winH <= 0 {
		panic("xobjdetect: NewWBDetectorSize requires a positive window")
	}
	return &WBDetector{
		WinW:            winW,
		WinH:            winH,
		Rounds:          defaultRounds,
		NumFeatures:     defaultNumFeatures,
		Seed:            1,
		ScaleFactor:     defaultScaleFactor,
		Step:            defaultStep,
		NMSOverlap:      defaultNMSOverlap,
		DetectThreshold: 0,
	}
}

// Trained reports whether the detector has been trained (directly or by loading
// a model).
func (d *WBDetector) Trained() bool { return d.trained }

// Evaluator returns an [ACFFeatureEvaluator] bound to the detector's feature
// pool. It panics if the detector is untrained.
func (d *WBDetector) Evaluator() *ACFFeatureEvaluator {
	if !d.trained {
		panic("xobjdetect: Evaluator on untrained detector")
	}
	return NewACFFeatureEvaluator(d.pool)
}

// Train learns the detector from positive samples (patches containing the
// object) and negative samples (patches that do not). Each patch is resized to
// the detection window before its integral-channel features are extracted. It
// returns an error if either set is empty.
func (d *WBDetector) Train(posSamples, negSamples []*cv.Mat) error {
	if len(posSamples) == 0 || len(negSamples) == 0 {
		return errors.New("xobjdetect: Train needs both positive and negative samples")
	}
	if d.NumFeatures <= 0 {
		d.NumFeatures = defaultNumFeatures
	}
	if d.Rounds <= 0 {
		d.Rounds = defaultRounds
	}
	rng := rand.New(rand.NewSource(d.Seed))
	d.pool = NewFeaturePool(d.WinW, d.WinH, d.NumFeatures, rng)
	eval := NewACFFeatureEvaluator(d.pool)

	pos := make([][]float64, len(posSamples))
	for i, s := range posSamples {
		pos[i] = eval.Sample(s)
	}
	neg := make([][]float64, len(negSamples))
	for i, s := range negSamples {
		neg[i] = eval.Sample(s)
	}

	d.boost = NewWaldBoost(d.Rounds)
	if err := d.boost.Train(pos, neg); err != nil {
		return err
	}
	d.trained = true
	return nil
}

// Detect scans img and returns the bounding boxes of every detected object
// together with the confidence score of each, sorted by descending score. The
// classifier is slid over a downscaling image pyramid and overlapping raw
// detections are merged with non-maximum suppression. It panics if the detector
// is untrained.
func (d *WBDetector) Detect(img *cv.Mat) (rects []cv.Rect, confidences []float64) {
	if !d.trained {
		panic("xobjdetect: Detect on untrained detector")
	}
	scaleFactor := d.ScaleFactor
	if scaleFactor <= 1 {
		scaleFactor = defaultScaleFactor
	}
	step := d.Step
	if step < 1 {
		step = 1
	}
	eval := NewACFFeatureEvaluator(d.pool)

	var rawRects []cv.Rect
	var rawScores []float64

	scale := 1.0
	for {
		rw := int(math.Round(float64(img.Cols) / scale))
		rh := int(math.Round(float64(img.Rows) / scale))
		if rw < d.WinW || rh < d.WinH {
			break
		}
		level := img
		if rw != img.Cols || rh != img.Rows {
			level = cv.Resize(img, rw, rh, cv.InterLinear)
		}
		eval.SetImage(level)
		for y := 0; y+d.WinH <= rh; y += step {
			for x := 0; x+d.WinW <= rw; x += step {
				feat := eval.EvaluateWindow(x, y)
				score, ok := d.boost.Predict(feat)
				if ok && score > d.DetectThreshold {
					rawRects = append(rawRects, cv.Rect{
						X:      int(math.Round(float64(x) * scale)),
						Y:      int(math.Round(float64(y) * scale)),
						Width:  int(math.Round(float64(d.WinW) * scale)),
						Height: int(math.Round(float64(d.WinH) * scale)),
					})
					rawScores = append(rawScores, score)
				}
			}
		}
		scale *= scaleFactor
	}

	return nonMaxSuppression(rawRects, rawScores, d.NMSOverlap)
}

// rectIoU returns the intersection-over-union of two rectangles in [0,1].
func rectIoU(a, b cv.Rect) float64 {
	ix0 := maxInt(a.X, b.X)
	iy0 := maxInt(a.Y, b.Y)
	ix1 := minInt(a.X+a.Width, b.X+b.Width)
	iy1 := minInt(a.Y+a.Height, b.Y+b.Height)
	iw := ix1 - ix0
	ih := iy1 - iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	union := float64(a.Width*a.Height+b.Width*b.Height) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// nonMaxSuppression greedily keeps the highest-scoring boxes, discarding any box
// that overlaps an already-kept box by more than overlap. Results are returned
// in descending score order.
func nonMaxSuppression(rects []cv.Rect, scores []float64, overlap float64) ([]cv.Rect, []float64) {
	order := make([]int, len(rects))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })

	var keptRects []cv.Rect
	var keptScores []float64
	for _, idx := range order {
		suppressed := false
		for _, k := range keptRects {
			if rectIoU(rects[idx], k) > overlap {
				suppressed = true
				break
			}
		}
		if !suppressed {
			keptRects = append(keptRects, rects[idx])
			keptScores = append(keptScores, scores[idx])
		}
	}
	return keptRects, keptScores
}

// modelSnapshot is the gob-encodable form of a trained detector.
type modelSnapshot struct {
	Version         int
	WinW, WinH      int
	Rounds          int
	NumFeatures     int
	Seed            int64
	ScaleFactor     float64
	Step            int
	NMSOverlap      float64
	DetectThreshold float64
	Pool            *FeaturePool
	Boost           *WaldBoost
}

const snapshotVersion = 1

// Write serialises a trained detector to w with encoding/gob. The stream can be
// restored with [WBDetector.Read] and reproduces detections exactly. It returns
// an error if the detector is untrained or encoding fails.
func (d *WBDetector) Write(w io.Writer) error {
	if !d.trained {
		return errors.New("xobjdetect: Write on untrained detector")
	}
	snap := modelSnapshot{
		Version:         snapshotVersion,
		WinW:            d.WinW,
		WinH:            d.WinH,
		Rounds:          d.Rounds,
		NumFeatures:     d.NumFeatures,
		Seed:            d.Seed,
		ScaleFactor:     d.ScaleFactor,
		Step:            d.Step,
		NMSOverlap:      d.NMSOverlap,
		DetectThreshold: d.DetectThreshold,
		Pool:            d.pool,
		Boost:           d.boost,
	}
	if err := gob.NewEncoder(w).Encode(snap); err != nil {
		return err
	}
	return nil
}

// Read restores a detector previously written by [WBDetector.Write] from r,
// replacing the receiver's state. It returns an error if decoding fails or the
// stream version is unrecognised.
func (d *WBDetector) Read(r io.Reader) error {
	var snap modelSnapshot
	if err := gob.NewDecoder(r).Decode(&snap); err != nil {
		return err
	}
	if snap.Version != snapshotVersion {
		return errors.New("xobjdetect: Read unsupported model version")
	}
	if snap.Pool == nil || snap.Boost == nil || !snap.Boost.Trained {
		return errors.New("xobjdetect: Read incomplete model")
	}
	d.WinW = snap.WinW
	d.WinH = snap.WinH
	d.Rounds = snap.Rounds
	d.NumFeatures = snap.NumFeatures
	d.Seed = snap.Seed
	d.ScaleFactor = snap.ScaleFactor
	d.Step = snap.Step
	d.NMSOverlap = snap.NMSOverlap
	d.DetectThreshold = snap.DetectThreshold
	d.pool = snap.Pool
	d.boost = snap.Boost
	d.trained = true
	return nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

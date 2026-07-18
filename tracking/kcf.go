package tracking

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MatchResult reports the best location found by [MatchTemplateNCC]: the
// top-left corner of the match, the template size that produced it and the
// normalized-cross-correlation score in [-1, 1] (1 is a perfect match).
type MatchResult struct {
	// X is the column of the best-match top-left corner.
	X int
	// Y is the row of the best-match top-left corner.
	Y int
	// Width is the template width.
	Width int
	// Height is the template height.
	Height int
	// Score is the normalized cross-correlation at the best location, in [-1, 1].
	Score float64
}

// BoundingBox returns the matched region as a Rect.
func (r MatchResult) BoundingBox() Rect {
	return Rect{X: r.X, Y: r.Y, Width: r.Width, Height: r.Height}
}

// MatchTemplateNCC slides templ over img and returns the position of maximum
// normalized cross-correlation. Both inputs are converted to grayscale; the
// score at each candidate location is the mean-subtracted correlation
// coefficient, which is invariant to uniform brightness and contrast changes and
// lies in [-1, 1]. templ must not be larger than img. The computation is
// deterministic.
func MatchTemplateNCC(img, templ *cv.Mat) MatchResult {
	if img == nil || img.Empty() || templ == nil || templ.Empty() {
		panic("tracking: MatchTemplateNCC requires non-empty images")
	}
	if templ.Rows > img.Rows || templ.Cols > img.Cols {
		panic("tracking: MatchTemplateNCC template larger than image")
	}
	gi := trackingToGrayF(img)
	gt := trackingToGrayF(templ)
	tpl, tMean, tNorm := trackingPrepTemplate(gt, nil)

	best := math.Inf(-1)
	bestX, bestY := 0, 0
	for y := 0; y <= gi.rows-gt.rows; y++ {
		for x := 0; x <= gi.cols-gt.cols; x++ {
			s := trackingNCCAt(gi, tpl, gt.rows, gt.cols, x, y, tMean, tNorm, nil)
			if s > best {
				best = s
				bestX, bestY = x, y
			}
		}
	}
	return MatchResult{X: bestX, Y: bestY, Width: gt.cols, Height: gt.rows, Score: best}
}

// trackingPrepTemplate returns the template's raw values (optionally multiplied
// by a per-pixel window), its mean and its L2 norm after mean subtraction. These
// are precomputed once so the sliding search only touches the candidate patch.
func trackingPrepTemplate(gt *trackingGray, window []float64) (vals []float64, mean, norm float64) {
	n := gt.rows * gt.cols
	vals = make([]float64, n)
	var sum float64
	for i := 0; i < n; i++ {
		v := gt.data[i]
		if window != nil {
			v *= window[i]
		}
		vals[i] = v
		sum += v
	}
	mean = sum / float64(n)
	var sq float64
	for i := 0; i < n; i++ {
		d := vals[i] - mean
		sq += d * d
	}
	return vals, mean, math.Sqrt(sq)
}

// trackingNCCAt computes the normalized cross-correlation between the prepared
// template and the candidate patch of img whose top-left corner is (x, y).
func trackingNCCAt(img *trackingGray, tpl []float64, tRows, tCols, x, y int, tMean, tNorm float64, window []float64) float64 {
	n := tRows * tCols
	// Candidate patch mean.
	var sum float64
	patch := make([]float64, n)
	for j := 0; j < tRows; j++ {
		for i := 0; i < tCols; i++ {
			v := img.at(y+j, x+i)
			if window != nil {
				v *= window[j*tCols+i]
			}
			patch[j*tCols+i] = v
			sum += v
		}
	}
	mean := sum / float64(n)
	var cross, pNorm float64
	for k := 0; k < n; k++ {
		pd := patch[k] - mean
		cross += (tpl[k] - tMean) * pd
		pNorm += pd * pd
	}
	denom := tNorm * math.Sqrt(pNorm)
	if denom < 1e-12 {
		return 0
	}
	return cross / denom
}

// hannWindow returns a separable 2-D raised-cosine (Hann) window of the given
// size, flattened row-major. It tapers a patch towards zero at its borders,
// suppressing edge effects in the correlation response.
func hannWindow(rows, cols int) []float64 {
	wr := make([]float64, rows)
	wc := make([]float64, cols)
	for i := 0; i < rows; i++ {
		if rows == 1 {
			wr[i] = 1
		} else {
			wr[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(rows-1))
		}
	}
	for i := 0; i < cols; i++ {
		if cols == 1 {
			wc[i] = 1
		} else {
			wc[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(cols-1))
		}
	}
	out := make([]float64, rows*cols)
	for j := 0; j < rows; j++ {
		for i := 0; i < cols; i++ {
			out[j*cols+i] = wr[j] * wc[i]
		}
	}
	return out
}

// KCFParams configures the [KCFTracker].
type KCFParams struct {
	// SearchRadius is how far, in pixels, the tracker searches around the last
	// position each frame. Zero is treated as half the template's larger side.
	SearchRadius int
	// LearningRate blends the current appearance into the stored template each
	// successful update, in [0, 1]; 0 keeps the initial template fixed. Zero is
	// treated as 0.1.
	LearningRate float64
}

// DefaultKCFParams returns the default parameters (adaptive search radius,
// learning rate 0.1).
func DefaultKCFParams() KCFParams { return KCFParams{SearchRadius: 0, LearningRate: 0.1} }

// KCFTracker is a lightweight template correlation tracker ("KCF-lite"). It
// stores a cosine-windowed appearance model of the target and, each frame,
// locates it by maximising the windowed normalized cross-correlation over a
// search region around the previous position, then optionally blends the new
// appearance into the model. It is deterministic and CPU-only. Unlike the full
// kernelized correlation filter it does not use the Fourier-domain kernel trick,
// trading peak efficiency for a compact, dependency-free implementation.
type KCFTracker struct {
	params  KCFParams
	tplVals []float64
	tplMean float64
	tplNorm float64
	window  []float64
	box     Rect
	rows    int
	cols    int
	inited  bool
}

// NewKCFTracker creates an uninitialised tracker with the given parameters.
// Call [KCFTracker.Init] with the first frame and target box before
// [KCFTracker.Update].
func NewKCFTracker(params KCFParams) *KCFTracker {
	return &KCFTracker{params: params}
}

// Init sets the target from box in img and builds the initial appearance model.
// box is clamped to the image; it panics if the clamped box is empty.
func (t *KCFTracker) Init(img *cv.Mat, box Rect) {
	if img == nil || img.Empty() {
		panic("tracking: KCFTracker.Init requires a non-empty image")
	}
	b := box.clampTo(img.Rows, img.Cols)
	if b.Empty() {
		panic("tracking: KCFTracker.Init box is empty after clamping")
	}
	gi := trackingToGrayF(img)
	patch := trackingCrop(gi, b)
	t.rows, t.cols = b.Height, b.Width
	t.window = hannWindow(t.rows, t.cols)
	t.tplVals, t.tplMean, t.tplNorm = trackingPrepTemplate(patch, t.window)
	t.box = b
	t.inited = true
}

// Update locates the target in the next frame img, returns the new bounding box
// and the correlation confidence in [-1, 1], and updates the internal model. If
// the tracker was not initialised it panics. The returned box keeps the original
// target size.
func (t *KCFTracker) Update(img *cv.Mat) (Rect, float64) {
	if !t.inited {
		panic("tracking: KCFTracker.Update called before Init")
	}
	if img == nil || img.Empty() {
		panic("tracking: KCFTracker.Update requires a non-empty image")
	}
	gi := trackingToGrayF(img)
	sr := t.params.SearchRadius
	if sr <= 0 {
		sr = max(t.rows, t.cols) / 2
		if sr < 1 {
			sr = 1
		}
	}
	best := math.Inf(-1)
	bestX, bestY := t.box.X, t.box.Y
	for dy := -sr; dy <= sr; dy++ {
		for dx := -sr; dx <= sr; dx++ {
			x := t.box.X + dx
			y := t.box.Y + dy
			if x < 0 || y < 0 || x+t.cols > gi.cols || y+t.rows > gi.rows {
				continue
			}
			s := trackingNCCAt(gi, t.tplVals, t.rows, t.cols, x, y, t.tplMean, t.tplNorm, t.window)
			if s > best {
				best = s
				bestX, bestY = x, y
			}
		}
	}
	t.box = Rect{X: bestX, Y: bestY, Width: t.cols, Height: t.rows}

	// Blend the newly observed appearance into the model.
	lr := t.params.LearningRate
	if lr <= 0 {
		lr = 0.1
	}
	if lr > 0 && best > 0 {
		patch := trackingCrop(gi, t.box)
		nv, _, _ := trackingPrepTemplate(patch, t.window)
		for i := range t.tplVals {
			t.tplVals[i] = (1-lr)*t.tplVals[i] + lr*nv[i]
		}
		// Recompute mean and norm of the blended template.
		var sum float64
		for _, v := range t.tplVals {
			sum += v
		}
		t.tplMean = sum / float64(len(t.tplVals))
		var sq float64
		for _, v := range t.tplVals {
			d := v - t.tplMean
			sq += d * d
		}
		t.tplNorm = math.Sqrt(sq)
	}
	return t.box, best
}

// BoundingBox returns the tracker's current target box.
func (t *KCFTracker) BoundingBox() Rect { return t.box }

// trackingCrop extracts the sub-region r of a float grayscale image as a new
// image. r must lie inside the image.
func trackingCrop(g *trackingGray, r Rect) *trackingGray {
	out := trackingNewGray(r.Height, r.Width)
	for j := 0; j < r.Height; j++ {
		for i := 0; i < r.Width; i++ {
			out.data[j*r.Width+i] = g.at(r.Y+j, r.X+i)
		}
	}
	return out
}

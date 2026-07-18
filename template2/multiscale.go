package template2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Pyramid is a sequence of successively downscaled copies of an image, together
// with the scale factor of each level relative to the original. Level 0 is the
// original image (scale 1). Build one with [BuildPyramid].
type Pyramid struct {
	// Levels holds the image at each pyramid level, coarsening with index.
	Levels []*cv.Mat
	// Scales[i] is the size of Levels[i] relative to the original (<= 1).
	Scales []float64
}

// BuildPyramid builds a Gaussian-free image pyramid of the given number of
// levels. Each level after the first is the previous level scaled by
// scaleFactor (which must be in (0,1)) using bilinear interpolation. Levels stop
// early if a dimension would fall below one pixel. levels must be >= 1.
func BuildPyramid(src *cv.Mat, levels int, scaleFactor float64) *Pyramid {
	if levels < 1 {
		levels = 1
	}
	if scaleFactor <= 0 || scaleFactor >= 1 {
		scaleFactor = 0.5
	}
	p := &Pyramid{}
	p.Levels = append(p.Levels, src.Clone())
	p.Scales = append(p.Scales, 1)
	scale := 1.0
	cur := src
	for i := 1; i < levels; i++ {
		scale *= scaleFactor
		w := int(math.Round(float64(src.Cols) * scale))
		h := int(math.Round(float64(src.Rows) * scale))
		if w < 1 || h < 1 {
			break
		}
		next := cv.Resize(cur, w, h, cv.InterLinear)
		p.Levels = append(p.Levels, next)
		p.Scales = append(p.Scales, scale)
		cur = next
	}
	return p
}

// NumLevels returns the number of levels in the pyramid.
func (p *Pyramid) NumLevels() int {
	return len(p.Levels)
}

// Level returns the image at pyramid level i. It panics if i is out of range.
func (p *Pyramid) Level(i int) *cv.Mat {
	return p.Levels[i]
}

// MultiScaleParams configures [MatchMultiScale].
type MultiScaleParams struct {
	// Method is the similarity measure to use.
	Method Method
	// MinScale and MaxScale bound the template scale factors to search
	// (relative to the template's native size). Both must be positive.
	MinScale float64
	MaxScale float64
	// Levels is the number of scale steps sampled between MinScale and MaxScale
	// (inclusive). It must be >= 1.
	Levels int
	// Threshold is the score cut-off applied at every scale (see
	// [FindMatches] for the polarity convention).
	Threshold float64
	// NMSIoU is the intersection-over-union overlap above which overlapping
	// detections are suppressed after pooling across scales. A value <= 0
	// disables suppression.
	NMSIoU float64
}

// DefaultMultiScaleParams returns sensible defaults: zero-mean normalised
// cross-correlation over scales 0.5 to 1.5 in 11 steps, a 0.7 score threshold
// and 0.3 IoU suppression.
func DefaultMultiScaleParams() MultiScaleParams {
	return MultiScaleParams{
		Method:    MethodZNCC,
		MinScale:  0.5,
		MaxScale:  1.5,
		Levels:    11,
		Threshold: 0.7,
		NMSIoU:    0.3,
	}
}

// BuildScales returns the Levels scale factors sampled uniformly from minScale
// to maxScale inclusive. With Levels == 1 it returns the midpoint. minScale and
// maxScale may be given in either order.
func BuildScales(minScale, maxScale float64, levels int) []float64 {
	if levels < 1 {
		levels = 1
	}
	if minScale > maxScale {
		minScale, maxScale = maxScale, minScale
	}
	if levels == 1 {
		return []float64{(minScale + maxScale) / 2}
	}
	out := make([]float64, levels)
	step := (maxScale - minScale) / float64(levels-1)
	for i := 0; i < levels; i++ {
		out[i] = minScale + step*float64(i)
	}
	return out
}

// MatchMultiScale searches for templ in src across a range of template scales
// and returns the pooled detections that pass the score threshold, optionally
// de-duplicated by non-maximum suppression. Each returned [Match] records the
// scale at which it was found and the corresponding scaled template dimensions.
//
// At each scale the template is resized with bilinear interpolation; scales that
// would make the template larger than src are skipped. The result is sorted
// best-first.
func MatchMultiScale(src, templ *cv.Mat, params MultiScaleParams) ([]Match, error) {
	if src.Empty() || templ.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != templ.Channels {
		return nil, ErrChannelMismatch
	}
	if !params.Method.Valid() {
		return nil, ErrInvalidMethod
	}
	if params.MinScale <= 0 || params.MaxScale <= 0 {
		return nil, ErrTemplateLarger
	}
	higher := params.Method.HigherIsBetter()
	scales := BuildScales(params.MinScale, params.MaxScale, params.Levels)

	var all []Match
	for _, sc := range scales {
		w := int(math.Round(float64(templ.Cols) * sc))
		h := int(math.Round(float64(templ.Rows) * sc))
		if w < 1 || h < 1 {
			continue
		}
		if w > src.Cols || h > src.Rows {
			continue
		}
		scaled := templ
		if w != templ.Cols || h != templ.Rows {
			scaled = cv.Resize(templ, w, h, cv.InterLinear)
		}
		scores, err := MatchTemplate(src, scaled, params.Method)
		if err != nil {
			return nil, err
		}
		matches := ExtractMatches(scores, w, h, params.Threshold, higher)
		for i := range matches {
			matches[i].Scale = sc
		}
		all = append(all, matches...)
	}

	sortMatches(all, higher)
	if params.NMSIoU > 0 {
		all = NonMaxSuppression(all, params.NMSIoU, higher)
	}
	return all, nil
}

// BestMatchMultiScale returns the single strongest detection of templ in src
// across scales 0.5 to 1.5 (11 steps) under the given method. It ignores the
// score threshold and always returns the best candidate found, or an error if
// no scale fits inside src.
func BestMatchMultiScale(src, templ *cv.Mat, method Method) (Match, error) {
	params := DefaultMultiScaleParams()
	params.Method = method
	if method.HigherIsBetter() {
		params.Threshold = math.Inf(-1)
	} else {
		params.Threshold = math.Inf(1)
	}
	params.NMSIoU = 0
	matches, err := MatchMultiScale(src, templ, params)
	if err != nil {
		return Match{}, err
	}
	if len(matches) == 0 {
		return Match{}, ErrTemplateLarger
	}
	return matches[0], nil
}

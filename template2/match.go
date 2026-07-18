package template2

import (
	"errors"
	"image"
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Errors returned by the matching routines.
var (
	// ErrEmptyImage is returned when the source or template image has no
	// samples.
	ErrEmptyImage = errors.New("template2: empty image")
	// ErrTemplateLarger is returned when the template does not fit inside the
	// source image.
	ErrTemplateLarger = errors.New("template2: template larger than source")
	// ErrChannelMismatch is returned when the source and template have a
	// different number of channels.
	ErrChannelMismatch = errors.New("template2: source and template channel mismatch")
	// ErrInvalidMethod is returned when an unknown [Method] is supplied.
	ErrInvalidMethod = errors.New("template2: invalid method")
)

// Match describes one detected template location. X and Y are the top-left
// corner of the matched patch in source pixel coordinates; Width and Height are
// the template dimensions at the matched scale (rounded to whole pixels). Score
// is the similarity value under the [Method] that produced it. Scale and Angle
// record the template scale factor and rotation (in degrees) at which the match
// was found; a plain single-pass match reports Scale == 1 and Angle == 0.
type Match struct {
	// X is the column of the matched patch's top-left corner.
	X int
	// Y is the row of the matched patch's top-left corner.
	Y int
	// Width is the matched patch width in pixels.
	Width int
	// Height is the matched patch height in pixels.
	Height int
	// Score is the similarity value under the producing method.
	Score float64
	// Scale is the template scale factor at which the match was found.
	Scale float64
	// Angle is the template rotation in degrees at which the match was found.
	Angle float64
}

// Rect returns the matched patch as a standard-library rectangle spanning
// [X, X+Width) × [Y, Y+Height).
func (m Match) Rect() image.Rectangle {
	return image.Rect(m.X, m.Y, m.X+m.Width, m.Y+m.Height)
}

// Area returns the number of pixels covered by the matched patch.
func (m Match) Area() int {
	return m.Width * m.Height
}

// CenterX returns the (possibly fractional) column of the patch centre.
func (m Match) CenterX() float64 {
	return float64(m.X) + float64(m.Width)/2
}

// CenterY returns the (possibly fractional) row of the patch centre.
func (m Match) CenterY() float64 {
	return float64(m.Y) + float64(m.Height)/2
}

// Center returns the integer-rounded pixel at the centre of the matched patch.
func (m Match) Center() cv.Point {
	return cv.Point{X: int(math.Round(m.CenterX())), Y: int(math.Round(m.CenterY()))}
}

// IoU returns the intersection-over-union overlap of two matched rectangles, a
// value in [0,1]. Disjoint rectangles overlap 0; identical rectangles overlap 1.
func (m Match) IoU(o Match) float64 {
	ix0 := maxInt(m.X, o.X)
	iy0 := maxInt(m.Y, o.Y)
	ix1 := minInt(m.X+m.Width, o.X+o.Width)
	iy1 := minInt(m.Y+m.Height, o.Y+o.Height)
	iw := ix1 - ix0
	ih := iy1 - iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	union := float64(m.Area()+o.Area()) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// Overlaps reports whether the intersection-over-union overlap of m and o is at
// least iouThreshold.
func (m Match) Overlaps(o Match, iouThreshold float64) bool {
	return m.IoU(o) >= iouThreshold
}

// template2stats holds the sums accumulated over one overlapped patch.
type template2stats struct {
	sumI, sumI2, sumIT, sumAbs, sumSq float64
}

// template2tmplStats holds the scale-independent template moments.
type template2tmplStats struct {
	n      float64 // number of samples
	sumT   float64
	sumT2  float64
	meanT  float64
	tNorm  float64 // sqrt(sumT2)
	tVar   float64 // sumT2 - n*meanT^2
	tStdSS float64 // sqrt(tVar)
}

// template2computeTmpl computes the template moments once.
func template2computeTmpl(templ *cv.Mat) template2tmplStats {
	var sumT, sumT2 float64
	for _, v := range templ.Data {
		f := float64(v)
		sumT += f
		sumT2 += f * f
	}
	n := float64(len(templ.Data))
	meanT := sumT / n
	tVar := sumT2 - n*meanT*meanT
	if tVar < 0 {
		tVar = 0
	}
	return template2tmplStats{
		n:      n,
		sumT:   sumT,
		sumT2:  sumT2,
		meanT:  meanT,
		tNorm:  math.Sqrt(sumT2),
		tVar:   tVar,
		tStdSS: math.Sqrt(tVar),
	}
}

// template2score converts accumulated patch statistics into a score under the
// given method.
func template2score(method Method, s template2stats, t template2tmplStats) float64 {
	switch method {
	case MethodSAD:
		return s.sumAbs
	case MethodSSD:
		return s.sumSq
	case MethodSSDNormed:
		denom := math.Sqrt(s.sumI2 * t.sumT2)
		if denom == 0 {
			return 0
		}
		return s.sumSq / denom
	case MethodCrossCorr:
		return s.sumIT
	case MethodNCC:
		denom := math.Sqrt(s.sumI2 * t.sumT2)
		if denom == 0 {
			return 0
		}
		return s.sumIT / denom
	case MethodCorrCoeff:
		// Sum((I-meanI)(T-meanT)) = sumIT - meanT*sumI.
		return s.sumIT - t.meanT*s.sumI
	case MethodZNCC:
		meanI := s.sumI / t.n
		num := s.sumIT - t.n*meanI*t.meanT
		iVar := s.sumI2 - s.sumI*s.sumI/t.n
		if iVar < 0 {
			iVar = 0
		}
		denom := math.Sqrt(iVar) * t.tStdSS
		if denom == 0 {
			return 0
		}
		return num / denom
	default:
		return 0
	}
}

// MatchTemplate slides templ over src and returns a dense score map of shape
// (src.Rows-templ.Rows+1) × (src.Cols-templ.Cols+1). Each entry holds the
// similarity of templ against the patch of src whose top-left corner is that
// position, under the chosen [Method]. The two images must have the same channel
// count and templ must fit inside src.
//
// The measure is computed over every channel sample, matching the parent
// package's [cv.MatchTemplate]. Use [BestMatch] to locate the strongest match
// or [cv.MinMaxLoc] on the returned map.
func MatchTemplate(src, templ *cv.Mat, method Method) (*cv.FloatMat, error) {
	if src.Empty() || templ.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != templ.Channels {
		return nil, ErrChannelMismatch
	}
	if templ.Rows > src.Rows || templ.Cols > src.Cols {
		return nil, ErrTemplateLarger
	}
	if !method.Valid() {
		return nil, ErrInvalidMethod
	}

	ch := src.Channels
	tStats := template2computeTmpl(templ)
	resRows := src.Rows - templ.Rows + 1
	resCols := src.Cols - templ.Cols + 1
	res := cv.NewFloatMat(resRows, resCols)

	for ry := 0; ry < resRows; ry++ {
		for rx := 0; rx < resCols; rx++ {
			var s template2stats
			for ty := 0; ty < templ.Rows; ty++ {
				srcRow := ((ry+ty)*src.Cols + rx) * ch
				tRow := (ty * templ.Cols) * ch
				count := templ.Cols * ch
				for k := 0; k < count; k++ {
					iv := float64(src.Data[srcRow+k])
					tv := float64(templ.Data[tRow+k])
					d := iv - tv
					s.sumI += iv
					s.sumI2 += iv * iv
					s.sumIT += iv * tv
					s.sumAbs += math.Abs(d)
					s.sumSq += d * d
				}
			}
			res.Data[ry*resCols+rx] = template2score(method, s, tStats)
		}
	}
	return res, nil
}

// MatchSAD returns the sum-of-absolute-differences score map. It is shorthand
// for [MatchTemplate] with [MethodSAD].
func MatchSAD(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return MatchTemplate(src, templ, MethodSAD)
}

// MatchSSD returns the sum-of-squared-differences score map. It is shorthand
// for [MatchTemplate] with [MethodSSD].
func MatchSSD(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return MatchTemplate(src, templ, MethodSSD)
}

// MatchSSDNormed returns the normalised sum-of-squared-differences score map.
// It is shorthand for [MatchTemplate] with [MethodSSDNormed].
func MatchSSDNormed(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return MatchTemplate(src, templ, MethodSSDNormed)
}

// MatchCrossCorr returns the raw cross-correlation score map. It is shorthand
// for [MatchTemplate] with [MethodCrossCorr].
func MatchCrossCorr(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return MatchTemplate(src, templ, MethodCrossCorr)
}

// MatchNCC returns the normalised cross-correlation score map. It is shorthand
// for [MatchTemplate] with [MethodNCC].
func MatchNCC(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return MatchTemplate(src, templ, MethodNCC)
}

// MatchCorrCoeff returns the mean-subtracted correlation-coefficient score map.
// It is shorthand for [MatchTemplate] with [MethodCorrCoeff].
func MatchCorrCoeff(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return MatchTemplate(src, templ, MethodCorrCoeff)
}

// MatchZNCC returns the zero-mean normalised cross-correlation score map. It is
// shorthand for [MatchTemplate] with [MethodZNCC].
func MatchZNCC(src, templ *cv.Mat) (*cv.FloatMat, error) {
	return MatchTemplate(src, templ, MethodZNCC)
}

// LocateExtremum scans a score map and returns the location and value of its
// best entry: the maximum when higherIsBetter is true, otherwise the minimum.
// It returns ok == false for an empty map.
func LocateExtremum(scores *cv.FloatMat, higherIsBetter bool) (x, y int, value float64, ok bool) {
	if scores == nil || len(scores.Data) == 0 {
		return 0, 0, 0, false
	}
	best := scores.Data[0]
	bx, by := 0, 0
	for yy := 0; yy < scores.Rows; yy++ {
		for xx := 0; xx < scores.Cols; xx++ {
			v := scores.Data[yy*scores.Cols+xx]
			if (higherIsBetter && v > best) || (!higherIsBetter && v < best) {
				best = v
				bx, by = xx, yy
			}
		}
	}
	return bx, by, best, true
}

// BestMatch computes the score map for the given method and returns the single
// strongest [Match]. The returned match has Scale 1 and Angle 0.
func BestMatch(src, templ *cv.Mat, method Method) (Match, error) {
	scores, err := MatchTemplate(src, templ, method)
	if err != nil {
		return Match{}, err
	}
	x, y, v, ok := LocateExtremum(scores, method.HigherIsBetter())
	if !ok {
		return Match{}, ErrEmptyImage
	}
	return Match{
		X: x, Y: y,
		Width:  templ.Cols,
		Height: templ.Rows,
		Score:  v,
		Scale:  1,
		Angle:  0,
	}, nil
}

// passesThreshold reports whether score meets the threshold given the polarity.
func passesThreshold(score, threshold float64, higherIsBetter bool) bool {
	if higherIsBetter {
		return score >= threshold
	}
	return score <= threshold
}

// ExtractMatches converts a score map into the list of positions that pass the
// threshold. width and height are the template dimensions used to build each
// [Match] rectangle. When higherIsBetter is true a position is kept if its score
// is >= threshold, otherwise if its score is <= threshold. The result is sorted
// best-first.
func ExtractMatches(scores *cv.FloatMat, width, height int, threshold float64, higherIsBetter bool) []Match {
	if scores == nil {
		return nil
	}
	var out []Match
	for y := 0; y < scores.Rows; y++ {
		for x := 0; x < scores.Cols; x++ {
			v := scores.Data[y*scores.Cols+x]
			if passesThreshold(v, threshold, higherIsBetter) {
				out = append(out, Match{
					X: x, Y: y, Width: width, Height: height,
					Score: v, Scale: 1, Angle: 0,
				})
			}
		}
	}
	sortMatches(out, higherIsBetter)
	return out
}

// FindMatches computes the score map for the given method and returns every
// location whose score passes threshold, sorted best-first. For the difference
// measures ([MethodSAD], [MethodSSD], [MethodSSDNormed]) a location is kept when
// its score is <= threshold; for the correlation measures when its score is
// >= threshold.
func FindMatches(src, templ *cv.Mat, method Method, threshold float64) ([]Match, error) {
	scores, err := MatchTemplate(src, templ, method)
	if err != nil {
		return nil, err
	}
	return ExtractMatches(scores, templ.Cols, templ.Rows, threshold, method.HigherIsBetter()), nil
}

// sortMatches orders matches best-first in place.
func sortMatches(matches []Match, higherIsBetter bool) {
	sort.SliceStable(matches, func(i, j int) bool {
		if higherIsBetter {
			return matches[i].Score > matches[j].Score
		}
		return matches[i].Score < matches[j].Score
	})
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

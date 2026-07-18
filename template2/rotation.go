package template2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// RotateTemplate rotates templ counter-clockwise by angleDeg degrees about its
// centre and returns a new [cv.Mat] whose canvas is expanded so the whole
// rotated template fits. Regions outside the original template are filled with
// zero; sampling uses bilinear interpolation. Passing 0 returns a clone of the
// original size.
func RotateTemplate(templ *cv.Mat, angleDeg float64) *cv.Mat {
	if angleDeg == 0 {
		return templ.Clone()
	}
	w, h := templ.Cols, templ.Rows
	cx := float64(w-1) / 2
	cy := float64(h-1) / 2
	m := cv.GetRotationMatrix2D(cx, cy, angleDeg, 1)
	cos := math.Abs(m[0])
	sin := math.Abs(m[1])
	newW := int(math.Round(float64(h)*sin + float64(w)*cos))
	newH := int(math.Round(float64(h)*cos + float64(w)*sin))
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}
	// Shift so the original centre maps to the centre of the enlarged canvas.
	m[2] += float64(newW-1)/2 - cx
	m[5] += float64(newH-1)/2 - cy
	return cv.WarpAffine(templ, m, newW, newH, cv.InterLinear)
}

// BuildAngles returns the rotation angles (in degrees) sampled from minAngle to
// maxAngle inclusive with the given step. minAngle and maxAngle may be given in
// either order; step is treated as a positive magnitude. A zero or negative
// step yields a single angle at the midpoint.
func BuildAngles(minAngle, maxAngle, step float64) []float64 {
	if minAngle > maxAngle {
		minAngle, maxAngle = maxAngle, minAngle
	}
	if step <= 0 {
		return []float64{(minAngle + maxAngle) / 2}
	}
	var out []float64
	for a := minAngle; a <= maxAngle+1e-9; a += step {
		out = append(out, a)
	}
	return out
}

// RotationParams configures [MatchRotationInvariant].
type RotationParams struct {
	// Method is the similarity measure to use.
	Method Method
	// MinAngle and MaxAngle bound the template rotations searched, in degrees.
	MinAngle, MaxAngle float64
	// AngleStep is the spacing between sampled angles, in degrees. It must be
	// positive.
	AngleStep float64
	// Threshold is the score cut-off applied at every angle (see [FindMatches]
	// for the polarity convention).
	Threshold float64
	// NMSIoU is the intersection-over-union overlap above which overlapping
	// detections are suppressed after pooling across angles. A value <= 0
	// disables suppression.
	NMSIoU float64
}

// DefaultRotationParams returns sensible defaults: zero-mean normalised
// cross-correlation over a full 0..350 degree sweep in 10 degree steps, a 0.7
// score threshold and 0.3 IoU suppression.
func DefaultRotationParams() RotationParams {
	return RotationParams{
		Method:    MethodZNCC,
		MinAngle:  0,
		MaxAngle:  350,
		AngleStep: 10,
		Threshold: 0.7,
		NMSIoU:    0.3,
	}
}

// MatchRotationInvariant searches for templ in src across a range of template
// rotations and returns the pooled detections that pass the score threshold,
// optionally de-duplicated by non-maximum suppression. Each returned [Match]
// records the angle at which it was found and the bounding-box dimensions of the
// rotated template.
//
// At each angle the template is rotated with [RotateTemplate]; rotations whose
// bounding box would exceed src are skipped. The result is sorted best-first.
//
// Note that rotated templates are zero-padded at the corners, which slightly
// depresses scores for correlation measures; the effect is largest for
// low-fill templates rotated near 45 degrees.
func MatchRotationInvariant(src, templ *cv.Mat, params RotationParams) ([]Match, error) {
	if src.Empty() || templ.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != templ.Channels {
		return nil, ErrChannelMismatch
	}
	if !params.Method.Valid() {
		return nil, ErrInvalidMethod
	}
	higher := params.Method.HigherIsBetter()
	angles := BuildAngles(params.MinAngle, params.MaxAngle, params.AngleStep)

	var all []Match
	for _, ang := range angles {
		rot := RotateTemplate(templ, ang)
		if rot.Rows > src.Rows || rot.Cols > src.Cols {
			continue
		}
		scores, err := MatchTemplate(src, rot, params.Method)
		if err != nil {
			return nil, err
		}
		matches := ExtractMatches(scores, rot.Cols, rot.Rows, params.Threshold, higher)
		for i := range matches {
			matches[i].Angle = ang
		}
		all = append(all, matches...)
	}

	sortMatches(all, higher)
	if params.NMSIoU > 0 {
		all = NonMaxSuppression(all, params.NMSIoU, higher)
	}
	return all, nil
}

// BestMatchRotated returns the single strongest detection of templ in src across
// a full 0..350 degree sweep in 10 degree steps under the given method. It
// ignores the score threshold and always returns the best candidate found, or an
// error if no rotation fits inside src.
func BestMatchRotated(src, templ *cv.Mat, method Method) (Match, error) {
	params := DefaultRotationParams()
	params.Method = method
	if method.HigherIsBetter() {
		params.Threshold = math.Inf(-1)
	} else {
		params.Threshold = math.Inf(1)
	}
	params.NMSIoU = 0
	matches, err := MatchRotationInvariant(src, templ, params)
	if err != nil {
		return Match{}, err
	}
	if len(matches) == 0 {
		return Match{}, ErrTemplateLarger
	}
	return matches[0], nil
}

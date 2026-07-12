package linedescriptor

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// PointF is a sub-pixel image coordinate (X is the column, Y the row). It
// complements [cv.Point], which is integer-only, and is used for the
// octave-space and back-projected endpoints of a [KeyLineEx] where fractional
// precision matters.
type PointF struct {
	X float64
	Y float64
}

// KeyLineEx is a fully-populated line segment mirroring every field of OpenCV's
// cv::line_descriptor::KeyLine. It embeds the compact [KeyLine] (whose
// StartPoint, EndPoint, Angle, Length, Response and Octave remain available by
// promotion, always expressed in the coordinates of the original, full
// resolution image) and adds the octave-space geometry that the multi-octave
// detector and descriptor need.
//
// The extra fields correspond to the upstream members sPointInOctaveX/Y,
// ePointInOctaveX/Y, lineLength, numOfPixels, class_id, pt and size that the
// single-scale [KeyLine] omits.
type KeyLineEx struct {
	KeyLine

	// ClassID is a caller-assignable label used to group or identify segments;
	// the detector fills it with the segment's index within its octave.
	ClassID int
	// Pt is the midpoint of the segment in original-image coordinates.
	Pt PointF
	// Size is a rough scale measure of the segment, set to its octave-space
	// length so that segments found in coarser octaves report a larger size.
	Size float64
	// StartPointF and EndPointF are the endpoints in original-image coordinates
	// at sub-pixel precision (the rounded integer versions live in the embedded
	// KeyLine).
	StartPointF PointF
	EndPointF   PointF
	// StartPointInOctave and EndPointInOctave are the endpoints expressed in the
	// coordinate system of the octave image the segment was actually detected
	// in, before back-projection to full resolution.
	StartPointInOctave PointF
	EndPointInOctave   PointF
	// LineLength is the Euclidean segment length measured in the octave image
	// (so LineLength × scale ≈ Length).
	LineLength float64
	// NumOfPixels is the number of pixels the segment spans in its octave image,
	// i.e. round(LineLength).
	NumOfPixels int
}

// ScalePyramid is a Gaussian-style scale pyramid of grayscale images used for
// multi-octave line detection and description. Levels[0] is the original image
// (reduced to luma) and each subsequent level is downsampled by the cumulative
// factor recorded in the matching entry of Scales, so a coordinate (x, y) in
// Levels[o] corresponds to (x×Scales[o], y×Scales[o]) in the original image.
type ScalePyramid struct {
	// Levels holds one single-channel image per octave, coarsest last.
	Levels []*cv.Mat
	// Scales[o] is the downsampling factor of Levels[o] relative to the
	// original image (Scales[0] == 1).
	Scales []float64
}

// BuildScalePyramid constructs a numOctaves-level [ScalePyramid] from img by
// repeatedly shrinking it by scaleFactor. img may be 1- or 3-channel; every
// level is stored as a single-channel luma image. scaleFactor must be greater
// than 1 and numOctaves must be at least 1. Octaves whose downsampled size
// would collapse below 1 pixel are dropped, so the returned pyramid may have
// fewer than numOctaves levels.
//
// Each level is lightly Gaussian-blurred before it is resampled, approximating
// the anti-aliased pyramid the upstream module builds so that lines survive to
// the coarser octaves instead of aliasing away.
func BuildScalePyramid(img *cv.Mat, numOctaves int, scaleFactor float64) *ScalePyramid {
	if numOctaves < 1 {
		panic("linedescriptor: BuildScalePyramid requires numOctaves >= 1")
	}
	if scaleFactor <= 1 {
		panic("linedescriptor: BuildScalePyramid requires scaleFactor > 1")
	}
	base := toGray(img)
	p := &ScalePyramid{
		Levels: []*cv.Mat{base},
		Scales: []float64{1},
	}
	for o := 1; o < numOctaves; o++ {
		scale := math.Pow(scaleFactor, float64(o))
		w := int(math.Round(float64(base.Cols) / scale))
		h := int(math.Round(float64(base.Rows) / scale))
		if w < 2 || h < 2 {
			break
		}
		blurred := cv.GaussianBlur(base, 3, 0.8)
		level := cv.Resize(blurred, w, h, cv.InterLinear)
		p.Levels = append(p.Levels, level)
		p.Scales = append(p.Scales, scale)
	}
	return p
}

// DetectWithScale runs the detector on a single downscaled octave of img and
// returns the segments back-projected into original-image coordinates. octave 0
// is the original resolution; octave o shrinks the image by 2^o before
// detection, so coarser octaves recover longer, blurrier structure while
// suppressing fine texture. Every returned [KeyLine] carries the given octave in
// its [KeyLine.Octave] field, and its endpoints, length and response are scaled
// back up to the original image. octave must be non-negative.
//
// This is the single-scale building block that [LSDDetector.DetectPyramid]
// invokes for every level of a [ScalePyramid].
func (d *LSDDetector) DetectWithScale(img *cv.Mat, octave int) []KeyLine {
	if octave < 0 {
		panic("linedescriptor: DetectWithScale requires octave >= 0")
	}
	gray := toGray(img)
	scale := math.Pow(2, float64(octave))
	small := gray
	if octave > 0 {
		w := int(math.Round(float64(gray.Cols) / scale))
		h := int(math.Round(float64(gray.Rows) / scale))
		if w < 2 || h < 2 {
			return nil
		}
		blurred := cv.GaussianBlur(gray, 3, 0.8)
		small = cv.Resize(blurred, w, h, cv.InterLinear)
	}
	lines := d.Detect(small)
	out := make([]KeyLine, len(lines))
	for i, kl := range lines {
		scaled := newKeyLine(
			float64(kl.StartPoint.X)*scale, float64(kl.StartPoint.Y)*scale,
			float64(kl.EndPoint.X)*scale, float64(kl.EndPoint.Y)*scale,
		)
		scaled.Octave = octave
		out[i] = scaled
	}
	return out
}

// DetectPyramid performs true multi-octave line detection. It builds a
// numOctaves-level [ScalePyramid] of img (shrinking by scaleFactor per octave),
// detects segments independently in every level and returns them as
// fully-populated [KeyLineEx] values whose octave-space endpoints are recorded
// and whose original-image geometry is back-projected. Unlike the single-scale
// [LSDDetector.Detect] (where every [KeyLine.Octave] is 0), the segments here
// carry the real octave they were found in, so callers can reason about scale.
//
// Results are sorted by descending [KeyLine.Response] (original-image length),
// with ties broken by ascending octave and then by [KeyLineEx.ClassID], so the
// output is deterministic. scaleFactor must exceed 1 and numOctaves must be at
// least 1.
func (d *LSDDetector) DetectPyramid(img *cv.Mat, numOctaves int, scaleFactor float64) []KeyLineEx {
	pyr := BuildScalePyramid(img, numOctaves, scaleFactor)
	var out []KeyLineEx
	for octave, level := range pyr.Levels {
		scale := pyr.Scales[octave]
		lines := d.Detect(level)
		for id, kl := range lines {
			out = append(out, makeKeyLineEx(kl, octave, id, scale))
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Response != out[j].Response {
			return out[i].Response > out[j].Response
		}
		if out[i].Octave != out[j].Octave {
			return out[i].Octave < out[j].Octave
		}
		return out[i].ClassID < out[j].ClassID
	})
	return out
}

// makeKeyLineEx promotes a single-octave detection (kl, in the coordinates of
// the octave image with the given cumulative scale) into a fully populated
// KeyLineEx expressed in both octave and original-image coordinates.
func makeKeyLineEx(kl KeyLine, octave, classID int, scale float64) KeyLineEx {
	sxO := float64(kl.StartPoint.X)
	syO := float64(kl.StartPoint.Y)
	exO := float64(kl.EndPoint.X)
	eyO := float64(kl.EndPoint.Y)

	sx := sxO * scale
	sy := syO * scale
	ex := exO * scale
	ey := eyO * scale

	base := newKeyLine(sx, sy, ex, ey)
	base.Octave = octave

	octLen := math.Hypot(exO-sxO, eyO-syO)
	return KeyLineEx{
		KeyLine:            base,
		ClassID:            classID,
		Pt:                 PointF{X: (sx + ex) / 2, Y: (sy + ey) / 2},
		Size:               octLen,
		StartPointF:        PointF{X: sx, Y: sy},
		EndPointF:          PointF{X: ex, Y: ey},
		StartPointInOctave: PointF{X: sxO, Y: syO},
		EndPointInOctave:   PointF{X: exO, Y: eyO},
		LineLength:         octLen,
		NumOfPixels:        int(math.Round(octLen)),
	}
}

// ToKeyLines extracts the embedded original-image [KeyLine] of every segment,
// discarding the octave-space extras. It is a convenience for feeding
// multi-octave results into routines such as [DrawKeylines] or
// [BinaryDescriptor.Compute] that expect plain [KeyLine] values.
func ToKeyLines(lines []KeyLineEx) []KeyLine {
	out := make([]KeyLine, len(lines))
	for i, l := range lines {
		out[i] = l.KeyLine
	}
	return out
}

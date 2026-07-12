package linedescriptor

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BinaryDescriptorParams collects the tunables of the LBD descriptor, mirroring
// cv::line_descriptor::BinaryDescriptor::Params. It bundles the settings that
// are otherwise scattered across [BinaryDescriptor] and the multi-octave
// pipeline so they can be passed around as one value and fed to
// [NewBinaryDescriptorWithParams].
type BinaryDescriptorParams struct {
	// NumOfOctaves is the number of scale-pyramid octaves the descriptor is
	// meant to operate over (used by the multi-octave pipeline). It must be at
	// least 1.
	NumOfOctaves int
	// WidthOfBand is the thickness in pixels of each parallel band of the
	// support region, corresponding to upstream widthOfBand_. It maps to
	// [BinaryDescriptor.BandWidth].
	WidthOfBand int
	// NumOfBands is the number of parallel bands across the support region,
	// mapping to [BinaryDescriptor.NumBands].
	NumOfBands int
	// ReductionRatio is the pyramid downsampling factor between consecutive
	// octaves, corresponding to upstream reductionRatio_. It must exceed 1.
	ReductionRatio float64
}

// DefaultBinaryDescriptorParams returns the upstream default configuration: a
// single octave, 8 bands of 7 pixels each and a pyramid reduction ratio of 2.
func DefaultBinaryDescriptorParams() BinaryDescriptorParams {
	return BinaryDescriptorParams{
		NumOfOctaves:   1,
		WidthOfBand:    7,
		NumOfBands:     8,
		ReductionRatio: 2,
	}
}

// Validate reports the first problem with the parameters, or nil if they are
// usable. It checks that every field lies in its permitted range.
func (p BinaryDescriptorParams) Validate() error {
	switch {
	case p.NumOfOctaves < 1:
		return &paramError{"NumOfOctaves must be >= 1"}
	case p.WidthOfBand < 1:
		return &paramError{"WidthOfBand must be >= 1"}
	case p.NumOfBands < 1:
		return &paramError{"NumOfBands must be >= 1"}
	case p.ReductionRatio <= 1:
		return &paramError{"ReductionRatio must be > 1"}
	default:
		return nil
	}
}

// paramError is a small error type describing an invalid parameter value.
type paramError struct{ msg string }

func (e *paramError) Error() string { return "linedescriptor: " + e.msg }

// NewBinaryDescriptorWithParams builds a [BinaryDescriptor] whose band geometry
// is taken from p. It panics if p is invalid (see [BinaryDescriptorParams.Validate]).
func NewBinaryDescriptorWithParams(p BinaryDescriptorParams) *BinaryDescriptor {
	if err := p.Validate(); err != nil {
		panic(err.Error())
	}
	return &BinaryDescriptor{
		NumBands:  p.NumOfBands,
		BandWidth: p.WidthOfBand,
	}
}

// ComputeMultiOctave describes each multi-octave segment using the image data of
// the octave it was actually detected in, so that a segment found in a coarse
// octave is sampled at that coarse resolution rather than in the full-size
// image. For every [KeyLineEx] the descriptor uses pyr.Levels[line.Octave] and
// the segment's octave-space endpoints ([KeyLineEx.StartPointInOctave] /
// [KeyLineEx.EndPointInOctave]); a segment whose octave is missing from the
// pyramid yields an all-zero code. The lines are returned unchanged alongside
// one binary code per line, aligned by index.
//
// This is the per-octave counterpart of [BinaryDescriptor.Compute], which always
// samples at full resolution.
func (bd *BinaryDescriptor) ComputeMultiOctave(pyr *ScalePyramid, lines []KeyLineEx) ([]KeyLineEx, [][]byte) {
	if bd.NumBands <= 0 || bd.BandWidth <= 0 {
		panic("linedescriptor: BinaryDescriptor requires positive NumBands and BandWidth")
	}
	// Compute each octave's gradients at most once.
	type grad struct {
		gx, gy     []float64
		rows, cols int
	}
	cache := make(map[int]grad)
	gradFor := func(octave int) (grad, bool) {
		if g, ok := cache[octave]; ok {
			return g, true
		}
		if octave < 0 || octave >= len(pyr.Levels) {
			return grad{}, false
		}
		gx, gy, rows, cols := gradients(pyr.Levels[octave])
		g := grad{gx: gx, gy: gy, rows: rows, cols: cols}
		cache[octave] = g
		return g, true
	}

	out := make([][]byte, len(lines))
	for i, line := range lines {
		g, ok := gradFor(line.Octave)
		if !ok {
			out[i] = make([]byte, bd.DescriptorSize())
			continue
		}
		octLine := octaveKeyLine(line)
		out[i] = bd.describe(octLine, g.gx, g.gy, g.rows, g.cols)
	}
	return lines, out
}

// octaveKeyLine builds a plain KeyLine whose endpoints are the octave-space
// coordinates of line, suitable for description against that octave's image.
func octaveKeyLine(line KeyLineEx) KeyLine {
	return newKeyLine(
		line.StartPointInOctave.X, line.StartPointInOctave.Y,
		line.EndPointInOctave.X, line.EndPointInOctave.Y,
	)
}

// DrawKeylinesEx renders multi-octave segments onto a colour copy of img,
// scaling the drawn thickness by each segment's octave so that segments from
// coarser octaves (which correspond to larger image structure) appear bolder.
// It is the [KeyLineEx] analogue of [DrawKeylines].
func DrawKeylinesEx(img *cv.Mat, lines []KeyLineEx, color cv.Scalar, baseThickness int) *cv.Mat {
	if baseThickness < 1 {
		baseThickness = 1
	}
	canvas := promoteRGB(img)
	for _, l := range lines {
		th := baseThickness + int(math.Round(float64(l.Octave)*float64(baseThickness)*0.5))
		cv.Line(canvas, l.StartPoint, l.EndPoint, color, th)
	}
	return canvas
}

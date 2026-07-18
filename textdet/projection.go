package textdet

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// HorizontalProjection returns the per-row foreground counts of a binary
// single-channel image (any non-zero sample counts as foreground). The result
// has one entry per row; peaks correspond to text lines and valleys to the gaps
// between them. It returns [ErrEmpty] for an empty image.
func HorizontalProjection(binary *cv.Mat) ([]int, error) {
	fg, rows, cols, err := textdetForeground(binary)
	if err != nil {
		return nil, err
	}
	prof := make([]int, rows)
	for y := 0; y < rows; y++ {
		row := 0
		base := y * cols
		for x := 0; x < cols; x++ {
			if fg[base+x] {
				row++
			}
		}
		prof[y] = row
	}
	return prof, nil
}

// VerticalProjection returns the per-column foreground counts of a binary
// single-channel image. Within a single text line, peaks correspond to glyph
// strokes and valleys to the spaces between characters and words. It returns
// [ErrEmpty] for an empty image.
func VerticalProjection(binary *cv.Mat) ([]int, error) {
	fg, rows, cols, err := textdetForeground(binary)
	if err != nil {
		return nil, err
	}
	prof := make([]int, cols)
	for y := 0; y < rows; y++ {
		base := y * cols
		for x := 0; x < cols; x++ {
			if fg[base+x] {
				prof[x]++
			}
		}
	}
	return prof, nil
}

// SmoothProfile returns a moving-average smoothing of prof with a window of
// (2*radius+1) samples and replicated borders. A radius of 0 returns a copy.
// It returns [ErrInvalidArgument] for radius < 0.
func SmoothProfile(prof []int, radius int) ([]float64, error) {
	if radius < 0 {
		return nil, ErrInvalidArgument
	}
	n := len(prof)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for k := -radius; k <= radius; k++ {
			j := i + k
			if j < 0 {
				j = 0
			} else if j >= n {
				j = n - 1
			}
			sum += float64(prof[j])
		}
		out[i] = sum / float64(2*radius+1)
	}
	return out, nil
}

// Band is a contiguous run of indices in a projection profile whose values
// exceed the segmentation threshold, such as a text line (horizontal profile)
// or a word/character (vertical profile).
type Band struct {
	// Start is the first index of the band (inclusive).
	Start int
	// End is the last index of the band (inclusive).
	End int
}

// Length returns the number of indices covered by the band.
func (b Band) Length() int { return b.End - b.Start + 1 }

// SegmentBands splits a projection profile into bands of consecutive indices
// whose value is strictly greater than threshold. Bands separated by a gap of
// fewer than minGap sub-threshold indices are merged, and bands shorter than
// minLength indices are discarded. It returns [ErrInvalidArgument] if minGap < 0
// or minLength < 0.
func SegmentBands(prof []int, threshold, minGap, minLength int) ([]Band, error) {
	if minGap < 0 || minLength < 0 {
		return nil, ErrInvalidArgument
	}
	var raw []Band
	inBand := false
	start := 0
	for i, v := range prof {
		if v > threshold {
			if !inBand {
				inBand = true
				start = i
			}
		} else if inBand {
			inBand = false
			raw = append(raw, Band{Start: start, End: i - 1})
		}
	}
	if inBand {
		raw = append(raw, Band{Start: start, End: len(prof) - 1})
	}
	// Merge bands separated by a small gap.
	var merged []Band
	for _, b := range raw {
		if len(merged) > 0 && b.Start-merged[len(merged)-1].End-1 < minGap {
			merged[len(merged)-1].End = b.End
		} else {
			merged = append(merged, b)
		}
	}
	// Drop short bands.
	out := merged[:0]
	for _, b := range merged {
		if b.Length() >= minLength {
			out = append(out, b)
		}
	}
	return out, nil
}

// SegmentTextLines locates text lines in a binary page image by thresholding
// its horizontal projection. A row is part of a line when its foreground count
// exceeds rowThreshold; lines separated by fewer than minGap blank rows are
// merged and lines shorter than minHeight rows are discarded. The returned
// rectangles span the full image width. It returns [ErrEmpty] for an empty image.
func SegmentTextLines(binary *cv.Mat, rowThreshold, minGap, minHeight int) ([]cv.Rect, error) {
	prof, err := HorizontalProjection(binary)
	if err != nil {
		return nil, err
	}
	bands, err := SegmentBands(prof, rowThreshold, minGap, minHeight)
	if err != nil {
		return nil, err
	}
	rects := make([]cv.Rect, len(bands))
	for i, b := range bands {
		rects[i] = cv.Rect{X: 0, Y: b.Start, Width: binary.Cols, Height: b.Length()}
	}
	return rects, nil
}

// SegmentWords locates words within a single text-line image by thresholding
// its vertical projection. A column belongs to a word when its foreground count
// exceeds colThreshold; columns separated by fewer than minGap blank columns
// join the same word (so intra-character gaps do not split it) and words
// narrower than minWidth columns are discarded. The returned rectangles span
// the full image height. It returns [ErrEmpty] for an empty image.
func SegmentWords(lineImage *cv.Mat, colThreshold, minGap, minWidth int) ([]cv.Rect, error) {
	prof, err := VerticalProjection(lineImage)
	if err != nil {
		return nil, err
	}
	bands, err := SegmentBands(prof, colThreshold, minGap, minWidth)
	if err != nil {
		return nil, err
	}
	rects := make([]cv.Rect, len(bands))
	for i, b := range bands {
		rects[i] = cv.Rect{X: b.Start, Y: 0, Width: b.Length(), Height: lineImage.Rows}
	}
	return rects, nil
}

// SegmentCharacters locates individual characters within a single text-line
// image by thresholding its vertical projection with no gap merging, so every
// run of ink columns wider than minWidth becomes its own box. The returned
// rectangles span the full image height. It returns [ErrEmpty] for an empty
// image.
func SegmentCharacters(lineImage *cv.Mat, colThreshold, minWidth int) ([]cv.Rect, error) {
	prof, err := VerticalProjection(lineImage)
	if err != nil {
		return nil, err
	}
	bands, err := SegmentBands(prof, colThreshold, 1, minWidth)
	if err != nil {
		return nil, err
	}
	rects := make([]cv.Rect, len(bands))
	for i, b := range bands {
		rects[i] = cv.Rect{X: b.Start, Y: 0, Width: b.Length(), Height: lineImage.Rows}
	}
	return rects, nil
}

// ProjectionVariance returns the variance of the horizontal projection of a
// binary image rotated by angle degrees about its centre. Foreground pixels are
// projected onto the rotated vertical axis and binned per row; the variance of
// those bin counts peaks when text lines are horizontal, which is the basis of
// projection-profile skew estimation. It returns [ErrEmpty] for an empty image.
func ProjectionVariance(binary *cv.Mat, angle float64) (float64, error) {
	fg, rows, cols, err := textdetForeground(binary)
	if err != nil {
		return 0, err
	}
	rad := angle * math.Pi / 180
	sin, cos := math.Sin(rad), math.Cos(rad)
	cx := float64(cols-1) / 2
	cy := float64(rows-1) / 2
	// The projected row index ranges over roughly [-diag, diag]; offset into a
	// non-negative bin array.
	diag := int(math.Ceil(math.Hypot(float64(rows), float64(cols)))) + 2
	bins := make([]float64, 2*diag+1)
	total := 0.0
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if !fg[y*cols+x] {
				continue
			}
			rx := float64(x) - cx
			ry := float64(y) - cy
			// Rotated vertical coordinate.
			r := -rx*sin + ry*cos
			idx := int(math.Round(r)) + diag
			if idx < 0 {
				idx = 0
			} else if idx >= len(bins) {
				idx = len(bins) - 1
			}
			bins[idx]++
			total++
		}
	}
	if total == 0 {
		return 0, nil
	}
	mean := total / float64(len(bins))
	varSum := 0.0
	for _, v := range bins {
		d := v - mean
		varSum += d * d
	}
	return varSum / float64(len(bins)), nil
}

// EstimateSkew estimates the skew angle of a binary text image in degrees by
// maximising the variance of its horizontal projection over candidate angles.
// It scans angles in [-maxAngle, +maxAngle] with the given step (both in
// degrees) and returns the angle whose projection variance is highest; a
// positive result means the text is rotated counter-clockwise relative to
// horizontal. It returns [ErrInvalidArgument] if maxAngle <= 0 or step <= 0.
func EstimateSkew(binary *cv.Mat, maxAngle, step float64) (float64, error) {
	if maxAngle <= 0 || step <= 0 {
		return 0, ErrInvalidArgument
	}
	// First pass: find the maximum projection variance.
	bestVar := -1.0
	for a := -maxAngle; a <= maxAngle+1e-9; a += step {
		v, err := ProjectionVariance(binary, a)
		if err != nil {
			return 0, err
		}
		if v > bestVar {
			bestVar = v
		}
	}
	// Second pass: among angles that reach the maximum (variance may be flat
	// across a plateau for coarse structures), return the one closest to
	// horizontal for a stable, symmetric result.
	const eps = 1e-9
	best := 0.0
	haveBest := false
	for a := -maxAngle; a <= maxAngle+1e-9; a += step {
		v, err := ProjectionVariance(binary, a)
		if err != nil {
			return 0, err
		}
		if v >= bestVar-eps {
			if !haveBest || math.Abs(a) < math.Abs(best) {
				best = a
				haveBest = true
			}
		}
	}
	return best, nil
}

// RotateImage rotates src about its centre by angle degrees (counter-clockwise)
// using bilinear interpolation, keeping the original dimensions. Areas outside
// the source are filled with zero. It returns [ErrEmpty] for an empty image.
func RotateImage(src *cv.Mat, angle float64) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmpty
	}
	cx := float64(src.Cols-1) / 2
	cy := float64(src.Rows-1) / 2
	m := cv.GetRotationMatrix2D(cx, cy, angle, 1.0)
	return cv.WarpAffine(src, m, src.Cols, src.Rows, cv.InterLinear), nil
}

// CorrectSkew estimates the skew of a binary text image with [EstimateSkew] and
// returns a deskewed copy rotated to bring the text back to horizontal, along
// with the detected angle in degrees. The binary mask is used only to measure
// the angle; the rotation is applied to that same mask. It returns
// [ErrInvalidArgument] for the same conditions as [EstimateSkew].
func CorrectSkew(binary *cv.Mat, maxAngle, step float64) (*cv.Mat, float64, error) {
	angle, err := EstimateSkew(binary, maxAngle, step)
	if err != nil {
		return nil, 0, err
	}
	// A positive skew (counter-clockwise text) is undone by rotating clockwise,
	// i.e. by -angle in the counter-clockwise convention of RotateImage.
	corrected, err := RotateImage(binary, -angle)
	if err != nil {
		return nil, 0, err
	}
	return corrected, angle, nil
}

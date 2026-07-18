package inpaint

import (
	"image/color"

	cv "github.com/malcolmston/opencv"
)

// inpaintLumaBuf returns the per-pixel BT.601 luma of img as a float64 slice in
// row-major order.
func inpaintLumaBuf(img *cv.Mat) []float64 {
	buf := make([]float64, img.Rows*img.Cols)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			buf[y*img.Cols+x] = inpaintLuma(img, y, x)
		}
	}
	return buf
}

// inpaintMorphLine applies a grayscale morphological operation (dilate=true for
// max, false for min) with a 1-D structuring element of half-length half, either
// horizontal or vertical, using edge replication.
func inpaintMorphLine(buf []float64, rows, cols, half int, horizontal, dilate bool) []float64 {
	out := make([]float64, len(buf))
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var acc float64
			first := true
			for k := -half; k <= half; k++ {
				ny, nx := y, x
				if horizontal {
					nx = inpaintClampInt(x+k, 0, cols-1)
				} else {
					ny = inpaintClampInt(y+k, 0, rows-1)
				}
				v := buf[ny*cols+nx]
				if first {
					acc = v
					first = false
				} else if (dilate && v > acc) || (!dilate && v < acc) {
					acc = v
				}
			}
			out[y*cols+x] = acc
		}
	}
	return out
}

// ScratchOptions configures [DetectScratches].
type ScratchOptions struct {
	// Threshold is the minimum top-hat contrast (in luma units) for a pixel to be
	// marked as a scratch. A non-positive value uses 20.
	Threshold float64
	// MaxWidth is the maximum scratch width in pixels; the structuring element is
	// 2*MaxWidth+1 long across the scratch. A non-positive value uses 3.
	MaxWidth int
	// Vertical enables detection of vertical scratches (using a horizontal
	// structuring element).
	Vertical bool
	// Horizontal enables detection of horizontal scratches.
	Horizontal bool
	// DetectBright marks scratches brighter than their surroundings.
	DetectBright bool
	// DetectDark marks scratches darker than their surroundings.
	DetectDark bool
}

// DefaultScratchOptions returns settings that detect both bright and dark
// vertical scratches up to 3 pixels wide (the common film-scratch case).
func DefaultScratchOptions() ScratchOptions {
	return ScratchOptions{Threshold: 20, MaxWidth: 3, Vertical: true, DetectBright: true, DetectDark: true}
}

// DetectScratches locates thin line defects (film scratches, pen strokes) in img
// via a directional grayscale top-hat: opening with a line structuring element
// removes structures thinner than the element, and the residual isolates thin
// lines running across it. A pixel is selected when the bright (image minus
// opening) or dark (closing minus image) residual exceeds the threshold, for the
// enabled orientations. The returned [Mask] is suitable for passing to any
// inpainting routine. img may be single- or three-channel.
func DetectScratches(img *cv.Mat, opts ScratchOptions) *Mask {
	inpaintRequireImage(img, "DetectScratches")
	if opts.Threshold <= 0 {
		opts.Threshold = 20
	}
	if opts.MaxWidth <= 0 {
		opts.MaxWidth = 3
	}
	if !opts.Vertical && !opts.Horizontal {
		opts.Vertical = true
	}
	if !opts.DetectBright && !opts.DetectDark {
		opts.DetectBright = true
		opts.DetectDark = true
	}
	rows, cols := img.Rows, img.Cols
	luma := inpaintLumaBuf(img)
	half := opts.MaxWidth
	out := NewMask(rows, cols)

	orientations := make([]bool, 0, 2)
	if opts.Vertical {
		orientations = append(orientations, true) // horizontal SE detects vertical lines
	}
	if opts.Horizontal {
		orientations = append(orientations, false)
	}
	for _, horiz := range orientations {
		eroded := inpaintMorphLine(luma, rows, cols, half, horiz, false)
		opening := inpaintMorphLine(eroded, rows, cols, half, horiz, true)
		dilated := inpaintMorphLine(luma, rows, cols, half, horiz, true)
		closing := inpaintMorphLine(dilated, rows, cols, half, horiz, false)
		for i := 0; i < rows*cols; i++ {
			if opts.DetectBright && luma[i]-opening[i] >= opts.Threshold {
				out.Data[i] = true
			}
			if opts.DetectDark && closing[i]-luma[i] >= opts.Threshold {
				out.Data[i] = true
			}
		}
	}
	return out
}

// TextOptions configures [DetectText].
type TextOptions struct {
	// GradientThreshold is the minimum morphological-gradient magnitude (in luma
	// units) for a pixel to count as a text edge. A non-positive value uses 40.
	GradientThreshold float64
	// DilateRadius grows and horizontally connects detected edges into solid text
	// regions. A non-positive value uses 2.
	DilateRadius int
}

// DefaultTextOptions returns default text-detection settings (gradient threshold
// 40, dilate radius 2).
func DefaultTextOptions() TextOptions {
	return TextOptions{GradientThreshold: 40, DilateRadius: 2}
}

// DetectText locates high-contrast text or caption overlays in img. It computes
// the morphological gradient of the luma (dilation minus erosion with a 3x3
// element), thresholds it to find character edges, then grows the result so the
// strokes of each glyph merge into a filled region. The returned [Mask] can be
// fed to an inpainting routine to erase burned-in captions or watermarks. img
// may be single- or three-channel.
func DetectText(img *cv.Mat, opts TextOptions) *Mask {
	inpaintRequireImage(img, "DetectText")
	if opts.GradientThreshold <= 0 {
		opts.GradientThreshold = 40
	}
	if opts.DilateRadius <= 0 {
		opts.DilateRadius = 2
	}
	rows, cols := img.Rows, img.Cols
	luma := inpaintLumaBuf(img)
	// 3x3 morphological gradient.
	dil := inpaintMorphLine(inpaintMorphLine(luma, rows, cols, 1, true, true), rows, cols, 1, false, true)
	ero := inpaintMorphLine(inpaintMorphLine(luma, rows, cols, 1, true, false), rows, cols, 1, false, false)
	edges := NewMask(rows, cols)
	for i := 0; i < rows*cols; i++ {
		if dil[i]-ero[i] >= opts.GradientThreshold {
			edges.Data[i] = true
		}
	}
	return edges.Dilate(opts.DilateRadius)
}

// BlotchOptions configures [DetectBlotches].
type BlotchOptions struct {
	// Threshold is the minimum top-hat contrast (in luma units) for a speck. A
	// non-positive value uses 25.
	Threshold float64
	// MaxSize is the maximum blotch radius in pixels; the square structuring
	// element is 2*MaxSize+1 on a side. A non-positive value uses 2.
	MaxSize int
	// DetectBright marks specks brighter than their surroundings (dust flashes).
	DetectBright bool
	// DetectDark marks specks darker than their surroundings (dirt).
	DetectDark bool
}

// DefaultBlotchOptions returns settings that detect both bright and dark specks
// up to radius 2 (dust and dirt on scanned film).
func DefaultBlotchOptions() BlotchOptions {
	return BlotchOptions{Threshold: 25, MaxSize: 2, DetectBright: true, DetectDark: true}
}

// DetectBlotches locates small isolated dust/dirt specks in img using a square
// grayscale top-hat: an opening with a square element the size of the largest
// wanted speck removes such specks, and the residual isolates them. Bright specks
// come from the white top-hat and dark specks from the black top-hat, gated by
// the options. The returned [Mask] can be fed to an inpainting routine. img may
// be single- or three-channel.
func DetectBlotches(img *cv.Mat, opts BlotchOptions) *Mask {
	inpaintRequireImage(img, "DetectBlotches")
	if opts.Threshold <= 0 {
		opts.Threshold = 25
	}
	if opts.MaxSize <= 0 {
		opts.MaxSize = 2
	}
	if !opts.DetectBright && !opts.DetectDark {
		opts.DetectBright = true
		opts.DetectDark = true
	}
	rows, cols := img.Rows, img.Cols
	luma := inpaintLumaBuf(img)
	half := opts.MaxSize
	// Square opening/closing = successive horizontal then vertical 1-D ops.
	erode := inpaintMorphLine(inpaintMorphLine(luma, rows, cols, half, true, false), rows, cols, half, false, false)
	opening := inpaintMorphLine(inpaintMorphLine(erode, rows, cols, half, true, true), rows, cols, half, false, true)
	dilate := inpaintMorphLine(inpaintMorphLine(luma, rows, cols, half, true, true), rows, cols, half, false, true)
	closing := inpaintMorphLine(inpaintMorphLine(dilate, rows, cols, half, true, false), rows, cols, half, false, false)
	out := NewMask(rows, cols)
	for i := 0; i < rows*cols; i++ {
		if opts.DetectBright && luma[i]-opening[i] >= opts.Threshold {
			out.Data[i] = true
		}
		if opts.DetectDark && closing[i]-luma[i] >= opts.Threshold {
			out.Data[i] = true
		}
	}
	return out
}

// inpaintColor8 returns the 8-bit RGB components of a color.Color.
func inpaintColor8(c color.Color) (r, g, b uint8) {
	rr, gg, bb, _ := c.RGBA()
	return uint8(rr >> 8), uint8(gg >> 8), uint8(bb >> 8)
}

// absDiffU8 returns |a-b| for two uint8 values.
func absDiffU8(a, b uint8) int {
	if a > b {
		return int(a - b)
	}
	return int(b - a)
}

// MaskFromColor selects the pixels of img whose colour is within tol (per
// channel, inclusive) of target — useful for erasing a solid-coloured logo,
// subtitle or chroma-key region. img must be three-channel; target is any
// image/color colour, taken as RGB.
func MaskFromColor(img *cv.Mat, target color.Color, tol uint8) *Mask {
	inpaintRequireImage(img, "MaskFromColor")
	inpaintRequireChannels(img, 3, "MaskFromColor")
	tr, tg, tb := inpaintColor8(target)
	out := NewMask(img.Rows, img.Cols)
	t := int(tol)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			if absDiffU8(img.At(y, x, 0), tr) <= t &&
				absDiffU8(img.At(y, x, 1), tg) <= t &&
				absDiffU8(img.At(y, x, 2), tb) <= t {
				out.Set(y, x, true)
			}
		}
	}
	return out
}

// MaskFromColorRange selects the pixels of img whose every channel lies within
// the inclusive range [lower, upper] (taken as RGB) — an axis-aligned colour box
// selector, the building block of chroma keying. img must be three-channel. It
// panics if any lower channel exceeds the matching upper channel.
func MaskFromColorRange(img *cv.Mat, lower, upper color.Color) *Mask {
	inpaintRequireImage(img, "MaskFromColorRange")
	inpaintRequireChannels(img, 3, "MaskFromColorRange")
	lr, lg, lb := inpaintColor8(lower)
	ur, ug, ub := inpaintColor8(upper)
	if lr > ur || lg > ug || lb > ub {
		panic("inpaint: MaskFromColorRange requires lower <= upper per channel")
	}
	out := NewMask(img.Rows, img.Cols)
	for y := 0; y < img.Rows; y++ {
		for x := 0; x < img.Cols; x++ {
			r := img.At(y, x, 0)
			g := img.At(y, x, 1)
			b := img.At(y, x, 2)
			if r >= lr && r <= ur && g >= lg && g <= ug && b >= lb && b <= ub {
				out.Set(y, x, true)
			}
		}
	}
	return out
}

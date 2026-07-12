package structured_light

import (
	"errors"
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Default thresholds used by [NewGrayCodePattern]; see the [GrayCodePattern]
// fields for their meaning. They match the defaults of OpenCV's
// cv::structured_light::GrayCodePattern.
const (
	// DefaultWhiteThreshold is the minimum contrast between a pattern sample
	// and its inverse for a bit to be considered reliable.
	DefaultWhiteThreshold = 5
	// DefaultBlackThreshold is the minimum contrast between the white and black
	// reference for a camera pixel to be considered lit (not in shadow).
	DefaultBlackThreshold = 40
)

// GrayCodeParams configures the projector resolution a [GrayCodePattern]
// encodes. Width and Height are the projector's column and row counts; both
// must be at least 2.
type GrayCodeParams struct {
	// Width is the number of projector columns to encode.
	Width int
	// Height is the number of projector rows to encode.
	Height int
}

// GrayCodePattern generates and decodes a binary reflected Gray-code pattern set
// for a projector of the configured resolution. Construct it with
// [NewGrayCodePattern]. The zero value is not usable.
type GrayCodePattern struct {
	// Width and Height are the encoded projector resolution.
	Width, Height int
	// numCol and numRow are the number of Gray-code bits (pattern images,
	// before adding inverses) needed to address every column / row.
	numCol, numRow int

	// WhiteThreshold is the minimum absolute difference between a pattern
	// sample and its inverse for the corresponding bit to be trusted during
	// decoding. Pixels with lower contrast are marked invalid.
	WhiteThreshold int
	// BlackThreshold is the minimum difference between the white and black
	// reference samples for a camera pixel to be treated as lit. Pixels below
	// it are considered shadow and marked invalid.
	BlackThreshold int
}

// NewGrayCodePattern returns a [GrayCodePattern] for the given projector
// resolution with the default robust-bit and shadow thresholds. It panics if
// either dimension is smaller than 2.
func NewGrayCodePattern(p GrayCodeParams) *GrayCodePattern {
	if p.Width < 2 || p.Height < 2 {
		panic(fmt.Sprintf("structured_light: GrayCode requires Width>=2 and Height>=2, got %dx%d", p.Width, p.Height))
	}
	return &GrayCodePattern{
		Width:          p.Width,
		Height:         p.Height,
		numCol:         numBits(p.Width),
		numRow:         numBits(p.Height),
		WhiteThreshold: DefaultWhiteThreshold,
		BlackThreshold: DefaultBlackThreshold,
	}
}

// NumColBits returns the number of Gray-code bits used to encode a projector
// column (before inverses are added).
func (g *GrayCodePattern) NumColBits() int { return g.numCol }

// NumRowBits returns the number of Gray-code bits used to encode a projector
// row (before inverses are added).
func (g *GrayCodePattern) NumRowBits() int { return g.numRow }

// NumberOfPatternImages returns the number of pattern images
// [GrayCodePattern.Generate] produces, namely 2*(NumColBits+NumRowBits): a
// pattern and an inverse for every column and row bit. It does not count the
// white/black reference pair.
func (g *GrayCodePattern) NumberOfPatternImages() int {
	return 2 * (g.numCol + g.numRow)
}

// Generate returns the full stack of projection pattern images, each a
// single-channel [github.com/malcolmston/opencv.Mat] of size Height×Width whose
// samples are 0 or 255. The layout is column bits first then row bits, each bit
// immediately followed by its photometric inverse:
//
//	[col0 col0inv col1 col1inv ... row0 row0inv row1 row1inv ...]
//
// [GrayCodePattern.Decode] expects a captured stack in exactly this order.
func (g *GrayCodePattern) Generate() []*cv.Mat {
	out := make([]*cv.Mat, 0, g.NumberOfPatternImages())
	// Column patterns: the encoded value is the pixel's x coordinate.
	for bit := 0; bit < g.numCol; bit++ {
		shift := g.numCol - 1 - bit // MSB first
		normal := cv.NewMat(g.Height, g.Width, 1)
		inverse := cv.NewMat(g.Height, g.Width, 1)
		for x := 0; x < g.Width; x++ {
			gray := binaryToGray(uint(x))
			var v uint8
			if (gray>>uint(shift))&1 == 1 {
				v = 255
			}
			inv := uint8(255) - v
			for y := 0; y < g.Height; y++ {
				normal.Set(y, x, 0, v)
				inverse.Set(y, x, 0, inv)
			}
		}
		out = append(out, normal, inverse)
	}
	// Row patterns: the encoded value is the pixel's y coordinate.
	for bit := 0; bit < g.numRow; bit++ {
		shift := g.numRow - 1 - bit
		normal := cv.NewMat(g.Height, g.Width, 1)
		inverse := cv.NewMat(g.Height, g.Width, 1)
		for y := 0; y < g.Height; y++ {
			gray := binaryToGray(uint(y))
			var v uint8
			if (gray>>uint(shift))&1 == 1 {
				v = 255
			}
			inv := uint8(255) - v
			for x := 0; x < g.Width; x++ {
				normal.Set(y, x, 0, v)
				inverse.Set(y, x, 0, inv)
			}
		}
		out = append(out, normal, inverse)
	}
	return out
}

// ReferenceImages returns the fully-lit (white, all 255) and fully-dark (black,
// all 0) reference patterns of size Height×Width. The captured versions of
// these two images are passed to [GrayCodePattern.Decode] to build the shadow
// mask and to measure per-pixel contrast.
func (g *GrayCodePattern) ReferenceImages() (white, black *cv.Mat) {
	white = cv.NewMat(g.Height, g.Width, 1)
	for i := range white.Data {
		white.Data[i] = 255
	}
	black = cv.NewMat(g.Height, g.Width, 1)
	return white, black
}

// Decoded holds the result of [GrayCodePattern.Decode]. All slices are indexed
// in row-major order (index = y*Cols + x) over the camera image.
type Decoded struct {
	// Rows and Cols are the camera image dimensions.
	Rows, Cols int
	// Col holds, for each camera pixel, the decoded projector column, or -1 if
	// the pixel is invalid (in shadow or with an unreliable bit).
	Col []int
	// Row holds, for each camera pixel, the decoded projector row, or -1 if the
	// pixel is invalid.
	Row []int
	// Mask reports which camera pixels decoded to a valid projector coordinate.
	Mask []bool
}

// At returns the decoded projector (col, row) for camera pixel (x, y) and
// whether that pixel is valid.
func (d *Decoded) At(y, x int) (col, row int, ok bool) {
	i := y*d.Cols + x
	return d.Col[i], d.Row[i], d.Mask[i]
}

// Decode recovers the camera→projector correspondence from a captured pattern
// stack. captured must contain exactly [GrayCodePattern.NumberOfPatternImages]
// images in the order produced by [GrayCodePattern.Generate]; white and black
// are the captured references. Every image must share the camera resolution and
// be single-channel or convertible to grayscale.
//
// For each camera pixel Decode:
//
//  1. marks the pixel lit only if white-black exceeds BlackThreshold (shadow
//     masking);
//  2. decides each Gray-code bit by comparing the pattern with its inverse,
//     marking the pixel invalid if their contrast is below WhiteThreshold (the
//     robust-bit test);
//  3. converts the recovered Gray codes to binary projector column and row,
//     marking the pixel invalid if either falls outside [0,Width)/[0,Height).
//
// It returns an error if the stack size or image dimensions are inconsistent.
func (g *GrayCodePattern) Decode(captured []*cv.Mat, white, black *cv.Mat) (*Decoded, error) {
	want := g.NumberOfPatternImages()
	if len(captured) != want {
		return nil, fmt.Errorf("structured_light: Decode expects %d captured images, got %d", want, len(captured))
	}
	if white == nil || black == nil {
		return nil, errors.New("structured_light: Decode requires non-nil white and black references")
	}
	rows, cols := white.Rows, white.Cols
	if black.Rows != rows || black.Cols != cols {
		return nil, errors.New("structured_light: Decode white/black references differ in size")
	}
	// Convert every input to grayscale planes up front.
	grays := make([]*cv.Mat, len(captured))
	for i, m := range captured {
		if m == nil {
			return nil, fmt.Errorf("structured_light: Decode captured[%d] is nil", i)
		}
		if m.Rows != rows || m.Cols != cols {
			return nil, fmt.Errorf("structured_light: Decode captured[%d] is %dx%d, want %dx%d", i, m.Rows, m.Cols, rows, cols)
		}
		grays[i] = toGray(m)
	}
	wg := toGray(white)
	bg := toGray(black)

	n := rows * cols
	res := &Decoded{
		Rows: rows,
		Cols: cols,
		Col:  make([]int, n),
		Row:  make([]int, n),
		Mask: make([]bool, n),
	}

	for p := 0; p < n; p++ {
		res.Col[p], res.Row[p] = -1, -1

		// Shadow mask from the reference pair.
		if int(wg.Data[p])-int(bg.Data[p]) <= g.BlackThreshold {
			continue
		}

		col, colOK := g.decodeAxis(grays, 0, g.numCol, p)
		if !colOK || col >= g.Width {
			continue
		}
		row, rowOK := g.decodeAxis(grays, g.numCol, g.numRow, p)
		if !rowOK || row >= g.Height {
			continue
		}
		res.Col[p] = col
		res.Row[p] = row
		res.Mask[p] = true
	}
	return res, nil
}

// decodeAxis reconstructs one coordinate (column or row) at pixel p by reading
// nbits Gray-code bits starting at bit index startBit in the pattern stack. It
// returns the binary coordinate and whether every bit passed the robust-bit
// test.
func (g *GrayCodePattern) decodeAxis(grays []*cv.Mat, startBit, nbits, p int) (int, bool) {
	var gray uint
	for bit := 0; bit < nbits; bit++ {
		idx := 2 * (startBit + bit)
		normal := int(grays[idx].Data[p])
		inverse := int(grays[idx+1].Data[p])
		diff := normal - inverse
		if abs(diff) < g.WhiteThreshold {
			return 0, false // unreliable bit
		}
		gray <<= 1
		if diff > 0 {
			gray |= 1
		}
	}
	return int(grayToBinary(gray)), true
}

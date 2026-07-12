package structured_light

import (
	"errors"
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Encoding selects how a [CodePattern] maps a projector coordinate to the bit
// sequence projected for it.
type Encoding int

const (
	// EncodingGray uses binary reflected Gray code: adjacent coordinates differ
	// in exactly one projected bit, so a single mis-decoded bit can only move the
	// result to an adjacent coordinate. This is the robust default and matches
	// [GrayCodePattern].
	EncodingGray Encoding = iota
	// EncodingBinary uses the natural binary representation of the coordinate.
	// It is simpler and occasionally requested for teaching or for comparison,
	// but a single bad bit can cause a large coordinate error, so it is less
	// robust than Gray code in practice.
	EncodingBinary
)

// String renders the encoding name.
func (e Encoding) String() string {
	switch e {
	case EncodingGray:
		return "gray"
	case EncodingBinary:
		return "binary"
	default:
		return fmt.Sprintf("Encoding(%d)", int(e))
	}
}

// CodePattern generates and decodes a column/row-encoding binary pattern set
// under a selectable [Encoding] (reflected Gray or natural binary), letting a
// caller compare the two schemes with an otherwise identical pipeline. The
// image layout, reference pair and decoded output match [GrayCodePattern]; only
// the coordinate-to-bits mapping changes. Construct it with [NewCodePattern].
type CodePattern struct {
	// Width and Height are the encoded projector resolution.
	Width, Height int
	// Encoding selects the coordinate-to-bit mapping.
	Encoding Encoding
	// WhiteThreshold is the minimum pattern/inverse contrast for a trusted bit.
	WhiteThreshold int
	// BlackThreshold is the minimum white/black contrast for a lit pixel.
	BlackThreshold int

	numCol, numRow int
}

// NewCodePattern returns a [CodePattern] for the given resolution and encoding
// with the default robust-bit and shadow thresholds. It panics if either
// dimension is smaller than 2.
func NewCodePattern(p GrayCodeParams, enc Encoding) *CodePattern {
	if p.Width < 2 || p.Height < 2 {
		panic(fmt.Sprintf("structured_light: CodePattern requires Width>=2 and Height>=2, got %dx%d", p.Width, p.Height))
	}
	return &CodePattern{
		Width:          p.Width,
		Height:         p.Height,
		Encoding:       enc,
		WhiteThreshold: DefaultWhiteThreshold,
		BlackThreshold: DefaultBlackThreshold,
		numCol:         numBits(p.Width),
		numRow:         numBits(p.Height),
	}
}

// NumColBits returns the number of code bits used to encode a projector column.
func (c *CodePattern) NumColBits() int { return c.numCol }

// NumRowBits returns the number of code bits used to encode a projector row.
func (c *CodePattern) NumRowBits() int { return c.numRow }

// NumberOfPatternImages returns 2*(NumColBits+NumRowBits), the number of images
// [CodePattern.Generate] produces (a pattern and its inverse per bit), excluding
// the white/black reference pair.
func (c *CodePattern) NumberOfPatternImages() int { return 2 * (c.numCol + c.numRow) }

// encode maps a coordinate to its projected code word under the pattern's
// encoding.
func (c *CodePattern) encode(v uint) uint {
	if c.Encoding == EncodingGray {
		return binaryToGray(v)
	}
	return v
}

// Generate returns the pattern stack, column bits first then row bits, each bit
// immediately followed by its photometric inverse — the same order and 0/255
// convention as [GrayCodePattern.Generate], but using the selected encoding.
func (c *CodePattern) Generate() []*cv.Mat {
	out := make([]*cv.Mat, 0, c.NumberOfPatternImages())
	for bit := 0; bit < c.numCol; bit++ {
		shift := c.numCol - 1 - bit
		normal := cv.NewMat(c.Height, c.Width, 1)
		inverse := cv.NewMat(c.Height, c.Width, 1)
		for x := 0; x < c.Width; x++ {
			code := c.encode(uint(x))
			var val uint8
			if (code>>uint(shift))&1 == 1 {
				val = 255
			}
			inv := uint8(255) - val
			for y := 0; y < c.Height; y++ {
				normal.Set(y, x, 0, val)
				inverse.Set(y, x, 0, inv)
			}
		}
		out = append(out, normal, inverse)
	}
	for bit := 0; bit < c.numRow; bit++ {
		shift := c.numRow - 1 - bit
		normal := cv.NewMat(c.Height, c.Width, 1)
		inverse := cv.NewMat(c.Height, c.Width, 1)
		for y := 0; y < c.Height; y++ {
			code := c.encode(uint(y))
			var val uint8
			if (code>>uint(shift))&1 == 1 {
				val = 255
			}
			inv := uint8(255) - val
			for x := 0; x < c.Width; x++ {
				normal.Set(y, x, 0, val)
				inverse.Set(y, x, 0, inv)
			}
		}
		out = append(out, normal, inverse)
	}
	return out
}

// ReferenceImages returns the all-white and all-black reference patterns, as in
// [GrayCodePattern.ReferenceImages].
func (c *CodePattern) ReferenceImages() (white, black *cv.Mat) {
	white = cv.NewMat(c.Height, c.Width, 1)
	for i := range white.Data {
		white.Data[i] = 255
	}
	black = cv.NewMat(c.Height, c.Width, 1)
	return white, black
}

// Decode recovers the camera→projector correspondence from a captured stack,
// mirroring [GrayCodePattern.Decode] but honouring the pattern's [Encoding] when
// converting recovered code words back to coordinates. captured must hold
// exactly [CodePattern.NumberOfPatternImages] images in generation order; white
// and black are the captured references. It returns an error on inconsistent
// sizes.
func (c *CodePattern) Decode(captured []*cv.Mat, white, black *cv.Mat) (*Decoded, error) {
	want := c.NumberOfPatternImages()
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
	res := &Decoded{Rows: rows, Cols: cols, Col: make([]int, n), Row: make([]int, n), Mask: make([]bool, n)}
	for p := 0; p < n; p++ {
		res.Col[p], res.Row[p] = -1, -1
		if int(wg.Data[p])-int(bg.Data[p]) <= c.BlackThreshold {
			continue
		}
		col, colOK := c.decodeAxis(grays, 0, c.numCol, p)
		if !colOK || col >= c.Width {
			continue
		}
		row, rowOK := c.decodeAxis(grays, c.numCol, c.numRow, p)
		if !rowOK || row >= c.Height {
			continue
		}
		res.Col[p], res.Row[p], res.Mask[p] = col, row, true
	}
	return res, nil
}

// decodeAxis reconstructs one coordinate from nbits code bits starting at
// startBit, applying the robust-bit test and converting from the pattern's
// encoding.
func (c *CodePattern) decodeAxis(grays []*cv.Mat, startBit, nbits, p int) (int, bool) {
	var code uint
	for bit := 0; bit < nbits; bit++ {
		idx := 2 * (startBit + bit)
		diff := int(grays[idx].Data[p]) - int(grays[idx+1].Data[p])
		if abs(diff) < c.WhiteThreshold {
			return 0, false
		}
		code <<= 1
		if diff > 0 {
			code |= 1
		}
	}
	if c.Encoding == EncodingGray {
		return int(grayToBinary(code)), true
	}
	return int(code), true
}

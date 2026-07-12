package datamatrix

import cv "github.com/malcolmston/opencv"

// Options controls how a symbol is rendered into a bitmap.
type Options struct {
	// ModulePixels is the side length, in pixels, of one module. Values <= 0
	// are replaced by the default of 8.
	ModulePixels int
	// QuietZoneModules is the width, in modules, of the mandatory white quiet
	// zone drawn around the symbol. Negative values are treated as 0.
	QuietZoneModules int
	// Channels is the number of channels of the produced Mat: 1 (grayscale)
	// or 3 (RGB). Any other value is replaced by 1.
	Channels int
}

// DefaultOptions returns the rendering options used by [Encode]: 8 pixels per
// module, a 2-module quiet zone and a single-channel (grayscale) bitmap.
func DefaultOptions() Options {
	return Options{ModulePixels: 8, QuietZoneModules: 2, Channels: 1}
}

func (o Options) normalized() Options {
	if o.ModulePixels <= 0 {
		o.ModulePixels = 8
	}
	if o.QuietZoneModules < 0 {
		o.QuietZoneModules = 0
	}
	if o.Channels != 1 && o.Channels != 3 {
		o.Channels = 1
	}
	return o
}

// Symbol is a fully-laid-out square Data Matrix symbol. Modules[r][c] is true
// where a module is dark (black). Row 0 is the top edge and column 0 the left
// edge; the solid finder pattern occupies the left column and the bottom row,
// while the alternating timing pattern occupies the top row and right column.
type Symbol struct {
	// Size is the symbol side length in modules, finder pattern included.
	Size int
	// Modules holds the module colours: true = dark.
	Modules [][]bool
}

// newSymbol allocates a symbol of the given side length with its finder and
// timing patterns already drawn.
func newSymbol(size int) *Symbol {
	mods := make([][]bool, size)
	for r := range mods {
		mods[r] = make([]bool, size)
	}
	s := &Symbol{Size: size, Modules: mods}
	s.drawFinder()
	return s
}

// drawFinder draws the solid "L" finder (left column and bottom row) and the
// alternating timing pattern (top row and right column).
func (s *Symbol) drawFinder() {
	n := s.Size
	for i := 0; i < n; i++ {
		s.Modules[n-1][i] = true     // bottom solid edge
		s.Modules[i][0] = true       // left solid edge
		s.Modules[0][i] = i%2 == 0   // top timing: dark on even columns
		s.Modules[i][n-1] = i%2 == 1 // right timing: dark on odd rows
	}
}

// setMapping stores a data module at mapping coordinate (r, c), which maps to
// symbol coordinate (r+1, c+1) since a one-module border surrounds the region.
func (s *Symbol) setMapping(r, c int, dark bool) {
	s.Modules[r+1][c+1] = dark
}

// getMapping reads the data module at mapping coordinate (r, c).
func (s *Symbol) getMapping(r, c int) bool {
	return s.Modules[r+1][c+1]
}

// encodeASCII converts text into ECC200 ASCII-encodation data codewords.
// Consecutive digit pairs are packed into a single codeword; every other
// character occupies one codeword. Non-ASCII bytes are rejected.
func encodeASCII(text string) ([]int, error) {
	var cw []int
	i := 0
	for i < len(text) {
		c := text[i]
		if isDigit(c) && i+1 < len(text) && isDigit(text[i+1]) {
			val := int(c-'0')*10 + int(text[i+1]-'0')
			cw = append(cw, 130+val)
			i += 2
			continue
		}
		if c > 127 {
			return nil, errNonASCII
		}
		cw = append(cw, int(c)+1)
		i++
	}
	return cw, nil
}

// padCodewords pads the data codewords out to dataCW using the ECC200 pad
// scheme: a single end-of-message codeword (129) followed by the 253-state
// randomised pad codewords.
func padCodewords(cw []int, dataCW int) []int {
	out := make([]int, len(cw), dataCW)
	copy(out, cw)
	if len(out) < dataCW {
		out = append(out, 129) // end of message
		for len(out) < dataCW {
			pos := len(out) + 1 // 1-based codeword position
			r := ((149 * pos) % 253) + 1
			v := 129 + r
			if v > 254 {
				v -= 254
			}
			out = append(out, v)
		}
	}
	return out
}

// encodeToSymbol performs the full encode pipeline: ASCII encodation, symbol
// selection, padding, Reed-Solomon generation and module placement.
func encodeToSymbol(text string) (*Symbol, symbolSpec, error) {
	data, err := encodeASCII(text)
	if err != nil {
		return nil, symbolSpec{}, err
	}
	spec, ok := smallestSymbolFor(len(data))
	if !ok {
		return nil, symbolSpec{}, errTooLong
	}
	data = padCodewords(data, spec.DataCW)
	full := rsEncode(data, spec.ECCW)

	sym := newSymbol(spec.Size)
	pl := buildPlacement(spec.MappingSize(), spec.MappingSize(), spec.TotalCW())
	for k, v := range full {
		for b := 0; b < 8; b++ {
			bit := (v >> (7 - b)) & 1
			pos := pl.pos[k][b]
			sym.setMapping(pos.Row, pos.Col, bit == 1)
		}
	}
	for _, f := range pl.fixed {
		sym.setMapping(f.Row, f.Col, true)
	}
	return sym, spec, nil
}

// render rasterises the symbol into a Mat according to opts.
func (s *Symbol) render(opts Options) *cv.Mat {
	opts = opts.normalized()
	mp := opts.ModulePixels
	qz := opts.QuietZoneModules
	ch := opts.Channels
	dim := (s.Size + 2*qz) * mp
	m := cv.NewMat(dim, dim, ch)
	m.SetTo(255) // white background
	for r := 0; r < s.Size; r++ {
		for c := 0; c < s.Size; c++ {
			if !s.Modules[r][c] {
				continue
			}
			y0 := (r + qz) * mp
			x0 := (c + qz) * mp
			for dy := 0; dy < mp; dy++ {
				for dx := 0; dx < mp; dx++ {
					base := ((y0+dy)*m.Cols + (x0 + dx)) * ch
					for cc := 0; cc < ch; cc++ {
						m.Data[base+cc] = 0 // black module
					}
				}
			}
		}
	}
	return m
}

// EncodeSymbol encodes text into a laid-out [Symbol] (the module grid) and
// also reports the chosen symbol specification's side length. It is the
// lower-level counterpart to [Encode] for callers that want the module grid
// rather than a bitmap.
func EncodeSymbol(text string) (*Symbol, error) {
	sym, _, err := encodeToSymbol(text)
	if err != nil {
		return nil, err
	}
	return sym, nil
}

// Encode encodes an ASCII string into an ECC200 Data Matrix bitmap using
// [DefaultOptions]. Modules are drawn black on a white background and the
// smallest supported square symbol that fits the data is chosen automatically.
func Encode(text string) (*cv.Mat, error) {
	return EncodeWithOptions(text, DefaultOptions())
}

// EncodeWithOptions is like [Encode] but lets the caller control the module
// pixel size, quiet-zone width and channel count.
func EncodeWithOptions(text string, opts Options) (*cv.Mat, error) {
	sym, _, err := encodeToSymbol(text)
	if err != nil {
		return nil, err
	}
	return sym.render(opts), nil
}

func isDigit(b byte) bool { return b >= '0' && b <= '9' }

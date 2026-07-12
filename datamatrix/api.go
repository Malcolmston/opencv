package datamatrix

import cv "github.com/malcolmston/opencv"

// StructuredAppend describes a symbol's place in a structured-append group, in
// which a long message is split across up to 16 Data Matrix symbols that a
// reader reassembles. Position and Total are 1..16 and FileID is a two-byte
// group identifier (each byte 1..254) shared by every symbol of the group.
type StructuredAppend struct {
	// Position is the 1-based index of this symbol within the group.
	Position int
	// Total is the number of symbols in the group.
	Total int
	// FileID identifies the group; every member symbol shares the same value.
	FileID [2]byte
}

// EncodeOptions controls the extended [EncodeText] encoder.
type EncodeOptions struct {
	// Scheme selects the encodation scheme; SchemeAuto minimises codewords.
	Scheme Scheme
	// Size constrains symbol-shape selection.
	Size SizePreference
	// UseECI enables emitting an Extended Channel Interpretation before the data.
	UseECI bool
	// ECI is the ECI number to emit when UseECI is set (0..999999).
	ECI int
	// GS1 emits a leading FNC1 (GS1 application-identifier data); ASCII group
	// separators (0x1D) in the input are encoded as FNC1 codewords.
	GS1 bool
	// Append, when non-nil, marks the symbol as part of a structured-append set.
	Append *StructuredAppend
	// Render controls rasterisation of the returned bitmap (see [Options]).
	Render Options
}

// DecodedResult is the outcome of decoding one symbol, including any metadata.
type DecodedResult struct {
	// Text is the decoded content interpreted as a Go string of bytes.
	Text string
	// Bytes is the raw decoded byte content.
	Bytes []byte
	// ECI is the Extended Channel Interpretation number, or -1 if none was set.
	ECI int
	// GS1 reports whether the symbol began with a GS1 FNC1 indicator.
	GS1 bool
	// Append carries structured-append information, or nil if the symbol is
	// standalone.
	Append *StructuredAppend
	// SizeName is the symbol size as "rowsxcols", e.g. "16x48".
	SizeName string
}

// EncodeText encodes text into an ECC200 Data Matrix bitmap using the supplied
// options, supporting every encodation scheme, square and rectangular symbol
// size, ECI, GS1 and structured append. The zero-value EncodeOptions selects
// automatic scheme selection, the smallest fitting square symbol and the
// default rendering.
func EncodeText(text string, opts EncodeOptions) (*cv.Mat, error) {
	sym, err := EncodeTextSymbol(text, opts)
	if err != nil {
		return nil, err
	}
	return sym.Render(opts.Render), nil
}

// EncodeTextSymbol is the lower-level counterpart to [EncodeText]: it returns
// the laid-out [EncodedSymbol] (module grid) for callers that want the modules
// rather than a bitmap.
func EncodeTextSymbol(text string, opts EncodeOptions) (*EncodedSymbol, error) {
	data, info, err := buildDataCodewords([]byte(text), opts)
	if err != nil {
		return nil, err
	}
	full := interleavedCodewords(info, data)
	grid := placeCodewords(info, full)
	return &EncodedSymbol{
		Rows:     info.symbolRows(),
		Cols:     info.symbolCols(),
		SizeName: info.name(),
		Modules:  grid,
	}, nil
}

// toResult converts an internal decoded stream into the public result type.
func toResult(s *decodedStream, sizeName string) *DecodedResult {
	return &DecodedResult{
		Text:     string(s.bytes),
		Bytes:    s.bytes,
		ECI:      s.eci,
		GS1:      s.gs1,
		Append:   s.append,
		SizeName: sizeName,
	}
}

// DecodeGrid decodes a fully-sampled module grid of any supported square or
// rectangular size, returning the content and metadata. The grid must include
// the finder/timing patterns and be one of the standard symbol dimensions. It
// runs Reed-Solomon error correction (per interleaved block) before decoding.
func DecodeGrid(modules [][]bool) (*DecodedResult, error) {
	rows := len(modules)
	if rows == 0 {
		return nil, errBadMatrix
	}
	cols := len(modules[0])
	for _, r := range modules {
		if len(r) != cols {
			return nil, errBadMatrix
		}
	}
	info, ok := symbolByDimensions(rows, cols)
	if !ok {
		return nil, errBadMatrix
	}
	data, _, err := recoverData(info, readCodewordStream(info, modules))
	if err != nil {
		return nil, err
	}
	stream, err := decodeDataCodewords(data)
	if err != nil {
		return nil, err
	}
	return toResult(stream, info.name()), nil
}

// sampleGridRC samples a rows x cols module grid at module centres inside the
// given pixel bounding box.
func sampleGridRC(m *cv.Mat, minX, minY, w, h, rows, cols int) [][]bool {
	grid := make([][]bool, rows)
	for r := 0; r < rows; r++ {
		grid[r] = make([]bool, cols)
		py := minY + int((float64(r)+0.5)*float64(h)/float64(rows))
		for c := 0; c < cols; c++ {
			px := minX + int((float64(c)+0.5)*float64(w)/float64(cols))
			grid[r][c] = isDark(m, py, px)
		}
	}
	return grid
}

// decodeRegion attempts to read and decode a symbol occupying the given dark
// bounding box, trying every supported symbol size.
func decodeRegion(m *cv.Mat, minX, minY, w, h int) (*DecodedResult, error) {
	for _, info := range symbolTable {
		nr, nc := info.symbolRows(), info.symbolCols()
		if h < nr || w < nc {
			continue
		}
		grid := sampleGridRC(m, minX, minY, w, h, nr, nc)
		if !finderMatchesLayout(grid, info) {
			continue
		}
		data, _, err := recoverData(info, readCodewordStream(info, grid))
		if err != nil {
			continue
		}
		stream, err := decodeDataCodewords(data)
		if err != nil {
			continue
		}
		return toResult(stream, info.name()), nil
	}
	return nil, errNotFound
}

// DecodeText locates a single Data Matrix symbol of any supported size in the
// image and decodes it, returning the content and metadata. The image may be
// scaled by an integer factor and surrounded by a white quiet zone, with dark
// modules on a light background (as produced by [EncodeText]).
func DecodeText(m *cv.Mat) (*DecodedResult, error) {
	if m == nil || m.Empty() {
		return nil, errNotFound
	}
	minX, minY, maxX, maxY, ok := darkBoundingBox(m)
	if !ok {
		return nil, errNotFound
	}
	return decodeRegion(m, minX, minY, maxX-minX+1, maxY-minY+1)
}

// DecodeAll locates and decodes every Data Matrix symbol in the image, returning
// one result per symbol. Symbols must be separated by white space (a quiet zone)
// so they can be isolated by a recursive X-Y cut. Results are returned in
// top-to-bottom, left-to-right order; an error is returned only when no symbol
// can be decoded at all.
func DecodeAll(m *cv.Mat) ([]*DecodedResult, error) {
	if m == nil || m.Empty() {
		return nil, errNoSymbols
	}
	var out []*DecodedResult
	for _, rc := range segmentSymbols(m) {
		res, err := decodeRegion(m, rc.minX, rc.minY, rc.maxX-rc.minX+1, rc.maxY-rc.minY+1)
		if err == nil {
			out = append(out, res)
		}
	}
	if len(out) == 0 {
		return nil, errNoSymbols
	}
	return out, nil
}

// rect is an inclusive pixel bounding box.
type rect struct{ minX, minY, maxX, maxY int }

// rowHasDark reports whether row y has any dark pixel within [x0, x1).
func rowHasDark(m *cv.Mat, y, x0, x1 int) bool {
	for x := x0; x < x1; x++ {
		if isDark(m, y, x) {
			return true
		}
	}
	return false
}

// colHasDark reports whether column x has any dark pixel within [y0, y1).
func colHasDark(m *cv.Mat, x, y0, y1 int) bool {
	for y := y0; y < y1; y++ {
		if isDark(m, y, x) {
			return true
		}
	}
	return false
}

// darkBox returns the tight dark bounding box within the given region, or ok
// false when the region has no dark pixels.
func darkBox(m *cv.Mat, r rect) (rect, bool) {
	out := rect{minX: r.maxX, minY: r.maxY, maxX: r.minX - 1, maxY: r.minY - 1}
	found := false
	for y := r.minY; y <= r.maxY; y++ {
		for x := r.minX; x <= r.maxX; x++ {
			if isDark(m, y, x) {
				found = true
				if x < out.minX {
					out.minX = x
				}
				if x > out.maxX {
					out.maxX = x
				}
				if y < out.minY {
					out.minY = y
				}
				if y > out.maxY {
					out.maxY = y
				}
			}
		}
	}
	return out, found
}

// segmentSymbols partitions the image into candidate symbol bounding boxes with
// a recursive X-Y cut: it splits on fully white rows, then fully white columns,
// alternating until neither axis splits, and returns the tight dark box of each
// leaf region.
func segmentSymbols(m *cv.Mat) []rect {
	var out []rect
	var stack []rect
	stack = append(stack, rect{0, 0, m.Cols - 1, m.Rows - 1})
	for len(stack) > 0 {
		r := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		box, ok := darkBox(m, r)
		if !ok {
			continue
		}
		// Split the tight box into horizontal bands separated by white rows.
		bands := splitRows(m, box)
		if len(bands) > 1 {
			stack = append(stack, bands...)
			continue
		}
		cols := splitCols(m, box)
		if len(cols) > 1 {
			stack = append(stack, cols...)
			continue
		}
		out = append(out, box)
	}
	return out
}

// splitRows returns the maximal horizontal bands of r that contain dark pixels,
// separated by at least one fully white row.
func splitRows(m *cv.Mat, r rect) []rect {
	var bands []rect
	y := r.minY
	for y <= r.maxY {
		if !rowHasDark(m, y, r.minX, r.maxX+1) {
			y++
			continue
		}
		start := y
		for y <= r.maxY && rowHasDark(m, y, r.minX, r.maxX+1) {
			y++
		}
		bands = append(bands, rect{r.minX, start, r.maxX, y - 1})
	}
	return bands
}

// splitCols returns the maximal vertical bands of r that contain dark pixels,
// separated by at least one fully white column.
func splitCols(m *cv.Mat, r rect) []rect {
	var bands []rect
	x := r.minX
	for x <= r.maxX {
		if !colHasDark(m, x, r.minY, r.maxY+1) {
			x++
			continue
		}
		start := x
		for x <= r.maxX && colHasDark(m, x, r.minY, r.maxY+1) {
			x++
		}
		bands = append(bands, rect{start, r.minY, x - 1, r.maxY})
	}
	return bands
}

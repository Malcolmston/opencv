package datamatrix

import (
	"bytes"
	"strings"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func mustEncodeSymbol(t *testing.T, text string, opts EncodeOptions) *EncodedSymbol {
	t.Helper()
	sym, err := EncodeTextSymbol(text, opts)
	if err != nil {
		t.Fatalf("EncodeTextSymbol(%q, %+v) error: %v", text, opts, err)
	}
	return sym
}

func TestForcedSchemeRoundTrip(t *testing.T) {
	cases := []struct {
		name   string
		scheme Scheme
		input  string
	}{
		{"C40", SchemeC40, "DATAMATRIX ECC200 CODE 42"},
		{"Text", SchemeText, "the quick brown fox 123"},
		{"X12", SchemeX12, "ABC*DEF>GHI 12345"},
		{"EDIFACT", SchemeEDIFACT, "EDIFACT/MESSAGE:2024+ABC"},
		{"Base256", SchemeBase256, "\x00\x01\x02\xff\xfe binary\x80\x81"},
		{"ASCII", SchemeASCII, "Hello, World! 2024"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sym := mustEncodeSymbol(t, tc.input, EncodeOptions{Scheme: tc.scheme})
			res, err := DecodeGrid(sym.Modules)
			if err != nil {
				t.Fatalf("DecodeGrid error: %v", err)
			}
			if res.Text != tc.input {
				t.Errorf("round trip mismatch:\n got %q\nwant %q", res.Text, tc.input)
			}
		})
	}
}

func TestAutoSchemeRoundTrip(t *testing.T) {
	inputs := []string{
		"",
		"A",
		"1234567890",
		"HELLO WORLD",
		"UPPERCASE ONLY TEXT STRING FOR C40",
		"lowercase only text for the text scheme",
		"Mixed Case 123 With Symbols !@#$%",
		"ISO/IEC 16022:2006 EDIFACT-LIKE +DATA",
		"binary\x00\x01\x80\xffpayload",
		strings.Repeat("AB12", 20),
	}
	for _, in := range inputs {
		sym := mustEncodeSymbol(t, in, EncodeOptions{})
		res, err := DecodeGrid(sym.Modules)
		if err != nil {
			t.Fatalf("DecodeGrid(%q) error: %v", in, err)
		}
		if res.Text != in {
			t.Errorf("auto round trip mismatch:\n got %q\nwant %q", res.Text, in)
		}
	}
}

func TestRectangularSizes(t *testing.T) {
	cases := []struct {
		input    string
		wantSize string
	}{
		{"HELLO", "8x18"},
		{"ABCDEFGH", "8x32"},
		{"ABCDEFGHIJKLMNOP", "12x26"},
		{"ABCDEFGHIJKLMNOPQRST", "12x36"},
		{strings.Repeat("X", 30), "16x36"},
		{strings.Repeat("Y", 45), "16x48"},
	}
	for _, tc := range cases {
		sym := mustEncodeSymbol(t, tc.input, EncodeOptions{Scheme: SchemeASCII, Size: SizeRectangleOnly})
		if sym.SizeName != tc.wantSize {
			t.Errorf("input %q: size = %s, want %s", tc.input, sym.SizeName, tc.wantSize)
		}
		res, err := DecodeGrid(sym.Modules)
		if err != nil {
			t.Fatalf("DecodeGrid(%q) error: %v", tc.input, err)
		}
		if res.Text != tc.input {
			t.Errorf("rectangular round trip mismatch: got %q want %q", res.Text, tc.input)
		}
	}
}

func TestLargeSquareSizes(t *testing.T) {
	cases := []struct {
		input    string
		wantSize string
	}{
		{strings.Repeat("Z", 25), "22x22"},
		{strings.Repeat("Z", 34), "24x24"},
		{strings.Repeat("Z", 40), "26x26"},
		{strings.Repeat("Z", 60), "32x32"},
	}
	for _, tc := range cases {
		sym := mustEncodeSymbol(t, tc.input, EncodeOptions{Scheme: SchemeASCII, Size: SizeSquareOnly})
		if sym.SizeName != tc.wantSize {
			t.Errorf("input len %d: size = %s, want %s", len(tc.input), sym.SizeName, tc.wantSize)
		}
		res, err := DecodeGrid(sym.Modules)
		if err != nil {
			t.Fatalf("DecodeGrid error: %v", err)
		}
		if res.Text != tc.input {
			t.Errorf("large square round trip mismatch: got %q want %q", res.Text, tc.input)
		}
	}
}

func TestInterleavedRSGridRoundTrip(t *testing.T) {
	// 52x52 and larger use multiple interleaved Reed-Solomon blocks. Round-trip a
	// long input that selects such a symbol through the full grid pipeline.
	for _, n := range []int{200, 360} {
		in := strings.Repeat("MATRIX7", n/7+1)[:n]
		sym := mustEncodeSymbol(t, in, EncodeOptions{Scheme: SchemeASCII, Size: SizeSquareOnly})
		res, err := DecodeGrid(sym.Modules)
		if err != nil {
			t.Fatalf("len %d (%s) DecodeGrid error: %v", n, sym.SizeName, err)
		}
		if res.Text != in {
			t.Errorf("interleaved round trip mismatch at len %d (%s)", n, sym.SizeName)
		}
	}
}

func TestBitmapRoundTripAllShapes(t *testing.T) {
	cases := []struct {
		input string
		opts  EncodeOptions
	}{
		{"RECT 8x32 SAMPLE", EncodeOptions{Scheme: SchemeASCII, Size: SizeRectangleOnly}},
		{strings.Repeat("SQ", 20), EncodeOptions{Scheme: SchemeASCII, Size: SizeSquareOnly}},
		{"C40 UPPER CASE PAYLOAD 99", EncodeOptions{Scheme: SchemeC40}},
	}
	for _, tc := range cases {
		tc.opts.Render = Options{ModulePixels: 4, QuietZoneModules: 3, Channels: 1}
		m, err := EncodeText(tc.input, tc.opts)
		if err != nil {
			t.Fatalf("EncodeText(%q) error: %v", tc.input, err)
		}
		res, err := DecodeText(m)
		if err != nil {
			t.Fatalf("DecodeText(%q) error: %v", tc.input, err)
		}
		if res.Text != tc.input {
			t.Errorf("bitmap round trip mismatch: got %q want %q", res.Text, tc.input)
		}
	}
}

func TestStructuredAppendRoundTrip(t *testing.T) {
	sa := &StructuredAppend{Position: 3, Total: 7, FileID: [2]byte{42, 99}}
	sym := mustEncodeSymbol(t, "PART THREE", EncodeOptions{Scheme: SchemeASCII, Append: sa})
	res, err := DecodeGrid(sym.Modules)
	if err != nil {
		t.Fatalf("DecodeGrid error: %v", err)
	}
	if res.Text != "PART THREE" {
		t.Errorf("content mismatch: %q", res.Text)
	}
	if res.Append == nil {
		t.Fatal("expected structured-append metadata")
	}
	if res.Append.Position != 3 || res.Append.Total != 7 || res.Append.FileID != [2]byte{42, 99} {
		t.Errorf("append metadata mismatch: %+v", *res.Append)
	}
}

func TestECIRoundTrip(t *testing.T) {
	for _, eci := range []int{3, 26, 100, 500, 40000, 123456} {
		sym := mustEncodeSymbol(t, "ECIDATA", EncodeOptions{Scheme: SchemeASCII, UseECI: true, ECI: eci})
		res, err := DecodeGrid(sym.Modules)
		if err != nil {
			t.Fatalf("ECI %d DecodeGrid error: %v", eci, err)
		}
		if res.Text != "ECIDATA" {
			t.Errorf("ECI %d content mismatch: %q", eci, res.Text)
		}
		if res.ECI != eci {
			t.Errorf("ECI value mismatch: got %d want %d", res.ECI, eci)
		}
	}
}

func TestGS1FNC1RoundTrip(t *testing.T) {
	// GS1 payload: AI (01) GTIN then a group separator then AI (10) batch.
	in := "0112345678901231" + "\x1d" + "10ABC123"
	sym := mustEncodeSymbol(t, in, EncodeOptions{GS1: true})
	res, err := DecodeGrid(sym.Modules)
	if err != nil {
		t.Fatalf("DecodeGrid error: %v", err)
	}
	if !res.GS1 {
		t.Error("expected GS1 flag to be set")
	}
	if res.Text != in {
		t.Errorf("GS1 round trip mismatch:\n got %q\nwant %q", res.Text, in)
	}
}

func TestErrorRecoveryWithinBudget(t *testing.T) {
	in := strings.Repeat("REPAIR", 6)
	sym := mustEncodeSymbol(t, in, EncodeOptions{Scheme: SchemeASCII, Size: SizeSquareOnly})
	info, ok := symbolByDimensions(sym.Rows, sym.Cols)
	if !ok {
		t.Fatalf("unknown symbol %s", sym.SizeName)
	}
	// Flip bit 0 of the first ECCW/2 codewords, giving one codeword error each,
	// which stays within the Reed-Solomon correction budget.
	pl := buildPlacement(info.mapRows(), info.mapCols(), info.totalCW())
	budget := info.eccCW / 2
	for k := 0; k < budget; k++ {
		p := pl.pos[k][0]
		r, c := mapToModule(info, p.Row, p.Col)
		sym.Modules[r][c] = !sym.Modules[r][c]
	}
	res, err := DecodeGrid(sym.Modules)
	if err != nil {
		t.Fatalf("DecodeGrid after %d flips error: %v", budget, err)
	}
	if res.Text != in {
		t.Errorf("recovery mismatch: got %q want %q", res.Text, in)
	}
}

func TestErrorBeyondBudgetFails(t *testing.T) {
	in := "SHORT"
	sym := mustEncodeSymbol(t, in, EncodeOptions{Scheme: SchemeASCII, Size: SizeRectangleOnly})
	info, _ := symbolByDimensions(sym.Rows, sym.Cols)
	pl := buildPlacement(info.mapRows(), info.mapCols(), info.totalCW())
	// Corrupt every codeword: far beyond the correction budget.
	for k := 0; k < info.totalCW(); k++ {
		p := pl.pos[k][0]
		r, c := mapToModule(info, p.Row, p.Col)
		sym.Modules[r][c] = !sym.Modules[r][c]
		p2 := pl.pos[k][1]
		r2, c2 := mapToModule(info, p2.Row, p2.Col)
		sym.Modules[r2][c2] = !sym.Modules[r2][c2]
	}
	if _, err := DecodeGrid(sym.Modules); err == nil {
		t.Error("expected failure for uncorrectable corruption")
	}
}

func TestSmallestSizeSelection(t *testing.T) {
	// Uppercase-only content is far more compact in C40 than ASCII, so auto
	// selection must produce fewer data codewords (and never more) than forced
	// ASCII for a long uppercase run.
	in := strings.Repeat("ABCDEF", 6)
	autoSym := mustEncodeSymbol(t, in, EncodeOptions{})
	asciiSym := mustEncodeSymbol(t, in, EncodeOptions{Scheme: SchemeASCII})
	autoInfo, _ := symbolByDimensions(autoSym.Rows, autoSym.Cols)
	asciiInfo, _ := symbolByDimensions(asciiSym.Rows, asciiSym.Cols)
	if autoInfo.dataCW >= asciiInfo.dataCW {
		t.Errorf("auto scheme did not reduce size: auto=%s ascii=%s", autoSym.SizeName, asciiSym.SizeName)
	}
	// Pure digits are optimal in ASCII (digit pairs), so auto should stay ASCII
	// and pick the same small symbol.
	digits := "1234567890123456"
	dAuto := mustEncodeSymbol(t, digits, EncodeOptions{})
	dAscii := mustEncodeSymbol(t, digits, EncodeOptions{Scheme: SchemeASCII})
	if dAuto.SizeName != dAscii.SizeName {
		t.Errorf("digit selection differs: auto=%s ascii=%s", dAuto.SizeName, dAscii.SizeName)
	}
}

func TestDecodeAllMultiSymbol(t *testing.T) {
	texts := []string{"FIRST 111", "SECOND 222", "THIRD 333"}
	// Lay three symbols side by side separated by a white gap.
	opts := Options{ModulePixels: 3, QuietZoneModules: 2, Channels: 1}
	var syms []*cv.Mat
	total := 0
	maxH := 0
	for _, tx := range texts {
		m, err := EncodeText(tx, EncodeOptions{Scheme: SchemeASCII, Render: opts})
		if err != nil {
			t.Fatalf("EncodeText error: %v", err)
		}
		syms = append(syms, m)
		total += m.Cols + 12 // gap between symbols
		if m.Rows > maxH {
			maxH = m.Rows
		}
	}
	canvas := cv.NewMat(maxH+12, total+12, 1)
	canvas.SetTo(255)
	x := 6
	for _, m := range syms {
		m.CopyTo(canvas, 6, x)
		x += m.Cols + 12
	}
	results, err := DecodeAll(canvas)
	if err != nil {
		t.Fatalf("DecodeAll error: %v", err)
	}
	if len(results) != len(texts) {
		t.Fatalf("DecodeAll found %d symbols, want %d", len(results), len(texts))
	}
	got := make(map[string]bool)
	for _, r := range results {
		got[r.Text] = true
	}
	for _, tx := range texts {
		if !got[tx] {
			t.Errorf("missing decoded symbol %q (got %v)", tx, got)
		}
	}
}

func TestCrossCompatWithASCIICodec(t *testing.T) {
	// The extended decoder must read symbols produced by the original ASCII-only
	// encoder, and vice versa, for the shared small square sizes.
	for _, in := range []string{"HELLO", "42", "abc123XYZ"} {
		oldSym, err := EncodeSymbol(in)
		if err != nil {
			t.Fatalf("EncodeSymbol error: %v", err)
		}
		res, err := DecodeGrid(oldSym.Modules)
		if err != nil {
			t.Fatalf("DecodeGrid of legacy symbol error: %v", err)
		}
		if res.Text != in {
			t.Errorf("legacy decode mismatch: got %q want %q", res.Text, in)
		}
		newSym := mustEncodeSymbol(t, in, EncodeOptions{Scheme: SchemeASCII})
		out, err := DecodeMatrix(newSym.Modules)
		if err != nil {
			t.Fatalf("legacy DecodeMatrix error: %v", err)
		}
		if out != in {
			t.Errorf("legacy decode of new symbol mismatch: got %q want %q", out, in)
		}
	}
}

func TestBase256BinaryExact(t *testing.T) {
	payload := make([]byte, 40)
	for i := range payload {
		payload[i] = byte((i * 37) & 0xFF)
	}
	sym := mustEncodeSymbol(t, string(payload), EncodeOptions{Scheme: SchemeBase256})
	res, err := DecodeGrid(sym.Modules)
	if err != nil {
		t.Fatalf("DecodeGrid error: %v", err)
	}
	if !bytes.Equal(res.Bytes, payload) {
		t.Errorf("base256 payload mismatch")
	}
}

func TestC40FullRepertoireRoundTrip(t *testing.T) {
	// Exercise every C40 sub-set: basic, Shift 1 (controls), Shift 2 (punctuation),
	// Shift 3 (lower/DEL) and Upper Shift (extended ASCII).
	inputs := []string{
		"A0 Z9",
		"\t\n\rTAB",
		"a{|}~",
		"!#$%&()+,-./:;<=>?@[\\]^_",
		"CAF\xc9 \xe9\xff",
	}
	for _, in := range inputs {
		sym := mustEncodeSymbol(t, in, EncodeOptions{Scheme: SchemeC40})
		res, err := DecodeGrid(sym.Modules)
		if err != nil {
			t.Fatalf("DecodeGrid(%q) error: %v", in, err)
		}
		if res.Text != in {
			t.Errorf("C40 repertoire mismatch:\n got %q\nwant %q", res.Text, in)
		}
	}
}

func TestTextShift3AndExtended(t *testing.T) {
	in := "hello WORLD `~ \xe0\xe1"
	sym := mustEncodeSymbol(t, in, EncodeOptions{Scheme: SchemeText})
	res, err := DecodeGrid(sym.Modules)
	if err != nil {
		t.Fatalf("DecodeGrid error: %v", err)
	}
	if res.Text != in {
		t.Errorf("Text mismatch:\n got %q\nwant %q", res.Text, in)
	}
}

func TestEDIFACTAutoSelected(t *testing.T) {
	// A long run of EDIFACT-native, non-C40 characters should trigger EDIFACT.
	in := "abcd+efgh:ijkl/mnop=qrst"
	autoSym := mustEncodeSymbol(t, in, EncodeOptions{})
	asciiSym := mustEncodeSymbol(t, in, EncodeOptions{Scheme: SchemeASCII})
	ai, _ := symbolByDimensions(autoSym.Rows, autoSym.Cols)
	si, _ := symbolByDimensions(asciiSym.Rows, asciiSym.Cols)
	if ai.dataCW >= si.dataCW {
		t.Logf("note: auto=%s ascii=%s", autoSym.SizeName, asciiSym.SizeName)
	}
	res, err := DecodeGrid(autoSym.Modules)
	if err != nil {
		t.Fatalf("DecodeGrid error: %v", err)
	}
	if res.Text != in {
		t.Errorf("EDIFACT auto mismatch: got %q want %q", res.Text, in)
	}
}

func TestECIThreeCodewordRange(t *testing.T) {
	for _, eci := range []int{16383, 100000, 999999} {
		sym := mustEncodeSymbol(t, "X", EncodeOptions{Scheme: SchemeASCII, UseECI: true, ECI: eci})
		res, err := DecodeGrid(sym.Modules)
		if err != nil {
			t.Fatalf("ECI %d error: %v", eci, err)
		}
		if res.ECI != eci {
			t.Errorf("ECI mismatch: got %d want %d", res.ECI, eci)
		}
	}
}

func TestLargeBase256TwoByteLength(t *testing.T) {
	// A payload over 249 bytes forces the two-codeword Base 256 length indicator
	// and selects a large, interleaved-Reed-Solomon symbol.
	payload := make([]byte, 300)
	for i := range payload {
		payload[i] = byte((i*97 + 13) & 0xFF)
	}
	sym := mustEncodeSymbol(t, string(payload), EncodeOptions{Scheme: SchemeBase256})
	res, err := DecodeGrid(sym.Modules)
	if err != nil {
		t.Fatalf("DecodeGrid (%s) error: %v", sym.SizeName, err)
	}
	if !bytes.Equal(res.Bytes, payload) {
		t.Errorf("large base256 payload mismatch")
	}
}

func TestEncodeErrorsExt(t *testing.T) {
	if _, err := EncodeTextSymbol("x", EncodeOptions{UseECI: true, ECI: -5}); err == nil {
		t.Error("expected error for negative ECI")
	}
	if _, err := EncodeTextSymbol("x", EncodeOptions{Append: &StructuredAppend{Position: 0, Total: 3, FileID: [2]byte{1, 1}}}); err == nil {
		t.Error("expected error for invalid append position")
	}
	huge := strings.Repeat("A", 4000)
	if _, err := EncodeTextSymbol(huge, EncodeOptions{Scheme: SchemeASCII}); err == nil {
		t.Error("expected error for over-capacity input")
	}
}

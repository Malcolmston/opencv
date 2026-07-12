package datamatrix

import (
	"fmt"
	"reflect"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// roundTripInputs exercises digit packing, mixed ASCII and short/long strings
// across several symbol sizes.
var roundTripInputs = []string{
	"",
	"A",
	"12",
	"HELLO",
	"1234567890",
	"Data Matrix!",
	"abc123XYZ",
	"9876543210987654321",
	"Test 42 mixed 07 content",
}

func TestGeneratorPolynomialMatchesECC200(t *testing.T) {
	// The 10x10 symbol uses five error-correction codewords whose generator
	// polynomial is the well-known {228, 48, 15, 111, 62}. A match confirms the
	// GF(256) field parameters and first-root exponent are the ECC200 values.
	// rsGeneratorPoly stores coefficients highest-degree first with an implicit
	// leading 1; the ECC200 table lists them constant-term first without the
	// leading coefficient. Dropping the leading 1 and reversing must match.
	got := rsGeneratorPoly(5)
	body := reverseInts(got[1:])
	want := []int{228, 48, 15, 111, 62}
	if !reflect.DeepEqual(body, want) {
		t.Fatalf("generator body = %v, want %v (full poly %v)", body, want, got)
	}
}

func TestEncodeDecodeMatrixRoundTrip(t *testing.T) {
	for _, in := range roundTripInputs {
		sym, err := EncodeSymbol(in)
		if err != nil {
			t.Fatalf("EncodeSymbol(%q) error: %v", in, err)
		}
		out, err := DecodeMatrix(sym.Modules)
		if err != nil {
			t.Fatalf("DecodeMatrix(%q) error: %v", in, err)
		}
		if out != in {
			t.Errorf("round trip mismatch: got %q, want %q", out, in)
		}
	}
}

func TestEncodeDetectAndDecodeRoundTrip(t *testing.T) {
	for _, in := range roundTripInputs {
		m, err := Encode(in)
		if err != nil {
			t.Fatalf("Encode(%q) error: %v", in, err)
		}
		out, err := DetectAndDecode(m)
		if err != nil {
			t.Fatalf("DetectAndDecode(%q) error: %v", in, err)
		}
		if out != in {
			t.Errorf("bitmap round trip mismatch: got %q, want %q", out, in)
		}
	}
}

func TestDetectAndDecodeScaledAndQuietZoned(t *testing.T) {
	for _, opts := range []Options{
		{ModulePixels: 1, QuietZoneModules: 0, Channels: 1},
		{ModulePixels: 3, QuietZoneModules: 4, Channels: 1},
		{ModulePixels: 10, QuietZoneModules: 2, Channels: 3},
	} {
		in := "SCALE 99 test"
		m, err := EncodeWithOptions(in, opts)
		if err != nil {
			t.Fatalf("EncodeWithOptions error: %v", err)
		}
		out, err := DetectAndDecode(m)
		if err != nil {
			t.Fatalf("DetectAndDecode (opts %+v) error: %v", opts, err)
		}
		if out != in {
			t.Errorf("opts %+v: got %q, want %q", opts, out, in)
		}
	}
}

func TestReedSolomonRecoversFlippedModules(t *testing.T) {
	in := "HELLO"
	sym, err := EncodeSymbol(in)
	if err != nil {
		t.Fatalf("EncodeSymbol error: %v", err)
	}
	spec, _ := symbolBySize(sym.Size)
	// The symbol can correct up to ECCW/2 codeword errors. Flip a handful of
	// interior data modules (leaving the finder pattern intact) within that
	// budget and confirm the string is still recovered.
	budget := spec.ECCW / 2
	flips := 0
	inner := spec.MappingSize()
	// Flip modules spread across distinct codewords by stepping through the
	// interior on a stride.
	for r := 0; r < inner && flips < budget; r += 3 {
		for c := 0; c < inner && flips < budget; c += 5 {
			sym.setMapping(r, c, !sym.getMapping(r, c))
			flips++
		}
	}
	if flips == 0 {
		t.Fatal("expected to flip at least one module")
	}
	out, err := DecodeMatrix(sym.Modules)
	if err != nil {
		t.Fatalf("DecodeMatrix after %d flips error: %v", flips, err)
	}
	if out != in {
		t.Errorf("after %d flips: got %q, want %q", flips, out, in)
	}
}

func TestReedSolomonSingleBitErrorEachSize(t *testing.T) {
	// A single flipped data module must be correctable in every supported size.
	inputs := map[int]string{
		10: "AB",
		12: "TEST",
		14: "ABCDEFG",
		16: "MixedCase12",
		18: "eighteen size ok",
		20: "twenty by twenty ok!!",
	}
	for size, in := range inputs {
		sym, err := EncodeSymbol(in)
		if err != nil {
			t.Fatalf("EncodeSymbol(%q) error: %v", in, err)
		}
		if sym.Size != size {
			t.Fatalf("input %q selected size %d, want %d", in, sym.Size, size)
		}
		sym.setMapping(0, 0, !sym.getMapping(0, 0))
		out, err := DecodeMatrix(sym.Modules)
		if err != nil {
			t.Fatalf("size %d: DecodeMatrix error: %v", size, err)
		}
		if out != in {
			t.Errorf("size %d: got %q, want %q", size, out, in)
		}
	}
}

func TestSmallestSymbolAutoSelection(t *testing.T) {
	cases := []struct {
		in       string
		wantSize int
	}{
		{"A", 10},         // 1 data codeword -> 10x10
		{"AB", 10},        // 2 codewords -> still 10x10
		{"ABC", 10},       // exactly 3 -> 10x10
		{"ABCD", 12},      // 4 codewords -> 12x12
		{"ABCDE", 12},     // 5 -> 12x12
		{"ABCDEF", 14},    // 6 -> 14x14
		{"ABCDEFGH", 14},  // 8 -> 14x14
		{"ABCDEFGHI", 16}, // 9 -> 16x16
		{"123456", 10},    // three digit-pairs -> 3 codewords -> 10x10
		{"12345678", 12},  // four pairs -> 4 codewords -> 12x12
	}
	for _, tc := range cases {
		sym, err := EncodeSymbol(tc.in)
		if err != nil {
			t.Fatalf("EncodeSymbol(%q) error: %v", tc.in, err)
		}
		if sym.Size != tc.wantSize {
			t.Errorf("EncodeSymbol(%q) size = %d, want %d", tc.in, sym.Size, tc.wantSize)
		}
	}
}

func TestSymbolSizeGrowsWithLength(t *testing.T) {
	prev := 0
	for _, in := range []string{"A", "ABCD", "ABCDEF", "ABCDEFGHI", "ABCDEFGHIJKLMNOP", "ABCDEFGHIJKLMNOPQRST"} {
		sym, err := EncodeSymbol(in)
		if err != nil {
			t.Fatalf("EncodeSymbol(%q) error: %v", in, err)
		}
		if sym.Size < prev {
			t.Errorf("size decreased for longer input %q: %d < %d", in, sym.Size, prev)
		}
		prev = sym.Size
	}
}

func TestEncodeErrors(t *testing.T) {
	if _, err := Encode("café"); err == nil {
		t.Error("expected error for non-ASCII input")
	}
	// Longer than the largest symbol (22 ASCII codewords) can hold.
	if _, err := Encode("this string is definitely far too long to fit"); err == nil {
		t.Error("expected error for over-capacity input")
	}
}

func TestDecodeMatrixValidation(t *testing.T) {
	if _, err := DecodeMatrix(nil); err == nil {
		t.Error("expected error for nil matrix")
	}
	if _, err := DecodeMatrix([][]bool{{true, false}, {false, true}}); err == nil {
		t.Error("expected error for unsupported size")
	}
	jagged := [][]bool{make([]bool, 10), make([]bool, 9)}
	if _, err := DecodeMatrix(jagged); err == nil {
		t.Error("expected error for jagged matrix")
	}
}

func TestDetectAndDecodeNoSymbol(t *testing.T) {
	blank := cv.NewMat(40, 40, 1)
	blank.SetTo(255)
	if _, err := DetectAndDecode(blank); err == nil {
		t.Error("expected error for image with no symbol")
	}
	if _, err := DetectAndDecode(nil); err == nil {
		t.Error("expected error for nil image")
	}
}

func TestTooManyErrorsFail(t *testing.T) {
	sym, err := EncodeSymbol("AB")
	if err != nil {
		t.Fatalf("EncodeSymbol error: %v", err)
	}
	// 10x10 corrects at most 2 codeword errors; corrupt every interior module.
	inner := sym.Size - 2
	for r := 0; r < inner; r++ {
		for c := 0; c < inner; c++ {
			sym.setMapping(r, c, !sym.getMapping(r, c))
		}
	}
	if _, err := DecodeMatrix(sym.Modules); err == nil {
		t.Error("expected decode failure for uncorrectable corruption")
	}
}

func TestReedSolomonEncodeDecodeUnit(t *testing.T) {
	data := []int{66, 67, 68} // arbitrary data codewords
	full := rsEncode(data, 5)
	if len(full) != 8 {
		t.Fatalf("rsEncode length = %d, want 8", len(full))
	}
	// Inject two errors and recover.
	full[1] ^= 0x5a
	full[6] ^= 0x13
	corrected, n, err := rsCorrect(full, 5)
	if err != nil {
		t.Fatalf("rsCorrect error: %v", err)
	}
	if n != 2 {
		t.Errorf("corrected %d errors, want 2", n)
	}
	if !reflect.DeepEqual(corrected[:3], data) {
		t.Errorf("corrected data = %v, want %v", corrected[:3], data)
	}
}

func ExampleEncode() {
	m, err := Encode("HELLO")
	if err != nil {
		panic(err)
	}
	text, err := DetectAndDecode(m)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%dx%d bitmap decodes to %q\n", m.Rows, m.Cols, text)
	// Output: 128x128 bitmap decodes to "HELLO"
}

func ExampleDecodeMatrix() {
	sym, _ := EncodeSymbol("42")
	fmt.Printf("symbol is %dx%d modules\n", sym.Size, sym.Size)
	text, _ := DecodeMatrix(sym.Modules)
	fmt.Printf("decoded: %q\n", text)
	// Output:
	// symbol is 10x10 modules
	// decoded: "42"
}

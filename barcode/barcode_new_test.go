package barcode

import (
	"strings"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- Advanced QR ----------------------------------------------------------

func TestQRAdvancedAllVersionsAndLevels(t *testing.T) {
	levels := []QRECCLevel{QRECCLow, QRECCMedium, QRECCQuartile, QRECCHigh}
	for v := 1; v <= 10; v++ {
		for _, lvl := range levels {
			// Byte payload sized to a fraction of the version/level capacity.
			capBytes := QRDataCapacity(v, lvl) - 3
			if capBytes < 1 {
				continue
			}
			n := capBytes / 2
			if n < 1 {
				n = 1
			}
			if n > 40 {
				n = 40
			}
			text := strings.Repeat("Zx9", n)
			if len(text) > capBytes {
				text = text[:capBytes]
			}
			img, err := QREncodeAdvanced(text, v, lvl)
			if err != nil {
				t.Fatalf("v%d lvl%d encode: %v", v, lvl, err)
			}
			got, ok := QRDetectAndDecodeAdvanced(img)
			if !ok || got != text {
				t.Errorf("v%d lvl%d: ok=%v match=%v (len %d)", v, lvl, ok, got == text, len(text))
			}
		}
	}
}

func TestQRAdvancedNumericMode(t *testing.T) {
	for _, s := range []string{"0", "42", "12345", "9876543210", strings.Repeat("1234567890", 5)} {
		img, err := QREncodeAdvanced(s, 0, QRECCMedium)
		if err != nil {
			t.Fatalf("encode %q: %v", s, err)
		}
		got, ok := QRDetectAndDecodeAdvanced(img)
		if !ok || got != s {
			t.Errorf("numeric %q: got %q ok=%v", s, got, ok)
		}
	}
}

func TestQRAdvancedAlphanumericMode(t *testing.T) {
	for _, s := range []string{"HELLO WORLD", "ABC123", "HTTP://EXAMPLE.COM", "A", "PRICE $12.50/EA"} {
		img, err := QREncodeAdvanced(s, 0, QRECCQuartile)
		if err != nil {
			t.Fatalf("encode %q: %v", s, err)
		}
		got, ok := QRDetectAndDecodeAdvanced(img)
		if !ok || got != s {
			t.Errorf("alnum %q: got %q ok=%v", s, got, ok)
		}
	}
}

func TestQRAdvancedByteMode(t *testing.T) {
	for _, s := range []string{"Hello, World!", "lower+UPPER=mix", "camelCase text", "a~b|c"} {
		img, err := QREncodeAdvanced(s, 0, QRECCHigh)
		if err != nil {
			t.Fatalf("encode %q: %v", s, err)
		}
		got, ok := QRDetectAndDecodeAdvanced(img)
		if !ok || got != s {
			t.Errorf("byte %q: got %q ok=%v", s, got, ok)
		}
	}
}

func TestQRAdvancedModeSelection(t *testing.T) {
	if selectMode("12345") != modeNumeric {
		t.Error("digits should select numeric mode")
	}
	if selectMode("ABC 123") != modeAlpha {
		t.Error("upper alnum should select alphanumeric mode")
	}
	if selectMode("abc") != modeByteAdv {
		t.Error("lowercase should select byte mode")
	}
}

func TestQRAdvancedModeCompactness(t *testing.T) {
	// A long numeric string that fits a small version in numeric mode would not
	// fit the same version in byte mode; check numeric mode is materially denser.
	digits := strings.Repeat("1234567890", 6) // 60 digits
	nv, _ := selectVersionEx(digits, 0, QRECCLow)
	// The same length as arbitrary bytes needs a larger version.
	bytes := strings.Repeat("aB6", 20) // 60 bytes, byte mode
	bv, _ := selectVersionEx(bytes, 0, QRECCLow)
	if nv >= bv {
		t.Errorf("numeric mode (v%d) should be denser than byte mode (v%d)", nv, bv)
	}
}

func TestQRAdvancedErrorCorrection(t *testing.T) {
	// Level H over a version 3 symbol corrects many modules; flip several data
	// modules and require exact recovery.
	text := "ERRORCORRECTION 99"
	img, err := QREncodeAdvanced(text, 3, QRECCHigh)
	if err != nil {
		t.Fatal(err)
	}
	flips := [][2]int{{9, 9}, {11, 12}, {13, 10}, {15, 14}, {17, 11}}
	for _, f := range flips {
		flipModule(img, f[0], f[1])
	}
	got, ok := QRDetectAndDecodeAdvanced(img)
	if !ok || got != text {
		t.Errorf("error-corrected advanced decode: ok=%v got=%q", ok, got)
	}
}

func TestQRAdvancedRotation(t *testing.T) {
	text := "ROTATE ME 2026"
	base, err := QREncodeAdvanced(text, 7, QRECCMedium)
	if err != nil {
		t.Fatal(err)
	}
	for _, code := range []cv.RotateCode{cv.Rotate90CW, cv.Rotate180, cv.Rotate90CCW} {
		img := cv.Rotate(base, code)
		got, ok := QRDetectAndDecodeAdvanced(img)
		if !ok || got != text {
			t.Errorf("rotate %v: ok=%v got=%q", code, ok, got)
		}
	}
}

func TestQRAdvancedCapacityAndErrors(t *testing.T) {
	if QRDataCapacity(5, QRECCLow) != 108 {
		t.Errorf("v5-L data capacity = %d, want 108", QRDataCapacity(5, QRECCLow))
	}
	if QRDataCapacity(10, QRECCHigh) != 122 {
		t.Errorf("v10-H data capacity = %d, want 122", QRDataCapacity(10, QRECCHigh))
	}
	if QRDataCapacity(11, QRECCLow) != 0 {
		t.Error("unsupported version should have 0 capacity")
	}
	if _, err := QREncodeAdvanced(strings.Repeat("x", 5000), 0, QRECCLow); err == nil {
		t.Error("oversized payload should error")
	}
	if _, err := QREncodeAdvanced("hi", 99, QRECCLow); err == nil {
		t.Error("bad version should error")
	}
}

func TestQRAdvancedVersionInfoBCH(t *testing.T) {
	// The version information must survive a BCH round-trip: newTemplateEx writes
	// it and it can be read back exactly for versions 7-10.
	for v := 7; v <= 10; v++ {
		q := newTemplateEx(v)
		size := q.size
		got := 0
		for i := 0; i < 6; i++ {
			for j := 0; j < 3; j++ {
				if q.modules[i][size-11+j] {
					got |= 1 << (i*3 + j)
				}
			}
		}
		if got != versionInfoBits[v] {
			t.Errorf("v%d version info = %#x, want %#x", v, got, versionInfoBits[v])
		}
	}
}

func TestQRAdvancedDeterministic(t *testing.T) {
	a, _ := QREncodeAdvanced("Deterministic 42", 5, QRECCQuartile)
	b, _ := QREncodeAdvanced("Deterministic 42", 5, QRECCQuartile)
	if len(a.Data) != len(b.Data) {
		t.Fatal("size mismatch")
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatal("QREncodeAdvanced is not deterministic")
		}
	}
}

// --- Code 39 --------------------------------------------------------------

func TestCode39RoundTrip(t *testing.T) {
	for _, s := range []string{"A", "CODE39", "HELLO WORLD", "ABC-123", "PRICE $9.99", "42/7+3"} {
		img, err := EncodeCode39(s)
		if err != nil {
			t.Fatalf("encode %q: %v", s, err)
		}
		got, ok := DecodeCode39(img)
		if !ok || got != s {
			t.Errorf("Code 39 %q: got %q ok=%v", s, got, ok)
		}
	}
}

func TestCode39Errors(t *testing.T) {
	if _, err := EncodeCode39("lower"); err == nil {
		t.Error("lowercase should error")
	}
	if _, err := EncodeCode39("A*B"); err == nil {
		t.Error("embedded '*' should error")
	}
	if _, err := EncodeCode39(""); err == nil {
		t.Error("empty should error")
	}
}

// --- Codabar --------------------------------------------------------------

func TestCodabarRoundTrip(t *testing.T) {
	for _, s := range []string{"A1234B", "A123456789A", "C0301B", "D12-34D", "A$50.00A", "A:/.+B"} {
		img, err := EncodeCodabar(s)
		if err != nil {
			t.Fatalf("encode %q: %v", s, err)
		}
		got, ok := DecodeCodabar(img)
		if !ok || got != s {
			t.Errorf("Codabar %q: got %q ok=%v", s, got, ok)
		}
	}
}

func TestCodabarErrors(t *testing.T) {
	if _, err := EncodeCodabar("1234"); err == nil {
		t.Error("missing start/stop should error")
	}
	if _, err := EncodeCodabar("A1B2A"); err == nil {
		t.Error("interior start/stop letter should error")
	}
}

// --- ITF ------------------------------------------------------------------

func TestITFRoundTrip(t *testing.T) {
	cases := []struct{ in, want string }{
		{"1234", "1234"},
		{"0123456789", "0123456789"},
		{"12345", "012345"}, // odd padded with a leading zero
		{"00", "00"},
	}
	for _, c := range cases {
		img, err := EncodeITF(c.in)
		if err != nil {
			t.Fatalf("encode %q: %v", c.in, err)
		}
		got, ok := DecodeITF(img)
		if !ok || got != c.want {
			t.Errorf("ITF %q: got %q ok=%v, want %q", c.in, got, ok, c.want)
		}
	}
}

func TestITFErrors(t *testing.T) {
	if _, err := EncodeITF("12A4"); err == nil {
		t.Error("non-digit should error")
	}
	if _, err := EncodeITF(""); err == nil {
		t.Error("empty should error")
	}
}

// --- Code 93 --------------------------------------------------------------

func TestCode93RoundTrip(t *testing.T) {
	for _, s := range []string{"A", "CODE93", "HELLO WORLD", "ABC-123", "TEST/42", "1234567890"} {
		img, err := EncodeCode93(s)
		if err != nil {
			t.Fatalf("encode %q: %v", s, err)
		}
		got, ok := DecodeCode93(img)
		if !ok || got != s {
			t.Errorf("Code 93 %q: got %q ok=%v", s, got, ok)
		}
	}
}

func TestCode93ChecksVary(t *testing.T) {
	// Two inputs differing in one character should generally produce different
	// check characters, exercising the weighted sums.
	c1, k1 := code93Checks([]int{1, 2, 3})
	c2, k2 := code93Checks([]int{1, 2, 4})
	if c1 == c2 && k1 == k2 {
		t.Error("check characters did not change with the data")
	}
}

// --- EAN-8 ----------------------------------------------------------------

func TestEAN8RoundTrip(t *testing.T) {
	cases := []struct{ in, want string }{
		{"1234567", ""}, // check computed
		{"96385074", "96385074"},
		{"0000000", ""},
	}
	for _, c := range cases {
		img, err := EncodeEAN8(c.in)
		if err != nil {
			t.Fatalf("encode %q: %v", c.in, err)
		}
		got, ok := DecodeEAN8(img)
		if !ok {
			t.Errorf("EAN-8 %q: decode failed", c.in)
			continue
		}
		if c.want != "" && got != c.want {
			t.Errorf("EAN-8 %q: got %q, want %q", c.in, got, c.want)
		}
		if len(got) != 8 {
			t.Errorf("EAN-8 %q: got %q, want 8 digits", c.in, got)
		}
		// Re-encoding the full 8-digit result must validate.
		if _, err := EncodeEAN8(got); err != nil {
			t.Errorf("EAN-8 re-encode %q: %v", got, err)
		}
	}
}

func TestEAN8Errors(t *testing.T) {
	if _, err := EncodeEAN8("123"); err == nil {
		t.Error("wrong length should error")
	}
	if _, err := EncodeEAN8("12345671"); err == nil {
		t.Error("bad check digit should error")
	}
}

// --- UPC-A ----------------------------------------------------------------

func TestUPCARoundTrip(t *testing.T) {
	for _, in := range []string{"03600029145", "01234567890", "12345678901"} {
		img, err := EncodeUPCA(in)
		if err != nil {
			t.Fatalf("encode %q: %v", in, err)
		}
		got, ok := DecodeUPCA(img)
		if !ok || len(got) != 12 || got[:11] != in {
			t.Errorf("UPC-A %q: got %q ok=%v", in, got, ok)
		}
		if _, err := EncodeUPCA(got); err != nil {
			t.Errorf("UPC-A re-encode %q: %v", got, err)
		}
	}
}

func TestUPCAErrors(t *testing.T) {
	if _, err := EncodeUPCA("123"); err == nil {
		t.Error("wrong length should error")
	}
	if _, err := EncodeUPCA("036000291455"); err == nil {
		t.Error("bad check digit should error")
	}
}

// --- MSI ------------------------------------------------------------------

func TestMSIRoundTrip(t *testing.T) {
	for _, in := range []string{"1234", "80523", "0", "999999"} {
		img, err := EncodeMSI(in)
		if err != nil {
			t.Fatalf("encode %q: %v", in, err)
		}
		got, ok := DecodeMSI(img)
		if !ok || !strings.HasPrefix(got, in) || len(got) != len(in)+1 {
			t.Errorf("MSI %q: got %q ok=%v", in, got, ok)
		}
	}
}

func TestMSIErrors(t *testing.T) {
	if _, err := EncodeMSI("12A"); err == nil {
		t.Error("non-digit should error")
	}
	if _, err := EncodeMSI(""); err == nil {
		t.Error("empty should error")
	}
}

// --- Code 11 --------------------------------------------------------------

func TestCode11RoundTrip(t *testing.T) {
	for _, in := range []string{"123", "0123456789", "12-34", "555-1212"} {
		img, err := EncodeCode11(in)
		if err != nil {
			t.Fatalf("encode %q: %v", in, err)
		}
		got, ok := DecodeCode11(img)
		if !ok || got != in {
			t.Errorf("Code 11 %q: got %q ok=%v", in, got, ok)
		}
	}
}

func TestCode11Errors(t *testing.T) {
	if _, err := EncodeCode11("12A"); err == nil {
		t.Error("non-digit should error")
	}
	if _, err := EncodeCode11(""); err == nil {
		t.Error("empty should error")
	}
}

// --- Multi ----------------------------------------------------------------

// vstack stacks two grayscale barcodes vertically into one image, centring each
// horizontally and padding widths to the wider of the two.
func vstack(a, b *cv.Mat) *cv.Mat {
	w := a.Cols
	if b.Cols > w {
		w = b.Cols
	}
	h := a.Rows + b.Rows
	m := cv.NewMat(h, w, 1)
	m.SetTo(255)
	a.CopyTo(m, 0, (w-a.Cols)/2)
	b.CopyTo(m, a.Rows, (w-b.Cols)/2)
	return m
}

func TestDetectAndDecodeMultiSingle(t *testing.T) {
	img, _ := EncodeEAN13("590123412345")
	res := DetectAndDecodeMulti(img)
	found := false
	for _, r := range res {
		if r.Type == "EAN-13" && r.Text == "5901234123457" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected EAN-13 in %v", res)
	}
}

func TestDetectAndDecodeMultiStacked(t *testing.T) {
	ean, _ := EncodeEAN13("400638133393")
	c39, _ := EncodeCode39("HELLO")
	img := vstack(ean, c39)
	res := DetectAndDecodeMulti(img)
	var haveEAN, haveC39 bool
	for _, r := range res {
		if r.Type == "EAN-13" && r.Text == "4006381333931" {
			haveEAN = true
		}
		if r.Type == "Code 39" && r.Text == "HELLO" {
			haveC39 = true
		}
	}
	if !haveEAN || !haveC39 {
		t.Errorf("stacked decode missing symbols: %v", res)
	}
}

func TestDetectAndDecodeMultiEmpty(t *testing.T) {
	blank := cv.NewMat(60, 200, 1)
	blank.SetTo(255)
	if res := DetectAndDecodeMulti(blank); len(res) != 0 {
		t.Errorf("blank image should decode nothing, got %v", res)
	}
}

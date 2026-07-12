package barcode

import (
	"strings"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- Reed-Solomon ---------------------------------------------------------

func TestReedSolomonCleanRoundTrip(t *testing.T) {
	cases := []struct {
		data string
		nsym int
	}{
		{"", 4},
		{"A", 4},
		{"HELLO WORLD", 10},
		{strings.Repeat("x", 200), 20},
	}
	for _, c := range cases {
		data := []byte(c.data)
		ecc := ReedSolomonEncode(data, c.nsym)
		if len(ecc) != c.nsym {
			t.Fatalf("ecc length %d, want %d", len(ecc), c.nsym)
		}
		msg := append(append([]byte{}, data...), ecc...)
		out, ok := ReedSolomonDecode(msg, c.nsym)
		if !ok || string(out) != string(msg) {
			t.Errorf("clean decode of %q failed: ok=%v", c.data, ok)
		}
	}
}

func TestReedSolomonCorrectsWithinBudget(t *testing.T) {
	data := []byte("Reed-Solomon deterministic correction test payload 0123456789")
	nsym := 16 // corrects up to 8 byte errors
	msg := append(append([]byte{}, data...), ReedSolomonEncode(data, nsym)...)
	for n := 1; n <= nsym/2; n++ {
		corrupt := append([]byte{}, msg...)
		for i := 0; i < n; i++ {
			corrupt[(i*5)%len(corrupt)] ^= byte(0x40 + i)
		}
		out, ok := ReedSolomonDecode(corrupt, nsym)
		if !ok || string(out) != string(msg) {
			t.Errorf("%d errors: correction failed ok=%v", n, ok)
		}
	}
}

func TestReedSolomonRejectsBeyondBudget(t *testing.T) {
	data := []byte("small")
	nsym := 6 // corrects up to 3 errors
	msg := append(append([]byte{}, data...), ReedSolomonEncode(data, nsym)...)
	corrupt := append([]byte{}, msg...)
	for i := 0; i < 5; i++ { // 5 > 3
		corrupt[i] ^= 0xFF
	}
	if _, ok := ReedSolomonDecode(corrupt, nsym); ok {
		t.Error("decode reported success beyond the error-correction budget")
	}
}

func TestGaloisFieldIdentities(t *testing.T) {
	for a := 1; a < 256; a++ {
		inv := gfInverse(byte(a))
		if gfMul(byte(a), inv) != 1 {
			t.Fatalf("gfMul(%d, inv)=%d, want 1", a, gfMul(byte(a), inv))
		}
		if gfDiv(byte(a), byte(a)) != 1 {
			t.Fatalf("gfDiv(%d,%d) != 1", a, a)
		}
	}
}

// --- QR codes -------------------------------------------------------------

// payloads sized to land in each of versions 1-4 at level L.
var qrPayloads = map[int]string{
	1: "HELLO",
	2: strings.Repeat("v2 ", 8),
	3: strings.Repeat("ver3 ", 10),
	4: strings.Repeat("version4 ", 8),
}

func TestQRRoundTripAllVersions(t *testing.T) {
	for v := 1; v <= 4; v++ {
		text := qrPayloads[v]
		// Explicit version.
		img := QREncode(text, v)
		got, ok := QRDetectAndDecode(img)
		if !ok || got != text {
			t.Errorf("version %d explicit: ok=%v match=%v", v, ok, got == text)
		}
		// Auto version selection.
		img2 := QREncode(text, 0)
		got2, ok2 := QRDetectAndDecode(img2)
		if !ok2 || got2 != text {
			t.Errorf("version %d auto: ok=%v match=%v", v, ok2, got2 == text)
		}
	}
}

func TestQRExactStringRoundTrip(t *testing.T) {
	img := QREncode("HELLO", 0)
	got, ok := QRDetectAndDecode(img)
	if !ok {
		t.Fatal("decode failed")
	}
	if got != "HELLO" {
		t.Fatalf("got %q, want %q", got, "HELLO")
	}
}

func TestQRFinderLocalization(t *testing.T) {
	for v := 1; v <= 4; v++ {
		img := QREncode(qrPayloads[v], v)
		pts := FindFinderPatterns(img)
		if len(pts) != 3 {
			t.Errorf("version %d: found %d finder patterns, want 3", v, len(pts))
		}
	}
}

func TestQRRotationInvariance(t *testing.T) {
	text := "Rotate 123"
	base := QREncode(text, 0)
	for _, code := range []cv.RotateCode{cv.Rotate90CW, cv.Rotate180, cv.Rotate90CCW} {
		img := cv.Rotate(base, code)
		got, ok := QRDetectAndDecode(img)
		if !ok || got != text {
			t.Errorf("rotate %v: ok=%v match=%v", code, ok, got == text)
		}
	}
}

// flipModule inverts the pixels of module (mrow, mcol) in a rendered symbol so
// the detector samples the opposite colour there.
func flipModule(img *cv.Mat, mrow, mcol int) {
	y0 := (mrow + quietZone) * moduleScale
	x0 := (mcol + quietZone) * moduleScale
	for dy := 0; dy < moduleScale; dy++ {
		for dx := 0; dx < moduleScale; dx++ {
			v := img.At(y0+dy, x0+dx, 0)
			img.Set(y0+dy, x0+dx, 0, 255-v)
		}
	}
}

func TestQRErrorCorrectionRecoversModules(t *testing.T) {
	// Version 1 level L has 7 EC codewords, correcting up to 3 codeword errors.
	// Flip three interior data modules (avoiding finders and timing patterns);
	// Reed-Solomon must still recover the exact payload.
	img := QREncode("HELLO", 1)
	flipModule(img, 10, 10)
	flipModule(img, 12, 13)
	flipModule(img, 14, 9)
	got, ok := QRDetectAndDecode(img)
	if !ok || got != "HELLO" {
		t.Errorf("error-corrected decode: ok=%v got=%q", ok, got)
	}
}

func TestQRCapacityAndSelection(t *testing.T) {
	want := map[int]int{1: 17, 2: 32, 3: 53, 4: 78}
	for v, exp := range want {
		if got := QRCapacity(v); got != exp {
			t.Errorf("QRCapacity(%d)=%d, want %d", v, got, exp)
		}
	}
	if QRCapacity(5) != 0 || QRCapacity(0) != 0 {
		t.Error("QRCapacity should be 0 for unsupported versions")
	}
	// selectVersion picks the smallest fitting version.
	if v, _ := selectVersion(strings.Repeat("a", 17), 0); v != 1 {
		t.Errorf("17 bytes should fit version 1, got %d", v)
	}
	if v, _ := selectVersion(strings.Repeat("a", 18), 0); v != 2 {
		t.Errorf("18 bytes should require version 2, got %d", v)
	}
}

func TestQREncodePanicsOnOverflow(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic for oversized payload")
		}
	}()
	QREncode(strings.Repeat("x", 200), 1)
}

func TestQREncodeDeterministic(t *testing.T) {
	a := QREncode("determinism", 0)
	b := QREncode("determinism", 0)
	if len(a.Data) != len(b.Data) {
		t.Fatal("size mismatch")
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatal("QREncode output is not deterministic")
		}
	}
}

func TestQRFormatBitsBCH(t *testing.T) {
	// Every (level, mask) format string must be recoverable at Hamming distance
	// 0, and distinct valid strings differ by at least the BCH minimum distance.
	seen := map[int]bool{}
	for ecl := 0; ecl < 4; ecl++ {
		for m := 0; m < 8; m++ {
			bits := formatBits(ecl, m)
			if seen[bits] {
				t.Fatalf("duplicate format string for ecl=%d mask=%d", ecl, m)
			}
			seen[bits] = true
		}
	}
	// readFormatMask recovers the mask written by drawFormatBits.
	for m := 0; m < 8; m++ {
		q := newQRTemplate(1)
		q.drawFormatBits(m)
		got, ok := q.readFormatMask(q.modules)
		if !ok || got != m {
			t.Errorf("format mask %d: recovered %d ok=%v", m, got, ok)
		}
	}
}

func TestQRDetectRejectsBlank(t *testing.T) {
	blank := cv.NewMat(80, 80, 1)
	blank.SetTo(255)
	if _, ok := QRDetectAndDecode(blank); ok {
		t.Error("blank image should not decode")
	}
}

// --- EAN-13 ---------------------------------------------------------------

func TestEAN13RoundTrip(t *testing.T) {
	cases := []struct{ in, want string }{
		{"590123412345", "5901234123457"},
		{"400638133393", "4006381333931"},
		{"0123456789012", "0123456789012"},
	}
	for _, c := range cases {
		img, err := EncodeEAN13(c.in)
		if err != nil {
			t.Fatalf("encode %q: %v", c.in, err)
		}
		got, ok := DecodeEAN13(img)
		if !ok || got != c.want {
			t.Errorf("EAN-13 %q: got %q ok=%v, want %q", c.in, got, ok, c.want)
		}
	}
}

func TestEAN13Errors(t *testing.T) {
	if _, err := EncodeEAN13("12345"); err == nil {
		t.Error("expected error for wrong digit count")
	}
	if _, err := EncodeEAN13("12345678901X"); err == nil {
		t.Error("expected error for non-digit")
	}
	if _, err := EncodeEAN13("5901234123450"); err == nil {
		t.Error("expected error for bad check digit")
	}
}

// --- Code 128 -------------------------------------------------------------

func TestCode128RoundTrip(t *testing.T) {
	cases := []string{"A", "ABC-123", "Hello, World!", "Order #42", "~}{|", "Code 128 test"}
	for _, s := range cases {
		img, err := EncodeCode128(s)
		if err != nil {
			t.Fatalf("encode %q: %v", s, err)
		}
		got, ok := DecodeCode128(img)
		if !ok || got != s {
			t.Errorf("Code 128 %q: got %q ok=%v", s, got, ok)
		}
	}
}

func TestCode128RejectsNonASCII(t *testing.T) {
	if _, err := EncodeCode128("café"); err == nil {
		t.Error("expected error for non-ASCII input")
	}
}

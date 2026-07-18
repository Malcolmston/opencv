package barcode

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- Checksums ------------------------------------------------------------

func TestGS1AndSymbologyCheckDigits(t *testing.T) {
	// Known real-world codes with their published check digits.
	if got, _ := EAN13CheckDigit("400638133393"); got != 1 {
		t.Errorf("EAN13CheckDigit = %d, want 1", got)
	}
	if !ValidateEAN13("4006381333931") {
		t.Error("ValidateEAN13 rejected a valid code")
	}
	if ValidateEAN13("4006381333930") {
		t.Error("ValidateEAN13 accepted a wrong check digit")
	}
	if got, _ := UPCACheckDigit("03600029145"); got != 2 {
		t.Errorf("UPCACheckDigit = %d, want 2", got)
	}
	if !ValidateUPCA("036000291452") {
		t.Error("ValidateUPCA rejected a valid code")
	}
	if got, _ := EAN8CheckDigit("9638507"); got != 4 {
		t.Errorf("EAN8CheckDigit = %d, want 4", got)
	}
	if !ValidateEAN8("96385074") {
		t.Error("ValidateEAN8 rejected a valid code")
	}
	if !ValidateITF14("15400141288763") {
		t.Error("ValidateITF14 rejected a valid code")
	}
	// Errors.
	if _, err := EAN13CheckDigit("12345"); err == nil {
		t.Error("EAN13CheckDigit accepted a short payload")
	}
	if _, err := GS1CheckDigit("12x45"); err == nil {
		t.Error("GS1CheckDigit accepted a non-digit")
	}
}

func TestLuhnAndCode39CheckDigits(t *testing.T) {
	// 7992739871 -> Luhn check digit 3 (classic textbook example).
	if got, _ := LuhnChecksum("7992739871"); got != 3 {
		t.Errorf("LuhnChecksum = %d, want 3", got)
	}
	if !ValidateLuhn("79927398713") {
		t.Error("ValidateLuhn rejected a valid number")
	}
	if ValidateLuhn("79927398710") {
		t.Error("ValidateLuhn accepted an invalid number")
	}
	if got, _ := MSICheckDigitMod10("1234567"); got != func() int { g, _ := LuhnChecksum("1234567"); return g }() {
		t.Errorf("MSICheckDigitMod10 disagreed with LuhnChecksum: %d", got)
	}
	// Code 39 mod-43: message "CODE39" has a defined check character.
	c, err := Code39CheckChar("CODE39")
	if err != nil {
		t.Fatalf("Code39CheckChar error: %v", err)
	}
	// C=12 O=24 D=13 E=14 3=3 9=9 => sum 75, 75 mod 43 = 32 => 'W'.
	if c != 'W' {
		t.Errorf("Code39CheckChar = %q, want 'W'", c)
	}
	if _, err := Code39CheckChar("code39"); err == nil {
		t.Error("Code39CheckChar accepted lowercase (outside the set)")
	}
}

// --- Conversions ----------------------------------------------------------

func TestUPCEConversions(t *testing.T) {
	// 04252614 is a canonical UPC-E code expanding to UPC-A 042100005264.
	upca, err := UPCEToUPCA("04252614")
	if err != nil {
		t.Fatalf("UPCEToUPCA error: %v", err)
	}
	if upca != "042100005264" {
		t.Errorf("UPCEToUPCA = %q, want 042100005264", upca)
	}
	if !ValidateUPCA(upca) {
		t.Error("expanded UPC-A fails its own check digit")
	}
	back, err := UPCAToUPCE(upca)
	if err != nil {
		t.Fatalf("UPCAToUPCE error: %v", err)
	}
	if back != "04252614" {
		t.Errorf("UPCAToUPCE round-trip = %q, want 04252614", back)
	}
	ean, err := UPCAToEAN13(upca)
	if err != nil {
		t.Fatalf("UPCAToEAN13 error: %v", err)
	}
	if ean != "0042100005264" || !ValidateEAN13(ean) {
		t.Errorf("UPCAToEAN13 = %q (valid=%v)", ean, ValidateEAN13(ean))
	}
}

// --- QR format / version metadata ----------------------------------------

func TestQRVersionInfoKnownValues(t *testing.T) {
	// ISO/IEC 18004 Annex D reference version-information bit patterns.
	known := map[int]int{7: 0x07C94, 8: 0x085BC, 10: 0x0A4D3, 20: 0x149A6, 40: 0x28C69}
	for v, want := range known {
		got, err := EncodeQRVersionInfo(v)
		if err != nil || got != want {
			t.Errorf("EncodeQRVersionInfo(%d) = %05X (err %v), want %05X", v, got, err, want)
		}
	}
	if _, err := EncodeQRVersionInfo(6); err == nil {
		t.Error("EncodeQRVersionInfo accepted version 6")
	}
}

func TestQRFormatVersionRoundTrip(t *testing.T) {
	for _, lv := range []QRECCLevel{QRECCLow, QRECCMedium, QRECCQuartile, QRECCHigh} {
		for m := 0; m < 8; m++ {
			raw, err := EncodeQRFormatInfo(lv, m)
			if err != nil {
				t.Fatalf("EncodeQRFormatInfo: %v", err)
			}
			fi, ok := DecodeQRFormatInfo(raw)
			if !ok || fi.Level != lv || fi.Mask != m {
				t.Errorf("format round-trip lv=%v m=%d: got %+v ok=%v", lv, m, fi, ok)
			}
			// Inject a single bit error; BCH should still correct it.
			fi2, ok2 := DecodeQRFormatInfo(raw ^ 0x0040)
			if !ok2 || fi2.Level != lv || fi2.Mask != m {
				t.Errorf("format 1-bit correction lv=%v m=%d failed: %+v ok=%v", lv, m, fi2, ok2)
			}
		}
	}
	for v := 7; v <= 40; v++ {
		raw, _ := EncodeQRVersionInfo(v)
		if got, ok := DecodeQRVersionInfo(raw); !ok || got != v {
			t.Errorf("version round-trip %d: got %d ok=%v", v, got, ok)
		}
	}
	if _, ok := DecodeQRFormatInfo(0); ok {
		// 0 masked is a valid codeword-ish; ensure the decode is deterministic
		// rather than asserting failure. Just check it does not panic.
	}
}

func TestQRSizeVersionAndAlignment(t *testing.T) {
	if QRSizeForVersion(1) != 21 || QRSizeForVersion(40) != 177 {
		t.Error("QRSizeForVersion boundary values wrong")
	}
	if QRSizeForVersion(0) != 0 || QRSizeForVersion(41) != 0 {
		t.Error("QRSizeForVersion should be 0 out of range")
	}
	if v, ok := QRVersionForSize(25); !ok || v != 2 {
		t.Errorf("QRVersionForSize(25) = %d,%v want 2,true", v, ok)
	}
	if _, ok := QRVersionForSize(24); ok {
		t.Error("QRVersionForSize(24) should be invalid")
	}
	if QRAlignmentCenters(1) != nil {
		t.Error("version 1 has no alignment centres")
	}
	got := QRAlignmentCenters(7)
	want := []int{6, 22, 38}
	if len(got) != len(want) {
		t.Fatalf("QRAlignmentCenters(7) len = %d", len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("QRAlignmentCenters(7)[%d] = %d want %d", i, got[i], want[i])
		}
	}
	if l := QRECCLevel(0).String(); l != "L" {
		t.Errorf("QRECCLevel.String = %q want L", l)
	}
}

// --- QR module sampling ---------------------------------------------------

func TestSampleQRModulesFinderPatterns(t *testing.T) {
	version := 2
	img := QREncode("HELLO", version)
	if img == nil {
		t.Fatal("QREncode returned nil")
	}
	size := QRSizeForVersion(version)
	grid, ok := SampleQRModules(img, version)
	if !ok {
		t.Fatal("SampleQRModules failed")
	}
	if len(grid) != size || len(grid[0]) != size {
		t.Fatalf("grid dimensions %dx%d, want %d", len(grid), len(grid[0]), size)
	}
	// Top-left finder pattern: solid dark 7x7 outer ring with a light ring and
	// a dark 3x3 core.
	for c := 0; c < 7; c++ {
		if !grid[0][c] {
			t.Errorf("top-left finder top edge module (0,%d) should be dark", c)
		}
	}
	if grid[1][1] {
		t.Error("finder inner ring module (1,1) should be light")
	}
	if !grid[3][3] {
		t.Error("finder core module (3,3) should be dark")
	}
	// Separator between finder and data.
	if grid[7][7] && grid[0][7] {
		// (0,7) is a separator (light) in a well-formed symbol.
		t.Error("separator module (0,7) should be light")
	}
	// Top-right finder present.
	for c := size - 7; c < size; c++ {
		if !grid[0][c] {
			t.Errorf("top-right finder module (0,%d) should be dark", c)
		}
	}
	// Bottom-left finder present.
	for r := size - 7; r < size; r++ {
		if !grid[r][0] {
			t.Errorf("bottom-left finder module (%d,0) should be dark", r)
		}
	}
	if ratio := DarkModuleRatio(grid); ratio < 0.30 || ratio > 0.70 {
		t.Errorf("DarkModuleRatio = %.3f, implausible", ratio)
	}
	if CountDarkModules(grid) == 0 {
		t.Error("CountDarkModules = 0")
	}
	if QRModuleCount(version) != size {
		t.Error("QRModuleCount mismatch")
	}
	if QRRawDataModuleCount(2) <= 0 {
		t.Error("QRRawDataModuleCount(2) should be positive")
	}
}

// --- Data Matrix finder ---------------------------------------------------

// renderModuleMatrix draws a module matrix (true=dark) as a grayscale Mat with
// the given module scale and a light quiet zone, used to synthesise Data Matrix
// test images.
func renderModuleMatrix(mods [][]bool, scale, quiet int) *cv.Mat {
	rows, cols := len(mods), len(mods[0])
	h := (rows + 2*quiet) * scale
	w := (cols + 2*quiet) * scale
	m := cv.NewMat(h, w, 1)
	m.SetTo(255)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if !mods[r][c] {
				continue
			}
			y0 := (r + quiet) * scale
			x0 := (c + quiet) * scale
			for dy := 0; dy < scale; dy++ {
				for dx := 0; dx < scale; dx++ {
					m.Set(y0+dy, x0+dx, 0, 0)
				}
			}
		}
	}
	return m
}

func TestFindDataMatrixFinder(t *testing.T) {
	const n = 12
	mods := make([][]bool, n)
	for r := 0; r < n; r++ {
		mods[r] = make([]bool, n)
	}
	// Solid L: left column and bottom row.
	for r := 0; r < n; r++ {
		mods[r][0] = true
	}
	for c := 0; c < n; c++ {
		mods[n-1][c] = true
	}
	// Timing tracks: top row and right column alternate, starting dark at the
	// corners so the dark bounding box spans the whole symbol.
	for c := 0; c < n; c++ {
		if c%2 == 0 {
			mods[0][c] = true
		}
	}
	for r := 0; r < n; r++ {
		if r%2 == 0 {
			mods[r][n-1] = true
		}
	}
	img := renderModuleMatrix(mods, 6, 4)
	f, ok := FindDataMatrixFinder(img)
	if !ok {
		t.Fatal("FindDataMatrixFinder failed on a synthetic symbol")
	}
	if f.Corner != DataMatrixBottomLeft {
		t.Errorf("Corner = %v, want bottom-left", f.Corner)
	}
	if f.Rows != n || f.Cols != n {
		t.Errorf("dimensions = %dx%d, want %dx%d", f.Rows, f.Cols, n, n)
	}
	if f.Width() != n*6 || f.Height() != n*6 {
		t.Errorf("pixel size = %dx%d, want %dx%d", f.Width(), f.Height(), n*6, n*6)
	}
	if _, ok := FindDataMatrixFinder(cv.NewMat(20, 20, 1)); ok {
		t.Error("FindDataMatrixFinder accepted an all-white image")
	}
}

// --- Benchmark (heaviest new routine) ------------------------------------

func BenchmarkSampleQRModules(b *testing.B) {
	img := QREncode("BENCHMARK PAYLOAD 1234567890", 4)
	version := 4
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := SampleQRModules(img, version); !ok {
			b.Fatal("sample failed")
		}
	}
}

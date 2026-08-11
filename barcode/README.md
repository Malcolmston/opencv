# barcode — provides 1D and 2D barcode generation and detection on top of the standard-library-only

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/barcode.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/barcode)

Package barcode provides 1D and 2D barcode generation and detection on top of
the standard-library-only OpenCV port github.com/malcolmston/opencv (imported
here as cv). It mirrors the spirit of OpenCV's barcode and objdetect QR
modules while depending only on that root package and the Go standard library.

- Versions 1-4 (21x21 to 33x33 modules). - Byte (8-bit) encoding mode. -
Error-correction level L (a single Reed-Solomon block per symbol).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/barcode"
```

## Usage

This is the package's own `ExampleDecodeCode128`, so it compiles and its output is
asserted on every `go test ./barcode/`.

```go
img, err := barcode.EncodeCode128("Wiki-128")
	if err != nil {
		fmt.Println(err)
		return
	}
	text, ok := barcode.DecodeCode128(img)
	fmt.Printf("%q ok=%v\n", text, ok)

	_, err = barcode.EncodeCode128("")
	fmt.Println("empty:", err)
```

```
"Wiki-128" ok=true
empty: barcode: Code 128 payload must not be empty
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Code39CheckChar(message string) (byte, error)` | Code39CheckChar computes the Code 39 modulo-43 check character for a message made of the 43 valid Code 39 data characters (digits, uppercase letters… |
| `func CountDarkModules(grid [][]bool) int` | CountDarkModules returns the number of true (dark) entries in a module matrix. |
| `func DarkModuleRatio(grid [][]bool) float64` | DarkModuleRatio returns the fraction of dark modules in a module matrix, a value in [0,1]. |
| `func DecodeCodabar(img *cv.Mat) (string, bool)` | DecodeCodabar scans a rendered Codabar barcode and returns the decoded text (including the framing start/stop letters) and true on success, or ("",… |
| `func DecodeCode11(img *cv.Mat) (string, bool)` | DecodeCode11 scans a rendered Code 11 barcode and returns the decoded text (without the framing or check character) and true on success, or ("",… |
| `func DecodeCode128(img *cv.Mat) (string, bool)` | DecodeCode128 scans a rendered Code 128 (code set B) barcode and returns the decoded text and true on success, or ("", false) if no valid symbol is… |
| `func DecodeCode39(img *cv.Mat) (string, bool)` | DecodeCode39 scans a rendered Code 39 barcode and returns the decoded text (without the framing '*' characters) and true on success, or ("", false)… |
| `func DecodeCode93(img *cv.Mat) (string, bool)` | DecodeCode93 scans a rendered Code 93 barcode and returns the decoded text (without the framing, check characters or termination bar) and true on… |
| `func DecodeEAN13(img *cv.Mat) (string, bool)` | DecodeEAN13 scans a rendered EAN-13 barcode and returns the 13-digit string and true on success, or ("", false) if no valid symbol is found. |
| `func DecodeEAN8(img *cv.Mat) (string, bool)` | DecodeEAN8 scans a rendered EAN-8 barcode and returns the eight-digit string and true on success, or ("", false) if no valid symbol is found. |
| `func DecodeITF(img *cv.Mat) (string, bool)` | DecodeITF scans a rendered Interleaved 2 of 5 barcode and returns the decoded digit string and true on success, or ("", false) if no valid symbol is… |
| `func DecodeMSI(img *cv.Mat) (string, bool)` | DecodeMSI scans a rendered MSI barcode and returns the digit string (including the trailing check digit) and true on success, or ("", false) if no… |
| `func DecodeQRVersionInfo(raw int) (int, bool)` | DecodeQRVersionInfo recovers a QR version number (7-40) from an 18-bit version-information value, tolerating up to three bit errors by choosing the… |
| `func DecodeUPCA(img *cv.Mat) (string, bool)` | DecodeUPCA scans a rendered UPC-A barcode and returns the twelve-digit string and true on success, or ("", false) if no valid symbol is found. |
| `func EAN13CheckDigit(payload string) (int, error)` | EAN13CheckDigit returns the modulo-10 check digit of a 12-digit EAN-13 payload. |
| `func EAN8CheckDigit(payload string) (int, error)` | EAN8CheckDigit returns the modulo-10 check digit of a 7-digit EAN-8 payload. |
| `func EncodeCodabar(text string) (*cv.Mat, error)` | EncodeCodabar renders text as a Codabar barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with quiet zones. |
| `func EncodeCode11(text string) (*cv.Mat, error)` | EncodeCode11 renders text as a Code 11 barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with quiet zones. |
| `func EncodeCode128(text string) (*cv.Mat, error)` | EncodeCode128 renders text as a Code 128 (code set B) barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with quiet… |
| `func EncodeCode39(text string) (*cv.Mat, error)` | EncodeCode39 renders text as a Code 39 barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with quiet zones. |
| `func EncodeCode93(text string) (*cv.Mat, error)` | EncodeCode93 renders text as a Code 93 barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with quiet zones. |
| `func EncodeEAN13(digits string) (*cv.Mat, error)` | EncodeEAN13 renders digits as an EAN-13 barcode and returns it as a single-channel grayscale `cv.Mat` (bars are 0, spaces 255) with quiet zones. |
| `func EncodeEAN8(digits string) (*cv.Mat, error)` | EncodeEAN8 renders digits as an EAN-8 barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with quiet zones. |
| `func EncodeITF(digits string) (*cv.Mat, error)` | EncodeITF renders a numeric string as an Interleaved 2 of 5 barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with… |
| `func EncodeMSI(digits string) (*cv.Mat, error)` | EncodeMSI renders digits as an MSI (Modified Plessey) barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with quiet… |
| `func EncodeQRFormatInfo(level QRECCLevel, mask int) (int, error)` | EncodeQRFormatInfo returns the 15-bit BCH-protected, XOR-masked format information value for the given error-correction level and data-mask number. |
| `func EncodeQRVersionInfo(version int) (int, error)` | EncodeQRVersionInfo returns the 18-bit BCH-protected version-information value carried by QR versions 7 through 40. |
| `func EncodeUPCA(digits string) (*cv.Mat, error)` | EncodeUPCA renders digits as a UPC-A barcode and returns it as a single-channel grayscale `cv.Mat` (bars 0, spaces 255) with quiet zones. |
| `func FindFinderPatterns(img *cv.Mat) []cv.Point` | FindFinderPatterns locates the QR finder patterns in img and returns their centre points (in pixels), strongest first. |
| `func GS1CheckDigit(payload string) (int, error)` | GS1CheckDigit computes the standard GS1 modulo-10 check digit for a numeric payload (the code without its trailing check digit). |
| `func ITF14CheckDigit(payload string) (int, error)` | ITF14CheckDigit returns the modulo-10 check digit of a 13-digit ITF-14 payload. |
| `func LuhnChecksum(payload string) (int, error)` | LuhnChecksum returns the Luhn (modulo-10, "double every second digit") check digit for a numeric payload that does not yet include the check digit. |
| `func MSICheckDigitMod10(payload string) (int, error)` | MSICheckDigitMod10 returns the MSI Modulo-10 check digit of a numeric payload. |
| `func QRAlignmentCenters(version int) []int` | QRAlignmentCenters returns the row/column centre coordinates at which a QR symbol of the given version places alignment patterns (an alignment… |
| `func QRCapacity(version int) int` | QRCapacity reports the maximum byte-mode payload length (in bytes) for a supported QR version (1-4). |
| `func QRDataCapacity(version int, level QRECCLevel) int` | QRDataCapacity returns the number of data (non-error-correction) codewords available for the given version (1-10) and error-correction level. |
| `func QRDetectAndDecode(img *cv.Mat) (string, bool)` | QRDetectAndDecode locates a QR Code symbol in img, samples its modules and decodes the byte-mode payload, returning the text and true on success. |
| `func QRDetectAndDecodeAdvanced(img *cv.Mat) (string, bool)` | QRDetectAndDecodeAdvanced locates a QR Code symbol in img, samples its modules and decodes it, handling versions 1-10, all error-correction levels… |
| `func QREncode(text string, version int) *cv.Mat` | QREncode renders text as a QR Code symbol and returns it as a single-channel grayscale `cv.Mat` (dark modules are 0, light modules are 255) with a… |
| `func QREncodeAdvanced(text string, version int, level QRECCLevel) (*cv.Mat, error)` | QREncodeAdvanced renders text as a QR Code symbol and returns it as a single-channel grayscale `cv.Mat` (dark modules 0, light 255) with a 4-module… |
| `func QRModuleCount(version int) int` | QRModuleCount returns the number of modules along one side of a QR symbol of the given version (identical to `QRSizeForVersion`); it returns 0 for… |
| `func QRRawDataModuleCount(version int) int` | QRRawDataModuleCount returns the number of data-carrying modules of a QR symbol of the given version: the total module count minus the function… |
| `func QRSizeForVersion(version int) int` | QRSizeForVersion returns the side length in modules of a QR symbol of the given version, which is 17+4*version. |
| `func QRVersionForSize(size int) (int, bool)` | QRVersionForSize returns the version of a square QR symbol with the given side length in modules. |
| `func ReedSolomonDecode(msg []byte, nsym int) ([]byte, bool)` | ReedSolomonDecode takes a received codeword (data codewords followed by nsym EC codewords) and attempts to correct up to nsym/2 byte errors. |
| `func ReedSolomonEncode(data []byte, nsym int) []byte` | ReedSolomonEncode computes nsym Reed-Solomon error-correction codewords for the given data codewords and returns them (length nsym). |
| `func SampleQRModules(img *cv.Mat, version int) ([][]bool, bool)` | SampleQRModules samples a rendered QR image into its version*version module matrix, where a true element denotes a dark module. |
| `func UPCACheckDigit(payload string) (int, error)` | UPCACheckDigit returns the modulo-10 check digit of an 11-digit UPC-A payload. |
| `func UPCAToEAN13(upca string) (string, error)` | UPCAToEAN13 embeds a 12-digit UPC-A code into the 13-digit EAN-13 space by prefixing a leading zero. |
| `func UPCAToUPCE(upca string) (string, error)` | UPCAToUPCE compresses a 12-digit UPC-A code into its 8-digit UPC-E form when the code is compressible (its number system is 0 or 1 and its… |
| `func UPCEToUPCA(upce string) (string, error)` | UPCEToUPCA expands an 8-digit UPC-E code (a number-system digit of 0 or 1, six data digits and a check digit) into its equivalent 12-digit UPC-A code. |
| `func ValidateEAN13(code string) bool` | ValidateEAN13 reports whether a full 13-digit EAN-13 code has a correct check digit. |
| `func ValidateEAN8(code string) bool` | ValidateEAN8 reports whether a full 8-digit EAN-8 code has a correct check digit. |
| `func ValidateITF14(code string) bool` | ValidateITF14 reports whether a full 14-digit ITF-14 code has a correct check digit. |
| `func ValidateLuhn(code string) bool` | ValidateLuhn reports whether a full numeric string (payload followed by its Luhn check digit) is self-consistent. |
| `func ValidateUPCA(code string) bool` | ValidateUPCA reports whether a full 12-digit UPC-A code has a correct check digit. |

### Types

| Type | What it is |
| --- | --- |
| `Barcode1D` | Barcode1D is a single decoded 1D barcode: its symbology name and text. |
| `DataMatrixCorner` | DataMatrixCorner identifies which corner of a Data Matrix symbol holds the solid "L" finder pattern. |
| `DataMatrixFinder` | DataMatrixFinder describes a detected Data Matrix finder pattern: the symbol's bounding box in pixel coordinates, which corner carries the solid L,… |
| `QRECCLevel` | QRECCLevel names a QR error-correction level. |
| `QRFormatInfo` | QRFormatInfo is the decoded 5-bit QR format information: the error-correction level and the data-mask pattern number (0-7). |

<details>
<summary><code>Barcode1D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DetectAndDecodeMulti(img *cv.Mat) []Barcode1D` | DetectAndDecodeMulti scans img for multiple 1D barcodes and returns every distinct symbology/text it can decode. |

</details>

<details>
<summary><code>DataMatrixCorner</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c DataMatrixCorner) String() string` | String returns a human-readable name for the corner. |

</details>

<details>
<summary><code>DataMatrixFinder</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FindDataMatrixFinder(img *cv.Mat) (DataMatrixFinder, bool)` | FindDataMatrixFinder locates the solid "L" finder pattern of a Data Matrix symbol in img. |
| `func (f DataMatrixFinder) Height() int` | Height returns the symbol's pixel height (MaxY-MinY+1). |
| `func (f DataMatrixFinder) Width() int` | Width returns the symbol's pixel width (MaxX-MinX+1). |

</details>

<details>
<summary><code>QRECCLevel</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (l QRECCLevel) String() string` | String returns the single-letter name of the error-correction level ("L", "M", "Q" or "H"). |

</details>

<details>
<summary><code>QRFormatInfo</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DecodeQRFormatInfo(raw int) (QRFormatInfo, bool)` | DecodeQRFormatInfo recovers the error-correction level and mask from a 15-bit format-information value, tolerating up to three bit errors by choosing… |
| `func (f QRFormatInfo) String() string` | String renders the format information as e.g. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/barcode).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.

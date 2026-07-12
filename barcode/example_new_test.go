package barcode_test

import (
	"fmt"

	"github.com/malcolmston/opencv/barcode"
)

// ExampleQREncodeAdvanced shows a version-7, level-Q QR round trip carrying a
// mixed-case byte-mode payload.
func ExampleQREncodeAdvanced() {
	img, err := barcode.QREncodeAdvanced("Hello, Advanced QR!", 0, barcode.QRECCQuartile)
	if err != nil {
		fmt.Println(err)
		return
	}
	text, ok := barcode.QRDetectAndDecodeAdvanced(img)
	fmt.Println(text, ok)
	// Output: Hello, Advanced QR! true
}

// ExampleQREncodeAdvanced_numeric shows the encoder automatically selecting the
// compact numeric mode for an all-digit payload.
func ExampleQREncodeAdvanced_numeric() {
	img, _ := barcode.QREncodeAdvanced("8675309", 0, barcode.QRECCMedium)
	text, ok := barcode.QRDetectAndDecodeAdvanced(img)
	fmt.Println(text, ok)
	// Output: 8675309 true
}

// ExampleEncodeCode39 shows a Code 39 round trip.
func ExampleEncodeCode39() {
	img, _ := barcode.EncodeCode39("CODE 39")
	text, ok := barcode.DecodeCode39(img)
	fmt.Println(text, ok)
	// Output: CODE 39 true
}

// ExampleEncodeUPCA shows a UPC-A round trip; the check digit is appended to the
// 11 supplied digits.
func ExampleEncodeUPCA() {
	img, _ := barcode.EncodeUPCA("03600029145")
	digits, ok := barcode.DecodeUPCA(img)
	fmt.Println(digits, ok)
	// Output: 036000291452 true
}

// ExampleEncodeEAN8 shows an EAN-8 round trip.
func ExampleEncodeEAN8() {
	img, _ := barcode.EncodeEAN8("9638507")
	digits, ok := barcode.DecodeEAN8(img)
	fmt.Println(digits, ok)
	// Output: 96385074 true
}

// ExampleEncodeITF shows an Interleaved 2 of 5 round trip.
func ExampleEncodeITF() {
	img, _ := barcode.EncodeITF("1234567890")
	digits, ok := barcode.DecodeITF(img)
	fmt.Println(digits, ok)
	// Output: 1234567890 true
}

// ExampleEncodeCodabar shows a Codabar round trip including its A/B framing.
func ExampleEncodeCodabar() {
	img, _ := barcode.EncodeCodabar("A1234B")
	text, ok := barcode.DecodeCodabar(img)
	fmt.Println(text, ok)
	// Output: A1234B true
}

// ExampleEncodeCode93 shows a Code 93 round trip.
func ExampleEncodeCode93() {
	img, _ := barcode.EncodeCode93("CODE 93")
	text, ok := barcode.DecodeCode93(img)
	fmt.Println(text, ok)
	// Output: CODE 93 true
}

// ExampleDetectAndDecodeMulti decodes a single 1D barcode from an image; the
// return value is a slice so that stacked barcodes can all be reported.
func ExampleDetectAndDecodeMulti() {
	img, _ := barcode.EncodeCode39("SHIP-42")
	for _, r := range barcode.DetectAndDecodeMulti(img) {
		fmt.Printf("%s %s\n", r.Type, r.Text)
	}
	// Output: Code 39 SHIP-42
}

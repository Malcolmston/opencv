package barcode_test

import (
	"fmt"

	"github.com/malcolmston/opencv/barcode"
)

// ExampleQREncode shows a QR round trip: encode a string, then detect and decode
// it from the rendered image.
func ExampleQREncode() {
	img := barcode.QREncode("HELLO", 0)
	text, ok := barcode.QRDetectAndDecode(img)
	fmt.Println(text, ok)
	// Output: HELLO true
}

// ExampleFindFinderPatterns locates the three finder patterns of a QR symbol.
func ExampleFindFinderPatterns() {
	img := barcode.QREncode("HELLO", 1)
	points := barcode.FindFinderPatterns(img)
	fmt.Println(len(points))
	// Output: 3
}

// ExampleEncodeEAN13 shows an EAN-13 round trip; the check digit is appended to
// the 12 supplied digits.
func ExampleEncodeEAN13() {
	img, err := barcode.EncodeEAN13("590123412345")
	if err != nil {
		fmt.Println(err)
		return
	}
	digits, ok := barcode.DecodeEAN13(img)
	fmt.Println(digits, ok)
	// Output: 5901234123457 true
}

// ExampleEncodeCode128 shows a Code 128 (set B) round trip.
func ExampleEncodeCode128() {
	img, err := barcode.EncodeCode128("Order #42")
	if err != nil {
		fmt.Println(err)
		return
	}
	text, ok := barcode.DecodeCode128(img)
	fmt.Println(text, ok)
	// Output: Order #42 true
}

// ExampleReedSolomonDecode corrects a corrupted codeword within the error budget.
func ExampleReedSolomonDecode() {
	data := []byte("data")
	nsym := 4 // corrects up to 2 byte errors
	msg := append(append([]byte{}, data...), barcode.ReedSolomonEncode(data, nsym)...)
	msg[0] ^= 0xFF // introduce one error
	fixed, ok := barcode.ReedSolomonDecode(msg, nsym)
	fmt.Println(string(fixed[:len(data)]), ok)
	// Output: data true
}

package barcode

import (
	cv "github.com/malcolmston/opencv"
)

// This file provides DetectAndDecodeMulti, which reads potentially several 1D
// barcodes from a single image by scanning multiple horizontal bands and trying
// every linear decoder in the package on each. It is useful for label sheets
// that stack barcodes vertically, where a single middle-row scan would only see
// one of them.

// Barcode1D is a single decoded 1D barcode: its symbology name and text.
type Barcode1D struct {
	// Type is the symbology name, e.g. "EAN-13", "Code 128", "UPC-A".
	Type string
	// Text is the decoded payload.
	Text string
}

// oneDDecoder pairs a symbology name with its whole-image decode function.
type oneDDecoder struct {
	name   string
	decode func(*cv.Mat) (string, bool)
}

// oneDDecoders lists every linear decoder tried by DetectAndDecodeMulti. Each
// decoder validates guard patterns and/or check digits, so false positives
// across symbologies are highly unlikely.
var oneDDecoders = []oneDDecoder{
	{"EAN-13", DecodeEAN13},
	{"UPC-A", DecodeUPCA},
	{"EAN-8", DecodeEAN8},
	{"Code 128", DecodeCode128},
	{"Code 93", DecodeCode93},
	{"Code 39", DecodeCode39},
	{"Codabar", DecodeCodabar},
	{"ITF", DecodeITF},
	{"Code 11", DecodeCode11},
	{"MSI", DecodeMSI},
}

// DetectAndDecodeMulti scans img for multiple 1D barcodes and returns every
// distinct symbology/text it can decode. It splits the image into overlapping
// horizontal bands (plus the whole image) and runs each linear decoder on every
// band, so barcodes stacked vertically are all found. Results are de-duplicated
// on (Type, Text) and returned in a deterministic order. An image with no
// decodable 1D barcode yields an empty slice.
func DetectAndDecodeMulti(img *cv.Mat) []Barcode1D {
	if img == nil || img.Empty() {
		return nil
	}
	var results []Barcode1D
	seen := map[string]bool{}
	add := func(name, text string) {
		key := name + "\x00" + text
		if seen[key] {
			return
		}
		seen[key] = true
		results = append(results, Barcode1D{Type: name, Text: text})
	}

	// Candidate scan images: the whole image, then a set of horizontal bands.
	// Bands overlap so a barcode straddling a band boundary still centres in at
	// least one band.
	candidates := []*cv.Mat{img}
	const bands = 6
	if img.Rows >= 2*bands {
		bandH := img.Rows / bands
		for b := 0; b < bands; b++ {
			y := b * bandH
			h := bandH
			if b == bands-1 {
				h = img.Rows - y
			}
			candidates = append(candidates, img.Region(y, 0, h, img.Cols))
		}
		// Half-offset bands to catch barcodes centred on a boundary.
		for b := 0; b < bands-1; b++ {
			y := b*bandH + bandH/2
			candidates = append(candidates, img.Region(y, 0, bandH, img.Cols))
		}
	}

	for _, cand := range candidates {
		for _, d := range oneDDecoders {
			if text, ok := d.decode(cand); ok {
				add(d.name, text)
			}
		}
	}
	return results
}

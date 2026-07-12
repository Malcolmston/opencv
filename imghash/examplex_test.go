package imghash_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/imghash"
)

// ExampleWaveletHash shows that the Haar-wavelet hash of an image matches a copy
// of itself exactly (Hamming distance zero).
func ExampleWaveletHash() {
	img := photo(64, 64)
	h := imghash.NewWaveletHash()
	a := h.Compute(img)
	b := h.Compute(img.Clone())
	fmt.Printf("bytes=%d distance=%.0f\n", len(a), h.Compare(a, b))
	// Output: bytes=8 distance=0
}

// ExampleHaarDWT2D demonstrates the exact invertibility of the 2-D Haar
// transform: forward then inverse reconstructs the input.
func ExampleHaarDWT2D() {
	in := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	fwd := imghash.HaarDWT2D(in, 4)
	back := imghash.HaarIDWT2D(fwd, 4)
	fmt.Printf("%.0f %.0f %.0f\n", back[0], back[5], back[15])
	// Output: 1 6 16
}

// ExampleMarrHildreth72 shows the 72-bit multi-scale, multi-orientation
// descriptor is nine bytes long and identical for a copy of the same image.
func ExampleMarrHildreth72() {
	img := photo(64, 64)
	h := imghash.NewMarrHildrethHash72()
	a := h.Compute(img)
	fmt.Printf("bytes=%d distance=%.0f\n", len(a), h.Compare(a, imghash.MarrHildreth72(img.Clone())))
	// Output: bytes=9 distance=0
}

// ExampleHexEncode round-trips a hash through its hexadecimal text form.
func ExampleHexEncode() {
	img := photo(32, 32)
	h := imghash.Perceptual(img)
	s := imghash.HexEncode(h)
	back, _ := imghash.HexDecode(s)
	fmt.Printf("len=%d match=%v\n", len(s), string(back) == string(h))
	// Output: len=16 match=true
}

// ExampleSimilarity reports that an image is perfectly similar to itself.
func ExampleSimilarity() {
	img := photo(32, 32)
	h := imghash.Average(img)
	fmt.Printf("%.1f\n", imghash.Similarity(h, h))
	// Output: 1.0
}

// ExampleIsDuplicate flags a blurred copy as a near-duplicate but a
// checkerboard as distinct, both against a difference hash.
func ExampleIsDuplicate() {
	base := photo(64, 64)
	blurred := cv.GaussianBlur(base, 5, 0)

	checker := cv.NewMat(64, 64, 1)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			if ((x/8)+(y/8))%2 == 0 {
				checker.Set(y, x, 0, 255)
			}
		}
	}

	hb := imghash.Difference(base)
	fmt.Println(
		imghash.IsDuplicate(hb, imghash.Difference(blurred), 0.2),
		imghash.IsDuplicate(hb, imghash.Difference(checker), 0.2),
	)
	// Output: true false
}

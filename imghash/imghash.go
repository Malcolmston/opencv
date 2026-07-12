package imghash

import (
	"fmt"
	"math/bits"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// ImgHash is the common interface implemented by every perceptual hash in this
// package, mirroring OpenCV's cv::img_hash::ImgHashBase. A hasher reduces an
// image to a short, fixed-length fingerprint whose bytes can be stored and
// compared cheaply:
//
//	h := imghash.NewPHash()
//	a := h.Compute(imgA)
//	b := h.Compute(imgB)
//	dist := h.Compare(a, b) // small => perceptually similar
//
// Compute returns the fingerprint of img as a fresh byte slice; the length is
// fixed for a given hasher. Compare returns the distance between two
// fingerprints produced by the same hasher — smaller means more similar, and
// two fingerprints of identical images always compare as 0. The distance is a
// Hamming distance (number of differing bits) for the binary hashes and an L1
// distance for the real-valued hashes ([RadialVarianceHash], [ColorMomentHash]).
// Compare panics if the two slices have different lengths, which indicates they
// came from different hashers.
type ImgHash interface {
	// Compute returns the perceptual hash of img as a fresh byte slice.
	Compute(img *cv.Mat) []byte
	// Compare returns the distance between two hashes from this hasher.
	Compare(a, b []byte) float64
}

// requireImage panics if img is nil or empty, matching the fail-fast behaviour
// of the root package's pixel helpers.
func requireImage(img *cv.Mat, name string) {
	if img == nil || img.Empty() {
		panic(fmt.Sprintf("imghash: %s requires a non-empty image", name))
	}
}

// requireSameLen panics if the two hashes differ in length.
func requireSameLen(a, b []byte, name string) {
	if len(a) != len(b) {
		panic(fmt.Sprintf("imghash: %s requires equal-length hashes, got %d and %d", name, len(a), len(b)))
	}
}

// toGray returns a single-channel view of img. A one-channel image is returned
// unchanged; a three-channel image is reduced with the BT.601 luma weights via
// [cv.CvtColor]; any other channel count is averaged across channels. The
// result never aliases a multi-channel input.
func toGray(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return img
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		gray := cv.NewMat(img.Rows, img.Cols, 1)
		for p := 0; p < img.Total(); p++ {
			var sum int
			base := p * img.Channels
			for c := 0; c < img.Channels; c++ {
				sum += int(img.Data[base+c])
			}
			gray.Data[p] = uint8((sum + img.Channels/2) / img.Channels)
		}
		return gray
	}
}

// grayResize converts img to gray and resizes it to width×height with bilinear
// interpolation, the common front end of every hasher.
func grayResize(img *cv.Mat, width, height int) *cv.Mat {
	return cv.Resize(toGray(img), width, height, cv.InterLinear)
}

// packBits packs bits most-significant-bit first into bytes. Bit i lands in
// byte i/8 at position 7-(i%8). The output length is ceil(len(bits)/8).
func packBits(bitsIn []bool) []byte {
	out := make([]byte, (len(bitsIn)+7)/8)
	for i, b := range bitsIn {
		if b {
			out[i/8] |= 1 << uint(7-(i%8))
		}
	}
	return out
}

// hamming returns the number of differing bits between two equal-length byte
// slices.
func hamming(a, b []byte) int {
	dist := 0
	for i := range a {
		dist += bits.OnesCount8(a[i] ^ b[i])
	}
	return dist
}

// l1 returns the sum of absolute per-byte differences of two equal-length byte
// slices, the L1 distance used by the real-valued hashes.
func l1(a, b []byte) float64 {
	var sum float64
	for i := range a {
		d := int(a[i]) - int(b[i])
		if d < 0 {
			d = -d
		}
		sum += float64(d)
	}
	return sum
}

// median returns the median of vals. The input is copied, so the caller's slice
// is left unmodified. It returns 0 for an empty slice.
func median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := make([]float64, len(vals))
	copy(cp, vals)
	sort.Float64s(cp)
	n := len(cp)
	if n%2 == 1 {
		return cp[n/2]
	}
	return (cp[n/2-1] + cp[n/2]) / 2
}

// mean returns the arithmetic mean of vals, or 0 for an empty slice.
func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

package xfeatures2d

import (
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// BRIEF computes the Binary Robust Independent Elementary Features descriptor,
// a port of OpenCV's cv::xfeatures2d::BriefDescriptorExtractor.
//
// BRIEF describes the neighbourhood of a keypoint by comparing the smoothed
// intensities of a fixed set of point pairs sampled inside a square patch: bit
// b is 1 when the first point of pair b is darker than the second. The pattern
// is generated once with an isotropic-Gaussian random layout from a fixed seed,
// so the descriptor is fully deterministic. Like the original, BRIEF is not
// rotation invariant; feed it upright keypoints (or use [LATCH] or [FREAK] for
// a rotation-invariant binary descriptor).
//
// Two descriptors are compared with the [HammingDistance].
type BRIEF struct {
	// PatchSize is the side length in pixels of the square sampling patch.
	PatchSize int
	// Sigma is the standard deviation of the Gaussian pre-smoothing applied
	// before sampling, which makes the individual pixel tests less noise
	// sensitive.
	Sigma float64

	bytes int
	pairs [][4]float64 // x1,y1,x2,y2 offsets from the keypoint centre
}

// NewBRIEF returns a BRIEF extractor producing descriptors of the given byte
// length (typically 16, 32 or 64, i.e. 128, 256 or 512 bits). It panics if
// bytes is not positive.
func NewBRIEF(bytes int) *BRIEF {
	if bytes <= 0 {
		panic("xfeatures2d: NewBRIEF requires a positive byte length")
	}
	b := &BRIEF{PatchSize: 48, Sigma: 2.0, bytes: bytes}
	b.buildPattern(bytes * 8)
	return b
}

// buildPattern samples nbits point pairs inside the patch using an
// isotropic-Gaussian layout (standard deviation PatchSize/5) with a fixed seed.
func (b *BRIEF) buildPattern(nbits int) {
	rng := rand.New(rand.NewSource(int64(0x5eed1234) ^ int64(b.PatchSize)))
	sigma := float64(b.PatchSize) / 5.0
	half := float64(b.PatchSize) / 2
	clamp := func(v float64) float64 {
		if v > half {
			return half
		}
		if v < -half {
			return -half
		}
		return v
	}
	b.pairs = make([][4]float64, nbits)
	for i := 0; i < nbits; i++ {
		b.pairs[i] = [4]float64{
			clamp(rng.NormFloat64() * sigma),
			clamp(rng.NormFloat64() * sigma),
			clamp(rng.NormFloat64() * sigma),
			clamp(rng.NormFloat64() * sigma),
		}
	}
}

// DescriptorSizeBytes returns the number of bytes in each descriptor.
func (b *BRIEF) DescriptorSizeBytes() int { return b.bytes }

// Compute describes each keypoint of img and returns the keypoints unchanged
// together with their bit-packed descriptors (one []byte of length
// DescriptorSizeBytes per keypoint). The image is Gaussian pre-smoothed and
// sampled with border replication, so no keypoint is dropped. img may be
// single- or three-channel; a colour image is converted to gray.
func (b *BRIEF) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte) {
	gray := toGray(img)
	if b.Sigma > 0 {
		gray = cv.GaussianBlur(gray, gaussianKSize(b.Sigma), b.Sigma)
	}
	out := make([]KeyPoint, len(keypoints))
	descs := make([][]byte, len(keypoints))
	for k, kp := range keypoints {
		desc := make([]byte, b.bytes)
		fx := float64(kp.Pt.X)
		fy := float64(kp.Pt.Y)
		for bit, p := range b.pairs {
			v1 := bilinear(gray, fx+p[0], fy+p[1])
			v2 := bilinear(gray, fx+p[2], fy+p[3])
			if v1 < v2 {
				packBit(desc, bit)
			}
		}
		out[k] = kp
		descs[k] = desc
	}
	return out, descs
}

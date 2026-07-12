package features2d

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// Descriptor geometry shared by BRIEF and ORB. NumBits binary tests are packed
// into NumBits/8 bytes. The sampling points are drawn from an isotropic Gaussian
// inside a PatchSize×PatchSize window centred on the keypoint.
const (
	defaultPatchSize = 31
	defaultNumBits   = 256
	// patternSeed fixes the pseudo-random sampling pattern so descriptors are
	// deterministic and comparable across runs.
	patternSeed = 0x5eed2d
)

// testPair is one BRIEF intensity comparison: the descriptor bit is set when the
// smoothed sample at (Ax, Ay) is less than the sample at (Bx, By). Coordinates
// are relative to the keypoint centre.
type testPair struct {
	ax, ay float64
	bx, by float64
}

// buildPattern generates numBits deterministic test pairs whose coordinates are
// Gaussian-distributed within the patch, matching the classic BRIEF sampling
// scheme. The pattern is a package-level constant of the algorithm, not tied to
// any image.
func buildPattern(numBits, patchSize int) []testPair {
	half := float64(patchSize) / 2
	sigma := float64(patchSize) / 5.0
	rng := rand.New(rand.NewSource(patternSeed))
	sample := func() float64 {
		v := rng.NormFloat64() * sigma
		if v > half {
			v = half
		} else if v < -half {
			v = -half
		}
		return v
	}
	pairs := make([]testPair, numBits)
	for i := range pairs {
		pairs[i] = testPair{
			ax: sample(), ay: sample(),
			bx: sample(), by: sample(),
		}
	}
	return pairs
}

// defaultPattern is the shared BRIEF/ORB sampling pattern, built once.
var defaultPattern = buildPattern(defaultNumBits, defaultPatchSize)

// BRIEF computes a Binary Robust Independent Elementary Features descriptor.
// It does not detect keypoints; pass keypoints from a detector such as
// cv.FASTCorners. Each descriptor is a fixed set of intensity comparisons on a
// Gaussian-smoothed copy of the image, packed into bytes.
//
// The zero value is usable and applies the defaults (a 31-pixel patch and 256
// bits, i.e. 32-byte descriptors); construct a customised instance with
// [NewBRIEF].
type BRIEF struct {
	// PatchSize is the side length of the square sampling window in pixels. When
	// zero the default (31) is used.
	PatchSize int
	// BlurSigma is the standard deviation of the Gaussian pre-smoothing. When
	// zero it is derived from the 5×5 kernel (OpenCV's rule).
	BlurSigma float64
	// pattern is the (possibly customised) set of test pairs; nil means use
	// defaultPattern.
	pattern []testPair
}

// NewBRIEF returns a BRIEF descriptor extractor with the given patch size and a
// freshly generated deterministic pattern of numBits tests. numBits must be a
// positive multiple of 8.
func NewBRIEF(patchSize, numBits int) *BRIEF {
	if patchSize <= 0 {
		patchSize = defaultPatchSize
	}
	if numBits <= 0 || numBits%8 != 0 {
		panic("features2d: NewBRIEF numBits must be a positive multiple of 8")
	}
	return &BRIEF{PatchSize: patchSize, pattern: buildPattern(numBits, patchSize)}
}

// pat returns the effective sampling pattern.
func (b *BRIEF) pat() []testPair {
	if b.pattern != nil {
		return b.pattern
	}
	return defaultPattern
}

// patchSize returns the effective patch size.
func (b *BRIEF) patchSize() int {
	if b.PatchSize > 0 {
		return b.PatchSize
	}
	return defaultPatchSize
}

// Compute describes the given keypoints on img and returns the surviving
// keypoints together with their bit-packed descriptors, one row per keypoint.
// Keypoints whose orientation is set (Angle >= 0) have their sampling pattern
// steered by that angle; otherwise the unrotated pattern is used. The image may
// be single- or three-channel (it is converted to grayscale). Every keypoint is
// described (border samples are clamped), so the returned slices are parallel
// and have the same length as kps.
func (b *BRIEF) Compute(img *cv.Mat, kps []KeyPoint) ([]KeyPoint, [][]byte) {
	gray := toGray(img)
	smoothed := cv.GaussianBlur(gray, 5, b.BlurSigma)
	pattern := b.pat()
	out := make([][]byte, len(kps))
	for i, kp := range kps {
		out[i] = describeKeypoint(smoothed, kp, pattern)
	}
	kpsCopy := make([]KeyPoint, len(kps))
	copy(kpsCopy, kps)
	return kpsCopy, out
}

// describeKeypoint computes one bit-packed descriptor for kp by evaluating the
// pattern's test pairs on the smoothed image, steering the pattern by the
// keypoint angle when it is set.
func describeKeypoint(smoothed *cv.Mat, kp KeyPoint, pattern []testPair) []byte {
	cx, cy := kp.Pt.X, kp.Pt.Y
	cosA, sinA := 1.0, 0.0
	if kp.Angle >= 0 {
		rad := kp.Angle * math.Pi / 180
		cosA, sinA = math.Cos(rad), math.Sin(rad)
	}
	desc := make([]byte, (len(pattern)+7)/8)
	for i, tp := range pattern {
		ax := cx + int(math.Round(tp.ax*cosA-tp.ay*sinA))
		ay := cy + int(math.Round(tp.ax*sinA+tp.ay*cosA))
		bx := cx + int(math.Round(tp.bx*cosA-tp.by*sinA))
		by := cy + int(math.Round(tp.bx*sinA+tp.by*cosA))
		if sampleClamped(smoothed, ax, ay) < sampleClamped(smoothed, bx, by) {
			desc[i/8] |= 1 << uint(i%8)
		}
	}
	return desc
}

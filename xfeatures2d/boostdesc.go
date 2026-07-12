package xfeatures2d

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// BoostDesc computes a weight-free boosted gradient binary descriptor in the
// spirit of OpenCV's cv::xfeatures2d::BoostDesc.
//
// The original BoostDesc selects, by boosting, a set of gradient-based weak
// learners; each learner pools an oriented image-gradient response over a box
// region. This port keeps that weak-learner form: every descriptor bit compares
// the mean gradient magnitude of two small boxes (read in constant time from an
// integral image of the gradient magnitude), setting the bit when the first is
// larger. The set of boxes is a fixed pseudo-random arrangement rather than a
// learned one, and no per-bit weights are used (documented as the weight-free
// approximation, so no trained tables are embedded). The pattern is rotated to
// the keypoint orientation, giving rotation invariance.
//
// Two descriptors are compared with the [HammingDistance].
type BoostDesc struct {
	// HalfBox is the half side of each pooling box.
	HalfBox int
	// Window is the half extent within which box centres are placed.
	Window int

	bytes   int
	entries []boxPairEntry
}

// NewBoostDesc returns a BoostDesc extractor producing descriptors of the given
// byte length (typically 32 or 64). It panics if bytes is not positive.
func NewBoostDesc(bytes int) *BoostDesc {
	if bytes <= 0 {
		panic("xfeatures2d: NewBoostDesc requires a positive byte length")
	}
	b := &BoostDesc{HalfBox: 2, Window: 16, bytes: bytes}
	rng := rand.New(rand.NewSource(int64(0xb005de5c)))
	rc := func() int { return rng.Intn(2*b.Window+1) - b.Window }
	b.entries = make([]boxPairEntry, bytes*8)
	for i := range b.entries {
		b.entries[i] = boxPairEntry{rc(), rc(), rc(), rc()}
	}
	return b
}

// DescriptorSizeBytes returns the number of bytes in each descriptor.
func (b *BoostDesc) DescriptorSizeBytes() int { return b.bytes }

// Compute describes each keypoint of img and returns the keypoints (with Angle
// set to the intensity-centroid orientation) and their bit-packed descriptors.
// Sampling uses border replication, so no keypoint is dropped. img may be
// single- or three-channel; a colour image is converted to gray.
func (b *BoostDesc) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte) {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	gx, gy := gradientMaps(gray)
	mag := make([]float64, rows*cols)
	for i := range mag {
		mag[i] = math.Hypot(gx[i], gy[i])
	}
	fi := newFloatIntegral(mag, rows, cols)

	out := make([]KeyPoint, len(keypoints))
	descs := make([][]byte, len(keypoints))
	radius := b.Window + b.HalfBox

	for k, kp := range keypoints {
		fx := float64(kp.Pt.X)
		fy := float64(kp.Pt.Y)
		angle := intensityCentroidAngle(gray, kp.Pt.X, kp.Pt.Y, radius)
		ca, sa := math.Cos(angle), math.Sin(angle)
		desc := make([]byte, b.bytes)
		for bit, e := range b.entries {
			arx := float64(e.ax)*ca - float64(e.ay)*sa
			ary := float64(e.ax)*sa + float64(e.ay)*ca
			brx := float64(e.bx)*ca - float64(e.by)*sa
			bry := float64(e.bx)*sa + float64(e.by)*ca
			ma := fi.boxMean(int(math.Round(fx+arx)), int(math.Round(fy+ary)), b.HalfBox, b.HalfBox)
			mb := fi.boxMean(int(math.Round(fx+brx)), int(math.Round(fy+bry)), b.HalfBox, b.HalfBox)
			if ma > mb {
				packBit(desc, bit)
			}
		}
		deg := angle * 180 / math.Pi
		if deg < 0 {
			deg += 360
		}
		kp.Angle = deg
		out[k] = kp
		descs[k] = desc
	}
	return out, descs
}

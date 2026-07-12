package xfeatures2d

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// boxPairEntry is one average-box weak learner: the offsets (from the keypoint)
// of the centres of the two boxes whose mean intensities are compared.
type boxPairEntry struct {
	ax, ay, bx, by int
}

// boxPairPattern is a seeded arrangement of average-box weak learners shared by
// [BEBLID] and [TEBLID]. Each learner compares the mean intensity of two
// equally sized square boxes.
type boxPairPattern struct {
	half    int // box half-size
	window  int // half extent within which box centres are placed
	entries []boxPairEntry
}

// newBoxPairPattern samples nbits average-box learners with a fixed seed.
func newBoxPairPattern(nbits, window, half int, seed int64) *boxPairPattern {
	rng := rand.New(rand.NewSource(seed ^ int64(window) ^ int64(half<<8)))
	rc := func() int { return rng.Intn(2*window+1) - window }
	p := &boxPairPattern{half: half, window: window}
	p.entries = make([]boxPairEntry, nbits)
	for i := 0; i < nbits; i++ {
		p.entries[i] = boxPairEntry{rc(), rc(), rc(), rc()}
	}
	return p
}

// compute packs one descriptor per keypoint by thresholding, for every learner,
// the difference of the two box means at zero (the weight-free BEBLID/BAD
// response). When rotationInvariant is set the box centres are rotated by the
// keypoint orientation (its Angle if >= 0, else the intensity-centroid angle).
func (p *boxPairPattern) compute(gray *cv.Mat, keypoints []KeyPoint, bytes int, rotationInvariant bool) ([]KeyPoint, [][]byte) {
	it := newIntegral(gray)
	out := make([]KeyPoint, len(keypoints))
	descs := make([][]byte, len(keypoints))
	radius := p.window + p.half

	boxMeanAt := func(cx, cy float64) float64 {
		return it.boxMean(int(math.Round(cx)), int(math.Round(cy)), p.half)
	}

	for k, kp := range keypoints {
		fx := float64(kp.Pt.X)
		fy := float64(kp.Pt.Y)
		angle := 0.0
		if rotationInvariant {
			if kp.Angle >= 0 {
				angle = kp.Angle * math.Pi / 180
			} else {
				angle = intensityCentroidAngle(gray, kp.Pt.X, kp.Pt.Y, radius)
			}
		}
		ca, sa := math.Cos(angle), math.Sin(angle)
		desc := make([]byte, bytes)
		for bit, e := range p.entries {
			arx := float64(e.ax)*ca - float64(e.ay)*sa
			ary := float64(e.ax)*sa + float64(e.ay)*ca
			brx := float64(e.bx)*ca - float64(e.by)*sa
			bry := float64(e.bx)*sa + float64(e.by)*ca
			ma := boxMeanAt(fx+arx, fy+ary)
			mb := boxMeanAt(fx+brx, fy+bry)
			if ma-mb > 0 {
				packBit(desc, bit)
			}
		}
		if rotationInvariant {
			deg := angle * 180 / math.Pi
			if deg < 0 {
				deg += 360
			}
			kp.Angle = deg
		}
		out[k] = kp
		descs[k] = desc
	}
	return out, descs
}

// BEBLID computes the Boosted Efficient Binary Local Image Descriptor, a
// weight-free port of OpenCV's cv::xfeatures2d::BEBLID.
//
// Each descriptor bit is an average-box weak learner: the mean intensity of one
// small box is compared with that of another box elsewhere in the patch, and
// the bit is 1 when the first mean is larger. Box means are read in constant
// time through an integral image. OpenCV ships boxes and per-bit thresholds
// learned by boosting on a labelled patch set; this port keeps the identical
// weak-learner form but uses a fixed pseudo-random box arrangement and a zero
// threshold (documented as the untrained, weight-free approximation), so no
// training tables are embedded. The pattern is rotated to the keypoint
// orientation, giving rotation invariance.
//
// Two descriptors are compared with the [HammingDistance].
type BEBLID struct {
	pattern *boxPairPattern
	bytes   int
}

// NewBEBLID returns a BEBLID extractor producing descriptors of the given byte
// length (typically 32 or 64). It panics if bytes is not positive.
func NewBEBLID(bytes int) *BEBLID {
	if bytes <= 0 {
		panic("xfeatures2d: NewBEBLID requires a positive byte length")
	}
	return &BEBLID{
		pattern: newBoxPairPattern(bytes*8, 16, 3, 0x8eb11d),
		bytes:   bytes,
	}
}

// DescriptorSizeBytes returns the number of bytes in each descriptor.
func (b *BEBLID) DescriptorSizeBytes() int { return b.bytes }

// Compute describes each keypoint of img and returns the keypoints (with Angle
// set to the estimated orientation) and their bit-packed descriptors. img may
// be single- or three-channel; a colour image is converted to gray.
func (b *BEBLID) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte) {
	return b.pattern.compute(toGray(img), keypoints, b.bytes, true)
}

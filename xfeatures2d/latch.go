package xfeatures2d

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// latchTriplet is one LATCH comparison: an anchor mini-patch and two companion
// mini-patches, each given by the centre offset from the keypoint.
type latchTriplet struct {
	ax, ay int
	bx, by int
	cx, cy int
}

// LATCH computes the Learned Arrangements of Three patCH codes binary
// descriptor, a port of OpenCV's cv::xfeatures2d::LATCH.
//
// Instead of comparing single pixels like BRIEF, each LATCH bit compares three
// small square patches (an anchor and two companions): the bit is 1 when the
// anchor is closer, in sum-of-squared-differences, to the first companion than
// to the second. Using patches rather than pixels makes the bits markedly more
// noise tolerant. The triplet arrangement is generated once from a fixed seed,
// so the descriptor is deterministic. When a keypoint carries an orientation
// (Angle >= 0) or RotationInvariant is set, the triplet offsets are rotated by
// the keypoint's intensity-centroid orientation, giving rotation invariance.
//
// Two descriptors are compared with the [HammingDistance].
type LATCH struct {
	// HalfPatch is the half side length of each mini-patch (patch side is
	// 2*HalfPatch+1).
	HalfPatch int
	// Window is the half extent within which triplet patch centres are sampled.
	Window int
	// RotationInvariant, when true, rotates the pattern by each keypoint's
	// intensity-centroid orientation before sampling.
	RotationInvariant bool

	bytes    int
	triplets []latchTriplet
}

// NewLATCH returns a LATCH extractor producing descriptors of the given byte
// length (typically 1, 2, 4, 8, 16 or 32 bytes). It panics if bytes is not
// positive.
func NewLATCH(bytes int) *LATCH {
	if bytes <= 0 {
		panic("xfeatures2d: NewLATCH requires a positive byte length")
	}
	l := &LATCH{HalfPatch: 3, Window: 20, RotationInvariant: true, bytes: bytes}
	l.buildPattern(bytes * 8)
	return l
}

// buildPattern samples the triplet arrangement with a fixed seed.
func (l *LATCH) buildPattern(nbits int) {
	rng := rand.New(rand.NewSource(int64(0x1a7c4) ^ int64(l.Window)))
	rc := func() int { return rng.Intn(2*l.Window+1) - l.Window }
	l.triplets = make([]latchTriplet, nbits)
	for i := 0; i < nbits; i++ {
		l.triplets[i] = latchTriplet{rc(), rc(), rc(), rc(), rc(), rc()}
	}
}

// DescriptorSizeBytes returns the number of bytes in each descriptor.
func (l *LATCH) DescriptorSizeBytes() int { return l.bytes }

// patchSSD returns the sum of squared intensity differences between the
// mini-patch centred at (ax, ay) and the one centred at (bx, by), sampled from
// gray with border replication. The optional rotation (ca, sa) rotates the
// within-patch offsets.
func (l *LATCH) patchSSD(gray *cv.Mat, ax, ay, bx, by float64, ca, sa float64) float64 {
	var s float64
	for dy := -l.HalfPatch; dy <= l.HalfPatch; dy++ {
		for dx := -l.HalfPatch; dx <= l.HalfPatch; dx++ {
			rx := float64(dx)*ca - float64(dy)*sa
			ry := float64(dx)*sa + float64(dy)*ca
			va := bilinear(gray, ax+rx, ay+ry)
			vb := bilinear(gray, bx+rx, by+ry)
			d := va - vb
			s += d * d
		}
	}
	return s
}

// Compute describes each keypoint of img and returns the keypoints (with Angle
// set when RotationInvariant) and their bit-packed descriptors. Sampling uses
// border replication so no keypoint is dropped. img may be single- or
// three-channel; a colour image is converted to gray.
func (l *LATCH) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte) {
	gray := toGray(img)
	out := make([]KeyPoint, len(keypoints))
	descs := make([][]byte, len(keypoints))
	radius := l.Window + l.HalfPatch

	for k, kp := range keypoints {
		fx := float64(kp.Pt.X)
		fy := float64(kp.Pt.Y)
		angle := 0.0
		if l.RotationInvariant {
			if kp.Angle >= 0 {
				angle = kp.Angle * math.Pi / 180
			} else {
				angle = intensityCentroidAngle(gray, kp.Pt.X, kp.Pt.Y, radius)
			}
		}
		ca, sa := math.Cos(angle), math.Sin(angle)
		desc := make([]byte, l.bytes)
		for bit, t := range l.triplets {
			// Rotate the triplet centre offsets.
			arx := float64(t.ax)*ca - float64(t.ay)*sa
			ary := float64(t.ax)*sa + float64(t.ay)*ca
			brx := float64(t.bx)*ca - float64(t.by)*sa
			bry := float64(t.bx)*sa + float64(t.by)*ca
			crx := float64(t.cx)*ca - float64(t.cy)*sa
			cry := float64(t.cx)*sa + float64(t.cy)*ca
			d1 := l.patchSSD(gray, fx+arx, fy+ary, fx+brx, fy+bry, ca, sa)
			d2 := l.patchSSD(gray, fx+arx, fy+ary, fx+crx, fy+cry, ca, sa)
			if d1 < d2 {
				packBit(desc, bit)
			}
		}
		if l.RotationInvariant {
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

package features3

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// BRIEFPattern is a fixed set of intensity-comparison point pairs sampled inside
// a square patch, used to compute BRIEF and steered-BRIEF (ORB) binary
// descriptors. Each descriptor bit compares the smoothed intensity at
// (P1x[i], P1y[i]) with that at (P2x[i], P2y[i]) relative to the keypoint. Build
// one with [GenerateBRIEFPattern] or [DefaultBRIEFPattern].
type BRIEFPattern struct {
	// PatchSize is the side length of the sampling patch in pixels.
	PatchSize int
	// P1x, P1y, P2x, P2y hold the two sample offsets of each comparison, each
	// slice one entry per descriptor bit.
	P1x []int
	P1y []int
	P2x []int
	P2y []int
}

// NumBits returns the number of comparison bits (descriptor length in bits).
func (p *BRIEFPattern) NumBits() int {
	return len(p.P1x)
}

// NumBytes returns the number of bytes each packed descriptor occupies.
func (p *BRIEFPattern) NumBytes() int {
	return (len(p.P1x) + 7) / 8
}

// GenerateBRIEFPattern builds a BRIEF sampling pattern of numPairs comparisons
// inside a patchSize×patchSize patch. Offsets are drawn from a deterministic
// pseudo-random generator seeded by seed (isotropic Gaussian, standard deviation
// patchSize/5, clamped to the patch), so the same seed always yields the same
// pattern. It panics if patchSize or numPairs is not positive.
func GenerateBRIEFPattern(patchSize, numPairs int, seed int64) *BRIEFPattern {
	if patchSize <= 0 || numPairs <= 0 {
		panic("features3: GenerateBRIEFPattern requires positive patchSize and numPairs")
	}
	rng := rand.New(rand.NewSource(seed))
	half := patchSize / 2
	sigma := float64(patchSize) / 5.0
	clamp := func(v float64) int {
		i := int(math.Round(v))
		if i < -half {
			i = -half
		}
		if i > half {
			i = half
		}
		return i
	}
	p := &BRIEFPattern{
		PatchSize: patchSize,
		P1x:       make([]int, numPairs),
		P1y:       make([]int, numPairs),
		P2x:       make([]int, numPairs),
		P2y:       make([]int, numPairs),
	}
	for i := 0; i < numPairs; i++ {
		p.P1x[i] = clamp(rng.NormFloat64() * sigma)
		p.P1y[i] = clamp(rng.NormFloat64() * sigma)
		p.P2x[i] = clamp(rng.NormFloat64() * sigma)
		p.P2y[i] = clamp(rng.NormFloat64() * sigma)
	}
	return p
}

// DefaultBRIEFPattern returns the standard 256-bit BRIEF pattern in a 31×31
// patch generated with a fixed seed, matching the descriptor length used by ORB.
func DefaultBRIEFPattern() *BRIEFPattern {
	return GenerateBRIEFPattern(31, 256, 0x1D_2016)
}

// ComputeBRIEF computes an (unsteered) BRIEF binary descriptor for each keypoint
// using pattern. The image is Gaussian-smoothed (sigma 2) before sampling for
// robustness; each bit is set when the first sample of a pair is darker than the
// second. Descriptors are returned bit-packed as one []byte per keypoint (bit i
// in byte i/8, least-significant bit first). Sample coordinates are clamped to
// the image border. Colour input is converted to grayscale first.
func ComputeBRIEF(img *cv.Mat, kps []KeyPoint, pattern *BRIEFPattern) [][]byte {
	if pattern == nil {
		pattern = DefaultBRIEFPattern()
	}
	g := features3ToGray(img)
	sm := features3gaussianBlur(g, 2)
	out := make([][]byte, len(kps))
	nbytes := pattern.NumBytes()
	for ki, kp := range kps {
		cx := int(math.Round(kp.Pt.X))
		cy := int(math.Round(kp.Pt.Y))
		desc := make([]byte, nbytes)
		for i := 0; i < pattern.NumBits(); i++ {
			a := sm.atClamped(cx+pattern.P1x[i], cy+pattern.P1y[i])
			b := sm.atClamped(cx+pattern.P2x[i], cy+pattern.P2y[i])
			if a < b {
				desc[i/8] |= 1 << uint(i%8)
			}
		}
		out[ki] = desc
	}
	return out
}

// IntensityCentroidOrientation returns the orientation, in degrees in [0, 360),
// of the patch of the given radius around (x, y), computed from the intensity
// centroid as atan2(m01, m10). This is the orientation ORB assigns to a
// keypoint. Colour input is converted to grayscale first.
func IntensityCentroidOrientation(img *cv.Mat, x, y, radius int) float64 {
	g := features3ToGray(img)
	return features3orientation(g, x, y, radius)
}

// features3orientation computes the intensity-centroid angle in degrees on the
// working buffer.
func features3orientation(g *features3gray, x, y, radius int) float64 {
	var m01, m10 float64
	r2 := radius * radius
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			if dx*dx+dy*dy > r2 {
				continue
			}
			v := g.atClamped(x+dx, y+dy)
			m10 += float64(dx) * v
			m01 += float64(dy) * v
		}
	}
	deg := math.Atan2(m01, m10) * 180 / math.Pi
	if deg < 0 {
		deg += 360
	}
	return deg
}

// ComputeORB computes an ORB-style steered BRIEF binary descriptor for each
// keypoint. Every keypoint is assigned an orientation (its Angle when set,
// otherwise the intensity-centroid angle over a radius-derived patch) and the
// sampling pattern is rotated by that angle before comparison, giving a
// descriptor invariant to in-plane rotation. Like [ComputeBRIEF] the image is
// Gaussian-smoothed first and descriptors are returned bit-packed, one []byte
// per keypoint. Colour input is converted to grayscale first.
func ComputeORB(img *cv.Mat, kps []KeyPoint, pattern *BRIEFPattern) [][]byte {
	if pattern == nil {
		pattern = DefaultBRIEFPattern()
	}
	g := features3ToGray(img)
	sm := features3gaussianBlur(g, 2)
	radius := pattern.PatchSize / 2
	out := make([][]byte, len(kps))
	nbytes := pattern.NumBytes()
	for ki, kp := range kps {
		cx := int(math.Round(kp.Pt.X))
		cy := int(math.Round(kp.Pt.Y))
		angle := kp.Angle
		if angle < 0 {
			angle = features3orientation(g, cx, cy, radius)
		}
		rad := angle * math.Pi / 180
		cosA := math.Cos(rad)
		sinA := math.Sin(rad)
		desc := make([]byte, nbytes)
		for i := 0; i < pattern.NumBits(); i++ {
			a1x := cosA*float64(pattern.P1x[i]) - sinA*float64(pattern.P1y[i])
			a1y := sinA*float64(pattern.P1x[i]) + cosA*float64(pattern.P1y[i])
			a2x := cosA*float64(pattern.P2x[i]) - sinA*float64(pattern.P2y[i])
			a2y := sinA*float64(pattern.P2x[i]) + cosA*float64(pattern.P2y[i])
			a := sm.atClamped(cx+int(math.Round(a1x)), cy+int(math.Round(a1y)))
			b := sm.atClamped(cx+int(math.Round(a2x)), cy+int(math.Round(a2y)))
			if a < b {
				desc[i/8] |= 1 << uint(i%8)
			}
		}
		out[ki] = desc
	}
	return out
}

// MatchBinaryDescriptors matches each query descriptor to its nearest train
// descriptor by Hamming distance (brute force). A match is emitted only when the
// best distance is at most maxDistance; pass a negative maxDistance to accept
// every query's best match. Descriptors must all share the same byte length.
// Matches are returned in query order.
func MatchBinaryDescriptors(query, train [][]byte, maxDistance int) []Match {
	var matches []Match
	for qi, q := range query {
		bestIdx := -1
		bestDist := math.MaxInt32
		for ti, t := range train {
			d := HammingDistance(q, t)
			if d < bestDist {
				bestDist = d
				bestIdx = ti
			}
		}
		if bestIdx >= 0 && (maxDistance < 0 || bestDist <= maxDistance) {
			matches = append(matches, Match{QueryIdx: qi, TrainIdx: bestIdx, Distance: bestDist})
		}
	}
	return matches
}

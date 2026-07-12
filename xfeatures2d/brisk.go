package xfeatures2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// briskRingRadii and briskRingCounts define the concentric sampling pattern:
// briskRingCounts[i] points are placed evenly on a ring of radius
// briskRingRadii[i] (in pixels at unit scale). The first ring is the single
// centre sample.
var (
	briskRingRadii  = []float64{0, 2.9, 4.9, 7.4, 10.8}
	briskRingCounts = []int{1, 10, 14, 15, 20}
	// briskRingSigma is the Gaussian smoothing standard deviation applied when
	// sampling each ring, growing with the ring radius so outer samples are
	// smoothed more.
	briskRingSigma = []float64{0.5, 0.9, 1.4, 2.0, 2.9}
)

// briskPoint is one pattern sample: its offset from the keypoint (at unit scale)
// and the Gaussian sigma used to read it.
type briskPoint struct {
	x, y  float64
	sigma float64
}

// briskPair is an ordered index pair into the sampling points.
type briskPair struct {
	i, j int
}

// BRISK computes the Binary Robust Invariant Scalable Keypoints descriptor. It
// mirrors the descriptor stage of OpenCV's cv::BRISK.
//
// For each keypoint the pattern is scaled by the keypoint size, its sample
// intensities are read with per-ring Gaussian smoothing, long-distance pairs
// estimate an orientation, the pattern is rotated by that orientation, and
// short-distance pairs are compared to produce the descriptor bits. The result
// is therefore invariant to in-plane rotation and (through Size) to scale.
//
// The descriptor length is fixed: every keypoint yields DescriptorSizeBytes
// bytes. Build a detector/descriptor with [NewBRISK].
type BRISK struct {
	// AGASTThreshold is the corner threshold used by DetectAndCompute when it
	// detects keypoints with [AGAST].
	AGASTThreshold int

	points     []briskPoint
	shortPairs []briskPair
	longPairs  []briskPair
	maxRadius  float64
	baseSize   float64
}

// NewBRISK builds a BRISK detector/descriptor with the standard concentric-ring
// sampling pattern and the given AGAST corner threshold for detection.
func NewBRISK(agastThreshold int) *BRISK {
	b := &BRISK{AGASTThreshold: agastThreshold}
	b.buildPattern()
	return b
}

// buildPattern generates the sampling points and classifies all point pairs into
// short-distance (descriptor) and long-distance (orientation) pairs.
func (b *BRISK) buildPattern() {
	for ring := range briskRingRadii {
		r := briskRingRadii[ring]
		n := briskRingCounts[ring]
		sigma := briskRingSigma[ring]
		if n == 1 {
			b.points = append(b.points, briskPoint{x: 0, y: 0, sigma: sigma})
			continue
		}
		// Offset alternate rings by half a step to decorrelate samples.
		offset := 0.0
		if ring%2 == 0 {
			offset = math.Pi / float64(n)
		}
		for i := 0; i < n; i++ {
			theta := 2*math.Pi*float64(i)/float64(n) + offset
			b.points = append(b.points, briskPoint{
				x:     r * math.Cos(theta),
				y:     r * math.Sin(theta),
				sigma: sigma,
			})
		}
	}
	b.maxRadius = briskRingRadii[len(briskRingRadii)-1]
	b.baseSize = 2 * b.maxRadius

	maxDist := 2 * b.maxRadius
	shortThresh := 0.5 * maxDist
	longThresh := 0.75 * maxDist
	for i := 0; i < len(b.points); i++ {
		for j := i + 1; j < len(b.points); j++ {
			dx := b.points[i].x - b.points[j].x
			dy := b.points[i].y - b.points[j].y
			d := math.Hypot(dx, dy)
			switch {
			case d < shortThresh && d > 0:
				b.shortPairs = append(b.shortPairs, briskPair{i, j})
			case d > longThresh:
				b.longPairs = append(b.longPairs, briskPair{i, j})
			}
		}
	}
}

// DescriptorSizeBits returns the number of bits in each descriptor (one per
// short-distance sampling pair).
func (b *BRISK) DescriptorSizeBits() int {
	return len(b.shortPairs)
}

// DescriptorSizeBytes returns the number of bytes in each bit-packed descriptor.
func (b *BRISK) DescriptorSizeBytes() int {
	return (len(b.shortPairs) + 7) / 8
}

// scaleFor returns the pattern scale factor for a keypoint of the given Size.
// A keypoint whose Size equals the pattern's base diameter is sampled at unit
// scale; larger keypoints sample a proportionally larger neighbourhood.
func (b *BRISK) scaleFor(size float64) float64 {
	if size <= 0 {
		return 1
	}
	return size / b.baseSize
}

// smoothed reads the intensity of gray at (x, y) with Gaussian smoothing of
// standard deviation sigma, using border replication. Small sigmas fall back to
// bilinear interpolation.
func smoothed(gray *cv.Mat, x, y, sigma float64) float64 {
	if sigma < 0.6 {
		return bilinear(gray, x, y)
	}
	r := int(math.Ceil(2 * sigma))
	if r < 1 {
		r = 1
	}
	cx := int(math.Round(x))
	cy := int(math.Round(y))
	inv := 1 / (2 * sigma * sigma)
	var sum, wsum float64
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			w := math.Exp(-float64(dx*dx+dy*dy) * inv)
			sum += w * grayAtClamped(gray, cx+dx, cy+dy)
			wsum += w
		}
	}
	return sum / wsum
}

// orientation estimates the keypoint orientation (in radians) from the
// long-distance pairs of the sampled intensities. samples[i] is the smoothed
// intensity at point i (at the keypoint's scale).
func (b *BRISK) orientation(samples []float64) float64 {
	var sumX, sumY float64
	for _, p := range b.longPairs {
		dx := b.points[p.i].x - b.points[p.j].x
		dy := b.points[p.i].y - b.points[p.j].y
		d2 := dx*dx + dy*dy
		if d2 == 0 {
			continue
		}
		g := (samples[p.i] - samples[p.j]) / d2
		sumX += g * dx
		sumY += g * dy
	}
	if sumX == 0 && sumY == 0 {
		return 0
	}
	return math.Atan2(sumY, sumX)
}

// Compute describes each keypoint of img and returns the retained keypoints
// together with their bit-packed descriptors (one []byte per keypoint, all of
// length DescriptorSizeBytes). Keypoints whose scaled pattern would fall outside
// the image are dropped, so the returned keypoint slice may be shorter than the
// input and stays aligned with the descriptor rows. img may be single- or
// three-channel; a colour image is converted to gray.
func (b *BRISK) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte) {
	gray := toGray(img)
	nBytes := b.DescriptorSizeBytes()

	var keptKps []KeyPoint
	var descs [][]byte
	samples := make([]float64, len(b.points))

	for _, kp := range keypoints {
		scale := b.scaleFor(kp.Size)
		margin := scale*b.maxRadius + 2*maxRingSigma()*scale + 2
		fx := float64(kp.Pt.X)
		fy := float64(kp.Pt.Y)
		if fx-margin < 0 || fy-margin < 0 ||
			fx+margin >= float64(gray.Cols) || fy+margin >= float64(gray.Rows) {
			continue
		}

		// First, sample the unrotated pattern to estimate orientation.
		for i, p := range b.points {
			samples[i] = smoothed(gray, fx+p.x*scale, fy+p.y*scale, p.sigma*scale)
		}
		angle := b.orientation(samples)
		ca := math.Cos(angle)
		sa := math.Sin(angle)

		// Re-sample the pattern rotated by the orientation.
		rot := make([]float64, len(b.points))
		for i, p := range b.points {
			rx := p.x*ca - p.y*sa
			ry := p.x*sa + p.y*ca
			rot[i] = smoothed(gray, fx+rx*scale, fy+ry*scale, p.sigma*scale)
		}

		desc := make([]byte, nBytes)
		for bit, pair := range b.shortPairs {
			if rot[pair.i] > rot[pair.j] {
				desc[bit>>3] |= 1 << uint(bit&7)
			}
		}

		out := kp
		deg := angle * 180 / math.Pi
		if deg < 0 {
			deg += 360
		}
		out.Angle = deg
		keptKps = append(keptKps, out)
		descs = append(descs, desc)
	}
	return keptKps, descs
}

// DetectAndCompute detects keypoints with [AGAST] and describes them, returning
// the retained keypoints and their descriptors. img may be single- or
// three-channel.
func (b *BRISK) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]byte) {
	det := &AGAST{Threshold: b.AGASTThreshold, NonmaxSuppression: true}
	kps := det.Detect(img)
	// Give each keypoint the descriptor's base size so it is sampled at unit
	// scale.
	for i := range kps {
		kps[i].Size = b.baseSize
	}
	return b.Compute(img, kps)
}

// maxRingSigma returns the largest per-ring smoothing sigma in the pattern.
func maxRingSigma() float64 {
	m := 0.0
	for _, s := range briskRingSigma {
		if s > m {
			m = s
		}
	}
	return m
}

package xfeatures2d

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// freakReceptor is one receptive field of the FREAK retinal pattern: a sampling
// position (offset from the keypoint at unit scale) and the Gaussian sigma of
// the field.
type freakReceptor struct {
	x, y  float64
	sigma float64
}

// FREAK computes the Fast Retina Keypoint binary descriptor, a port of the
// descriptor stage of OpenCV's cv::xfeatures2d::FREAK.
//
// FREAK samples the neighbourhood with a retina-like pattern: several concentric
// hexagonal rings of overlapping receptive fields whose size grows towards the
// periphery. Each field is read as a Gaussian-smoothed intensity. A subset of
// symmetric long-distance field pairs estimates the keypoint orientation; the
// pattern is rotated by that orientation and a fixed list of field pairs is
// compared to yield the descriptor bits, so the descriptor is invariant to
// in-plane rotation and (through the keypoint Size) to scale.
//
// Two descriptors are compared with the [HammingDistance].
type FREAK struct {
	// AGASTThreshold is the corner threshold used by DetectAndCompute.
	AGASTThreshold int

	receptors  []freakReceptor
	pairs      []briskPair // descriptor comparison pairs
	orientPair []briskPair // symmetric pairs used for orientation
	maxRadius  float64
	baseSize   float64
}

// FREAK pattern parameters: rings of the retina, from the periphery inwards.
var (
	freakRingRadius = []float64{10.8, 8.1, 6.0, 4.3, 2.9, 1.4, 0}
	freakRingSigma  = []float64{2.6, 2.0, 1.5, 1.1, 0.8, 0.5, 0.4}
)

// NewFREAK returns a FREAK descriptor with the standard retinal pattern and the
// given AGAST corner threshold for detection.
func NewFREAK(agastThreshold int) *FREAK {
	f := &FREAK{AGASTThreshold: agastThreshold}
	f.buildPattern()
	return f
}

// buildPattern generates the receptive fields and selects the orientation and
// descriptor pairs deterministically from the geometry.
func (f *FREAK) buildPattern() {
	for ring := range freakRingRadius {
		r := freakRingRadius[ring]
		sigma := freakRingSigma[ring]
		if r == 0 {
			f.receptors = append(f.receptors, freakReceptor{0, 0, sigma})
			continue
		}
		// Six points per ring, alternate rings rotated by half a step, matching
		// the hexagonal FREAK grid.
		offset := 0.0
		if ring%2 == 1 {
			offset = math.Pi / 6
		}
		for i := 0; i < 6; i++ {
			theta := 2*math.Pi*float64(i)/6 + offset
			f.receptors = append(f.receptors, freakReceptor{
				x:     r * math.Cos(theta),
				y:     r * math.Sin(theta),
				sigma: sigma,
			})
		}
	}
	f.maxRadius = freakRingRadius[0]
	f.baseSize = 2 * f.maxRadius

	// All unordered pairs, split by distance.
	n := len(f.receptors)
	maxDist := 2 * f.maxRadius
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dx := f.receptors[i].x - f.receptors[j].x
			dy := f.receptors[i].y - f.receptors[j].y
			d := math.Hypot(dx, dy)
			if d > 0.7*maxDist {
				f.orientPair = append(f.orientPair, briskPair{i, j})
			}
		}
	}
	// Descriptor pairs: the 512 shortest-distance pairs give a compact,
	// well-localised code. Sort by distance for a deterministic selection.
	type pd struct {
		p briskPair
		d float64
	}
	var all []pd
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dx := f.receptors[i].x - f.receptors[j].x
			dy := f.receptors[i].y - f.receptors[j].y
			all = append(all, pd{briskPair{i, j}, math.Hypot(dx, dy)})
		}
	}
	sort.Slice(all, func(a, b int) bool {
		if all[a].d != all[b].d {
			return all[a].d < all[b].d
		}
		if all[a].p.i != all[b].p.i {
			return all[a].p.i < all[b].p.i
		}
		return all[a].p.j < all[b].p.j
	})
	limit := 512
	if limit > len(all) {
		limit = len(all)
	}
	for k := 0; k < limit; k++ {
		f.pairs = append(f.pairs, all[k].p)
	}
}

// DescriptorSizeBits returns the number of bits in each descriptor.
func (f *FREAK) DescriptorSizeBits() int { return len(f.pairs) }

// DescriptorSizeBytes returns the number of bytes in each bit-packed descriptor.
func (f *FREAK) DescriptorSizeBytes() int { return (len(f.pairs) + 7) / 8 }

func (f *FREAK) scaleFor(size float64) float64 {
	if size <= 0 {
		return 1
	}
	return size / f.baseSize
}

// orientation estimates the keypoint orientation (radians) from the symmetric
// long-distance pairs of sampled field intensities.
func (f *FREAK) orientation(samples []float64) float64 {
	var sumX, sumY float64
	for _, p := range f.orientPair {
		dx := f.receptors[p.i].x - f.receptors[p.j].x
		dy := f.receptors[p.i].y - f.receptors[p.j].y
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

// Compute describes each keypoint of img and returns the keypoints (with Angle
// set to the estimated orientation) and their bit-packed descriptors. Samples
// use Gaussian smoothing with border replication, so no keypoint is dropped.
// img may be single- or three-channel; a colour image is converted to gray.
func (f *FREAK) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte) {
	gray := toGray(img)
	nBytes := f.DescriptorSizeBytes()
	out := make([]KeyPoint, len(keypoints))
	descs := make([][]byte, len(keypoints))
	samples := make([]float64, len(f.receptors))
	rot := make([]float64, len(f.receptors))

	for k, kp := range keypoints {
		scale := f.scaleFor(kp.Size)
		fx := float64(kp.Pt.X)
		fy := float64(kp.Pt.Y)
		for i, p := range f.receptors {
			samples[i] = smoothed(gray, fx+p.x*scale, fy+p.y*scale, math.Max(p.sigma*scale, 0.4))
		}
		angle := f.orientation(samples)
		ca, sa := math.Cos(angle), math.Sin(angle)
		for i, p := range f.receptors {
			rx := p.x*ca - p.y*sa
			ry := p.x*sa + p.y*ca
			rot[i] = smoothed(gray, fx+rx*scale, fy+ry*scale, math.Max(p.sigma*scale, 0.4))
		}
		desc := make([]byte, nBytes)
		for bit, pair := range f.pairs {
			if rot[pair.i] > rot[pair.j] {
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

// DetectAndCompute detects keypoints with [AGAST] (sized to the FREAK base
// diameter) and describes them. img may be single- or three-channel.
func (f *FREAK) DetectAndCompute(img *cv.Mat) ([]KeyPoint, [][]byte) {
	det := &AGAST{Threshold: f.AGASTThreshold, NonmaxSuppression: true}
	kps := det.Detect(img)
	for i := range kps {
		kps[i].Size = f.baseSize
	}
	return f.Compute(img, kps)
}

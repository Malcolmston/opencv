package xfeatures2d

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// LUCID computes the Locally Uniform Comparison Image Descriptor, a port of
// OpenCV's cv::xfeatures2d::LUCID.
//
// LUCID blurs the image and, for the square neighbourhood of each keypoint,
// records the permutation that sorts the neighbourhood's pixel intensities —
// the rank of every sampled pixel. Because only the relative order of
// intensities is kept, the descriptor is invariant to any monotone change of
// illumination while remaining extremely cheap to build. Descriptors are
// compared with [LUCIDDistance] (the L1 distance between the two rank vectors),
// not the Hamming distance.
type LUCID struct {
	// KernelSize is the half side of the square neighbourhood that is described
	// (the neighbourhood side is 2*KernelSize+1).
	KernelSize int
	// BlurSize is the half side of the box blur applied before sampling; 0
	// disables blurring.
	BlurSize int
}

// NewLUCID returns a LUCID extractor with the given neighbourhood and blur half
// sizes. It panics if kernelSize is not positive.
func NewLUCID(kernelSize, blurSize int) *LUCID {
	if kernelSize <= 0 {
		panic("xfeatures2d: NewLUCID requires a positive kernelSize")
	}
	return &LUCID{KernelSize: kernelSize, BlurSize: blurSize}
}

// DescriptorSize returns the number of rank entries in each descriptor, one per
// pixel of the sampled neighbourhood.
func (l *LUCID) DescriptorSize() int {
	s := 2*l.KernelSize + 1
	return s * s
}

// Compute describes each keypoint of img and returns the keypoints unchanged
// together with their rank descriptors (one []byte of length DescriptorSize per
// keypoint; entry i is the rank of the i-th neighbourhood pixel among all
// sampled pixels). Sampling uses border replication, so no keypoint is dropped.
// img may be single- or three-channel; a colour image is converted to gray.
func (l *LUCID) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte) {
	gray := toGray(img)
	if l.BlurSize > 0 {
		ksize := 2*l.BlurSize + 1
		gray = cv.Blur(gray, ksize)
	}
	k := l.KernelSize
	side := 2*k + 1
	n := side * side
	out := make([]KeyPoint, len(keypoints))
	descs := make([][]byte, len(keypoints))

	type sample struct {
		idx int
		val float64
	}
	for kp := range keypoints {
		cx, cy := keypoints[kp].Pt.X, keypoints[kp].Pt.Y
		samples := make([]sample, 0, n)
		i := 0
		for dy := -k; dy <= k; dy++ {
			for dx := -k; dx <= k; dx++ {
				samples = append(samples, sample{i, grayAtClamped(gray, cx+dx, cy+dy)})
				i++
			}
		}
		// Rank by value; equal values keep their positional order for
		// determinism.
		sort.SliceStable(samples, func(a, b int) bool {
			return samples[a].val < samples[b].val
		})
		desc := make([]byte, n)
		for rank, s := range samples {
			// Scale the rank into a byte so descriptors of the usual small
			// neighbourhoods fit without loss (n <= 256 keeps it exact).
			if n <= 256 {
				desc[s.idx] = byte(rank)
			} else {
				desc[s.idx] = byte(rank * 255 / (n - 1))
			}
		}
		out[kp] = keypoints[kp]
		descs[kp] = desc
	}
	return out, descs
}

// LUCIDDistance returns the L1 distance between two LUCID rank descriptors, the
// natural dissimilarity for the rank-order representation. It panics if the
// descriptors differ in length.
func LUCIDDistance(a, b []byte) int {
	if len(a) != len(b) {
		panic("xfeatures2d: LUCIDDistance on descriptors of different length")
	}
	d := 0
	for i := range a {
		v := int(a[i]) - int(b[i])
		if v < 0 {
			v = -v
		}
		d += v
	}
	return d
}

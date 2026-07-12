package imghash

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

const (
	// rvSize is the side length of the working image the projections run on.
	rvSize = 32
	// rvAngles is the number of Radon projection angles (0..π), and hence the
	// length in bytes of the resulting hash.
	rvAngles = 40
)

// RadialVarianceHash implements a radial-variance hash in the spirit of the
// RASH descriptor and OpenCV's cv::img_hash::RadialVarianceHash. The image is
// reduced to grayscale and scaled to a fixed square, and a Radon transform is
// evaluated at rvAngles orientations spanning 0..π. For each orientation the
// variance of its 1-D projection is taken, giving a rotation-sensitive profile
// of how mass is distributed across the image. The profile is normalised to its
// own peak and quantised to one byte per angle, producing a 40-byte fingerprint
// compared by L1 distance.
//
// Because the profile is normalised by its own maximum, the hash is invariant
// to uniform brightness and contrast scaling; blurring perturbs it only
// slightly, while structurally different images produce clearly different
// profiles. Unlike OpenCV, which compares radial-variance hashes by peak cross
// correlation, this port compares them by the interface's L1 distance for
// consistency with the other real-valued hashes.
//
// The zero value is ready to use; [NewRadialVarianceHash] is provided for
// symmetry.
type RadialVarianceHash struct{}

// NewRadialVarianceHash returns a ready-to-use [RadialVarianceHash].
func NewRadialVarianceHash() RadialVarianceHash { return RadialVarianceHash{} }

// Compute returns the 40-byte radial-variance hash of img.
func (RadialVarianceHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "RadialVarianceHash.Compute")
	small := grayResize(img, rvSize, rvSize)

	cx := float64(rvSize-1) / 2
	cy := float64(rvSize-1) / 2
	// The projection axis has at most rvSize*sqrt(2) distinct offsets; centre
	// the bins so t=0 falls in the middle.
	nBins := int(math.Ceil(float64(rvSize)*math.Sqrt2)) + 1
	binOffset := float64(nBins-1) / 2

	variances := make([]float64, rvAngles)
	sum := make([]float64, nBins)
	count := make([]int, nBins)
	for a := 0; a < rvAngles; a++ {
		theta := math.Pi * float64(a) / float64(rvAngles)
		cos, sin := math.Cos(theta), math.Sin(theta)
		for i := range sum {
			sum[i] = 0
			count[i] = 0
		}
		for y := 0; y < rvSize; y++ {
			dy := float64(y) - cy
			for x := 0; x < rvSize; x++ {
				dx := float64(x) - cx
				t := dx*cos + dy*sin
				bin := int(math.Round(t + binOffset))
				if bin < 0 {
					bin = 0
				} else if bin >= nBins {
					bin = nBins - 1
				}
				sum[bin] += float64(small.Data[y*rvSize+x])
				count[bin]++
			}
		}
		// Projection value per populated bin, then its variance.
		proj := make([]float64, 0, nBins)
		for i := 0; i < nBins; i++ {
			if count[i] > 0 {
				proj = append(proj, sum[i]/float64(count[i]))
			}
		}
		variances[a] = variance(proj)
	}

	// Normalise to the peak and quantise to [0,255].
	maxV := 0.0
	for _, v := range variances {
		if v > maxV {
			maxV = v
		}
	}
	out := make([]byte, rvAngles)
	if maxV > 0 {
		for i, v := range variances {
			out[i] = uint8(math.Round(v / maxV * 255))
		}
	}
	return out
}

// Compare returns the L1 distance between two radial-variance hashes.
func (RadialVarianceHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "RadialVarianceHash.Compare")
	return l1(a, b)
}

// variance returns the population variance of vals, or 0 for an empty slice.
func variance(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	m := mean(vals)
	var s float64
	for _, v := range vals {
		d := v - m
		s += d * d
	}
	return s / float64(len(vals))
}

// RadialVariance is a convenience wrapper returning the [RadialVarianceHash] of
// img.
func RadialVariance(img *cv.Mat) []byte { return RadialVarianceHash{}.Compute(img) }

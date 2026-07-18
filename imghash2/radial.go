package imghash2

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

const (
	// rvSize is the side length of the working image the projections run on.
	rvSize = 32
	// defaultRadialAngles is the number of Radon projection angles spanning
	// 0..π used by a zero-value hasher.
	defaultRadialAngles = 40
)

// RadialVarianceHash implements a radial-variance hash in the spirit of the
// RASH descriptor and OpenCV's cv::img_hash::RadialVarianceHash. The image is
// reduced to grayscale and scaled to a fixed square, and a Radon transform is
// evaluated at Angles orientations spanning 0..π. For each orientation the
// variance of its 1-D projection is taken, giving a profile of how image mass
// is distributed across orientation. The profile is normalised to its own peak,
// yielding a [FloatHash] of Angles values in [0, 1].
//
// Because the profile is normalised by its own maximum, the descriptor is
// invariant to uniform brightness and contrast scaling; blurring perturbs it
// only slightly, while structurally different images produce clearly different
// profiles. Rotating the image cyclically shifts the profile, so comparing two
// profiles with [RadialCrossCorrelation] — which maximises correlation over all
// circular shifts — is robust to rotation, whereas [FloatHash.L1] is
// rotation-sensitive. The zero value uses 40 angles; [NewRadialVarianceHash]
// chooses the count explicitly.
type RadialVarianceHash struct {
	// Angles is the number of Radon projection orientations, and hence the
	// length of the resulting descriptor. A zero value means 40.
	Angles int
}

// NewRadialVarianceHash returns a ready-to-use [RadialVarianceHash] with the
// default 40 angles.
func NewRadialVarianceHash() RadialVarianceHash {
	return RadialVarianceHash{Angles: defaultRadialAngles}
}

// NewRadialVarianceHashAngles returns a [RadialVarianceHash] using the given
// number of projection angles. It panics if angles is less than 2.
func NewRadialVarianceHashAngles(angles int) RadialVarianceHash {
	if angles < 2 {
		panic(fmt.Sprintf("imghash2: NewRadialVarianceHashAngles requires angles >= 2, got %d", angles))
	}
	return RadialVarianceHash{Angles: angles}
}

// Name returns the identifier "radialvariance".
func (RadialVarianceHash) Name() string { return "radialvariance" }

// angleCount returns the effective number of angles, applying the default.
func (h RadialVarianceHash) angleCount() int {
	if h.Angles <= 0 {
		return defaultRadialAngles
	}
	return h.Angles
}

// ComputeFloat returns the radial-variance descriptor of img, a [FloatHash] of
// Angles values normalised so its peak is 1.
func (h RadialVarianceHash) ComputeFloat(img *cv.Mat) FloatHash {
	requireImage(img, "RadialVarianceHash.ComputeFloat")
	angles := h.angleCount()
	small := GrayResize(img, rvSize, rvSize)

	cx := float64(rvSize-1) / 2
	cy := float64(rvSize-1) / 2
	nBins := int(math.Ceil(float64(rvSize)*math.Sqrt2)) + 1
	binOffset := float64(nBins-1) / 2

	variances := make(FloatHash, angles)
	sum := make([]float64, nBins)
	count := make([]int, nBins)
	for a := 0; a < angles; a++ {
		theta := math.Pi * float64(a) / float64(angles)
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
		proj := make([]float64, 0, nBins)
		for i := 0; i < nBins; i++ {
			if count[i] > 0 {
				proj = append(proj, sum[i]/float64(count[i]))
			}
		}
		variances[a] = Variance(proj)
	}

	maxV := 0.0
	for _, v := range variances {
		if v > maxV {
			maxV = v
		}
	}
	if maxV > 0 {
		for i := range variances {
			variances[i] /= maxV
		}
	}
	return variances
}

// RadialVariance is a convenience wrapper returning the default
// [RadialVarianceHash] descriptor of img.
func RadialVariance(img *cv.Mat) FloatHash { return NewRadialVarianceHash().ComputeFloat(img) }

// RadialCrossCorrelation returns the peak Pearson cross-correlation between two
// radial-variance descriptors over all circular shifts, a value in [-1, 1]
// where 1 means the profiles match under some rotation. Because rotating an
// image cyclically shifts its radial profile, taking the maximum over shifts
// makes this a rotation-robust similarity measure, unlike a plain L1 distance.
// It panics if the descriptors differ in length or are empty.
func RadialCrossCorrelation(a, b FloatHash) float64 {
	requireSameFloatLen(a, b, "RadialCrossCorrelation")
	n := len(a)
	if n == 0 {
		panic("imghash2: RadialCrossCorrelation requires non-empty descriptors")
	}
	ma := Mean(a)
	mb := Mean(b)
	da := make([]float64, n)
	db := make([]float64, n)
	var na, nb float64
	for i := 0; i < n; i++ {
		da[i] = a[i] - ma
		db[i] = b[i] - mb
		na += da[i] * da[i]
		nb += db[i] * db[i]
	}
	denom := math.Sqrt(na * nb)
	if denom == 0 {
		// One profile is constant: correlation is defined as 1 iff both are.
		if na == 0 && nb == 0 {
			return 1
		}
		return 0
	}
	best := math.Inf(-1)
	for shift := 0; shift < n; shift++ {
		var dot float64
		for i := 0; i < n; i++ {
			dot += da[i] * db[(i+shift)%n]
		}
		if c := dot / denom; c > best {
			best = c
		}
	}
	return best
}

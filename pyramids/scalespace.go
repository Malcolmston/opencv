package pyramids

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DifferenceOfGaussians returns the difference of two Gaussian-blurred versions
// of f, blur(sigma1) - blur(sigma2). With sigma1 < sigma2 this is a band-pass
// filter approximating the Laplacian of Gaussian; a constant image yields all
// zeros. It panics if either sigma is not positive.
func DifferenceOfGaussians(f *cv.FloatMat, sigma1, sigma2 float64) *cv.FloatMat {
	pyramidsRequire(f, "DifferenceOfGaussians")
	g1 := GaussianBlurFloat(f, sigma1)
	g2 := GaussianBlurFloat(f, sigma2)
	return SubtractFloat(g1, g2)
}

// ScaleSpace is a stack of Gaussian-blurred versions of one image, all at the
// original resolution, with geometrically increasing standard deviations. It is
// the linear (Witkin) scale space and the basis for difference-of-Gaussian
// analysis.
type ScaleSpace struct {
	// Sigmas holds the standard deviation of each level, ascending.
	Sigmas []float64
	// Levels holds the blurred image at each sigma, same order as Sigmas.
	Levels []*cv.FloatMat
}

// BuildScaleSpace produces a Gaussian scale space of numLevels images. The i-th
// level is f blurred to sigma0 * k^i. Successive levels are obtained by
// incremental blurring, which is equivalent to blurring the original by the
// combined sigma. It panics if numLevels < 1, sigma0 <= 0, or k <= 1.
func BuildScaleSpace(f *cv.FloatMat, numLevels int, sigma0, k float64) *ScaleSpace {
	pyramidsRequire(f, "BuildScaleSpace")
	if numLevels < 1 {
		panic("pyramids: BuildScaleSpace: numLevels must be >= 1")
	}
	if sigma0 <= 0 {
		panic("pyramids: BuildScaleSpace: sigma0 must be positive")
	}
	if k <= 1 {
		panic("pyramids: BuildScaleSpace: k must be > 1")
	}
	ss := &ScaleSpace{
		Sigmas: make([]float64, numLevels),
		Levels: make([]*cv.FloatMat, numLevels),
	}
	sigma := sigma0
	prev := GaussianBlurFloat(f, sigma0)
	ss.Sigmas[0] = sigma0
	ss.Levels[0] = prev
	for i := 1; i < numLevels; i++ {
		nextSigma := sigma0 * math.Pow(k, float64(i))
		// Incremental blur sigma so that combined variance reaches nextSigma^2.
		delta := math.Sqrt(nextSigma*nextSigma - sigma*sigma)
		prev = GaussianBlurFloat(prev, delta)
		ss.Sigmas[i] = nextSigma
		ss.Levels[i] = prev
		sigma = nextSigma
	}
	return ss
}

// NumLevels returns the number of levels in the scale space.
func (s *ScaleSpace) NumLevels() int { return len(s.Levels) }

// Level returns the blurred image at index i. It panics if i is out of range.
func (s *ScaleSpace) Level(i int) *cv.FloatMat {
	if i < 0 || i >= len(s.Levels) {
		panic("pyramids: ScaleSpace.Level: index out of range")
	}
	return s.Levels[i]
}

// DoG returns the difference-of-Gaussian stack derived from adjacent scale
// levels: element i is Level(i+1) - Level(i). The result has NumLevels-1
// images, each band-pass and zero on flat regions.
func (s *ScaleSpace) DoG() []*cv.FloatMat {
	out := make([]*cv.FloatMat, 0, len(s.Levels)-1)
	for i := 0; i+1 < len(s.Levels); i++ {
		out = append(out, SubtractFloat(s.Levels[i+1], s.Levels[i]))
	}
	return out
}

// DoGScaleSpace is a difference-of-Gaussian stack together with the standard
// deviation associated with each difference image, suitable for scale-selective
// blob detection.
type DoGScaleSpace struct {
	// Images holds the DoG responses from fine (small sigma) to coarse.
	Images []*cv.FloatMat
	// Sigmas holds a representative sigma for each DoG image (the geometric
	// mean of the two Gaussians differenced).
	Sigmas []float64
}

// BuildDoGScaleSpace builds a difference-of-Gaussian stack from a Gaussian
// scale space of numDoG+1 blur levels, so it contains numDoG DoG images. The
// arguments sigma0 and k are passed through to [BuildScaleSpace]. It panics if
// numDoG < 1.
func BuildDoGScaleSpace(f *cv.FloatMat, numDoG int, sigma0, k float64) *DoGScaleSpace {
	if numDoG < 1 {
		panic("pyramids: BuildDoGScaleSpace: numDoG must be >= 1")
	}
	ss := BuildScaleSpace(f, numDoG+1, sigma0, k)
	out := &DoGScaleSpace{
		Images: make([]*cv.FloatMat, numDoG),
		Sigmas: make([]float64, numDoG),
	}
	for i := 0; i < numDoG; i++ {
		out.Images[i] = SubtractFloat(ss.Levels[i+1], ss.Levels[i])
		out.Sigmas[i] = math.Sqrt(ss.Sigmas[i] * ss.Sigmas[i+1])
	}
	return out
}

// NumImages returns the number of DoG images in the stack.
func (d *DoGScaleSpace) NumImages() int { return len(d.Images) }

// Keypoint is a scale-space blob detected in a difference-of-Gaussian stack.
type Keypoint struct {
	// X and Y are the pixel coordinates of the extremum (column, row).
	X, Y int
	// ScaleIndex is the index of the DoG image the extremum was found in.
	ScaleIndex int
	// Sigma is the characteristic scale (representative sigma) of that image.
	Sigma float64
	// Response is the signed DoG value at the extremum; its magnitude measures
	// blob strength and its sign distinguishes dark from bright blobs.
	Response float64
}

// DetectDoGExtrema finds blobs as local extrema of the difference-of-Gaussian
// stack. A pixel is reported when its DoG value exceeds threshold in magnitude
// and is strictly greater (or strictly smaller) than all 26 neighbours in the
// 3×3×3 spatial-and-scale window. Extrema on the spatial border and in the
// first/last scale are skipped because they lack a full neighbourhood. Results
// are returned in a deterministic order (by scale, then row, then column). It
// panics if threshold is negative or the stack has fewer than three images.
func DetectDoGExtrema(d *DoGScaleSpace, threshold float64) []Keypoint {
	if threshold < 0 {
		panic("pyramids: DetectDoGExtrema: threshold must be non-negative")
	}
	if len(d.Images) < 3 {
		panic("pyramids: DetectDoGExtrema: need at least three DoG images")
	}
	rows, cols := d.Images[0].Rows, d.Images[0].Cols
	var kps []Keypoint
	for s := 1; s < len(d.Images)-1; s++ {
		cur := d.Images[s]
		below := d.Images[s-1]
		above := d.Images[s+1]
		for y := 1; y < rows-1; y++ {
			for x := 1; x < cols-1; x++ {
				v := cur.Data[y*cols+x]
				if math.Abs(v) < threshold {
					continue
				}
				isMax, isMin := true, true
				for dy := -1; dy <= 1 && (isMax || isMin); dy++ {
					for dx := -1; dx <= 1; dx++ {
						idx := (y+dy)*cols + (x + dx)
						for _, lvl := range []*cv.FloatMat{below, cur, above} {
							nv := lvl.Data[idx]
							if lvl == cur && dx == 0 && dy == 0 {
								continue
							}
							if nv >= v {
								isMax = false
							}
							if nv <= v {
								isMin = false
							}
						}
					}
				}
				if isMax || isMin {
					kps = append(kps, Keypoint{
						X: x, Y: y, ScaleIndex: s, Sigma: d.Sigmas[s], Response: v,
					})
				}
			}
		}
	}
	return kps
}

package bgsegm

import (
	"math"
	"math/bits"
	"math/rand"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// BackgroundSubtractorLSBP is the Local SVD Binary Pattern background model of
// Guo, Wang, Qi, Zhang, Liu and Chang ("Background Subtraction using Local SVD
// Binary Pattern", 2016). It combines a per-pixel colour (intensity) model with
// a texture descriptor that is robust to illumination change. For every pixel a
// scalar local-SVD feature g is computed from the second singular value of its
// 3×3 intensity patch (zero on a flat patch, growing with local structure); the
// LSBP descriptor is the 8-bit pattern of which neighbours share the centre's g
// within Tau.
//
// The background is a ViBe-style sample consensus: each pixel stores NSamples
// past observations of (intensity, descriptor). A pixel is background when at
// least MinMatches stored samples match the observation both in intensity
// (absolute difference below IntensityThreshold) and in texture (descriptor
// Hamming distance at most LSBPThreshold). A near-exact intensity match within
// NoiseThreshold satisfies the texture test on its own, which keeps stationary
// background stable where a passing object momentarily distorts the local
// texture. Matching background samples are refreshed and propagated to a
// neighbour with the configured probabilities; the pseudo-random choices are
// seeded, so a given Seed and frame sequence always yield identical output.
//
// Construct one with [NewBackgroundSubtractorLSBP]; the zero value is not usable.
type BackgroundSubtractorLSBP struct {
	// ShadowParams supplies the embedded DetectShadows / ShadowValue /
	// ShadowThreshold configuration and their setters.
	ShadowParams

	// NSamples is the number of (intensity, descriptor) samples stored per pixel.
	NSamples int
	// MinMatches is the minimum number of matching samples for a background
	// classification.
	MinMatches int
	// IntensityThreshold is the absolute intensity difference within which a
	// stored sample matches the observation in colour.
	IntensityThreshold float64
	// LSBPThreshold is the maximum descriptor Hamming distance (0..8) that still
	// counts as a texture match.
	LSBPThreshold int
	// NoiseThreshold is a near-exact intensity distance below which the texture
	// test is skipped, absorbing sensor noise and thin texture artefacts.
	NoiseThreshold float64
	// Tau is the local-SVD-feature difference below which two neighbouring pixels
	// are deemed to share texture, setting a descriptor bit.
	Tau float64
	// SubsamplingFactor sets the update probability 1/SubsamplingFactor with
	// which a matched pixel refreshes one of its own samples and one neighbour's.
	SubsamplingFactor int
	// Seed seeds the deterministic pseudo-random sample replacement.
	Seed int64
	// OpenKernel, when greater than 1, morphologically opens the mask at that odd
	// size before Apply returns it (see [CleanupMask]).
	OpenKernel int

	rows, cols int
	samples    [][]lsbpSample
	rng        *rand.Rand
	inited     bool
}

// lsbpSample is one stored observation in a pixel's model.
type lsbpSample struct {
	intensity float64
	desc      uint8
}

// NewBackgroundSubtractorLSBP creates an LSBP subtractor. nSamples and
// intensityThreshold fall back to sensible defaults (8 and 30) when
// non-positive; detectShadows toggles shadow classification. The remaining
// tunables are initialised to working defaults on the returned value and may be
// overridden before the first Apply.
func NewBackgroundSubtractorLSBP(nSamples int, intensityThreshold float64, detectShadows bool) *BackgroundSubtractorLSBP {
	if nSamples <= 0 {
		nSamples = 8
	}
	if intensityThreshold <= 0 {
		intensityThreshold = 30
	}
	sp := defaultShadowParams()
	sp.DetectShadows = detectShadows
	return &BackgroundSubtractorLSBP{
		ShadowParams:       sp,
		NSamples:           nSamples,
		MinMatches:         2,
		IntensityThreshold: intensityThreshold,
		LSBPThreshold:      2,
		NoiseThreshold:     10,
		Tau:                1.0,
		SubsamplingFactor:  16,
		Seed:               1,
	}
}

func (b *BackgroundSubtractorLSBP) init(frame *cv.Mat, intensity []float64, desc []uint8) {
	b.rows, b.cols = frame.Rows, frame.Cols
	b.rng = rand.New(rand.NewSource(b.Seed))
	b.samples = make([][]lsbpSample, frame.Total())
	for p := range b.samples {
		row := make([]lsbpSample, b.NSamples)
		for i := range row {
			row[i] = lsbpSample{intensity: intensity[p], desc: desc[p]}
		}
		b.samples[p] = row
	}
	b.inited = true
}

// Apply computes the LSBP descriptors of frame, classifies each pixel against
// its sample consensus, updates the models and returns the foreground mask. See
// [BackgroundSubtractor].
func (b *BackgroundSubtractorLSBP) Apply(frame *cv.Mat) *cv.Mat {
	intensity := toIntensity(frame)
	g := localSVDFeature(intensity, frame.Rows, frame.Cols)
	desc := lsbpDescriptors(g, frame.Rows, frame.Cols, b.Tau)
	if !b.inited {
		b.init(frame, intensity, desc)
	} else {
		checkFrame(b.rows, b.cols, frame)
	}

	mask := newMask(b.rows, b.cols)
	for p := range b.samples {
		mask.Data[p] = b.classify(b.samples[p], intensity[p], desc[p])
	}
	// Update pass: refresh and propagate matched (background) pixels.
	for p := range b.samples {
		if mask.Data[p] != BackgroundValue {
			continue
		}
		if b.rng.Intn(b.SubsamplingFactor) == 0 {
			b.samples[p][b.rng.Intn(b.NSamples)] = lsbpSample{intensity: intensity[p], desc: desc[p]}
		}
		if b.rng.Intn(b.SubsamplingFactor) == 0 {
			np := b.randomNeighbour(p)
			b.samples[np][b.rng.Intn(b.NSamples)] = lsbpSample{intensity: intensity[p], desc: desc[p]}
		}
	}
	return applyCleanup(mask, b.OpenKernel)
}

// classify returns the mask sample for a pixel given its sample bank and the
// current observation.
func (b *BackgroundSubtractorLSBP) classify(samples []lsbpSample, v float64, desc uint8) uint8 {
	matches := 0
	nearest := math.Inf(1)
	var nearestI float64
	for _, s := range samples {
		d := math.Abs(v - s.intensity)
		if d < nearest {
			nearest = d
			nearestI = s.intensity
		}
		if d >= b.IntensityThreshold {
			continue
		}
		if d <= b.NoiseThreshold || hamming8(desc, s.desc) <= b.LSBPThreshold {
			matches++
			if matches >= b.MinMatches {
				return BackgroundValue
			}
		}
	}
	if b.isShadowOf(v, nearestI) {
		return b.shadowSample()
	}
	return ForegroundValue
}

// randomNeighbour returns a random 4-connected neighbour index of p, clamped to
// stay inside the image (falling back to p itself at a corner where every draw
// lands out of bounds).
func (b *BackgroundSubtractorLSBP) randomNeighbour(p int) int {
	y, x := p/b.cols, p%b.cols
	switch b.rng.Intn(4) {
	case 0:
		if y > 0 {
			return p - b.cols
		}
	case 1:
		if y < b.rows-1 {
			return p + b.cols
		}
	case 2:
		if x > 0 {
			return p - 1
		}
	default:
		if x < b.cols-1 {
			return p + 1
		}
	}
	return p
}

// GetBackgroundImage returns the per-pixel median stored intensity as a
// single-channel image, or nil before the first Apply.
func (b *BackgroundSubtractorLSBP) GetBackgroundImage() *cv.Mat {
	if !b.inited {
		return nil
	}
	out := cv.NewMat(b.rows, b.cols, 1)
	buf := make([]float64, b.NSamples)
	for p, samples := range b.samples {
		for i, s := range samples {
			buf[i] = s.intensity
		}
		sort.Float64s(buf)
		out.Data[p] = clampUint8(buf[len(buf)/2])
	}
	return out
}

// localSVDFeature computes, for every pixel, the second singular value of its
// 3×3 intensity patch (borders replicated). This scalar is zero on a flat patch
// and grows with local structure, forming the illumination-robust basis of the
// LSBP descriptor. The returned slice is row-major of length rows*cols.
func localSVDFeature(intensity []float64, rows, cols int) []float64 {
	g := make([]float64, rows*cols)
	var patch [3][3]float64
	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return intensity[y*cols+x]
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					patch[dy+1][dx+1] = at(y+dy, x+dx)
				}
			}
			sv := singularValues3(patch)
			g[y*cols+x] = sv[1] // second (middle) singular value
		}
	}
	return g
}

// lsbpDescriptors builds the 8-bit LSBP descriptor for every pixel from the
// local-SVD feature map: bit i is set when the i-th 8-neighbour's feature is
// within tau of the centre's. Borders are replicated. The returned slice is
// row-major of length rows*cols.
func lsbpDescriptors(g []float64, rows, cols int, tau float64) []uint8 {
	desc := make([]uint8, rows*cols)
	// Eight neighbour offsets in a fixed order around the centre.
	off := [8][2]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}}
	at := func(y, x int) float64 {
		if y < 0 {
			y = 0
		} else if y >= rows {
			y = rows - 1
		}
		if x < 0 {
			x = 0
		} else if x >= cols {
			x = cols - 1
		}
		return g[y*cols+x]
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			c := g[y*cols+x]
			var d uint8
			for i := 0; i < 8; i++ {
				if math.Abs(at(y+off[i][0], x+off[i][1])-c) < tau {
					d |= 1 << uint(i)
				}
			}
			desc[y*cols+x] = d
		}
	}
	return desc
}

// singularValues3 returns the three singular values of a 3×3 matrix in
// descending order. They are the square roots of the eigenvalues of AᵀA, found
// with a symmetric Jacobi eigensolver.
func singularValues3(a [3][3]float64) [3]float64 {
	// s = AᵀA (symmetric, positive semi-definite).
	var s [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			sum := 0.0
			for k := 0; k < 3; k++ {
				sum += a[k][i] * a[k][j]
			}
			s[i][j] = sum
		}
	}
	ev := jacobiEigenvalues3(s)
	sort.Sort(sort.Reverse(sort.Float64Slice(ev[:])))
	var out [3]float64
	for i := 0; i < 3; i++ {
		if ev[i] > 0 {
			out[i] = math.Sqrt(ev[i])
		}
	}
	return out
}

// jacobiEigenvalues3 returns the eigenvalues of a symmetric 3×3 matrix using
// cyclic Jacobi rotations. The order of the returned values is unspecified.
func jacobiEigenvalues3(m [3][3]float64) [3]float64 {
	a := m
	for sweep := 0; sweep < 12; sweep++ {
		off := math.Abs(a[0][1]) + math.Abs(a[0][2]) + math.Abs(a[1][2])
		if off < 1e-12 {
			break
		}
		for _, pq := range [3][2]int{{0, 1}, {0, 2}, {1, 2}} {
			p, q := pq[0], pq[1]
			apq := a[p][q]
			if math.Abs(apq) < 1e-18 {
				continue
			}
			theta := (a[q][q] - a[p][p]) / (2 * apq)
			t := sign(theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
			if theta == 0 {
				t = 1
			}
			c := 1 / math.Sqrt(t*t+1)
			sn := t * c
			app := a[p][p]
			aqq := a[q][q]
			a[p][p] = c*c*app - 2*sn*c*apq + sn*sn*aqq
			a[q][q] = sn*sn*app + 2*sn*c*apq + c*c*aqq
			a[p][q] = 0
			a[q][p] = 0
			for i := 0; i < 3; i++ {
				if i != p && i != q {
					aip := a[i][p]
					aiq := a[i][q]
					a[i][p] = c*aip - sn*aiq
					a[p][i] = a[i][p]
					a[i][q] = sn*aip + c*aiq
					a[q][i] = a[i][q]
				}
			}
		}
	}
	return [3]float64{a[0][0], a[1][1], a[2][2]}
}

// sign returns -1 for negative v and +1 otherwise.
func sign(v float64) float64 {
	if v < 0 {
		return -1
	}
	return 1
}

// hamming8 returns the number of differing bits between two 8-bit descriptors.
func hamming8(a, b uint8) int {
	return bits.OnesCount8(a ^ b)
}

var _ BackgroundSubtractor = (*BackgroundSubtractorLSBP)(nil)

package texture

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Direction identifies one of the four canonical neighbour orientations used to
// build a co-occurrence or run-length matrix. The angle is measured
// counter-clockwise from the positive x-axis; the corresponding pixel offset at
// a given distance d is returned by [Direction.Offset].
type Direction int

const (
	// Horizontal is the 0-degree direction: neighbour offset (d, 0).
	Horizontal Direction = iota
	// Diagonal45 is the 45-degree direction: neighbour offset (d, -d)
	// (up and to the right).
	Diagonal45
	// Vertical is the 90-degree direction: neighbour offset (0, -d) (up).
	Vertical
	// Diagonal135 is the 135-degree direction: neighbour offset (-d, -d)
	// (up and to the left).
	Diagonal135
)

// Offset returns the pixel column/row displacement (dx, dy) for this direction
// at the given distance. Positive dx moves right, positive dy moves down; the
// diagonal and vertical directions therefore return negative dy for "up". It
// panics if the direction is unknown.
func (dir Direction) Offset(distance int) (dx, dy int) {
	switch dir {
	case Horizontal:
		return distance, 0
	case Diagonal45:
		return distance, -distance
	case Vertical:
		return 0, -distance
	case Diagonal135:
		return -distance, -distance
	default:
		panic(fmt.Sprintf("texture: unknown Direction %d", int(dir)))
	}
}

// String returns a human-readable name for the direction.
func (dir Direction) String() string {
	switch dir {
	case Horizontal:
		return "Horizontal(0deg)"
	case Diagonal45:
		return "Diagonal45"
	case Vertical:
		return "Vertical(90deg)"
	case Diagonal135:
		return "Diagonal135"
	default:
		return fmt.Sprintf("Direction(%d)", int(dir))
	}
}

// GLCM is a gray-level co-occurrence matrix: a levels-by-levels table whose
// entry (i, j) counts how often a pixel of quantised gray level i is found next
// to a pixel of level j at a fixed spatial offset. It is the basis for the
// Haralick texture features. Construct one with [NewGLCM] or [ComputeGLCM].
type GLCM struct {
	levels int
	// counts holds the raw pair counts in row-major order (i*levels+j).
	counts []float64
	total  float64
}

// Levels returns the number of gray levels (the matrix side length).
func (g *GLCM) Levels() int { return g.levels }

// At returns the raw co-occurrence count for the ordered gray-level pair
// (i, j). It panics if either index is out of range.
func (g *GLCM) At(i, j int) float64 {
	if i < 0 || i >= g.levels || j < 0 || j >= g.levels {
		panic(fmt.Sprintf("texture: GLCM.At(%d,%d) out of range for %d levels", i, j, g.levels))
	}
	return g.counts[i*g.levels+j]
}

// Sum returns the total number of gray-level pairs tallied in the matrix.
func (g *GLCM) Sum() float64 { return g.total }

// Probabilities returns a fresh copy of the matrix normalised to sum to 1, in
// row-major order. If the matrix is empty every entry is 0.
func (g *GLCM) Probabilities() []float64 {
	p := make([]float64, len(g.counts))
	if g.total == 0 {
		return p
	}
	for i, c := range g.counts {
		p[i] = c / g.total
	}
	return p
}

// NewGLCM builds a co-occurrence matrix from img using an explicit pixel offset
// (dx, dy). The image is reduced to luminance and quantised into levels gray
// levels (levels must be >= 2). For every pixel whose (x+dx, y+dy) neighbour is
// in bounds, the ordered pair of quantised levels is tallied. If symmetric is
// true each pair is also counted in the reverse order, making the matrix
// symmetric (the usual choice, as it makes the descriptor direction-agnostic
// for +/- the offset). It panics on an empty image, levels < 2, or a zero
// offset.
func NewGLCM(img *cv.Mat, levels, dx, dy int, symmetric bool) *GLCM {
	textureRequire(img, "NewGLCM")
	if levels < 2 {
		panic(fmt.Sprintf("texture: NewGLCM requires levels >= 2, got %d", levels))
	}
	if dx == 0 && dy == 0 {
		panic("texture: NewGLCM requires a non-zero offset")
	}
	q := textureQuantize(textureLuma(img), levels)
	g := &GLCM{levels: levels, counts: make([]float64, levels*levels)}
	rows, cols := img.Rows, img.Cols
	for y := 0; y < rows; y++ {
		ny := y + dy
		if ny < 0 || ny >= rows {
			continue
		}
		for x := 0; x < cols; x++ {
			nx := x + dx
			if nx < 0 || nx >= cols {
				continue
			}
			a := q[y*cols+x]
			b := q[ny*cols+nx]
			g.counts[a*levels+b]++
			g.total++
			if symmetric {
				g.counts[b*levels+a]++
				g.total++
			}
		}
	}
	return g
}

// ComputeGLCM is a convenience wrapper over [NewGLCM] that takes a symbolic
// [Direction] and a distance instead of a raw offset. distance must be >= 1.
func ComputeGLCM(img *cv.Mat, levels int, dir Direction, distance int, symmetric bool) *GLCM {
	if distance < 1 {
		panic(fmt.Sprintf("texture: ComputeGLCM requires distance >= 1, got %d", distance))
	}
	dx, dy := dir.Offset(distance)
	return NewGLCM(img, levels, dx, dy, symmetric)
}

// marginals computes, from the normalised probabilities, the row/column means
// and standard deviations. Because the matrix may be symmetric the row and
// column marginals can differ only for asymmetric matrices, so both are
// returned.
func (g *GLCM) marginals(p []float64) (mux, muy, sx, sy float64) {
	L := g.levels
	px := make([]float64, L)
	py := make([]float64, L)
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			v := p[i*L+j]
			px[i] += v
			py[j] += v
		}
	}
	for i := 0; i < L; i++ {
		mux += float64(i) * px[i]
		muy += float64(i) * py[i]
	}
	for i := 0; i < L; i++ {
		sx += (float64(i) - mux) * (float64(i) - mux) * px[i]
		sy += (float64(i) - muy) * (float64(i) - muy) * py[i]
	}
	return mux, muy, math.Sqrt(sx), math.Sqrt(sy)
}

// Contrast returns the Haralick contrast (a.k.a. inertia): the sum of
// (i-j)^2 * p(i,j). It is large when the image has strong local intensity
// variation and 0 for a perfectly flat image.
func (g *GLCM) Contrast() float64 {
	p := g.Probabilities()
	L := g.levels
	var s float64
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			d := float64(i - j)
			s += d * d * p[i*L+j]
		}
	}
	return s
}

// Dissimilarity returns the sum of |i-j| * p(i,j), a linear analogue of
// contrast that weights large gray-level differences less heavily.
func (g *GLCM) Dissimilarity() float64 {
	p := g.Probabilities()
	L := g.levels
	var s float64
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			s += math.Abs(float64(i-j)) * p[i*L+j]
		}
	}
	return s
}

// Homogeneity returns the inverse difference moment, the sum of
// p(i,j)/(1+(i-j)^2). It is largest (1) for a diagonal matrix and small when
// off-diagonal pairs dominate, so it measures local uniformity.
func (g *GLCM) Homogeneity() float64 {
	p := g.Probabilities()
	L := g.levels
	var s float64
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			d := float64(i - j)
			s += p[i*L+j] / (1 + d*d)
		}
	}
	return s
}

// ASM returns the angular second moment, the sum of p(i,j)^2. It measures
// textural uniformity and is 1 for a single repeated pair. [GLCM.Energy] is its
// square root.
func (g *GLCM) ASM() float64 {
	p := g.Probabilities()
	var s float64
	for _, v := range p {
		s += v * v
	}
	return s
}

// Energy returns sqrt([GLCM.ASM]), the L2 norm of the probability matrix.
func (g *GLCM) Energy() float64 { return math.Sqrt(g.ASM()) }

// Entropy returns the Shannon entropy -sum p(i,j)*ln p(i,j) (natural log,
// nats). It is large for a busy, disordered texture and 0 when a single pair
// occurs.
func (g *GLCM) Entropy() float64 {
	p := g.Probabilities()
	var s float64
	for _, v := range p {
		if v > 0 {
			s -= v * math.Log(v)
		}
	}
	return s
}

// MaximumProbability returns the largest normalised entry of the matrix, the
// probability of the most common gray-level pair.
func (g *GLCM) MaximumProbability() float64 {
	p := g.Probabilities()
	var m float64
	for _, v := range p {
		if v > m {
			m = v
		}
	}
	return m
}

// Correlation returns the Haralick correlation, a measure in [-1, 1] of the
// linear dependency of gray levels between neighbouring pixels. It is computed
// as sum((i-mux)(j-muy)p(i,j))/(sx*sy); if either marginal standard deviation
// is zero (a flat image) it returns 0.
func (g *GLCM) Correlation() float64 {
	p := g.Probabilities()
	L := g.levels
	mux, muy, sx, sy := g.marginals(p)
	if sx == 0 || sy == 0 {
		return 0
	}
	var s float64
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			s += (float64(i) - mux) * (float64(j) - muy) * p[i*L+j]
		}
	}
	return s / (sx * sy)
}

// Autocorrelation returns the sum of i*j*p(i,j), a simple measure of the
// fineness and coarseness of texture.
func (g *GLCM) Autocorrelation() float64 {
	p := g.Probabilities()
	L := g.levels
	var s float64
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			s += float64(i) * float64(j) * p[i*L+j]
		}
	}
	return s
}

// ClusterShade returns the third-order cluster shade, sum((i+j-mux-muy)^3
// p(i,j)). Its sign reflects the skew of the co-occurrence distribution about
// its mean.
func (g *GLCM) ClusterShade() float64 {
	p := g.Probabilities()
	L := g.levels
	mux, muy, _, _ := g.marginals(p)
	var s float64
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			d := float64(i) + float64(j) - mux - muy
			s += d * d * d * p[i*L+j]
		}
	}
	return s
}

// ClusterProminence returns the fourth-order cluster prominence,
// sum((i+j-mux-muy)^4 p(i,j)); large values indicate an asymmetric,
// heavy-tailed co-occurrence distribution.
func (g *GLCM) ClusterProminence() float64 {
	p := g.Probabilities()
	L := g.levels
	mux, muy, _, _ := g.marginals(p)
	var s float64
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			d := float64(i) + float64(j) - mux - muy
			d2 := d * d
			s += d2 * d2 * p[i*L+j]
		}
	}
	return s
}

// sumProbs returns p_{x+y}(k) for k in [0, 2(L-1)].
func (g *GLCM) sumProbs(p []float64) []float64 {
	L := g.levels
	out := make([]float64, 2*L-1)
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			out[i+j] += p[i*L+j]
		}
	}
	return out
}

// diffProbs returns p_{x-y}(k) for k in [0, L-1].
func (g *GLCM) diffProbs(p []float64) []float64 {
	L := g.levels
	out := make([]float64, L)
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			d := i - j
			if d < 0 {
				d = -d
			}
			out[d] += p[i*L+j]
		}
	}
	return out
}

// SumAverage returns sum(k * p_{x+y}(k)), the mean of the distribution of
// neighbouring gray-level sums.
func (g *GLCM) SumAverage() float64 {
	s := g.sumProbs(g.Probabilities())
	var acc float64
	for k, v := range s {
		acc += float64(k) * v
	}
	return acc
}

// SumEntropy returns the entropy -sum p_{x+y}(k) ln p_{x+y}(k) of the
// gray-level-sum distribution (natural log).
func (g *GLCM) SumEntropy() float64 {
	s := g.sumProbs(g.Probabilities())
	var acc float64
	for _, v := range s {
		if v > 0 {
			acc -= v * math.Log(v)
		}
	}
	return acc
}

// SumVariance returns the Haralick sum variance, sum((k - SumEntropy)^2
// p_{x+y}(k)), using the sum entropy as the centre per Haralick's original
// definition.
func (g *GLCM) SumVariance() float64 {
	s := g.sumProbs(g.Probabilities())
	se := g.SumEntropy()
	var acc float64
	for k, v := range s {
		d := float64(k) - se
		acc += d * d * v
	}
	return acc
}

// DifferenceAverage returns sum(k * p_{x-y}(k)), the mean absolute gray-level
// difference between neighbouring pixels.
func (g *GLCM) DifferenceAverage() float64 {
	d := g.diffProbs(g.Probabilities())
	var acc float64
	for k, v := range d {
		acc += float64(k) * v
	}
	return acc
}

// DifferenceVariance returns the variance of the gray-level-difference
// distribution p_{x-y}, centred on its own mean [GLCM.DifferenceAverage].
func (g *GLCM) DifferenceVariance() float64 {
	d := g.diffProbs(g.Probabilities())
	mean := g.DifferenceAverage()
	var acc float64
	for k, v := range d {
		x := float64(k) - mean
		acc += x * x * v
	}
	return acc
}

// DifferenceEntropy returns the entropy -sum p_{x-y}(k) ln p_{x-y}(k) of the
// gray-level-difference distribution (natural log).
func (g *GLCM) DifferenceEntropy() float64 {
	d := g.diffProbs(g.Probabilities())
	var acc float64
	for _, v := range d {
		if v > 0 {
			acc -= v * math.Log(v)
		}
	}
	return acc
}

// HaralickFeatures bundles the standard scalar texture features derived from a
// single [GLCM]. Every field is documented on the method of the same name.
type HaralickFeatures struct {
	// ASM is the angular second moment; see [GLCM.ASM].
	ASM float64
	// Contrast is the inertia; see [GLCM.Contrast].
	Contrast float64
	// Correlation is the gray-level linear correlation; see [GLCM.Correlation].
	Correlation float64
	// Variance is the sum of (i-mu)^2 p(i,j); a measure of gray-level spread.
	Variance float64
	// Homogeneity is the inverse difference moment; see [GLCM.Homogeneity].
	Homogeneity float64
	// SumAverage; see [GLCM.SumAverage].
	SumAverage float64
	// SumVariance; see [GLCM.SumVariance].
	SumVariance float64
	// SumEntropy; see [GLCM.SumEntropy].
	SumEntropy float64
	// Entropy; see [GLCM.Entropy].
	Entropy float64
	// DifferenceVariance; see [GLCM.DifferenceVariance].
	DifferenceVariance float64
	// DifferenceEntropy; see [GLCM.DifferenceEntropy].
	DifferenceEntropy float64
	// Dissimilarity; see [GLCM.Dissimilarity].
	Dissimilarity float64
	// Energy is sqrt(ASM); see [GLCM.Energy].
	Energy float64
	// MaximumProbability; see [GLCM.MaximumProbability].
	MaximumProbability float64
}

// Variance returns the Haralick variance sum((i-mu)^2 p(i,j)) where mu is the
// overall matrix mean of the row index; it quantifies the spread of gray
// levels in the co-occurrence distribution.
func (g *GLCM) Variance() float64 {
	p := g.Probabilities()
	L := g.levels
	mux, _, _, _ := g.marginals(p)
	var s float64
	for i := 0; i < L; i++ {
		for j := 0; j < L; j++ {
			d := float64(i) - mux
			s += d * d * p[i*L+j]
		}
	}
	return s
}

// Haralick computes the full bundle of scalar features in a single pass-worth
// of calls and returns them as a [HaralickFeatures] value.
func (g *GLCM) Haralick() HaralickFeatures {
	return HaralickFeatures{
		ASM:                g.ASM(),
		Contrast:           g.Contrast(),
		Correlation:        g.Correlation(),
		Variance:           g.Variance(),
		Homogeneity:        g.Homogeneity(),
		SumAverage:         g.SumAverage(),
		SumVariance:        g.SumVariance(),
		SumEntropy:         g.SumEntropy(),
		Entropy:            g.Entropy(),
		DifferenceVariance: g.DifferenceVariance(),
		DifferenceEntropy:  g.DifferenceEntropy(),
		Dissimilarity:      g.Dissimilarity(),
		Energy:             g.Energy(),
		MaximumProbability: g.MaximumProbability(),
	}
}

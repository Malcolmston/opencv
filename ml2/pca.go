package ml2

import (
	"errors"
	"sort"
)

// PCA performs principal-component analysis: it finds the orthogonal directions
// of greatest variance in a dataset and projects data onto the leading ones,
// a standard tool for compressing high-dimensional image descriptors before
// classification. The covariance matrix is diagonalised with a symmetric Jacobi
// eigensolver.
type PCA struct {
	n int
	// Mean holds the per-feature mean subtracted before projection; populated
	// by Fit.
	Mean []float64
	// Components holds the principal axes, one per row, ordered by descending
	// explained variance; populated by Fit.
	Components [][]float64
	// ExplainedVariance holds the variance captured by each kept component.
	ExplainedVariance []float64
	// TotalVariance is the sum of all eigenvalues (variance of the full data).
	TotalVariance float64
}

// NewPCA returns a PCA that keeps nComponents axes. It panics if nComponents
// < 1.
func NewPCA(nComponents int) *PCA {
	if nComponents < 1 {
		panic("ml2: NewPCA requires nComponents >= 1")
	}
	return &PCA{n: nComponents}
}

// eigenPair couples an eigenvalue with its eigenvector for sorting.
type ml2eigenPair struct {
	value  float64
	vector []float64
}

// symmetricEigenDescending returns the eigenpairs of a symmetric matrix sorted
// by descending eigenvalue.
func ml2symmetricEigenDescending(sym [][]float64) []ml2eigenPair {
	vals, vecs := ml2jacobiEigen(sym)
	d := len(vals)
	pairs := make([]ml2eigenPair, d)
	for i := 0; i < d; i++ {
		vec := make([]float64, d)
		for r := 0; r < d; r++ {
			vec[r] = vecs[r][i]
		}
		pairs[i] = ml2eigenPair{value: vals[i], vector: vec}
	}
	sort.SliceStable(pairs, func(a, b int) bool { return pairs[a].value > pairs[b].value })
	return pairs
}

// Fit learns the mean and principal components of x. It returns an error if x
// is empty or has fewer features than the requested number of components.
func (p *PCA) Fit(x [][]float64) error {
	if len(x) == 0 {
		return errors.New("ml2: PCA.Fit given no samples")
	}
	d := len(x[0])
	if p.n > d {
		return errors.New("ml2: PCA.Fit requires nComponents <= number of features")
	}
	p.Mean = ml2columnMean(x)
	cov := ml2covariance(x)
	pairs := ml2symmetricEigenDescending(cov)
	p.TotalVariance = 0
	for _, pr := range pairs {
		if pr.value > 0 {
			p.TotalVariance += pr.value
		}
	}
	p.Components = make([][]float64, p.n)
	p.ExplainedVariance = make([]float64, p.n)
	for i := 0; i < p.n; i++ {
		p.Components[i] = ml2signCanonical(pairs[i].vector)
		p.ExplainedVariance[i] = pairs[i].value
	}
	return nil
}

// ml2signCanonical flips a vector so its largest-magnitude entry is positive,
// giving a deterministic sign convention for eigenvectors.
func ml2signCanonical(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	maxAbs, idx := 0.0, 0
	for i, x := range out {
		a := x
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs, idx = a, i
		}
	}
	if out[idx] < 0 {
		for i := range out {
			out[i] = -out[i]
		}
	}
	return out
}

// Transform projects x onto the principal components, returning a matrix with
// nComponents columns. It panics before Fit.
func (p *PCA) Transform(x [][]float64) [][]float64 {
	if p.Components == nil {
		panic("ml2: PCA.Transform before Fit")
	}
	out := make([][]float64, len(x))
	for i, row := range x {
		centered := make([]float64, len(row))
		for j := range row {
			centered[j] = row[j] - p.Mean[j]
		}
		proj := make([]float64, p.n)
		for c := 0; c < p.n; c++ {
			proj[c] = ml2dot(p.Components[c], centered)
		}
		out[i] = proj
	}
	return out
}

// FitTransform is Fit followed by Transform on the same data. It returns an
// error if Fit fails.
func (p *PCA) FitTransform(x [][]float64) ([][]float64, error) {
	if err := p.Fit(x); err != nil {
		return nil, err
	}
	return p.Transform(x), nil
}

// InverseTransform maps projected data back into the original feature space
// (approximately, since discarded components are lost). It panics before Fit.
func (p *PCA) InverseTransform(z [][]float64) [][]float64 {
	if p.Components == nil {
		panic("ml2: PCA.InverseTransform before Fit")
	}
	d := len(p.Mean)
	out := make([][]float64, len(z))
	for i, row := range z {
		rec := make([]float64, d)
		copy(rec, p.Mean)
		for c := 0; c < p.n; c++ {
			for j := 0; j < d; j++ {
				rec[j] += row[c] * p.Components[c][j]
			}
		}
		out[i] = rec
	}
	return out
}

// ExplainedVarianceRatio returns, per kept component, the fraction of the total
// dataset variance it explains. It panics before Fit.
func (p *PCA) ExplainedVarianceRatio() []float64 {
	if p.Components == nil {
		panic("ml2: PCA.ExplainedVarianceRatio before Fit")
	}
	out := make([]float64, p.n)
	if p.TotalVariance == 0 {
		return out
	}
	for i := 0; i < p.n; i++ {
		out[i] = p.ExplainedVariance[i] / p.TotalVariance
	}
	return out
}

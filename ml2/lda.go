package ml2

import "errors"

// LDA performs Fisher's linear discriminant analysis: a supervised projection
// that maximises between-class scatter relative to within-class scatter, giving
// axes that separate labelled classes as well as a linear map can. It is the
// classic complement to [PCA] for building compact, discriminative image
// features. The generalised symmetric eigenproblem is reduced to a standard one
// via a Cholesky factorisation of the within-class scatter.
type LDA struct {
	n int
	// Mean holds the global per-feature mean subtracted before projection;
	// populated by Fit.
	Mean []float64
	// Components holds the discriminant axes, one per row, ordered by
	// descending discriminant power; populated by Fit.
	Components [][]float64
}

// NewLDA returns an LDA that keeps nComponents discriminant axes. The usable
// number is at most (numClasses − 1); requesting more yields as many as exist.
// It panics if nComponents < 1.
func NewLDA(nComponents int) *LDA {
	if nComponents < 1 {
		panic("ml2: NewLDA requires nComponents >= 1")
	}
	return &LDA{n: nComponents}
}

// Fit learns the discriminant axes from labelled data. It returns an error for
// empty, mismatched or single-class input, or if the within-class scatter is
// singular even after ridge regularisation.
func (l *LDA) Fit(samples [][]float64, labels []int) error {
	if len(samples) == 0 {
		return errors.New("ml2: LDA.Fit given no samples")
	}
	if len(samples) != len(labels) {
		return errors.New("ml2: LDA.Fit requires len(samples) == len(labels)")
	}
	classes := ml2numClasses(labels)
	if classes < 2 {
		return errors.New("ml2: LDA.Fit requires at least two classes")
	}
	d := len(samples[0])
	l.Mean = ml2columnMean(samples)

	// Per-class means and counts.
	classMean := make([][]float64, classes)
	counts := make([]int, classes)
	for c := 0; c < classes; c++ {
		classMean[c] = make([]float64, d)
	}
	for i, s := range samples {
		c := labels[i]
		counts[c]++
		for j := 0; j < d; j++ {
			classMean[c][j] += s[j]
		}
	}
	for c := 0; c < classes; c++ {
		if counts[c] == 0 {
			continue
		}
		for j := 0; j < d; j++ {
			classMean[c][j] /= float64(counts[c])
		}
	}

	// Within-class scatter Sw and between-class scatter Sb.
	sw := zeros(d)
	sb := zeros(d)
	for i, s := range samples {
		c := labels[i]
		for a := 0; a < d; a++ {
			da := s[a] - classMean[c][a]
			for b := 0; b < d; b++ {
				sw[a][b] += da * (s[b] - classMean[c][b])
			}
		}
	}
	for c := 0; c < classes; c++ {
		if counts[c] == 0 {
			continue
		}
		for a := 0; a < d; a++ {
			da := classMean[c][a] - l.Mean[a]
			for b := 0; b < d; b++ {
				sb[a][b] += float64(counts[c]) * da * (classMean[c][b] - l.Mean[b])
			}
		}
	}

	// Solve Sb v = λ Sw v by whitening with the Cholesky factor of Sw.
	swReg := ml2addRidge(sw, 1e-6)
	lchol, ok := ml2cholesky(swReg)
	if !ok {
		return errors.New("ml2: LDA.Fit within-class scatter is not positive definite")
	}
	// M = L⁻¹ Sb L⁻ᵀ, symmetric.
	// First compute Y = L⁻¹ Sb (solve L Y = Sb column by column).
	y := zeros(d)
	for col := 0; col < d; col++ {
		rhs := make([]float64, d)
		for r := 0; r < d; r++ {
			rhs[r] = sb[r][col]
		}
		sol := ml2forwardSolve(lchol, rhs)
		for r := 0; r < d; r++ {
			y[r][col] = sol[r]
		}
	}
	// Then M = Y L⁻ᵀ  ⇒ solve Lᵀ Mᵀ = Yᵀ, i.e. for each row of Y treated as rhs.
	mMat := zeros(d)
	for row := 0; row < d; row++ {
		rhs := make([]float64, d)
		for cc := 0; cc < d; cc++ {
			rhs[cc] = y[row][cc]
		}
		sol := ml2backSolveT(lchol, rhs)
		copy(mMat[row], sol)
	}
	// Symmetrise to remove tiny asymmetries from round-off.
	for a := 0; a < d; a++ {
		for b := a + 1; b < d; b++ {
			avg := (mMat[a][b] + mMat[b][a]) / 2
			mMat[a][b], mMat[b][a] = avg, avg
		}
	}

	pairs := ml2symmetricEigenDescending(mMat)
	maxComp := classes - 1
	keep := l.n
	if keep > maxComp {
		keep = maxComp
	}
	if keep > d {
		keep = d
	}
	l.Components = make([][]float64, keep)
	for i := 0; i < keep; i++ {
		// Back-transform the whitened eigenvector: v = L⁻ᵀ u.
		v := ml2backSolveT(lchol, pairs[i].vector)
		l.Components[i] = ml2signCanonical(ml2normalizeVec(v))
	}
	return nil
}

// Transform projects samples onto the discriminant axes. It panics before Fit.
func (l *LDA) Transform(samples [][]float64) [][]float64 {
	if l.Components == nil {
		panic("ml2: LDA.Transform before Fit")
	}
	out := make([][]float64, len(samples))
	for i, row := range samples {
		centered := make([]float64, len(row))
		for j := range row {
			centered[j] = row[j] - l.Mean[j]
		}
		proj := make([]float64, len(l.Components))
		for c := range l.Components {
			proj[c] = ml2dot(l.Components[c], centered)
		}
		out[i] = proj
	}
	return out
}

// FitTransform is Fit followed by Transform on the same data. It returns an
// error if Fit fails.
func (l *LDA) FitTransform(samples [][]float64, labels []int) ([][]float64, error) {
	if err := l.Fit(samples, labels); err != nil {
		return nil, err
	}
	return l.Transform(samples), nil
}

// zeros allocates a d-by-d zero matrix.
func zeros(d int) [][]float64 {
	m := make([][]float64, d)
	for i := range m {
		m[i] = make([]float64, d)
	}
	return m
}

// ml2normalizeVec returns v scaled to unit L2 norm (zero vector unchanged).
func ml2normalizeVec(v []float64) []float64 {
	return Normalize(v)
}

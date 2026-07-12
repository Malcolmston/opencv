package face

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// FisherFaceRecognizer implements the Fisherfaces method (Belhumeur, Hespanha &
// Kriegman, 1997): a principal-component analysis for dimensionality reduction
// followed by a linear discriminant analysis (LDA) that maximises between-class
// scatter relative to within-class scatter. The PCA step first reduces the data
// to N−C dimensions (N training images, C classes) so the within-class scatter
// matrix is non-singular; the LDA step then finds at most C−1 discriminant
// axes. Compared with [EigenFaceRecognizer], Fisherfaces explicitly models
// class structure and is markedly more robust to illumination changes.
//
// Faces are reduced to luma and resampled to the first training image's
// geometry, exactly as for Eigenfaces. A query is projected through both the
// PCA and LDA stages and classified by nearest neighbour (Euclidean distance)
// in the discriminant subspace. Construct with [NewFisherFaceRecognizer]; the
// zero value is not usable.
type FisherFaceRecognizer struct {
	numComponents int
	rows, cols    int
	pca           *pcaModel
	lda           [][]float64 // kept discriminant axes, each length pca.dim()
	projections   [][]float64
	labels        []int
	trained       bool
}

// NewFisherFaceRecognizer returns an untrained recognizer. numComponents caps
// the number of retained discriminant axes; pass numComponents <= 0 to keep the
// full C−1 axes, which is the usual choice.
func NewFisherFaceRecognizer(numComponents int) *FisherFaceRecognizer {
	return &FisherFaceRecognizer{numComponents: numComponents}
}

// Train fits the PCA-then-LDA pipeline to the labelled images. It panics on
// malformed input, and additionally requires at least two distinct labels (LDA
// is undefined for a single class).
func (r *FisherFaceRecognizer) Train(images []*cv.Mat, labels []int) {
	validateTraining(images, labels)

	g0 := toGrayMat(images[0])
	r.rows, r.cols = g0.Rows, g0.Cols

	vectors := make([][]float64, len(images))
	for i, im := range images {
		vectors[i] = imageVector(im, r.rows, r.cols)
	}

	classes := distinctLabels(labels)
	c := len(classes)
	if c < 2 {
		panic("face: FisherFaceRecognizer.Train requires at least two classes")
	}
	n := len(images)

	// PCA to N−C dimensions keeps the within-class scatter invertible.
	pcaDim := n - c
	if pcaDim < 1 {
		pcaDim = 1
	}
	r.pca = computePCA(vectors, pcaDim)
	d := r.pca.dim()

	// Project every sample into the PCA subspace.
	proj := make([][]float64, n)
	for i, v := range vectors {
		proj[i] = r.pca.project(v)
	}

	// Within- and between-class scatter in the PCA subspace.
	sw, sb := scatterMatrices(proj, labels, classes, d)

	// Regularise Sw slightly so its whitening transform is well defined even
	// when a class has a single sample.
	for i := 0; i < d; i++ {
		sw[i][i] += 1e-6
	}

	// Whiten by Sw: eigendecompose Sw = U·Λ·Uᵀ and form W = U·Λ^(-1/2), so
	// that Wᵀ·Sw·W = I.
	swVal, swVec := jacobiEigen(sw)
	whiten := make([][]float64, d) // whiten[i][rr]
	for i := 0; i < d; i++ {
		whiten[i] = make([]float64, d)
	}
	for rr := 0; rr < d; rr++ {
		lam := swVal[rr]
		if lam < 1e-12 {
			lam = 1e-12
		}
		scale := 1 / math.Sqrt(lam)
		for i := 0; i < d; i++ {
			whiten[i][rr] = swVec[rr][i] * scale
		}
	}

	// Transform the between-class scatter into the whitened space:
	// M = Wᵀ·Sb·W, then eigendecompose. Its top eigenvectors are the LDA
	// directions in whitened coordinates.
	m := whitenedBetween(whiten, sb, d)
	mVal, mVec := jacobiEigen(m)

	// Number of discriminant axes: at most C−1, further capped by the request
	// and the available positive eigenvalues.
	keep := c - 1
	if r.numComponents > 0 && r.numComponents < keep {
		keep = r.numComponents
	}
	if keep > d {
		keep = d
	}
	posLimit := 0
	for posLimit < len(mVal) && mVal[posLimit] > 1e-12 {
		posLimit++
	}
	if keep > posLimit {
		keep = posLimit
	}
	if keep < 1 {
		keep = 1
	}

	// Map each whitened discriminant eigenvector back to PCA-space:
	// axis = W·q. Store as rows so projecting a PCA vector is a dot product.
	r.lda = make([][]float64, keep)
	for k := 0; k < keep; k++ {
		q := mVec[k]
		axis := make([]float64, d)
		for i := 0; i < d; i++ {
			var s float64
			for rr := 0; rr < d; rr++ {
				s += whiten[i][rr] * q[rr]
			}
			axis[i] = s
		}
		r.lda[k] = axis
	}

	// Final training projections through both stages.
	r.projections = make([][]float64, n)
	for i := range proj {
		r.projections[i] = ldaProject(r.lda, proj[i])
	}
	r.labels = append([]int(nil), labels...)
	r.trained = true
}

// Predict projects the query through the PCA and LDA stages and returns the
// nearest training label with its Euclidean distance (lower is more confident).
// It panics if the recognizer is untrained.
func (r *FisherFaceRecognizer) Predict(img *cv.Mat) (int, float64) {
	if !r.trained {
		panic("face: FisherFaceRecognizer.Predict before Train")
	}
	v := imageVector(img, r.rows, r.cols)
	p := r.pca.project(v)
	q := ldaProject(r.lda, p)
	idx, dist := nearestNeighbor(r.projections, q, euclidean)
	return r.labels[idx], dist
}

// NumComponents returns the number of discriminant axes retained after
// training. It returns 0 before training.
func (r *FisherFaceRecognizer) NumComponents() int {
	if !r.trained {
		return 0
	}
	return len(r.lda)
}

// Project returns the coordinates of img in the trained discriminant subspace.
// It panics if the recognizer is untrained.
func (r *FisherFaceRecognizer) Project(img *cv.Mat) []float64 {
	if !r.trained {
		panic("face: FisherFaceRecognizer.Project before Train")
	}
	p := r.pca.project(imageVector(img, r.rows, r.cols))
	return ldaProject(r.lda, p)
}

// distinctLabels returns the sorted set of distinct labels.
func distinctLabels(labels []int) []int {
	seen := make(map[int]struct{}, len(labels))
	var out []int
	for _, l := range labels {
		if _, ok := seen[l]; !ok {
			seen[l] = struct{}{}
			out = append(out, l)
		}
	}
	sort.Ints(out)
	return out
}

// scatterMatrices computes the within-class (Sw) and between-class (Sb) scatter
// matrices of d-dimensional samples grouped by label.
func scatterMatrices(samples [][]float64, labels, classes []int, d int) (sw, sb [][]float64) {
	// Overall mean.
	overall := make([]float64, d)
	for _, s := range samples {
		for j := 0; j < d; j++ {
			overall[j] += s[j]
		}
	}
	n := float64(len(samples))
	for j := 0; j < d; j++ {
		overall[j] /= n
	}

	// Per-class means and counts.
	means := make(map[int][]float64, len(classes))
	counts := make(map[int]int, len(classes))
	for _, cl := range classes {
		means[cl] = make([]float64, d)
	}
	for i, s := range samples {
		cl := labels[i]
		mv := means[cl]
		for j := 0; j < d; j++ {
			mv[j] += s[j]
		}
		counts[cl]++
	}
	for _, cl := range classes {
		cnt := float64(counts[cl])
		mv := means[cl]
		for j := 0; j < d; j++ {
			mv[j] /= cnt
		}
	}

	sw = newMatrix(d, d)
	sb = newMatrix(d, d)

	// Within-class scatter: Σ_c Σ_{x∈c} (x−μ_c)(x−μ_c)ᵀ.
	diff := make([]float64, d)
	for i, s := range samples {
		mv := means[labels[i]]
		for j := 0; j < d; j++ {
			diff[j] = s[j] - mv[j]
		}
		addOuter(sw, diff, diff)
	}

	// Between-class scatter: Σ_c n_c (μ_c−μ)(μ_c−μ)ᵀ.
	for _, cl := range classes {
		mv := means[cl]
		for j := 0; j < d; j++ {
			diff[j] = mv[j] - overall[j]
		}
		addOuterScaled(sb, diff, float64(counts[cl]))
	}
	return sw, sb
}

// whitenedBetween computes M = Wᵀ·Sb·W for the d×d whitening matrix whiten
// (indexed whiten[i][r]) and between-class scatter sb.
func whitenedBetween(whiten, sb [][]float64, d int) [][]float64 {
	// T = Sb·W (d×d).
	t := newMatrix(d, d)
	for i := 0; i < d; i++ {
		for r := 0; r < d; r++ {
			var s float64
			for kk := 0; kk < d; kk++ {
				s += sb[i][kk] * whiten[kk][r]
			}
			t[i][r] = s
		}
	}
	// M = Wᵀ·T (d×d), symmetric by construction.
	m := newMatrix(d, d)
	for a := 0; a < d; a++ {
		for b := a; b < d; b++ {
			var s float64
			for i := 0; i < d; i++ {
				s += whiten[i][a] * t[i][b]
			}
			m[a][b] = s
			m[b][a] = s
		}
	}
	return m
}

// ldaProject applies the stored discriminant axes to a PCA-space vector.
func ldaProject(axes [][]float64, p []float64) []float64 {
	out := make([]float64, len(axes))
	for k, axis := range axes {
		var s float64
		for j := range axis {
			s += axis[j] * p[j]
		}
		out[k] = s
	}
	return out
}

// newMatrix allocates a zero rows×cols matrix.
func newMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

// addOuter adds the outer product u·vᵀ into m in place.
func addOuter(m [][]float64, u, v []float64) {
	for i := range u {
		ui := u[i]
		if ui == 0 {
			continue
		}
		mi := m[i]
		for j := range v {
			mi[j] += ui * v[j]
		}
	}
}

// addOuterScaled adds scale·(u·uᵀ) into m in place.
func addOuterScaled(m [][]float64, u []float64, scale float64) {
	for i := range u {
		ui := u[i] * scale
		if ui == 0 {
			continue
		}
		mi := m[i]
		for j := range u {
			mi[j] += ui * u[j]
		}
	}
}

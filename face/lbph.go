package face

import (
	cv "github.com/malcolmston/opencv"
)

// LBPHFaceRecognizer implements Local Binary Pattern Histograms face
// recognition (Ahonen, Hadid & Pietikäinen, 2004). Each face is turned into an
// LBP code image (see [LBP]/[LBPUniform]), split into a GridX×GridY grid of
// cells, and summarised by concatenating the per-cell code histograms into one
// spatially-aware feature vector. A query face is described the same way and
// classified by nearest neighbour under the chi-square histogram distance.
//
// Unlike the holistic recognizers, LBPH is a local, texture-based method: it
// needs no common geometry (histograms are size-independent) and is inherently
// robust to monotonic illumination changes, because LBP codes depend only on
// the ordering of neighbouring intensities. Construct with
// [NewLBPHFaceRecognizer] or [NewLBPHFaceRecognizerWithParams]; the zero value
// is not usable.
//
// The neighbourhood is fixed at the classic radius-1, 8-neighbour 3×3 sampling.
type LBPHFaceRecognizer struct {
	// GridX and GridY are the number of cells the LBP image is divided into
	// horizontally and vertically.
	GridX, GridY int
	// Uniform selects uniform LBP labels (59-bin cell histograms) over the
	// full 256-bin histograms.
	Uniform bool

	histograms [][]float64
	labels     []int
	trained    bool
}

// NewLBPHFaceRecognizer returns an untrained recognizer with the OpenCV default
// geometry: an 8×8 grid and full 256-bin (non-uniform) histograms.
func NewLBPHFaceRecognizer() *LBPHFaceRecognizer {
	return &LBPHFaceRecognizer{GridX: 8, GridY: 8, Uniform: false}
}

// NewLBPHFaceRecognizerWithParams returns an untrained recognizer with an
// explicit grid and histogram type. It panics if gridX or gridY is not
// positive.
func NewLBPHFaceRecognizerWithParams(gridX, gridY int, uniform bool) *LBPHFaceRecognizer {
	if gridX < 1 || gridY < 1 {
		panic("face: LBPH grid dimensions must be positive")
	}
	return &LBPHFaceRecognizer{GridX: gridX, GridY: gridY, Uniform: uniform}
}

// bins returns the number of histogram bins per cell for the configured mode.
func (r *LBPHFaceRecognizer) bins() int {
	if r.Uniform {
		return 59
	}
	return 256
}

// Train computes and stores the spatial LBP histogram of every labelled image.
// It panics on malformed input (see [FaceRecognizer]). Grid defaults are filled
// in when the struct was created without a constructor.
func (r *LBPHFaceRecognizer) Train(images []*cv.Mat, labels []int) {
	validateTraining(images, labels)
	if r.GridX < 1 {
		r.GridX = 8
	}
	if r.GridY < 1 {
		r.GridY = 8
	}
	r.histograms = make([][]float64, len(images))
	for i, im := range images {
		r.histograms[i] = r.spatialHistogram(im)
	}
	r.labels = append([]int(nil), labels...)
	r.trained = true
}

// Predict describes the query face and returns the label of the nearest stored
// histogram together with their chi-square distance (lower is more confident).
// It panics if the recognizer is untrained.
func (r *LBPHFaceRecognizer) Predict(img *cv.Mat) (int, float64) {
	if !r.trained {
		panic("face: LBPHFaceRecognizer.Predict before Train")
	}
	q := r.spatialHistogram(img)
	idx, dist := nearestNeighbor(r.histograms, q, chiSquareDistance)
	return r.labels[idx], dist
}

// spatialHistogram builds the concatenated per-cell LBP histogram for one
// image. The LBP image is partitioned into GridY rows by GridX columns of
// roughly equal size; each cell contributes an independent histogram, and the
// cells are concatenated in row-major order.
func (r *LBPHFaceRecognizer) spatialHistogram(img *cv.Mat) []float64 {
	var lbp *cv.Mat
	if r.Uniform {
		lbp = LBPUniform(img)
	} else {
		lbp = LBP(img)
	}
	bins := r.bins()
	rows, cols := lbp.Rows, lbp.Cols
	feat := make([]float64, r.GridX*r.GridY*bins)
	for gy := 0; gy < r.GridY; gy++ {
		y0 := gy * rows / r.GridY
		y1 := (gy + 1) * rows / r.GridY
		for gx := 0; gx < r.GridX; gx++ {
			x0 := gx * cols / r.GridX
			x1 := (gx + 1) * cols / r.GridX
			cellBase := (gy*r.GridX + gx) * bins
			for y := y0; y < y1; y++ {
				rowBase := y * cols
				for x := x0; x < x1; x++ {
					feat[cellBase+int(lbp.Data[rowBase+x])]++
				}
			}
		}
	}
	return feat
}

// chiSquareDistance returns the chi-square distance between two equal-length
// histograms: Σ (a−b)² / (a+b), summed over bins where a+b > 0. Smaller means
// more similar; identical histograms score 0.
func chiSquareDistance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		denom := a[i] + b[i]
		if denom > 0 {
			sum += d * d / denom
		}
	}
	return sum
}

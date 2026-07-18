package texture

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// RunLengthMatrix is a gray-level run-length matrix (GLRLM): entry (i, j)
// counts the number of runs — maximal sequences of collinear, consecutive
// pixels sharing the same quantised gray level — that have gray level i and
// length j, measured along a fixed [Direction]. It underlies the run-length
// texture statistics of Galloway. Build one with [NewRunLengthMatrix].
type RunLengthMatrix struct {
	levels int
	maxRun int
	// counts[i*maxRun + (j-1)] holds the number of runs of level i, length j.
	counts []float64
	pixels int // total number of pixels (Np)
	runs   float64
}

// Levels returns the number of gray levels.
func (rl *RunLengthMatrix) Levels() int { return rl.levels }

// MaxRun returns the longest run length the matrix can represent (its column
// count).
func (rl *RunLengthMatrix) MaxRun() int { return rl.maxRun }

// At returns the number of runs of gray level i (0-based) and length j (1-based).
// It panics if the indices are out of range.
func (rl *RunLengthMatrix) At(i, j int) float64 {
	if i < 0 || i >= rl.levels || j < 1 || j > rl.maxRun {
		panic(fmt.Sprintf("texture: RunLengthMatrix.At(%d,%d) out of range", i, j))
	}
	return rl.counts[i*rl.maxRun+(j-1)]
}

// TotalRuns returns Nr, the total number of runs tallied in the matrix.
func (rl *RunLengthMatrix) TotalRuns() float64 { return rl.runs }

// NewRunLengthMatrix builds the run-length matrix of img along dir. The image
// is reduced to luminance and quantised into levels gray levels (levels >= 2).
// A run is a maximal set of consecutive pixels of equal quantised level along
// the direction; each run is counted exactly once. It panics on an empty image
// or levels < 2.
func NewRunLengthMatrix(img *cv.Mat, levels int, dir Direction) *RunLengthMatrix {
	textureRequire(img, "NewRunLengthMatrix")
	if levels < 2 {
		panic(fmt.Sprintf("texture: NewRunLengthMatrix requires levels >= 2, got %d", levels))
	}
	rows, cols := img.Rows, img.Cols
	q := textureQuantize(textureLuma(img), levels)
	dx, dy := dir.Offset(1)
	maxRun := rows
	if cols > maxRun {
		maxRun = cols
	}
	rl := &RunLengthMatrix{
		levels: levels,
		maxRun: maxRun,
		counts: make([]float64, levels*maxRun),
		pixels: rows * cols,
	}
	inb := func(x, y int) bool { return x >= 0 && x < cols && y >= 0 && y < rows }
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Only start a run where the previous pixel along the direction is
			// out of bounds or a different level.
			px, py := x-dx, y-dy
			lvl := q[y*cols+x]
			if inb(px, py) && q[py*cols+px] == lvl {
				continue
			}
			// Walk the run forward.
			length := 0
			cx, cy := x, y
			for inb(cx, cy) && q[cy*cols+cx] == lvl {
				length++
				cx += dx
				cy += dy
			}
			if length > maxRun {
				length = maxRun
			}
			rl.counts[lvl*maxRun+(length-1)]++
			rl.runs++
		}
	}
	return rl
}

// ShortRunEmphasis returns SRE = (1/Nr) sum P(i,j)/j^2, which is large for
// textures dominated by short runs (fine texture).
func (rl *RunLengthMatrix) ShortRunEmphasis() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		for j := 1; j <= rl.maxRun; j++ {
			s += rl.counts[i*rl.maxRun+(j-1)] / float64(j*j)
		}
	}
	return s / rl.runs
}

// LongRunEmphasis returns LRE = (1/Nr) sum P(i,j)*j^2, large for textures
// dominated by long runs (coarse texture).
func (rl *RunLengthMatrix) LongRunEmphasis() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		for j := 1; j <= rl.maxRun; j++ {
			s += rl.counts[i*rl.maxRun+(j-1)] * float64(j*j)
		}
	}
	return s / rl.runs
}

// GrayLevelNonUniformity returns GLN = (1/Nr) sum_i (sum_j P(i,j))^2, which is
// low when runs are spread evenly across gray levels and high when a few levels
// dominate.
func (rl *RunLengthMatrix) GrayLevelNonUniformity() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		var rowsum float64
		for j := 1; j <= rl.maxRun; j++ {
			rowsum += rl.counts[i*rl.maxRun+(j-1)]
		}
		s += rowsum * rowsum
	}
	return s / rl.runs
}

// RunLengthNonUniformity returns RLN = (1/Nr) sum_j (sum_i P(i,j))^2, low when
// run lengths are evenly distributed and high when one length dominates.
func (rl *RunLengthMatrix) RunLengthNonUniformity() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for j := 1; j <= rl.maxRun; j++ {
		var colsum float64
		for i := 0; i < rl.levels; i++ {
			colsum += rl.counts[i*rl.maxRun+(j-1)]
		}
		s += colsum * colsum
	}
	return s / rl.runs
}

// RunPercentage returns RP = Nr/Np, the ratio of runs to pixels. It is 1 when
// every run has length 1 (no coherent texture) and small for coarse texture.
func (rl *RunLengthMatrix) RunPercentage() float64 {
	if rl.pixels == 0 {
		return 0
	}
	return rl.runs / float64(rl.pixels)
}

// LowGrayLevelRunEmphasis returns LGRE = (1/Nr) sum P(i,j)/(i+1)^2, emphasising
// runs of low gray level.
func (rl *RunLengthMatrix) LowGrayLevelRunEmphasis() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		gi := float64(i + 1)
		for j := 1; j <= rl.maxRun; j++ {
			s += rl.counts[i*rl.maxRun+(j-1)] / (gi * gi)
		}
	}
	return s / rl.runs
}

// HighGrayLevelRunEmphasis returns HGRE = (1/Nr) sum P(i,j)*(i+1)^2,
// emphasising runs of high gray level.
func (rl *RunLengthMatrix) HighGrayLevelRunEmphasis() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		gi := float64(i + 1)
		for j := 1; j <= rl.maxRun; j++ {
			s += rl.counts[i*rl.maxRun+(j-1)] * gi * gi
		}
	}
	return s / rl.runs
}

// ShortRunLowGrayLevelEmphasis returns SRLGE = (1/Nr) sum
// P(i,j)/((i+1)^2 j^2).
func (rl *RunLengthMatrix) ShortRunLowGrayLevelEmphasis() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		gi := float64(i + 1)
		for j := 1; j <= rl.maxRun; j++ {
			s += rl.counts[i*rl.maxRun+(j-1)] / (gi * gi * float64(j*j))
		}
	}
	return s / rl.runs
}

// ShortRunHighGrayLevelEmphasis returns SRHGE = (1/Nr) sum
// P(i,j)*(i+1)^2/j^2.
func (rl *RunLengthMatrix) ShortRunHighGrayLevelEmphasis() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		gi := float64(i + 1)
		for j := 1; j <= rl.maxRun; j++ {
			s += rl.counts[i*rl.maxRun+(j-1)] * gi * gi / float64(j*j)
		}
	}
	return s / rl.runs
}

// LongRunLowGrayLevelEmphasis returns LRLGE = (1/Nr) sum
// P(i,j)*j^2/(i+1)^2.
func (rl *RunLengthMatrix) LongRunLowGrayLevelEmphasis() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		gi := float64(i + 1)
		for j := 1; j <= rl.maxRun; j++ {
			s += rl.counts[i*rl.maxRun+(j-1)] * float64(j*j) / (gi * gi)
		}
	}
	return s / rl.runs
}

// LongRunHighGrayLevelEmphasis returns LRHGE = (1/Nr) sum
// P(i,j)*j^2*(i+1)^2.
func (rl *RunLengthMatrix) LongRunHighGrayLevelEmphasis() float64 {
	if rl.runs == 0 {
		return 0
	}
	var s float64
	for i := 0; i < rl.levels; i++ {
		gi := float64(i + 1)
		for j := 1; j <= rl.maxRun; j++ {
			s += rl.counts[i*rl.maxRun+(j-1)] * float64(j*j) * gi * gi
		}
	}
	return s / rl.runs
}

// RunLengthFeatures bundles the eleven standard Galloway run-length statistics.
// Each field is documented on the [RunLengthMatrix] method of the same name.
type RunLengthFeatures struct {
	ShortRunEmphasis              float64 // see [RunLengthMatrix.ShortRunEmphasis]
	LongRunEmphasis               float64 // see [RunLengthMatrix.LongRunEmphasis]
	GrayLevelNonUniformity        float64 // see [RunLengthMatrix.GrayLevelNonUniformity]
	RunLengthNonUniformity        float64 // see [RunLengthMatrix.RunLengthNonUniformity]
	RunPercentage                 float64 // see [RunLengthMatrix.RunPercentage]
	LowGrayLevelRunEmphasis       float64 // see [RunLengthMatrix.LowGrayLevelRunEmphasis]
	HighGrayLevelRunEmphasis      float64 // see [RunLengthMatrix.HighGrayLevelRunEmphasis]
	ShortRunLowGrayLevelEmphasis  float64 // see [RunLengthMatrix.ShortRunLowGrayLevelEmphasis]
	ShortRunHighGrayLevelEmphasis float64 // see [RunLengthMatrix.ShortRunHighGrayLevelEmphasis]
	LongRunLowGrayLevelEmphasis   float64 // see [RunLengthMatrix.LongRunLowGrayLevelEmphasis]
	LongRunHighGrayLevelEmphasis  float64 // see [RunLengthMatrix.LongRunHighGrayLevelEmphasis]
}

// Features computes all eleven run-length statistics and returns them as a
// [RunLengthFeatures] value.
func (rl *RunLengthMatrix) Features() RunLengthFeatures {
	return RunLengthFeatures{
		ShortRunEmphasis:              rl.ShortRunEmphasis(),
		LongRunEmphasis:               rl.LongRunEmphasis(),
		GrayLevelNonUniformity:        rl.GrayLevelNonUniformity(),
		RunLengthNonUniformity:        rl.RunLengthNonUniformity(),
		RunPercentage:                 rl.RunPercentage(),
		LowGrayLevelRunEmphasis:       rl.LowGrayLevelRunEmphasis(),
		HighGrayLevelRunEmphasis:      rl.HighGrayLevelRunEmphasis(),
		ShortRunLowGrayLevelEmphasis:  rl.ShortRunLowGrayLevelEmphasis(),
		ShortRunHighGrayLevelEmphasis: rl.ShortRunHighGrayLevelEmphasis(),
		LongRunLowGrayLevelEmphasis:   rl.LongRunLowGrayLevelEmphasis(),
		LongRunHighGrayLevelEmphasis:  rl.LongRunHighGrayLevelEmphasis(),
	}
}

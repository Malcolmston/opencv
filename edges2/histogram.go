package edges2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// OrientationHistogram builds a magnitude-weighted histogram of gradient
// orientations from a [GradientField]. The angular range is split into bins
// equal-width bins; when signed is true the range is [0,360) degrees, otherwise
// opposite directions are folded together over [0,180). Each pixel adds its
// gradient magnitude to the bin of its orientation. The returned slice has
// length bins. It panics if bins is not positive.
func OrientationHistogram(f *GradientField, bins int, signed bool) []float64 {
	if bins <= 0 {
		panic("edges2: OrientationHistogram requires bins > 0")
	}
	span := 180.0
	if signed {
		span = 360.0
	}
	hist := make([]float64, bins)
	for i := range f.Gx.Data {
		gx := f.Gx.Data[i]
		gy := f.Gy.Data[i]
		mag := math.Hypot(gx, gy)
		if mag == 0 {
			continue
		}
		a := math.Atan2(gy, gx) * 180 / math.Pi
		if signed {
			if a < 0 {
				a += 360
			}
		} else {
			if a < 0 {
				a += 180
			}
			if a >= 180 {
				a -= 180
			}
		}
		b := int(a / span * float64(bins))
		if b >= bins {
			b = bins - 1
		}
		hist[b] += mag
	}
	return hist
}

// HOGOptions configures the [HOG] descriptor.
type HOGOptions struct {
	// CellSize is the side length, in pixels, of the square cells over which
	// orientation histograms are pooled.
	CellSize int
	// BlockSize is the side length, in cells, of the square blocks that are
	// jointly normalised.
	BlockSize int
	// Bins is the number of orientation bins per cell histogram.
	Bins int
	// Signed selects the [0,360) orientation range when true, otherwise the
	// unsigned [0,180) range.
	Signed bool
}

// DefaultHOGOptions returns the conventional HOG parameters: 8×8-pixel cells,
// 2×2-cell blocks, nine unsigned orientation bins.
func DefaultHOGOptions() HOGOptions {
	return HOGOptions{CellSize: 8, BlockSize: 2, Bins: 9, Signed: false}
}

// HOG computes a dense histogram-of-oriented-gradients descriptor of a
// single-channel image. It pools magnitude-weighted gradient orientations into
// per-cell histograms, then L2-normalises overlapping blocks of cells (with a
// one-cell stride) and concatenates them into a single feature vector. The
// descriptor length is
// blocksY*blocksX*BlockSize*BlockSize*Bins, where blocksY and blocksX are the
// numbers of block positions along each axis. It panics on multi-channel input
// or non-positive options.
func HOG(src *cv.Mat, opts HOGOptions) []float64 {
	edges2RequireGray(src, "HOG")
	if opts.CellSize <= 0 || opts.BlockSize <= 0 || opts.Bins <= 0 {
		panic("edges2: HOG requires positive options")
	}
	f := Sobel(src)
	cellsY := src.Rows / opts.CellSize
	cellsX := src.Cols / opts.CellSize
	if cellsY < opts.BlockSize || cellsX < opts.BlockSize {
		panic("edges2: HOG image too small for the chosen cell and block size")
	}
	span := 180.0
	if opts.Signed {
		span = 360.0
	}
	// Per-cell orientation histograms.
	cellHist := make([][]float64, cellsY*cellsX)
	for i := range cellHist {
		cellHist[i] = make([]float64, opts.Bins)
	}
	for cy := 0; cy < cellsY; cy++ {
		for cx := 0; cx < cellsX; cx++ {
			h := cellHist[cy*cellsX+cx]
			for py := 0; py < opts.CellSize; py++ {
				y := cy*opts.CellSize + py
				for px := 0; px < opts.CellSize; px++ {
					x := cx*opts.CellSize + px
					gx, gy := f.At(y, x)
					mag := math.Hypot(gx, gy)
					if mag == 0 {
						continue
					}
					a := math.Atan2(gy, gx) * 180 / math.Pi
					if opts.Signed {
						if a < 0 {
							a += 360
						}
					} else {
						if a < 0 {
							a += 180
						}
						if a >= 180 {
							a -= 180
						}
					}
					b := int(a / span * float64(opts.Bins))
					if b >= opts.Bins {
						b = opts.Bins - 1
					}
					h[b] += mag
				}
			}
		}
	}
	// Block normalisation with a one-cell stride.
	blocksY := cellsY - opts.BlockSize + 1
	blocksX := cellsX - opts.BlockSize + 1
	perBlock := opts.BlockSize * opts.BlockSize * opts.Bins
	desc := make([]float64, 0, blocksY*blocksX*perBlock)
	const eps = 1e-6
	for by := 0; by < blocksY; by++ {
		for bx := 0; bx < blocksX; bx++ {
			block := make([]float64, 0, perBlock)
			var norm float64
			for iy := 0; iy < opts.BlockSize; iy++ {
				for ix := 0; ix < opts.BlockSize; ix++ {
					h := cellHist[(by+iy)*cellsX+(bx+ix)]
					for _, v := range h {
						block = append(block, v)
						norm += v * v
					}
				}
			}
			norm = math.Sqrt(norm + eps*eps)
			for _, v := range block {
				desc = append(desc, v/norm)
			}
		}
	}
	return desc
}

// StructuredEdges computes a structured edge-strength map by combining Sobel
// gradient magnitude across several Gaussian scales (sigma, 2*sigma, 4*sigma),
// which emphasises edges that persist across scales while attenuating isolated
// noise. The result is normalised to the [0,255] range and returned as an 8-bit
// [cv.Mat].
//
// This is a deterministic, model-free approximation and is not the trained
// structured-forest detector of Dollár and Zitnick; no learned model or
// external data is used.
func StructuredEdges(src *cv.Mat, sigma float64) *cv.Mat {
	edges2RequireGray(src, "StructuredEdges")
	if sigma <= 0 {
		sigma = 1.0
	}
	acc := NewFloatGrid(src.Rows, src.Cols)
	scales := []float64{sigma, 2 * sigma, 4 * sigma}
	for _, s := range scales {
		mag := Sobel(edges2Blur(src, s)).Magnitude()
		for i, v := range mag.Data {
			acc.Data[i] += v
		}
	}
	for i := range acc.Data {
		acc.Data[i] /= float64(len(scales))
	}
	return acc.ToMatNormalized()
}

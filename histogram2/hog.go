package histogram2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// HOGDescriptor computes Histogram of Oriented Gradients feature vectors, the
// classic dense descriptor of Dalal and Triggs used for pedestrian and object
// detection. Construct one with [NewHOGDescriptor] and evaluate it with
// [HOGDescriptor.Compute].
type HOGDescriptor struct {
	// CellSize is the side length in pixels of a square cell.
	CellSize int
	// BlockSize is the side length in cells of a square block used for
	// contrast normalisation.
	BlockSize int
	// Bins is the number of orientation bins spanning the unsigned angle
	// range [0,180).
	Bins int
}

// NewHOGDescriptor returns a HOG descriptor with the given cell size (pixels),
// block size (cells) and orientation-bin count. It panics if any parameter is
// not positive.
func NewHOGDescriptor(cellSize, blockSize, bins int) *HOGDescriptor {
	if cellSize <= 0 || blockSize <= 0 || bins <= 0 {
		panic("histogram2: NewHOGDescriptor requires positive parameters")
	}
	return &HOGDescriptor{CellSize: cellSize, BlockSize: blockSize, Bins: bins}
}

// FeatureLength returns the length of the descriptor vector [HOGDescriptor.Compute]
// produces for an image of the given size, which is
// nBlocksX*nBlocksY*BlockSize*BlockSize*Bins. It returns 0 when the image is too
// small to contain a single block.
func (h *HOGDescriptor) FeatureLength(rows, cols int) int {
	cellsX := cols / h.CellSize
	cellsY := rows / h.CellSize
	nbx := cellsX - h.BlockSize + 1
	nby := cellsY - h.BlockSize + 1
	if nbx <= 0 || nby <= 0 {
		return 0
	}
	return nbx * nby * h.BlockSize * h.BlockSize * h.Bins
}

// histogram2gray extracts a grayscale float plane from src: a three-channel
// image is converted with the ITU-R luma weights, a single-channel image is
// used directly, and any other channel count falls back to channel 0.
func histogram2gray(src *cv.Mat) []float64 {
	total := src.Total()
	out := make([]float64, total)
	switch src.Channels {
	case 1:
		for p := 0; p < total; p++ {
			out[p] = float64(src.Data[p])
		}
	case 3:
		g := cv.CvtColor(src, cv.ColorRGB2Gray)
		for p := 0; p < total; p++ {
			out[p] = float64(g.Data[p])
		}
	default:
		ch := src.Channels
		for p := 0; p < total; p++ {
			out[p] = float64(src.Data[p*ch])
		}
	}
	return out
}

// Compute evaluates the HOG descriptor of src and returns the concatenated,
// block-normalised feature vector. The image is reduced to grayscale, gradient
// magnitudes and orientations are accumulated into cell histograms with soft
// (linear) orientation binning, and overlapping blocks of cells are L2
// normalised. It returns [ErrEmptyImage] if src is empty and [ErrInvalidArgument]
// if the image is smaller than one block.
func (h *HOGDescriptor) Compute(src *cv.Mat) ([]float64, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	rows, cols := src.Rows, src.Cols
	cellsX := cols / h.CellSize
	cellsY := rows / h.CellSize
	if cellsX < h.BlockSize || cellsY < h.BlockSize {
		return nil, ErrInvalidArgument
	}
	gray := histogram2gray(src)

	// Cell orientation histograms, indexed [cellY][cellX][bin] flattened.
	cellHist := make([]float64, cellsY*cellsX*h.Bins)
	binWidth := 180.0 / float64(h.Bins)

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
		return gray[y*cols+x]
	}

	// Only pixels inside the cell grid contribute.
	usableW := cellsX * h.CellSize
	usableH := cellsY * h.CellSize
	for y := 0; y < usableH; y++ {
		cy := y / h.CellSize
		for x := 0; x < usableW; x++ {
			gx := at(y, x+1) - at(y, x-1)
			gy := at(y+1, x) - at(y-1, x)
			mag := math.Hypot(gx, gy)
			if mag == 0 {
				continue
			}
			ang := math.Atan2(gy, gx) * 180 / math.Pi
			if ang < 0 {
				ang += 180
			}
			if ang >= 180 {
				ang -= 180
			}
			// Soft binning between the two nearest bin centres.
			pos := ang/binWidth - 0.5
			b0 := int(math.Floor(pos))
			frac := pos - float64(b0)
			b1 := b0 + 1
			b0m := ((b0 % h.Bins) + h.Bins) % h.Bins
			b1m := ((b1 % h.Bins) + h.Bins) % h.Bins
			cx := x / h.CellSize
			base := (cy*cellsX + cx) * h.Bins
			cellHist[base+b0m] += mag * (1 - frac)
			cellHist[base+b1m] += mag * frac
		}
	}

	// Block normalisation over overlapping BlockSize x BlockSize cell windows.
	nbx := cellsX - h.BlockSize + 1
	nby := cellsY - h.BlockSize + 1
	blockLen := h.BlockSize * h.BlockSize * h.Bins
	feat := make([]float64, 0, nbx*nby*blockLen)
	const eps = 1e-6
	block := make([]float64, blockLen)
	for by := 0; by < nby; by++ {
		for bx := 0; bx < nbx; bx++ {
			k := 0
			var norm float64
			for iy := 0; iy < h.BlockSize; iy++ {
				for ix := 0; ix < h.BlockSize; ix++ {
					cbase := ((by+iy)*cellsX + (bx + ix)) * h.Bins
					for b := 0; b < h.Bins; b++ {
						v := cellHist[cbase+b]
						block[k] = v
						norm += v * v
						k++
					}
				}
			}
			inv := 1.0 / math.Sqrt(norm+eps*eps)
			for i := 0; i < blockLen; i++ {
				feat = append(feat, block[i]*inv)
			}
		}
	}
	return feat, nil
}

// HOG is a convenience wrapper that computes a Histogram of Oriented Gradients
// descriptor of src using the given cell size and orientation-bin count with a
// fixed 2x2 block size. It returns the same errors as [HOGDescriptor.Compute].
func HOG(src *cv.Mat, cellSize, bins int) ([]float64, error) {
	return NewHOGDescriptor(cellSize, 2, bins).Compute(src)
}

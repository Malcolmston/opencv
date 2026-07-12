package objdetect

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Size is a width/height pair in pixels, the analogue of OpenCV's cv::Size used
// to configure the HOG geometry.
type Size struct {
	Width  int
	Height int
}

// HOGDescriptor computes a Histogram of Oriented Gradients feature vector and
// runs a linear-SVM sliding-window detector, following Dalal & Triggs (2005).
//
// The five geometry fields must be mutually compatible: BlockSize and
// BlockStride must be whole multiples of CellSize, and WinSize minus BlockSize
// must be a whole multiple of BlockStride. Build a valid default configuration
// (the 64×128 person detector) with [NewHOGDescriptor].
//
// # Descriptor layout
//
// Let cellsX = WinSize.Width/CellSize.Width and cellsY analogously be the
// window's cell grid, and cpbX = BlockSize.Width/CellSize.Width, cpbY the cells
// per block. The window is tiled with blocks stepping by BlockStride, giving
//
//	blocksX = (WinSize.Width  - BlockSize.Width )/BlockStride.Width  + 1
//	blocksY = (WinSize.Height - BlockSize.Height)/BlockStride.Height + 1
//
// [HOGDescriptor.Compute] returns blocksX·blocksY·cpbX·cpbY·NBins values. They
// are ordered block-major, then cell-within-block in row-major order, then
// orientation bin ascending. The component for block (bx,by), block-local cell
// (cx,cy) and bin k is at index
//
//	(((by*blocksX + bx)*cpbY + cy)*cpbX + cx)*NBins + k
//
// Orientation bins span [0,180) (unsigned gradients); bin k covers
// [k·180/NBins, (k+1)·180/NBins). Each block is L2-Hys normalised: L2
// normalise, clip components to 0.2, then L2 normalise again.
type HOGDescriptor struct {
	// WinSize is the detection window in pixels.
	WinSize Size
	// BlockSize is the block size in pixels (a multiple of CellSize).
	BlockSize Size
	// BlockStride is the block step in pixels (a multiple of CellSize).
	BlockStride Size
	// CellSize is the histogram cell size in pixels.
	CellSize Size
	// NBins is the number of orientation bins over [0,180).
	NBins int
}

// NewHOGDescriptor returns the canonical OpenCV/Dalal–Triggs configuration:
// a 64×128 window, 16×16 blocks stepping by 8×8, 8×8 cells and 9 orientation
// bins. Its [HOGDescriptor.DescriptorSize] is 3780.
func NewHOGDescriptor() *HOGDescriptor {
	return &HOGDescriptor{
		WinSize:     Size{64, 128},
		BlockSize:   Size{16, 16},
		BlockStride: Size{8, 8},
		CellSize:    Size{8, 8},
		NBins:       9,
	}
}

// validate panics with a descriptive message if the geometry is inconsistent.
func (h *HOGDescriptor) validate() {
	bad := func(msg string) { panic("objdetect: HOGDescriptor " + msg) }
	if h.NBins <= 0 {
		bad("NBins must be positive")
	}
	for _, s := range []struct {
		name string
		v    Size
	}{{"WinSize", h.WinSize}, {"BlockSize", h.BlockSize}, {"BlockStride", h.BlockStride}, {"CellSize", h.CellSize}} {
		if s.v.Width <= 0 || s.v.Height <= 0 {
			bad(fmt.Sprintf("%s must be positive, got %dx%d", s.name, s.v.Width, s.v.Height))
		}
	}
	if h.BlockSize.Width%h.CellSize.Width != 0 || h.BlockSize.Height%h.CellSize.Height != 0 {
		bad("BlockSize must be a multiple of CellSize")
	}
	if h.BlockStride.Width%h.CellSize.Width != 0 || h.BlockStride.Height%h.CellSize.Height != 0 {
		bad("BlockStride must be a multiple of CellSize")
	}
	if (h.WinSize.Width-h.BlockSize.Width)%h.BlockStride.Width != 0 ||
		(h.WinSize.Height-h.BlockSize.Height)%h.BlockStride.Height != 0 {
		bad("WinSize-BlockSize must be a multiple of BlockStride")
	}
	if h.WinSize.Width < h.BlockSize.Width || h.WinSize.Height < h.BlockSize.Height {
		bad("WinSize must be at least BlockSize")
	}
}

// DescriptorSize returns the length of the vector produced by
// [HOGDescriptor.Compute] for this geometry. It panics if the geometry is
// invalid.
func (h *HOGDescriptor) DescriptorSize() int {
	h.validate()
	cpbX := h.BlockSize.Width / h.CellSize.Width
	cpbY := h.BlockSize.Height / h.CellSize.Height
	blocksX := (h.WinSize.Width-h.BlockSize.Width)/h.BlockStride.Width + 1
	blocksY := (h.WinSize.Height-h.BlockSize.Height)/h.BlockStride.Height + 1
	return blocksX * blocksY * cpbX * cpbY * h.NBins
}

// Compute returns the HOG descriptor for a single window taken from the
// top-left corner of img. The image must be at least WinSize; any extra area
// is ignored. It panics on an invalid geometry or too-small image.
func (h *HOGDescriptor) Compute(img *cv.Mat) []float64 {
	h.validate()
	g := matToGray(img)
	if g.w < h.WinSize.Width || g.h < h.WinSize.Height {
		panic(fmt.Sprintf("objdetect: Compute image %dx%d smaller than window %dx%d",
			g.w, g.h, h.WinSize.Width, h.WinSize.Height))
	}
	mag, ori := g.gradients()
	return h.window(mag, ori, g.w, 0, 0)
}

// window computes the descriptor for the window whose top-left is (x0,y0) in an
// image of stride iw, given precomputed gradient magnitude/orientation arrays.
func (h *HOGDescriptor) window(mag, ori []float64, iw, x0, y0 int) []float64 {
	cellsX := h.WinSize.Width / h.CellSize.Width
	cellsY := h.WinSize.Height / h.CellSize.Height
	binW := 180.0 / float64(h.NBins)

	// Per-cell orientation histograms for the whole window.
	cellHist := make([]float64, cellsX*cellsY*h.NBins)
	for cy := 0; cy < cellsY; cy++ {
		for cx := 0; cx < cellsX; cx++ {
			hb := (cy*cellsX + cx) * h.NBins
			for py := 0; py < h.CellSize.Height; py++ {
				iy := y0 + cy*h.CellSize.Height + py
				row := iy * iw
				for px := 0; px < h.CellSize.Width; px++ {
					ix := x0 + cx*h.CellSize.Width + px
					idx := row + ix
					b := int(ori[idx] / binW)
					if b >= h.NBins {
						b = h.NBins - 1
					}
					cellHist[hb+b] += mag[idx]
				}
			}
		}
	}

	// Group cells into overlapping blocks and L2-Hys normalise.
	cpbX := h.BlockSize.Width / h.CellSize.Width
	cpbY := h.BlockSize.Height / h.CellSize.Height
	strideCX := h.BlockStride.Width / h.CellSize.Width
	strideCY := h.BlockStride.Height / h.CellSize.Height
	blocksX := (h.WinSize.Width-h.BlockSize.Width)/h.BlockStride.Width + 1
	blocksY := (h.WinSize.Height-h.BlockSize.Height)/h.BlockStride.Height + 1

	desc := make([]float64, 0, blocksX*blocksY*cpbX*cpbY*h.NBins)
	block := make([]float64, cpbX*cpbY*h.NBins)
	for by := 0; by < blocksY; by++ {
		for bx := 0; bx < blocksX; bx++ {
			bi := 0
			for cy := 0; cy < cpbY; cy++ {
				for cx := 0; cx < cpbX; cx++ {
					ccx := bx*strideCX + cx
					ccy := by*strideCY + cy
					src := (ccy*cellsX + ccx) * h.NBins
					copy(block[bi:bi+h.NBins], cellHist[src:src+h.NBins])
					bi += h.NBins
				}
			}
			l2HysNormalize(block)
			desc = append(desc, block...)
		}
	}
	return desc
}

// l2HysNormalize normalises v in place with the L2-Hys scheme.
func l2HysNormalize(v []float64) {
	const eps = 1e-7
	var s float64
	for _, x := range v {
		s += x * x
	}
	n := math.Sqrt(s + eps)
	for i := range v {
		v[i] /= n
		if v[i] > 0.2 {
			v[i] = 0.2
		}
	}
	s = 0
	for _, x := range v {
		s += x * x
	}
	n = math.Sqrt(s + eps)
	for i := range v {
		v[i] /= n
	}
}

// DetectMultiScale slides the detection window over a downscaling image pyramid
// and returns the windows whose linear-SVM score meets hitThreshold, mapped
// back to the coordinates of img.
//
// svmWeights is the linear classifier: its length must equal
// [HOGDescriptor.DescriptorSize] (no bias) or that plus one, in which case the
// final element is an additive bias. The score of a window is the dot product
// of svmWeights with its descriptor, plus any bias. scale is the pyramid ratio
// between successive levels and must be greater than 1 (for example 1.05).
// Windows step by BlockStride at every level. No non-maximum suppression is
// applied; overlapping raw hits are returned as-is.
//
// It panics on an invalid geometry, a wrong-length svmWeights, or scale <= 1.
func (h *HOGDescriptor) DetectMultiScale(img *cv.Mat, svmWeights []float64, hitThreshold, scale float64) []cv.Rect {
	h.validate()
	if scale <= 1 {
		panic("objdetect: DetectMultiScale requires scale > 1")
	}
	descLen := h.DescriptorSize()
	var bias float64
	weights := svmWeights
	if len(weights) == descLen+1 {
		bias = weights[descLen]
		weights = weights[:descLen]
	}
	if len(weights) != descLen {
		panic(fmt.Sprintf("objdetect: DetectMultiScale svmWeights length %d, want %d or %d",
			len(svmWeights), descLen, descLen+1))
	}

	base := matToGray(img)
	var hits []cv.Rect
	s := 1.0
	for {
		lw := int(float64(base.w)/s + 0.5)
		lh := int(float64(base.h)/s + 0.5)
		if lw < h.WinSize.Width || lh < h.WinSize.Height {
			break
		}
		level := base
		if s != 1.0 {
			level = base.resize(lw, lh)
		}
		mag, ori := level.gradients()
		for y0 := 0; y0+h.WinSize.Height <= lh; y0 += h.BlockStride.Height {
			for x0 := 0; x0+h.WinSize.Width <= lw; x0 += h.BlockStride.Width {
				desc := h.window(mag, ori, lw, x0, y0)
				score := bias
				for i, w := range weights {
					score += w * desc[i]
				}
				if score >= hitThreshold {
					hits = append(hits, cv.Rect{
						X:      int(float64(x0)*s + 0.5),
						Y:      int(float64(y0)*s + 0.5),
						Width:  int(float64(h.WinSize.Width)*s + 0.5),
						Height: int(float64(h.WinSize.Height)*s + 0.5),
					})
				}
			}
		}
		s *= scale
	}
	return hits
}

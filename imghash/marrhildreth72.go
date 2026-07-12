package imghash

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// mh72Size is the working image side the multi-scale filter runs on.
const mh72Size = 32

// mh72 layout constants: the 72-bit descriptor is organised as
// scales × orientations × (grid×grid) = 2 × 4 × 9 = 72 bits.
const (
	mh72Scales = 2 // number of Gaussian pre-blur scales
	mh72Orient = 4 // number of oriented energy channels, spanning [0,π)
	mh72Grid   = 3 // spatial pooling grid per (scale, orientation) channel
)

// mh72Sigmas are the Gaussian pre-blur standard deviations defining the two
// analysis scales; a small sigma keys on fine edge structure and a larger one on
// coarse structure.
var mh72Sigmas = [mh72Scales]float64{1.0, 2.5}

// MarrHildrethHash72 implements the full 72-bit, multi-scale, multi-orientation
// Marr–Hildreth descriptor deferred by the simplified 64-bit [MarrHildrethHash].
// The image is reduced to grayscale and scaled to 32×32, then analysed at two
// Gaussian scales. At each scale the Laplacian-of-Gaussian response is computed
// and its gradient is steered into four orientation channels spanning [0,π), so
// that horizontal, diagonal, vertical and anti-diagonal edge energy are captured
// separately. Each of the eight (scale × orientation) energy maps is pooled into
// a 3×3 grid, and every cell becomes one bit, set when it exceeds the median of
// its own nine cells. The 2 × 4 × 9 = 72 bits form a 9-byte fingerprint compared
// by Hamming distance.
//
// Analysing several scales makes the descriptor sensitive to structure over a
// range of sizes, and steering the LoG gradient into orientation channels makes
// it sensitive to edge direction — the two properties the single-scale,
// orientation-blind [MarrHildrethHash] lacks. Thresholding each channel at its
// own median keeps every channel balanced and the whole fingerprint
// well conditioned.
//
// The zero value is ready to use; [NewMarrHildrethHash72] is provided for
// symmetry.
type MarrHildrethHash72 struct{}

// NewMarrHildrethHash72 returns a ready-to-use [MarrHildrethHash72].
func NewMarrHildrethHash72() MarrHildrethHash72 { return MarrHildrethHash72{} }

// Compute returns the 9-byte (72-bit) multi-scale, multi-orientation
// Marr–Hildreth hash of img.
func (MarrHildrethHash72) Compute(img *cv.Mat) []byte {
	requireImage(img, "MarrHildrethHash72.Compute")
	gray := grayResize(img, mh72Size, mh72Size)

	base := make([]float64, mh72Size*mh72Size)
	for i := range base {
		base[i] = float64(gray.Data[i])
	}

	bitsOut := make([]bool, mh72Scales*mh72Orient*mh72Grid*mh72Grid)
	bitIdx := 0

	for s := 0; s < mh72Scales; s++ {
		blurred := gaussianBlurF(base, mh72Size, mh72Sigmas[s])
		resp := logFilter(blurred, mh72Size)
		gx, gy := gradients(resp, mh72Size)

		// Accumulate oriented energy maps. An edge orientation θ and θ+π are
		// indistinguishable for a hash, so orientation is folded to [0,π) and
		// split into mh72Orient bins.
		energy := make([][]float64, mh72Orient)
		for o := range energy {
			energy[o] = make([]float64, mh72Size*mh72Size)
		}
		for i := 0; i < mh72Size*mh72Size; i++ {
			mag := math.Hypot(gx[i], gy[i])
			if mag == 0 {
				continue
			}
			theta := math.Atan2(gy[i], gx[i])
			if theta < 0 {
				theta += math.Pi // fold to [0,π)
			}
			bin := int(theta / math.Pi * float64(mh72Orient))
			if bin >= mh72Orient {
				bin = mh72Orient - 1
			}
			energy[bin][i] += mag
		}

		for o := 0; o < mh72Orient; o++ {
			cells := poolGrid(energy[o], mh72Size, mh72Grid)
			thr := median(cells)
			for _, c := range cells {
				bitsOut[bitIdx] = c > thr
				bitIdx++
			}
		}
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two 72-bit Marr–Hildreth hashes.
func (MarrHildrethHash72) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "MarrHildrethHash72.Compare")
	return float64(hamming(a, b))
}

// gaussianBlurF convolves an n×n float image with a separable Gaussian of the
// given standard deviation, replicating the border. sigma <= 0 returns a copy.
func gaussianBlurF(in []float64, n int, sigma float64) []float64 {
	if sigma <= 0 {
		out := make([]float64, len(in))
		copy(out, in)
		return out
	}
	radius := int(math.Ceil(3 * sigma))
	kernel := make([]float64, 2*radius+1)
	var ksum float64
	for i := -radius; i <= radius; i++ {
		w := math.Exp(-float64(i*i) / (2 * sigma * sigma))
		kernel[i+radius] = w
		ksum += w
	}
	for i := range kernel {
		kernel[i] /= ksum
	}

	// Horizontal pass.
	tmp := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			var acc float64
			for k := -radius; k <= radius; k++ {
				sx := clampIndex(x+k, n)
				acc += kernel[k+radius] * in[y*n+sx]
			}
			tmp[y*n+x] = acc
		}
	}
	// Vertical pass.
	out := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			var acc float64
			for k := -radius; k <= radius; k++ {
				sy := clampIndex(y+k, n)
				acc += kernel[k+radius] * tmp[sy*n+x]
			}
			out[y*n+x] = acc
		}
	}
	return out
}

// logFilter convolves an n×n float image with the 5×5 Laplacian-of-Gaussian
// kernel, replicating the border.
func logFilter(in []float64, n int) []float64 {
	out := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			var acc float64
			for ky := -2; ky <= 2; ky++ {
				sy := clampIndex(y+ky, n)
				for kx := -2; kx <= 2; kx++ {
					sx := clampIndex(x+kx, n)
					acc += logKernel[ky+2][kx+2] * in[sy*n+sx]
				}
			}
			out[y*n+x] = acc
		}
	}
	return out
}

// gradients returns the horizontal and vertical central-difference gradients of
// an n×n float image, replicating the border.
func gradients(in []float64, n int) (gx, gy []float64) {
	gx = make([]float64, n*n)
	gy = make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			xl := in[y*n+clampIndex(x-1, n)]
			xr := in[y*n+clampIndex(x+1, n)]
			yt := in[clampIndex(y-1, n)*n+x]
			yb := in[clampIndex(y+1, n)*n+x]
			gx[y*n+x] = (xr - xl) / 2
			gy[y*n+x] = (yb - yt) / 2
		}
	}
	return gx, gy
}

// poolGrid partitions an n×n float image into a grid×grid set of cells and
// returns the sum of each cell, row-major. n need not be a multiple of grid; the
// cell boundaries are computed by proportional division so every pixel is
// counted exactly once.
func poolGrid(in []float64, n, grid int) []float64 {
	cells := make([]float64, grid*grid)
	for y := 0; y < n; y++ {
		gy := y * grid / n
		for x := 0; x < n; x++ {
			gx := x * grid / n
			cells[gy*grid+gx] += in[y*n+x]
		}
	}
	return cells
}

// MarrHildreth72 is a convenience wrapper returning the [MarrHildrethHash72] of
// img.
func MarrHildreth72(img *cv.Mat) []byte { return MarrHildrethHash72{}.Compute(img) }

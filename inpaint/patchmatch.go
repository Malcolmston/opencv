package inpaint

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// NNF is an approximate nearest-neighbour field: for every valid patch centre in
// a source image it stores the offset to the best-matching patch centre in a
// target image, together with the match distance. It is produced by
// [PatchMatchNNF] and consumed by [NNF.Reconstruct]. Construct via those
// functions rather than directly.
type NNF struct {
	// Rows and Cols are the dimensions of the image the field is defined over.
	Rows, Cols int
	// PatchRadius is the half-size of the compared square patches.
	PatchRadius int
	offY, offX  []int
	dist        []float64
}

// idx maps a pixel to the flat field index.
func (n *NNF) idx(y, x int) int { return y*n.Cols + x }

// At returns the matched patch-centre coordinates (my, mx) in the target image
// for the source patch centred at (y, x). It panics if out of range.
func (n *NNF) At(y, x int) (my, mx int) {
	if y < 0 || y >= n.Rows || x < 0 || x >= n.Cols {
		panic("inpaint: NNF.At out of range")
	}
	i := n.idx(y, x)
	return y + n.offY[i], x + n.offX[i]
}

// Distance returns the patch match distance (sum of squared differences) at
// source centre (y, x). It panics if out of range.
func (n *NNF) Distance(y, x int) float64 {
	if y < 0 || y >= n.Rows || x < 0 || x >= n.Cols {
		panic("inpaint: NNF.Distance out of range")
	}
	return n.dist[n.idx(y, x)]
}

// Reconstruct synthesises an image the size of the field by copying, for each
// pixel, the value of the target-image pixel its best-matching patch centre
// points at (a single-vote reconstruction). target must be the same image the
// field was matched against. The result has target's channel count.
func (n *NNF) Reconstruct(target *cv.Mat) *cv.Mat {
	inpaintRequireImage(target, "NNF.Reconstruct")
	out := cv.NewMat(n.Rows, n.Cols, target.Channels)
	for y := 0; y < n.Rows; y++ {
		for x := 0; x < n.Cols; x++ {
			my, mx := n.At(y, x)
			my = inpaintClampInt(my, 0, target.Rows-1)
			mx = inpaintClampInt(mx, 0, target.Cols-1)
			for c := 0; c < target.Channels; c++ {
				out.Set(y, x, c, target.At(my, mx, c))
			}
		}
	}
	return out
}

// PatchMatchNNF computes an approximate nearest-neighbour field mapping each
// patch of a to its most similar patch in b using Barnes et al.'s (2009)
// PatchMatch: a random initial field is refined by alternating propagation (good
// matches spread to neighbours) and random search (exponentially shrinking
// random probes). Patches are squares of side 2*patchRadius+1 compared by sum of
// squared differences with edge replication. a and b must have the same channel
// count. iterations of 4..6 usually converge; a non-positive value uses 5. The
// search is seeded deterministically, so results are reproducible.
func PatchMatchNNF(a, b *cv.Mat, patchRadius, iterations int) *NNF {
	inpaintRequireImage(a, "PatchMatchNNF a")
	inpaintRequireImage(b, "PatchMatchNNF b")
	if a.Channels != b.Channels {
		panic("inpaint: PatchMatchNNF requires matching channel counts")
	}
	if patchRadius < 1 {
		patchRadius = 1
	}
	if iterations <= 0 {
		iterations = 5
	}
	offY, offX, dist := inpaintPatchMatchCore(a, b, patchRadius, iterations,
		func(y, x int) bool { return true }, rand.New(rand.NewSource(1)))
	return &NNF{Rows: a.Rows, Cols: a.Cols, PatchRadius: patchRadius, offY: offY, offX: offX, dist: dist}
}

// inpaintPatchSSD returns the sum of squared differences between the patch of a
// centred at (ay, ax) and the patch of b centred at (by, bx), radius half, with
// edge replication. It stops early once the running sum reaches cutoff.
func inpaintPatchSSD(a, b *cv.Mat, ay, ax, by, bx, half int, cutoff float64) float64 {
	var s float64
	ch := a.Channels
	for dy := -half; dy <= half; dy++ {
		for dx := -half; dx <= half; dx++ {
			for c := 0; c < ch; c++ {
				d := float64(inpaintAtRep(a, ay+dy, ax+dx, c)) - float64(inpaintAtRep(b, by+dy, bx+dx, c))
				s += d * d
			}
		}
		if s >= cutoff {
			return s
		}
	}
	return s
}

// inpaintPatchMatchCore runs PatchMatch from a into b, restricting target
// centres to those where validB reports true. It returns per-pixel offset and
// distance slices in row-major order.
func inpaintPatchMatchCore(a, b *cv.Mat, half, iterations int, validB func(y, x int) bool, rng *rand.Rand) (offY, offX []int, dist []float64) {
	rows, cols := a.Rows, a.Cols
	n := rows * cols
	offY = make([]int, n)
	offX = make([]int, n)
	dist = make([]float64, n)

	byLo, byHi := 0, b.Rows-1
	bxLo, bxHi := 0, b.Cols-1

	// Collect valid target centres so random init/search always lands on a legal
	// patch even when validB is restrictive.
	var validCenters [][2]int
	for y := byLo; y <= byHi; y++ {
		for x := bxLo; x <= bxHi; x++ {
			if validB(y, x) {
				validCenters = append(validCenters, [2]int{y, x})
			}
		}
	}
	if len(validCenters) == 0 {
		// No legal target: leave a zero field.
		return offY, offX, dist
	}

	// Random initialisation.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			vc := validCenters[rng.Intn(len(validCenters))]
			i := y*cols + x
			offY[i] = vc[0] - y
			offX[i] = vc[1] - x
			dist[i] = inpaintPatchSSD(a, b, y, x, vc[0], vc[1], half, math.Inf(1))
		}
	}

	try := func(y, x, cy, cx int) {
		if cy < byLo || cy > byHi || cx < bxLo || cx > bxHi || !validB(cy, cx) {
			return
		}
		i := y*cols + x
		d := inpaintPatchSSD(a, b, y, x, cy, cx, half, dist[i])
		if d < dist[i] {
			dist[i] = d
			offY[i] = cy - y
			offX[i] = cx - x
		}
	}

	maxDim := b.Rows
	if b.Cols > maxDim {
		maxDim = b.Cols
	}
	for it := 0; it < iterations; it++ {
		reverse := it%2 == 1
		yStart, yEnd, yStep := 0, rows, 1
		xStart, xEnd, xStep := 0, cols, 1
		if reverse {
			yStart, yEnd, yStep = rows-1, -1, -1
			xStart, xEnd, xStep = cols-1, -1, -1
		}
		for y := yStart; y != yEnd; y += yStep {
			for x := xStart; x != xEnd; x += xStep {
				i := y*cols + x
				// Propagation from the previously processed neighbours.
				if !reverse {
					if x > 0 {
						try(y, x, y+offY[i-1], x+offX[i-1])
					}
					if y > 0 {
						try(y, x, y+offY[i-cols], x+offX[i-cols])
					}
				} else {
					if x < cols-1 {
						try(y, x, y+offY[i+1], x+offX[i+1])
					}
					if y < rows-1 {
						try(y, x, y+offY[i+cols], x+offX[i+cols])
					}
				}
				// Random search with exponentially shrinking radius.
				cy := y + offY[i]
				cx := x + offX[i]
				for r := maxDim; r >= 1; r /= 2 {
					ry := cy + rng.Intn(2*r+1) - r
					rx := cx + rng.Intn(2*r+1) - r
					try(y, x, ry, rx)
				}
			}
		}
	}
	return offY, offX, dist
}

// PatchMatchOptions configures [InpaintPatchMatch].
type PatchMatchOptions struct {
	// PatchRadius is the half-size of the synthesis patch (side 2*PatchRadius+1).
	// A non-positive value uses 3.
	PatchRadius int
	// Iterations is the number of PatchMatch propagation/search passes per
	// synthesis pass. A non-positive value uses 5.
	Iterations int
	// Passes is the number of coarse-to-fine synthesis passes (search then vote).
	// A non-positive value uses 4.
	Passes int
}

// DefaultPatchMatchOptions returns default synthesis settings (PatchRadius 3,
// Iterations 5, Passes 4).
func DefaultPatchMatchOptions() PatchMatchOptions {
	return PatchMatchOptions{PatchRadius: 3, Iterations: 5, Passes: 4}
}

// InpaintPatchMatch fills the pixels of img selected by mask by iterated
// PatchMatch synthesis: the hole is first initialised harmonically, then each
// pass finds, for every hole patch, the most similar fully-known patch and
// updates the hole pixels toward the average of those matches' votes. Repeating
// this propagates coherent texture from the known region into the hole.
//
// img may be single- or three-channel; mask must match its size (true = fill).
// img is not modified — a filled clone is returned. A uniform surround is
// reproduced exactly. Results are deterministic (fixed internal seed).
func InpaintPatchMatch(img *cv.Mat, mask *Mask, opts PatchMatchOptions) *cv.Mat {
	inpaintRequireImage(img, "InpaintPatchMatch")
	inpaintRequireMaskMatch(img, mask, "InpaintPatchMatch")
	half := opts.PatchRadius
	if half <= 0 {
		half = 3
	}
	iters := opts.Iterations
	if iters <= 0 {
		iters = 5
	}
	passes := opts.Passes
	if passes <= 0 {
		passes = 4
	}

	out := img.Clone()
	inpaintHarmonicFill(out, mask)
	rows, cols, ch := out.Rows, out.Cols, out.Channels

	hole := make([]bool, rows*cols)
	copy(hole, mask.Data)

	// A target patch is valid only if entirely outside the hole (fully known).
	validB := func(cy, cx int) bool {
		for dy := -half; dy <= half; dy++ {
			for dx := -half; dx <= half; dx++ {
				yy := inpaintClampInt(cy+dy, 0, rows-1)
				xx := inpaintClampInt(cx+dx, 0, cols-1)
				if hole[yy*cols+xx] {
					return false
				}
			}
		}
		return true
	}

	rng := rand.New(rand.NewSource(1))
	acc := make([]float64, rows*cols*ch)
	wgt := make([]float64, rows*cols)
	for pass := 0; pass < passes; pass++ {
		offY, offX, _ := inpaintPatchMatchCore(out, out, half, iters, validB, rng)
		for i := range acc {
			acc[i] = 0
		}
		for i := range wgt {
			wgt[i] = 0
		}
		// Each source patch centred at a hole-covering location votes for the
		// hole pixels it covers, using its matched known patch.
		for cy := 0; cy < rows; cy++ {
			for cx := 0; cx < cols; cx++ {
				i := cy*cols + cx
				my := cy + offY[i]
				mx := cx + offX[i]
				for dy := -half; dy <= half; dy++ {
					for dx := -half; dx <= half; dx++ {
						py, px := cy+dy, cx+dx
						if py < 0 || py >= rows || px < 0 || px >= cols {
							continue
						}
						if !hole[py*cols+px] {
							continue
						}
						sy := inpaintClampInt(my+dy, 0, rows-1)
						sx := inpaintClampInt(mx+dx, 0, cols-1)
						for c := 0; c < ch; c++ {
							acc[(py*cols+px)*ch+c] += float64(out.At(sy, sx, c))
						}
						wgt[py*cols+px]++
					}
				}
			}
		}
		for i := 0; i < rows*cols; i++ {
			if !hole[i] || wgt[i] == 0 {
				continue
			}
			y, x := i/cols, i%cols
			for c := 0; c < ch; c++ {
				out.Set(y, x, c, inpaintClampU8(acc[i*ch+c]/wgt[i]))
			}
		}
	}
	return out
}

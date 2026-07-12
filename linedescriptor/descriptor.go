package linedescriptor

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// BinaryDescriptor computes an LBD-style ("Line Band Descriptor") binary code
// for each line segment. The exported fields control the shape of the support
// region; [NewBinaryDescriptor] fills them with defaults. All descriptors from
// one BinaryDescriptor share the same fixed length in bytes, so they can be
// compared with a [BinaryDescriptorMatcher].
type BinaryDescriptor struct {
	// NumBands is the number of parallel bands the support region is split into
	// across the width of the line. It must be positive.
	NumBands int
	// BandWidth is the thickness, in pixels, of each band measured
	// perpendicular to the line. The total support-region width is
	// NumBands × BandWidth.
	BandWidth int
}

// NewBinaryDescriptor returns a descriptor with the default geometry of 8 bands
// of 7 pixels each, yielding a 32-bit (4-byte) code per line.
func NewBinaryDescriptor() *BinaryDescriptor {
	return &BinaryDescriptor{NumBands: 8, BandWidth: 7}
}

// featuresPerBand is the number of scalar gradient statistics gathered per band
// (mean and standard deviation of the along-line and across-line gradient
// components). Each contributes one bit per band to the code.
const featuresPerBand = 4

// DescriptorSize returns the length in bytes of every code produced by this
// descriptor. The code has NumBands × featuresPerBand bits, rounded up to whole
// bytes.
func (bd *BinaryDescriptor) DescriptorSize() int {
	bits := bd.NumBands * featuresPerBand
	return (bits + 7) / 8
}

// Compute describes each line in lines using the gradients of img and returns
// the (unchanged) lines alongside one binary code per line, aligned by index.
// img may be 1- or 3-channel; colour is reduced to luma first.
//
// For a line the support region is a band running along the segment, centred on
// it and NumBands × BandWidth pixels wide. The region is sampled at unit steps
// along the line and across its full width; at every sample the image gradient
// is projected onto the line direction (the "along" component) and onto the
// perpendicular (the "across" component), and these are accumulated per band.
// Each band then contributes four statistics — the mean and standard deviation
// of its along and across components. Every statistic is binarised by comparing
// it to the mean of that same statistic across all bands, giving a code that is
// invariant to a translation of the segment (a shifted copy of the same image
// content produces an identical code). Bits are packed most-significant-first,
// grouped by statistic then by band.
//
// A degenerate (zero-length) line yields an all-zero code.
func (bd *BinaryDescriptor) Compute(img *cv.Mat, lines []KeyLine) ([]KeyLine, [][]byte) {
	if bd.NumBands <= 0 || bd.BandWidth <= 0 {
		panic("linedescriptor: BinaryDescriptor requires positive NumBands and BandWidth")
	}
	gray := toGray(img)
	gx, gy, rows, cols := gradients(gray)

	out := make([][]byte, len(lines))
	for i, kl := range lines {
		out[i] = bd.describe(kl, gx, gy, rows, cols)
	}
	return lines, out
}

// describe computes the packed binary code for a single line.
func (bd *BinaryDescriptor) describe(kl KeyLine, gx, gy []float64, rows, cols int) []byte {
	code := make([]byte, bd.DescriptorSize())
	length := kl.Length
	if length <= 0 {
		return code
	}

	sx := float64(kl.StartPoint.X)
	sy := float64(kl.StartPoint.Y)
	dirX := float64(kl.EndPoint.X-kl.StartPoint.X) / length
	dirY := float64(kl.EndPoint.Y-kl.StartPoint.Y) / length
	perpX, perpY := -dirY, dirX

	totalW := bd.NumBands * bd.BandWidth
	half := float64(totalW) / 2

	// Per-band accumulators.
	nb := bd.NumBands
	sumL := make([]float64, nb)
	sumO := make([]float64, nb)
	sumL2 := make([]float64, nb)
	sumO2 := make([]float64, nb)
	count := make([]float64, nb)

	numSamp := int(math.Round(length))
	if numSamp < 1 {
		numSamp = 1
	}
	for t := 0; t <= numSamp; t++ {
		along := float64(t) / float64(numSamp) * length
		bx := sx + dirX*along
		by := sy + dirY*along
		for r := 0; r < totalW; r++ {
			off := float64(r) - half + 0.5
			px := bx + perpX*off
			py := by + perpY*off
			gxv, gyv := sampleGrad(gx, gy, rows, cols, px, py)
			gdL := gxv*dirX + gyv*dirY
			gdO := gxv*perpX + gyv*perpY
			b := r / bd.BandWidth
			sumL[b] += gdL
			sumO[b] += gdO
			sumL2[b] += gdL * gdL
			sumO2[b] += gdO * gdO
			count[b]++
		}
	}

	// Per-band statistics.
	meanL := make([]float64, nb)
	meanO := make([]float64, nb)
	stdL := make([]float64, nb)
	stdO := make([]float64, nb)
	for b := 0; b < nb; b++ {
		if count[b] == 0 {
			continue
		}
		ml := sumL[b] / count[b]
		mo := sumO[b] / count[b]
		meanL[b] = ml
		meanO[b] = mo
		stdL[b] = math.Sqrt(math.Max(0, sumL2[b]/count[b]-ml*ml))
		stdO[b] = math.Sqrt(math.Max(0, sumO2[b]/count[b]-mo*mo))
	}

	// Binarise each statistic against its cross-band mean and pack the bits.
	bit := 0
	for _, feat := range [][]float64{meanL, meanO, stdL, stdO} {
		var acc float64
		for _, v := range feat {
			acc += v
		}
		mean := acc / float64(nb)
		for b := 0; b < nb; b++ {
			if feat[b] > mean {
				code[bit/8] |= 1 << uint(7-bit%8)
			}
			bit++
		}
	}
	return code
}

// sampleGrad returns the gradient (gx, gy) at the real-valued location (px, py)
// using nearest-neighbour lookup with the coordinates clamped to the image, so
// an integer translation of the sampling point reproduces the value exactly.
func sampleGrad(gx, gy []float64, rows, cols int, px, py float64) (float64, float64) {
	x := int(math.Round(px))
	y := int(math.Round(py))
	if x < 0 {
		x = 0
	} else if x >= cols {
		x = cols - 1
	}
	if y < 0 {
		y = 0
	} else if y >= rows {
		y = rows - 1
	}
	i := y*cols + x
	return gx[i], gy[i]
}

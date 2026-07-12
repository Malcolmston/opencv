package structured_light

import cv "github.com/malcolmston/opencv"

// numBits returns the number of bits needed to address n distinct values, i.e.
// ceil(log2(n)) for n>1 and 1 for n<=2. It is used to size the Gray-code stack.
func numBits(n int) int {
	b := 0
	for (1 << b) < n {
		b++
	}
	if b == 0 {
		b = 1
	}
	return b
}

// binaryToGray maps a non-negative integer to its binary reflected Gray code.
func binaryToGray(n uint) uint { return n ^ (n >> 1) }

// grayToBinary inverts binaryToGray, recovering the integer from its Gray code.
func grayToBinary(g uint) uint {
	b := g
	for g >>= 1; g != 0; g >>= 1 {
		b ^= g
	}
	return b
}

// abs returns the absolute value of an int.
func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// clampRound rounds v to the nearest integer and clamps it to the 0..255 range.
func clampRound(v float64) uint8 {
	v += 0.5
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// toGray returns a single-channel copy of m. A single-channel Mat is cloned; a
// multi-channel Mat is reduced with the BT.601 luma weights. This is a local
// reimplementation so the package never depends on a sibling subpackage.
func toGray(m *cv.Mat) *cv.Mat {
	if m.Channels == 1 {
		return m.Clone()
	}
	out := cv.NewMat(m.Rows, m.Cols, 1)
	for p := 0; p < m.Total(); p++ {
		base := p * m.Channels
		if m.Channels >= 3 {
			r := float64(m.Data[base+0])
			g := float64(m.Data[base+1])
			b := float64(m.Data[base+2])
			out.Data[p] = clampRound(0.299*r + 0.587*g + 0.114*b)
		} else {
			out.Data[p] = m.Data[base]
		}
	}
	return out
}

// CoordMapToMat renders a decoded coordinate map (one int per camera pixel, -1
// for invalid) as a single-channel [github.com/malcolmston/opencv.Mat] of size
// rows×cols. Valid coordinates are linearly scaled from [0, maxVal] to [1, 255];
// invalid pixels are 0. maxVal must be positive. It panics on a size mismatch or
// non-positive maxVal.
func CoordMapToMat(coord []int, rows, cols, maxVal int) *cv.Mat {
	if len(coord) != rows*cols {
		panic("structured_light: CoordMapToMat length != rows*cols")
	}
	if maxVal <= 0 {
		panic("structured_light: CoordMapToMat requires maxVal>0")
	}
	out := cv.NewMat(rows, cols, 1)
	for i, c := range coord {
		if c < 0 {
			continue
		}
		v := 1 + int(float64(c)/float64(maxVal)*254.0+0.5)
		if v > 255 {
			v = 255
		}
		out.Data[i] = uint8(v)
	}
	return out
}

// PhaseMapToMat renders a phase map (one float per pixel) as a single-channel
// [github.com/malcolmston/opencv.Mat] of size rows×cols, min-max normalized to
// the full 0..255 range. A constant map maps to all zeros. It panics on a size
// mismatch.
func PhaseMapToMat(phase []float64, rows, cols int) *cv.Mat {
	if len(phase) != rows*cols {
		panic("structured_light: PhaseMapToMat length != rows*cols")
	}
	out := cv.NewMat(rows, cols, 1)
	if len(phase) == 0 {
		return out
	}
	lo, hi := phase[0], phase[0]
	for _, v := range phase {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	span := hi - lo
	if span == 0 {
		return out
	}
	for i, v := range phase {
		out.Data[i] = clampRound((v - lo) / span * 255.0)
	}
	return out
}

// MaskToMat renders a boolean validity mask as a single-channel
// [github.com/malcolmston/opencv.Mat] of size rows×cols, with 255 for true and 0
// for false. It panics on a size mismatch.
func MaskToMat(mask []bool, rows, cols int) *cv.Mat {
	if len(mask) != rows*cols {
		panic("structured_light: MaskToMat length != rows*cols")
	}
	out := cv.NewMat(rows, cols, 1)
	for i, b := range mask {
		if b {
			out.Data[i] = 255
		}
	}
	return out
}

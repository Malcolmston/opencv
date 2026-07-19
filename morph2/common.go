package morph2

import cv "github.com/malcolmston/opencv"

// Connectivity selects the neighbourhood used by binary and reconstruction
// routines.
type Connectivity int

// Conn4 and Conn8 are the supported connectivity values.
const (
	// Conn4 is 4-connectivity: the horizontal and vertical neighbours only.
	Conn4 Connectivity = 4
	// Conn8 is 8-connectivity: all eight surrounding neighbours.
	Conn8 Connectivity = 8
)

// neighbourOffsets returns the (dy, dx) neighbour displacements for a
// connectivity, panicking on an invalid value.
func neighbourOffsets(conn Connectivity) [][2]int {
	switch conn {
	case Conn4:
		return [][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	case Conn8:
		return [][2]int{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}
	default:
		panic("morph2: connectivity must be Conn4 or Conn8")
	}
}

// requireGray panics unless m is a usable single-channel matrix.
func requireGray(m *cv.Mat) {
	if m == nil || m.Empty() {
		panic("morph2: empty source matrix")
	}
	if m.Channels != 1 {
		panic("morph2: operation requires a single-channel matrix")
	}
}

// requireSameSize panics unless a and b are single-channel and identically sized.
func requireSameSize(a, b *cv.Mat) {
	requireGray(a)
	requireGray(b)
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("morph2: matrices must have identical dimensions")
	}
}

// newLike allocates a zeroed single-channel matrix matching m's dimensions.
func newLike(m *cv.Mat) *cv.Mat { return cv.NewMat(m.Rows, m.Cols, 1) }

// idx returns the flat index of pixel (y, x) in a single-channel matrix of the
// given width.
func idx(y, x, cols int) int { return y*cols + x }

func minU8(a, b uint8) uint8 {
	if a < b {
		return a
	}
	return b
}

func maxU8(a, b uint8) uint8 {
	if a > b {
		return a
	}
	return b
}

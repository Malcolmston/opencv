package features3

import (
	"math/bits"

	cv "github.com/malcolmston/opencv"
)

// CensusTransform3x3 computes the 3×3 census (local binary) transform of an
// image. For each pixel the eight neighbours are visited in raster order
// (top-left to bottom-right, skipping the centre) and bit i is set when that
// neighbour is strictly darker than the centre. The result is one uint8 per
// pixel, indexed y*Cols+x, comparable between images with the Hamming distance.
// Border neighbours are replicated. Colour input is converted to grayscale.
func CensusTransform3x3(img *cv.Mat) []uint8 {
	g := features3ToGray(img)
	out := make([]uint8, g.Rows*g.Cols)
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			c := g.at(x, y)
			var code uint8
			bit := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					if g.atClamped(x+dx, y+dy) < c {
						code |= 1 << uint(bit)
					}
					bit++
				}
			}
			out[y*g.Cols+x] = code
		}
	}
	return out
}

// CensusTransform5x5 computes the 5×5 census transform of an image, packing the
// twenty-four neighbour comparisons of each pixel into a uint32 (bit i set when
// that neighbour is strictly darker than the centre). The result is one uint32
// per pixel, indexed y*Cols+x. Border neighbours are replicated. Colour input is
// converted to grayscale.
func CensusTransform5x5(img *cv.Mat) []uint32 {
	g := features3ToGray(img)
	out := make([]uint32, g.Rows*g.Cols)
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			c := g.at(x, y)
			var code uint32
			bit := 0
			for dy := -2; dy <= 2; dy++ {
				for dx := -2; dx <= 2; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					if g.atClamped(x+dx, y+dy) < c {
						code |= 1 << uint(bit)
					}
					bit++
				}
			}
			out[y*g.Cols+x] = code
		}
	}
	return out
}

// CensusField is a dense census transform with a configurable window, storing
// one bit-packed uint64 code per pixel. Bits holds the number of comparison bits
// used per pixel. Build one with [CensusTransform] or [ModifiedCensusTransform]
// and compare codes with [CensusField.Hamming].
type CensusField struct {
	// Rows and Cols are the image dimensions.
	Rows int
	Cols int
	// Bits is the number of neighbour-comparison bits stored per pixel.
	Bits int
	// Data holds one packed census code per pixel, indexed y*Cols+x.
	Data []uint64
}

// At returns the packed census code at column x, row y.
func (c *CensusField) At(x, y int) uint64 {
	return c.Data[y*c.Cols+x]
}

// Hamming returns the Hamming distance between the census codes at (x1, y1) and
// (x2, y2), a matching cost in the range [0, Bits].
func (c *CensusField) Hamming(x1, y1, x2, y2 int) int {
	return bits.OnesCount64(c.Data[y1*c.Cols+x1] ^ c.Data[y2*c.Cols+x2])
}

// CensusTransform computes the census transform of an image over a
// (2*halfWindow+1) square window, comparing every neighbour with the centre and
// setting the bit when the neighbour is strictly darker. Codes are packed into a
// [CensusField] of uint64s, so the window must have at most 64 neighbours
// (halfWindow <= 3); it panics otherwise. Colour input is converted to
// grayscale first and border neighbours are replicated.
func CensusTransform(img *cv.Mat, halfWindow int) *CensusField {
	if halfWindow < 1 {
		halfWindow = 1
	}
	side := 2*halfWindow + 1
	nbits := side*side - 1
	if nbits > 64 {
		panic("features3: CensusTransform window too large for uint64 (use halfWindow <= 3)")
	}
	g := features3ToGray(img)
	field := &CensusField{Rows: g.Rows, Cols: g.Cols, Bits: nbits, Data: make([]uint64, g.Rows*g.Cols)}
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			c := g.at(x, y)
			var code uint64
			bit := 0
			for dy := -halfWindow; dy <= halfWindow; dy++ {
				for dx := -halfWindow; dx <= halfWindow; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					if g.atClamped(x+dx, y+dy) < c {
						code |= 1 << uint(bit)
					}
					bit++
				}
			}
			field.Data[y*g.Cols+x] = code
		}
	}
	return field
}

// ModifiedCensusTransform computes the modified census transform, which compares
// each neighbour with the mean intensity of the window rather than the centre
// pixel. This makes the descriptor robust to a bright or dark centre outlier.
// The centre pixel is included as its own bit, so a (2*halfWindow+1) window uses
// (2*halfWindow+1)^2 bits, which must be at most 64 (halfWindow <= 3); it panics
// otherwise. Colour input is converted to grayscale first.
func ModifiedCensusTransform(img *cv.Mat, halfWindow int) *CensusField {
	if halfWindow < 1 {
		halfWindow = 1
	}
	side := 2*halfWindow + 1
	nbits := side * side
	if nbits > 64 {
		panic("features3: ModifiedCensusTransform window too large for uint64 (use halfWindow <= 3)")
	}
	g := features3ToGray(img)
	field := &CensusField{Rows: g.Rows, Cols: g.Cols, Bits: nbits, Data: make([]uint64, g.Rows*g.Cols)}
	for y := 0; y < g.Rows; y++ {
		for x := 0; x < g.Cols; x++ {
			var sum float64
			for dy := -halfWindow; dy <= halfWindow; dy++ {
				for dx := -halfWindow; dx <= halfWindow; dx++ {
					sum += g.atClamped(x+dx, y+dy)
				}
			}
			mean := sum / float64(nbits)
			var code uint64
			bit := 0
			for dy := -halfWindow; dy <= halfWindow; dy++ {
				for dx := -halfWindow; dx <= halfWindow; dx++ {
					if g.atClamped(x+dx, y+dy) < mean {
						code |= 1 << uint(bit)
					}
					bit++
				}
			}
			field.Data[y*g.Cols+x] = code
		}
	}
	return field
}

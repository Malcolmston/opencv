package bioinspired

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// BayerPattern names one of the four 2x2 colour-filter-array (Bayer) mosaic
// layouts. The retina colour path in OpenCV multiplexes the three colour planes
// onto a single mosaic (as a real photoreceptor array does) and later
// demultiplexes them; [MosaicBayer] and [DemosaicBayer] implement that colour
// multiplexing / demultiplexing.
//
// Each pattern gives the colour sampled at the four positions of the repeating
// 2x2 tile, read in row-major order: top-left, top-right, bottom-left,
// bottom-right. Channel indices follow the package convention that a
// three-channel [cv.Mat] is RGB (R=0, G=1, B=2).
type BayerPattern int

const (
	// BayerRGGB samples R G / G B over the 2x2 tile.
	BayerRGGB BayerPattern = iota
	// BayerBGGR samples B G / G R over the 2x2 tile.
	BayerBGGR
	// BayerGRBG samples G R / B G over the 2x2 tile.
	BayerGRBG
	// BayerGBRG samples G B / R G over the 2x2 tile.
	BayerGBRG
)

// String returns a short human-readable name for the pattern.
func (p BayerPattern) String() string {
	switch p {
	case BayerRGGB:
		return "RGGB"
	case BayerBGGR:
		return "BGGR"
	case BayerGRBG:
		return "GRBG"
	case BayerGBRG:
		return "GBRG"
	default:
		return fmt.Sprintf("BayerPattern(%d)", int(p))
	}
}

// bayerTiles maps each pattern to the colour channel index sampled at the four
// tile positions [ (0,0), (0,1), (1,0), (1,1) ].
var bayerTiles = map[BayerPattern][4]int{
	BayerRGGB: {0, 1, 1, 2},
	BayerBGGR: {2, 1, 1, 0},
	BayerGRBG: {1, 0, 2, 1},
	BayerGBRG: {1, 2, 0, 1},
}

// colorAt returns the colour channel index (0=R, 1=G, 2=B) sampled by the given
// Bayer pattern at pixel (y, x). y and x must be non-negative.
func colorAt(pattern BayerPattern, y, x int) int {
	tile, ok := bayerTiles[pattern]
	if !ok {
		panic(fmt.Sprintf("bioinspired: unknown Bayer pattern %d", int(pattern)))
	}
	return tile[(y&1)*2+(x&1)]
}

// MosaicBayer multiplexes a three-channel RGB image onto a single-channel Bayer
// mosaic: at each pixel it keeps only the colour dictated by the mosaic pattern,
// mimicking a colour-filter-array photoreceptor sheet where every photoreceptor
// senses just one primary. The result is a single-channel [cv.Mat] the same size
// as the input. It panics unless the input is a non-empty three-channel Mat.
//
// MosaicBayer is the inverse operation to [DemosaicBayer]; together they model
// the colour multiplexing / demultiplexing of the retina colour pathway.
func MosaicBayer(rgb *cv.Mat, pattern BayerPattern) *cv.Mat {
	if rgb.Empty() {
		panic("bioinspired: MosaicBayer given an empty Mat")
	}
	if rgb.Channels != 3 {
		panic(fmt.Sprintf("bioinspired: MosaicBayer requires a 3-channel Mat, got %d", rgb.Channels))
	}
	out := cv.NewMat(rgb.Rows, rgb.Cols, 1)
	for y := 0; y < rgb.Rows; y++ {
		for x := 0; x < rgb.Cols; x++ {
			out.Data[y*rgb.Cols+x] = rgb.At(y, x, colorAt(pattern, y, x))
		}
	}
	return out
}

// DemosaicBayer demultiplexes a single-channel Bayer mosaic back into a
// three-channel RGB image using bilinear interpolation: at every pixel the
// measured colour is taken directly from the mosaic and the two missing colours
// are reconstructed as the average of the same-coloured samples in the 3x3
// neighbourhood (out-of-image neighbours are ignored). For a smooth image this
// closely inverts [MosaicBayer]; on a linear gradient the interior
// reconstruction is exact up to 8-bit rounding.
//
// It panics unless the input is a non-empty single-channel Mat. The pattern must
// match the one used by [MosaicBayer].
func DemosaicBayer(mosaic *cv.Mat, pattern BayerPattern) *cv.Mat {
	if mosaic.Empty() {
		panic("bioinspired: DemosaicBayer given an empty Mat")
	}
	if mosaic.Channels != 1 {
		panic(fmt.Sprintf("bioinspired: DemosaicBayer requires a 1-channel Mat, got %d", mosaic.Channels))
	}
	rows, cols := mosaic.Rows, mosaic.Cols
	out := cv.NewMat(rows, cols, 3)
	at := func(y, x int) float64 { return float64(mosaic.Data[y*cols+x]) }
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			own := colorAt(pattern, y, x)
			var rgb [3]float64
			rgb[own] = at(y, x)
			// Interpolate the two missing colours from same-colour neighbours.
			for c := 0; c < 3; c++ {
				if c == own {
					continue
				}
				var sum float64
				var n int
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						if dy == 0 && dx == 0 {
							continue
						}
						yy, xx := y+dy, x+dx
						if yy < 0 || yy >= rows || xx < 0 || xx >= cols {
							continue
						}
						if colorAt(pattern, yy, xx) != c {
							continue
						}
						sum += at(yy, xx)
						n++
					}
				}
				if n == 0 {
					rgb[c] = rgb[own] // degenerate corner: fall back to the measured colour
				} else {
					rgb[c] = sum / float64(n)
				}
			}
			base := (y*cols + x) * 3
			out.Data[base+0] = clampRound(rgb[0])
			out.Data[base+1] = clampRound(rgb[1])
			out.Data[base+2] = clampRound(rgb[2])
		}
	}
	return out
}

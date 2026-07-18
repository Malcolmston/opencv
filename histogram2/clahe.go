package histogram2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// CLAHE is a Contrast Limited Adaptive Histogram Equalisation operator. Unlike
// global [EqualizeHist], it equalises many small tiles independently, clips
// each tile histogram to limit noise amplification, and bilinearly
// interpolates the per-tile mappings so tile boundaries do not show. Construct
// one with [NewCLAHE] and apply it with [CLAHE.Apply].
type CLAHE struct {
	clipLimit float64
	tilesX    int
	tilesY    int
}

// NewCLAHE returns a CLAHE operator. clipLimit is the contrast-limiting
// threshold relative to the average tile bin height (a value of 0 or below
// disables clipping); the OpenCV default is 40. tilesX and tilesY give the
// number of tiles across the width and height. It panics if the tile counts
// are not positive.
func NewCLAHE(clipLimit float64, tilesX, tilesY int) *CLAHE {
	if tilesX <= 0 || tilesY <= 0 {
		panic("histogram2: NewCLAHE requires positive tile counts")
	}
	return &CLAHE{clipLimit: clipLimit, tilesX: tilesX, tilesY: tilesY}
}

// SetClipLimit updates the contrast-limiting threshold used by [CLAHE.Apply].
func (c *CLAHE) SetClipLimit(clipLimit float64) {
	c.clipLimit = clipLimit
}

// SetTilesGridSize updates the tile grid used by [CLAHE.Apply]. It panics if
// either count is not positive.
func (c *CLAHE) SetTilesGridSize(tilesX, tilesY int) {
	if tilesX <= 0 || tilesY <= 0 {
		panic("histogram2: SetTilesGridSize requires positive tile counts")
	}
	c.tilesX = tilesX
	c.tilesY = tilesY
}

// Apply runs contrast-limited adaptive histogram equalisation on a
// single-channel image and returns a new single-channel image of the same
// size. It returns [ErrEmptyImage] if src is empty and [ErrChannelRange] if
// src is not single-channel.
func (c *CLAHE) Apply(src *cv.Mat) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if src.Channels != 1 {
		return nil, ErrChannelRange
	}
	rows, cols := src.Rows, src.Cols
	tw := float64(cols) / float64(c.tilesX)
	th := float64(rows) / float64(c.tilesY)

	// Build a mapping LUT for every tile.
	luts := make([][256]uint8, c.tilesX*c.tilesY)
	for ty := 0; ty < c.tilesY; ty++ {
		y0 := int(float64(ty) * th)
		y1 := int(float64(ty+1) * th)
		if ty == c.tilesY-1 {
			y1 = rows
		}
		for tx := 0; tx < c.tilesX; tx++ {
			x0 := int(float64(tx) * tw)
			x1 := int(float64(tx+1) * tw)
			if tx == c.tilesX-1 {
				x1 = cols
			}
			luts[ty*c.tilesX+tx] = c.tileLUT(src, x0, y0, x1, y1)
		}
	}

	dst := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		// Locate the two tile rows whose centres bracket this pixel.
		gy := (float64(y)+0.5)/th - 0.5
		ty0, ty1, wy := histogram2bracket(gy, c.tilesY)
		for x := 0; x < cols; x++ {
			gx := (float64(x)+0.5)/tw - 0.5
			tx0, tx1, wx := histogram2bracket(gx, c.tilesX)
			v := src.Data[y*cols+x]
			// Bilinear interpolation of the four surrounding tile LUTs.
			v00 := float64(luts[ty0*c.tilesX+tx0][v])
			v01 := float64(luts[ty0*c.tilesX+tx1][v])
			v10 := float64(luts[ty1*c.tilesX+tx0][v])
			v11 := float64(luts[ty1*c.tilesX+tx1][v])
			top := v00*(1-wx) + v01*wx
			bot := v10*(1-wx) + v11*wx
			dst.Data[y*cols+x] = histogram2clampByte(top*(1-wy) + bot*wy)
		}
	}
	return dst, nil
}

// histogram2bracket maps a fractional tile coordinate to the two tile indices
// whose centres surround it and the interpolation weight toward the second.
// Coordinates outside the tile centres clamp to the nearest tile with weight 0.
func histogram2bracket(g float64, n int) (i0, i1 int, w float64) {
	if g <= 0 {
		return 0, 0, 0
	}
	if g >= float64(n-1) {
		return n - 1, n - 1, 0
	}
	i0 = int(math.Floor(g))
	i1 = i0 + 1
	w = g - float64(i0)
	return i0, i1, w
}

// tileLUT computes the clipped, equalised mapping LUT for the tile spanning
// columns [x0,x1) and rows [y0,y1) of src.
func (c *CLAHE) tileLUT(src *cv.Mat, x0, y0, x1, y1 int) [256]uint8 {
	var hist [256]int
	cols := src.Cols
	for y := y0; y < y1; y++ {
		row := y * cols
		for x := x0; x < x1; x++ {
			hist[src.Data[row+x]]++
		}
	}
	nPixels := (x1 - x0) * (y1 - y0)

	if c.clipLimit > 0 {
		// Absolute clip limit in counts, at least one.
		limit := int(c.clipLimit * float64(nPixels) / 256.0)
		if limit < 1 {
			limit = 1
		}
		excess := 0
		for i := 0; i < 256; i++ {
			if hist[i] > limit {
				excess += hist[i] - limit
				hist[i] = limit
			}
		}
		// Redistribute the clipped mass uniformly across all bins.
		inc := excess / 256
		rem := excess % 256
		for i := 0; i < 256; i++ {
			hist[i] += inc
		}
		for i := 0; i < rem; i++ {
			hist[i]++
		}
	}

	var lut [256]uint8
	if nPixels == 0 {
		for i := range lut {
			lut[i] = uint8(i)
		}
		return lut
	}
	acc := 0
	scale := 255.0 / float64(nPixels)
	for i := 0; i < 256; i++ {
		acc += hist[i]
		lut[i] = histogram2clampByte(float64(acc) * scale)
	}
	return lut
}

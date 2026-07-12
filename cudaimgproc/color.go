package cudaimgproc

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// CvtColor converts src between colour spaces on the CPU and returns a new
// GpuMat, mirroring cuda::cvtColor. It accepts every [cv.ColorConversionCode]
// the root package supports (RGB/BGR/Gray, HSV, HLS, Lab and YCrCb, in both
// directions). The trailing Stream argument is accepted and ignored. It panics
// on an empty source or a channel count the code cannot use.
func CvtColor(src GpuMat, code cv.ColorConversionCode, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("CvtColor")
	return wrap(cv.CvtColor(m, code))
}

// SwapChannels reorders the channels of src in place-like fashion, returning a
// new GpuMat whose channel c is taken from source channel dstOrder[c]. This
// mirrors cuda::swapChannels, which is commonly used to convert between RGB and
// BGR memory layouts or to move the alpha channel. dstOrder must have exactly
// as many entries as src has channels, and each entry must index a valid source
// channel. The trailing Stream argument is accepted and ignored.
func SwapChannels(src GpuMat, dstOrder []int, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("SwapChannels")
	if len(dstOrder) != m.Channels {
		panic(fmt.Sprintf("cudaimgproc: SwapChannels dstOrder has %d entries, want %d", len(dstOrder), m.Channels))
	}
	for i, o := range dstOrder {
		if o < 0 || o >= m.Channels {
			panic(fmt.Sprintf("cudaimgproc: SwapChannels dstOrder[%d]=%d out of range", i, o))
		}
	}
	dst := cv.NewMat(m.Rows, m.Cols, m.Channels)
	n := m.Total()
	for p := 0; p < n; p++ {
		base := p * m.Channels
		for c := 0; c < m.Channels; c++ {
			dst.Data[base+c] = m.Data[base+dstOrder[c]]
		}
	}
	return wrap(dst)
}

// GammaCorrection applies sRGB gamma (de)correction, matching cuda::gammaCorrection.
// When forward is true it encodes linear light to sRGB (the OpenCV default);
// when false it decodes sRGB to linear light. The mapping is applied
// independently to each colour channel through a 256-entry lookup table; for a
// 4-channel image the fourth (alpha) channel is passed through unchanged. The
// trailing Stream argument is accepted and ignored. It panics unless src has 3
// or 4 channels.
func GammaCorrection(src GpuMat, forward bool, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("GammaCorrection")
	if m.Channels != 3 && m.Channels != 4 {
		panic(fmt.Sprintf("cudaimgproc: GammaCorrection requires 3 or 4 channels, got %d", m.Channels))
	}
	lut := gammaLUT(forward)
	dst := cv.NewMat(m.Rows, m.Cols, m.Channels)
	n := m.Total()
	for p := 0; p < n; p++ {
		base := p * m.Channels
		dst.Data[base+0] = lut[m.Data[base+0]]
		dst.Data[base+1] = lut[m.Data[base+1]]
		dst.Data[base+2] = lut[m.Data[base+2]]
		if m.Channels == 4 {
			dst.Data[base+3] = m.Data[base+3]
		}
	}
	return wrap(dst)
}

// gammaLUT builds the 256-entry sRGB encode (forward) or decode (inverse) table.
func gammaLUT(forward bool) [256]uint8 {
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		c := float64(i) / 255
		var v float64
		if forward {
			// Linear -> sRGB.
			if c <= 0.0031308 {
				v = 12.92 * c
			} else {
				v = 1.055*math.Pow(c, 1/2.4) - 0.055
			}
		} else {
			// sRGB -> linear.
			if c <= 0.04045 {
				v = c / 12.92
			} else {
				v = math.Pow((c+0.055)/1.055, 2.4)
			}
		}
		lut[i] = clampU8(v*255 + 0.5)
	}
	return lut
}

// BayerCode selects the Bayer colour-filter-array layout for [DemosaicBayer]
// and [CvtColorBayer]. The two letters name the colours of the top-left 2×2
// tile read left-to-right, top-to-bottom, following OpenCV's COLOR_Bayer**2BGR
// naming (which describes the second row of the tile). The produced image is
// RGB-ordered to match the rest of this port.
type BayerCode int

const (
	// BayerBG has the tile [[B,G],[G,R]].
	BayerBG BayerCode = iota
	// BayerGB has the tile [[G,B],[R,G]].
	BayerGB
	// BayerRG has the tile [[R,G],[G,B]].
	BayerRG
	// BayerGR has the tile [[G,R],[B,G]].
	BayerGR
)

// DemosaicBayer reconstructs a 3-channel RGB image from a single-channel Bayer
// mosaic using bilinear interpolation, mirroring cuda::demosaicing for the
// Bayer→BGR conversions. src must be single-channel. The trailing Stream
// argument is accepted and ignored. It panics on a non single-channel source.
func DemosaicBayer(src GpuMat, code BayerCode, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("DemosaicBayer")
	if m.Channels != 1 {
		panic(fmt.Sprintf("cudaimgproc: DemosaicBayer requires a single-channel source, got %d", m.Channels))
	}
	rows, cols := m.Rows, m.Cols
	dst := cv.NewMat(rows, cols, 3)
	// colorAt returns the Bayer colour (0=R,1=G,2=B) sampled at (y,x) for the
	// selected pattern.
	colorAt := bayerColorFunc(code)
	at := func(y, x int) float64 {
		if y < 0 {
			y = -y
		}
		if x < 0 {
			x = -x
		}
		if y >= rows {
			y = 2*rows - 2 - y
		}
		if x >= cols {
			x = 2*cols - 2 - x
		}
		if y < 0 {
			y = 0
		}
		if x < 0 {
			x = 0
		}
		return float64(m.Data[y*cols+x])
	}
	avg := func(vs ...float64) float64 {
		var s float64
		for _, v := range vs {
			s += v
		}
		return s / float64(len(vs))
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			col := colorAt(y, x)
			center := at(y, x)
			var r, g, b float64
			switch col {
			case 0: // red pixel
				r = center
				g = avg(at(y-1, x), at(y+1, x), at(y, x-1), at(y, x+1))
				b = avg(at(y-1, x-1), at(y-1, x+1), at(y+1, x-1), at(y+1, x+1))
			case 2: // blue pixel
				b = center
				g = avg(at(y-1, x), at(y+1, x), at(y, x-1), at(y, x+1))
				r = avg(at(y-1, x-1), at(y-1, x+1), at(y+1, x-1), at(y+1, x+1))
			default: // green pixel
				g = center
				// The two colours flanking a green pixel depend on whether the
				// red neighbours are horizontal or vertical.
				if colorAt(y, x-1) == 0 || colorAt(y, x+1) == 0 {
					r = avg(at(y, x-1), at(y, x+1))
					b = avg(at(y-1, x), at(y+1, x))
				} else {
					b = avg(at(y, x-1), at(y, x+1))
					r = avg(at(y-1, x), at(y+1, x))
				}
			}
			i := (y*cols + x) * 3
			dst.Data[i+0] = clampU8(r + 0.5)
			dst.Data[i+1] = clampU8(g + 0.5)
			dst.Data[i+2] = clampU8(b + 0.5)
		}
	}
	return wrap(dst)
}

// CvtColorBayer is an alias for [DemosaicBayer] kept for parity with call sites
// that use the cvtColor spelling for Bayer conversions.
func CvtColorBayer(src GpuMat, code BayerCode, streams ...Stream) GpuMat {
	return DemosaicBayer(src, code, streams...)
}

// bayerColorFunc returns a closure giving the CFA colour (0=R,1=G,2=B) at each
// pixel for the chosen pattern. The pattern names the top-left 2×2 tile; the
// parity of (y,x) selects the tile position.
func bayerColorFunc(code BayerCode) func(y, x int) int {
	// tile[row][col] holds the colour for the top-left 2×2 block.
	var tile [2][2]int
	switch code {
	case BayerBG:
		tile = [2][2]int{{2, 1}, {1, 0}}
	case BayerGB:
		tile = [2][2]int{{1, 2}, {0, 1}}
	case BayerRG:
		tile = [2][2]int{{0, 1}, {1, 2}}
	case BayerGR:
		tile = [2][2]int{{1, 0}, {2, 1}}
	default:
		panic(fmt.Sprintf("cudaimgproc: unknown BayerCode %d", code))
	}
	return func(y, x int) int { return tile[y&1][x&1] }
}

// AlphaCompType selects the Porter–Duff compositing operator used by
// [AlphaComp], matching OpenCV's ALPHA_* constants.
type AlphaCompType int

const (
	// AlphaOver is the classic "src over dst" operator.
	AlphaOver AlphaCompType = iota
	// AlphaIn keeps src only where dst is opaque.
	AlphaIn
	// AlphaOut keeps src only where dst is transparent.
	AlphaOut
	// AlphaAtop draws src atop dst, bounded by dst's coverage.
	AlphaAtop
	// AlphaXor keeps each image only where the other is transparent.
	AlphaXor
	// AlphaPlus adds the two premultiplied images.
	AlphaPlus
)

// AlphaComp composites two 4-channel (RGBA) images using the selected
// Porter–Duff operator, mirroring cuda::alphaComp. Both inputs must have the
// same dimensions and exactly 4 channels; alpha is taken from the fourth
// channel and normalised to [0,1]. The trailing Stream argument is accepted and
// ignored. It panics on mismatched or non-RGBA inputs.
func AlphaComp(img1, img2 GpuMat, alphaOp AlphaCompType, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	a := img1.requireHost("AlphaComp")
	b := img2.requireHost("AlphaComp")
	if a.Channels != 4 || b.Channels != 4 {
		panic("cudaimgproc: AlphaComp requires 4-channel (RGBA) inputs")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("cudaimgproc: AlphaComp size mismatch")
	}
	// Porter–Duff coefficients (fa applied to src, fb to dst) as a function of
	// the two coverage values.
	coeff := func(sa, da float64) (fa, fb float64) {
		switch alphaOp {
		case AlphaOver:
			return 1, 1 - sa
		case AlphaIn:
			return da, 0
		case AlphaOut:
			return 1 - da, 0
		case AlphaAtop:
			return da, 1 - sa
		case AlphaXor:
			return 1 - da, 1 - sa
		case AlphaPlus:
			return 1, 1
		default:
			panic(fmt.Sprintf("cudaimgproc: unknown AlphaCompType %d", alphaOp))
		}
	}
	dst := cv.NewMat(a.Rows, a.Cols, 4)
	n := a.Total()
	for p := 0; p < n; p++ {
		base := p * 4
		sa := float64(a.Data[base+3]) / 255
		da := float64(b.Data[base+3]) / 255
		fa, fb := coeff(sa, da)
		for c := 0; c < 3; c++ {
			// Composite premultiplied colour, then un-premultiply.
			sc := float64(a.Data[base+c]) * sa
			dc := float64(b.Data[base+c]) * da
			outA := sa*fa + da*fb
			var outC float64
			if outA > 0 {
				outC = (sc*fa + dc*fb) / outA
			}
			dst.Data[base+c] = clampU8(outC + 0.5)
		}
		outA := sa*fa + da*fb
		dst.Data[base+3] = clampU8(outA*255 + 0.5)
	}
	return wrap(dst)
}

// clampU8 rounds and clamps v into the [0,255] range.
func clampU8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

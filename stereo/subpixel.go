package stereo

import cv "github.com/malcolmston/opencv"

// SubpixelParabola fits a parabola through the three cost samples cLo = c(d-1),
// cMid = c(d) and cHi = c(d+1) at consecutive disparities and returns the offset
// of the parabola vertex from the centre sample, in pixels. The offset lies in
// [-0.5, 0.5] for a genuine local minimum (cMid <= cLo, cHi). It is the standard
// equiangular / parabolic sub-pixel interpolation used to refine a
// winner-take-all disparity:
//
//	offset = (cLo - cHi) / (2 * (cLo - 2*cMid + cHi))
//
// When the denominator is zero (a flat or degenerate triple) it returns 0.
func SubpixelParabola(cLo, cMid, cHi int) float64 {
	denom := cLo - 2*cMid + cHi
	if denom == 0 {
		return 0
	}
	off := float64(cLo-cHi) / float64(2*denom)
	if off < -0.5 {
		off = -0.5
	}
	if off > 0.5 {
		off = 0.5
	}
	return off
}

// RefineSubpixel refines an integer disparity map to sub-pixel precision using
// the aggregated (or data) costs in vol. For each valid pixel it re-reads the
// costs at the map's disparity and its two neighbours, applies
// [SubpixelParabola] and adds the fractional offset. Pixels holding
// [InvalidDisparity], border pixels, or disparities at the ends of the search
// range (where no neighbour cost exists) are copied through as their integer
// value (or [InvalidDisparityF] when invalid).
//
// disp must be single-channel and match vol in size; it panics otherwise.
func RefineSubpixel(disp *cv.Mat, vol *CostVolume) *DisparityF {
	if disp == nil || disp.Empty() {
		panic("stereo: RefineSubpixel given a nil or empty disparity map")
	}
	if disp.Channels != 1 {
		panic("stereo: RefineSubpixel requires a single-channel disparity map")
	}
	if disp.Rows != vol.Rows || disp.Cols != vol.Cols {
		panic("stereo: RefineSubpixel disparity/volume size mismatch")
	}
	out := NewDisparityF(disp.Rows, disp.Cols)
	numD := vol.NumDisparities
	minD := vol.MinDisparity
	for y := 0; y < disp.Rows; y++ {
		for x := 0; x < disp.Cols; x++ {
			p := y*disp.Cols + x
			d := int(disp.Data[p])
			if disp.Data[p] == InvalidDisparity {
				continue
			}
			idx := d - minD
			if idx <= 0 || idx >= numD-1 {
				out.Data[p] = float32(d)
				continue
			}
			base := p * numD
			off := SubpixelParabola(int(vol.Data[base+idx-1]), int(vol.Data[base+idx]), int(vol.Data[base+idx+1]))
			out.Data[p] = float32(float64(d) + off)
		}
	}
	return out
}

package stereo

import cv "github.com/malcolmston/opencv"

// ComputeConfidence estimates a per-pixel matching confidence from a cost volume
// using the peak-ratio (PKRN) measure: the ratio between the best and the
// second-best (non-adjacent) cost. A deep, isolated minimum — the second-best
// cost much larger than the best — yields high confidence; an ambiguous match
// with several similar costs yields low confidence. The confidence is scaled to
// an 8-bit map:
//
//	confidence = round( 255 * (1 - c_best / c_second) )
//
// clamped to [0, 255]. Left border pixels (where the full range is unavailable)
// and pixels whose second-best cost is zero receive 0. The volume may be a raw
// [MatchingCostVolume] or an aggregated [StereoSGM.Aggregate] volume.
func ComputeConfidence(vol *CostVolume) *cv.Mat {
	out := cv.NewMat(vol.Rows, vol.Cols, 1)
	numD := vol.NumDisparities
	borderLimit := vol.MinDisparity + numD - 1
	for y := 0; y < vol.Rows; y++ {
		for x := 0; x < vol.Cols; x++ {
			if x < borderLimit {
				continue
			}
			base := (y*vol.Cols + x) * numD
			bestIdx, bestC := 0, costSentinel
			for idx := 0; idx < numD; idx++ {
				if c := vol.Data[base+idx]; c < bestC {
					bestC, bestIdx = c, idx
				}
			}
			secondC := costSentinel
			for idx := 0; idx < numD; idx++ {
				if idx < bestIdx-1 || idx > bestIdx+1 {
					if c := vol.Data[base+idx]; c < secondC {
						secondC = c
					}
				}
			}
			if secondC == costSentinel || secondC <= 0 {
				continue
			}
			ratio := 1.0 - float64(bestC)/float64(secondC)
			conf := int(ratio*255.0 + 0.5)
			out.Data[y*vol.Cols+x] = uint8(clampInt(conf, 0, 255))
		}
	}
	return out
}

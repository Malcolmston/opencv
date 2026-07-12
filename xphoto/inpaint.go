package xphoto

import (
	cv "github.com/malcolmston/opencv"
)

// InpaintDefaultRadius is the default exemplar patch radius used by [Inpaint];
// the matching window is (2*radius+1) square.
const InpaintDefaultRadius = 2

// inpaintCenter is a candidate exemplar patch centre.
type inpaintCenter struct{ y, x int }

// Inpaint fills the masked region of src with plausible content using a
// SHIFTMAP-style exemplar search, approximating cv::xphoto::inpaint. Every
// pixel where mask is non-zero is treated as unknown and reconstructed; where
// mask is zero the original sample is kept.
//
// The region is filled from its boundary inward (an "onion peel"): in each
// layer, every unknown pixel that touches known content is filled by searching
// the known part of the image for the patch (translation, i.e. shift) whose
// overlap with the pixel's surroundings has the smallest sum-of-squared
// difference, and copying that patch's centre. This is a greedy, local
// approximation of the shift-map / patch-based fill; see the package Deferred
// note about the full global EM optimisation.
//
// mask must be single-channel and the same size as src. src may be single- or
// three-channel. The result is a new Mat; src and mask are not modified.
func Inpaint(src, mask *cv.Mat) *cv.Mat {
	return InpaintWithRadius(src, mask, InpaintDefaultRadius)
}

// InpaintWithRadius is [Inpaint] with an explicit exemplar patch radius. A
// larger radius matches more context (smoother, slower); radius <= 0 defaults
// to [InpaintDefaultRadius].
func InpaintWithRadius(src, mask *cv.Mat, radius int) *cv.Mat {
	requireNonEmpty(src, "Inpaint")
	requireNonEmpty(mask, "Inpaint")
	requireChannels(mask, 1, "Inpaint mask")
	requireSameSize(src, mask, "Inpaint")
	if radius <= 0 {
		radius = InpaintDefaultRadius
	}
	rows, cols, ch := src.Rows, src.Cols, src.Channels
	out := src.Clone()

	// known[i] reports whether pixel i currently holds valid content.
	known := make([]bool, rows*cols)
	unknownCount := 0
	for i := 0; i < rows*cols; i++ {
		if mask.Data[i] == 0 {
			known[i] = true
		} else {
			unknownCount++
		}
	}
	if unknownCount == 0 {
		return out
	}

	// Precompute candidate exemplar centres: pixels whose full patch lies inside
	// the image and is entirely known in the original image. These are the
	// source patches copied from.
	var candidates []inpaintCenter
	for y := radius; y < rows-radius; y++ {
		for x := radius; x < cols-radius; x++ {
			full := true
			for dy := -radius; dy <= radius && full; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					if !known[(y+dy)*cols+(x+dx)] {
						full = false
						break
					}
				}
			}
			if full {
				candidates = append(candidates, inpaintCenter{y, x})
			}
		}
	}

	// fills accumulates the (index -> pixel value) decisions for one layer so
	// that pixels filled this layer are not used as context until the next.
	type fill struct {
		idx int
		val []uint8
	}

	for unknownCount > 0 {
		var layer []fill
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				idx := y*cols + x
				if known[idx] {
					continue
				}
				if !touchesKnown(known, rows, cols, y, x) {
					continue
				}
				val := bestExemplar(out, known, candidates, y, x, radius, ch)
				layer = append(layer, fill{idx: idx, val: val})
			}
		}
		if len(layer) == 0 {
			// No unknown pixel touches known content (e.g. no candidates found):
			// fill everything remaining with its known-neighbour mean to
			// guarantee termination.
			for y := 0; y < rows; y++ {
				for x := 0; x < cols; x++ {
					idx := y*cols + x
					if known[idx] {
						continue
					}
					val := neighbourMean(out, known, rows, cols, y, x, ch)
					copy(out.Data[idx*ch:idx*ch+ch], val)
					known[idx] = true
					unknownCount--
				}
			}
			break
		}
		for _, f := range layer {
			copy(out.Data[f.idx*ch:f.idx*ch+ch], f.val)
			known[f.idx] = true
			unknownCount--
		}
	}
	return out
}

// touchesKnown reports whether pixel (y,x) has at least one 4-connected known
// neighbour.
func touchesKnown(known []bool, rows, cols, y, x int) bool {
	if y > 0 && known[(y-1)*cols+x] {
		return true
	}
	if y < rows-1 && known[(y+1)*cols+x] {
		return true
	}
	if x > 0 && known[y*cols+(x-1)] {
		return true
	}
	if x < cols-1 && known[y*cols+(x+1)] {
		return true
	}
	return false
}

// bestExemplar finds the candidate patch centre whose overlap with the known
// context around (y,x) has the smallest sum-of-squared difference and returns
// that centre's pixel value. When no candidate has any overlapping known
// context it falls back to the known-neighbour mean.
func bestExemplar(out *cv.Mat, known []bool, candidates []inpaintCenter, y, x, radius, ch int) []uint8 {
	rows, cols := out.Rows, out.Cols
	bestSSD := -1.0
	bestIdx := -1
	for ci := range candidates {
		cy, cx := candidates[ci].y, candidates[ci].x
		var ssd float64
		var overlap int
		for dy := -radius; dy <= radius; dy++ {
			ty := y + dy
			if ty < 0 || ty >= rows {
				continue
			}
			for dx := -radius; dx <= radius; dx++ {
				tx := x + dx
				if tx < 0 || tx >= cols {
					continue
				}
				if !known[ty*cols+tx] {
					continue
				}
				ti := (ty*cols + tx) * ch
				si := ((cy+dy)*cols + (cx + dx)) * ch
				for c := 0; c < ch; c++ {
					d := float64(out.Data[ti+c]) - float64(out.Data[si+c])
					ssd += d * d
				}
				overlap++
			}
		}
		if overlap == 0 {
			continue
		}
		// Normalise by overlap so patches with more visible context are not
		// unfairly penalised.
		norm := ssd / float64(overlap)
		if bestIdx < 0 || norm < bestSSD {
			bestSSD = norm
			bestIdx = ci
		}
	}
	if bestIdx < 0 {
		return neighbourMean(out, known, rows, cols, y, x, ch)
	}
	cy, cx := candidates[bestIdx].y, candidates[bestIdx].x
	src := (cy*cols + cx) * ch
	val := make([]uint8, ch)
	copy(val, out.Data[src:src+ch])
	return val
}

// neighbourMean returns the mean of the known 8-connected neighbours of (y,x),
// or mid-grey if none exist.
func neighbourMean(out *cv.Mat, known []bool, rows, cols, y, x, ch int) []uint8 {
	acc := make([]float64, ch)
	var n int
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dy == 0 && dx == 0 {
				continue
			}
			ny, nx := y+dy, x+dx
			if ny < 0 || ny >= rows || nx < 0 || nx >= cols {
				continue
			}
			if !known[ny*cols+nx] {
				continue
			}
			base := (ny*cols + nx) * ch
			for c := 0; c < ch; c++ {
				acc[c] += float64(out.Data[base+c])
			}
			n++
		}
	}
	val := make([]uint8, ch)
	if n == 0 {
		for c := range val {
			val[c] = 128
		}
		return val
	}
	for c := 0; c < ch; c++ {
		val[c] = clampU8(acc[c] / float64(n))
	}
	return val
}

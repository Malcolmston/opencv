package filters2

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// gatherWindow collects, into buf, the samples of channel c in the square
// window of the given radius around (y, x), using edge replication, and returns
// the filled slice.
func gatherWindow(src *cv.Mat, y, x, c, radius int, buf []uint8) []uint8 {
	buf = buf[:0]
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			buf = append(buf, atReplicate(src, y+dy, x+dx, c))
		}
	}
	return buf
}

// MedianFilter replaces every sample with the median of its square
// neighbourhood of the given odd size, an effective remedy for impulse
// (salt-and-pepper) noise. Each channel is processed independently. It panics
// on empty input or a non-positive even size.
func MedianFilter(src *cv.Mat, size int) *cv.Mat {
	requireNonEmpty(src, "MedianFilter")
	requireOddPositive(size, "MedianFilter")
	radius := size / 2
	dst := like(src)
	ch := src.Channels
	buf := make([]uint8, 0, size*size)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < ch; c++ {
				buf = gatherWindow(src, y, x, c, radius, buf)
				sort.Slice(buf, func(i, j int) bool { return buf[i] < buf[j] })
				dst.Data[(y*src.Cols+x)*ch+c] = buf[len(buf)/2]
			}
		}
	}
	return dst
}

// MinFilter (erosion by a square element) replaces every sample with the
// minimum of its neighbourhood of the given odd size, processing each channel
// independently. It panics on empty input or a non-positive even size.
func MinFilter(src *cv.Mat, size int) *cv.Mat {
	return rankExtremum(src, size, true, "MinFilter")
}

// MaxFilter (dilation by a square element) replaces every sample with the
// maximum of its neighbourhood of the given odd size, processing each channel
// independently. It panics on empty input or a non-positive even size.
func MaxFilter(src *cv.Mat, size int) *cv.Mat {
	return rankExtremum(src, size, false, "MaxFilter")
}

func rankExtremum(src *cv.Mat, size int, wantMin bool, op string) *cv.Mat {
	requireNonEmpty(src, op)
	requireOddPositive(size, op)
	radius := size / 2
	dst := like(src)
	ch := src.Channels
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < ch; c++ {
				best := atReplicate(src, y, x, c)
				for dy := -radius; dy <= radius; dy++ {
					for dx := -radius; dx <= radius; dx++ {
						v := atReplicate(src, y+dy, x+dx, c)
						if wantMin {
							if v < best {
								best = v
							}
						} else if v > best {
							best = v
						}
					}
				}
				dst.Data[(y*src.Cols+x)*ch+c] = best
			}
		}
	}
	return dst
}

// MidpointFilter replaces every sample with the midpoint (min+max)/2 of its
// neighbourhood of the given odd size, which suppresses Gaussian and uniform
// noise. Each channel is processed independently. It panics on empty input or a
// non-positive even size.
func MidpointFilter(src *cv.Mat, size int) *cv.Mat {
	requireNonEmpty(src, "MidpointFilter")
	requireOddPositive(size, "MidpointFilter")
	radius := size / 2
	dst := like(src)
	ch := src.Channels
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < ch; c++ {
				lo := atReplicate(src, y, x, c)
				hi := lo
				for dy := -radius; dy <= radius; dy++ {
					for dx := -radius; dx <= radius; dx++ {
						v := atReplicate(src, y+dy, x+dx, c)
						if v < lo {
							lo = v
						}
						if v > hi {
							hi = v
						}
					}
				}
				dst.Data[(y*src.Cols+x)*ch+c] = uint8((int(lo) + int(hi)) / 2)
			}
		}
	}
	return dst
}

// AlphaTrimmedMeanFilter replaces every sample with the mean of its
// neighbourhood of the given odd size after discarding the trim smallest and
// trim largest values, blending the robustness of the median (large trim) with
// the smoothing of the mean (trim == 0). trim must leave at least one value.
// Each channel is processed independently. It panics on empty input, a
// non-positive even size, or too large a trim.
func AlphaTrimmedMeanFilter(src *cv.Mat, size, trim int) *cv.Mat {
	requireNonEmpty(src, "AlphaTrimmedMeanFilter")
	requireOddPositive(size, "AlphaTrimmedMeanFilter")
	if trim < 0 || 2*trim >= size*size {
		panic("filters2: AlphaTrimmedMeanFilter: trim must leave at least one value")
	}
	radius := size / 2
	dst := like(src)
	ch := src.Channels
	buf := make([]uint8, 0, size*size)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < ch; c++ {
				buf = gatherWindow(src, y, x, c, radius, buf)
				sort.Slice(buf, func(i, j int) bool { return buf[i] < buf[j] })
				var sum int
				kept := 0
				for i := trim; i < len(buf)-trim; i++ {
					sum += int(buf[i])
					kept++
				}
				dst.Data[(y*src.Cols+x)*ch+c] = clampU8(float64(sum) / float64(kept))
			}
		}
	}
	return dst
}

// AdaptiveMedianFilter applies the adaptive median filter, which grows its
// window from 3×3 up to maxSize×maxSize (maxSize must be a positive odd
// integer) only where needed. It preserves detail better than a fixed median:
// a pixel is left unchanged unless it is an impulse (equal to the local minimum
// or maximum), in which case it is replaced by the local median. Operates on a
// single-channel image and panics on multi-channel or empty input, or a
// non-positive even maxSize.
func AdaptiveMedianFilter(src *cv.Mat, maxSize int) *cv.Mat {
	requireGray(src, "AdaptiveMedianFilter")
	requireOddPositive(maxSize, "AdaptiveMedianFilter")
	if maxSize < 3 {
		maxSize = 3
	}
	dst := like(src)
	buf := make([]uint8, 0, maxSize*maxSize)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			dst.Data[y*src.Cols+x] = adaptiveMedianAt(src, y, x, maxSize, buf)
		}
	}
	return dst
}

// adaptiveMedianAt evaluates the adaptive-median decision at a single pixel.
func adaptiveMedianAt(src *cv.Mat, y, x, maxSize int, buf []uint8) uint8 {
	zxy := atReplicate(src, y, x, 0)
	for size := 3; size <= maxSize; size += 2 {
		radius := size / 2
		buf = gatherWindow(src, y, x, 0, radius, buf)
		sort.Slice(buf, func(i, j int) bool { return buf[i] < buf[j] })
		zmin := buf[0]
		zmax := buf[len(buf)-1]
		zmed := buf[len(buf)/2]
		// Stage A: is the median itself an impulse?
		if zmed > zmin && zmed < zmax {
			// Stage B: is the centre pixel an impulse?
			if zxy > zmin && zxy < zmax {
				return zxy
			}
			return zmed
		}
		// Median is extreme: enlarge the window and retry.
	}
	// Window reached its maximum: fall back to the largest-window median.
	radius := maxSize / 2
	buf = gatherWindow(src, y, x, 0, radius, buf)
	sort.Slice(buf, func(i, j int) bool { return buf[i] < buf[j] })
	return buf[len(buf)/2]
}

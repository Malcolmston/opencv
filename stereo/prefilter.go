package stereo

import cv "github.com/malcolmston/opencv"

// PrefilterType selects the pre-normalisation applied to the input images before
// block matching, matching OpenCV's StereoBM pre-filter modes. Pre-filtering
// removes low-frequency intensity gradients (vignetting, exposure differences)
// so the sum-of-absolute-differences cost compares texture rather than
// brightness.
type PrefilterType int

// PrefilterNone, PrefilterXSobel and PrefilterNormalizedResponse are the
// supported pre-filter modes.
const (
	// PrefilterNone leaves the image unchanged.
	PrefilterNone PrefilterType = iota
	// PrefilterXSobel applies a horizontal Sobel derivative and clamps it to
	// [0, 2*capVal], like OpenCV's PREFILTER_XSOBEL. It emphasises vertical edges,
	// the features block matching relies on.
	PrefilterXSobel
	// PrefilterNormalizedResponse subtracts a local box-filter mean and clamps the
	// residual to [-capVal, capVal] (re-centred to [0, 2*capVal]), like OpenCV's
	// PREFILTER_NORMALIZED_RESPONSE.
	PrefilterNormalizedResponse
)

// ApplyXSobelPrefilter applies the horizontal Sobel pre-filter to a grayscale
// (or RGB, converted to gray) image. Each output sample is the clamped
// horizontal derivative recentred to preFilterCap:
//
//	g = Sobel_x(image);  out = clamp(g, -capVal, capVal) + capVal
//
// so a flat region maps to preFilterCap and strong vertical edges saturate at 0
// or 2*capVal. preFilterCap defaults to 31 when non-positive and is clamped to
// [1, 63]. It panics on empty or unsupported input.
func ApplyXSobelPrefilter(m *cv.Mat, preFilterCap int) *cv.Mat {
	capVal := normalizeCap(preFilterCap)
	g := grayMat(m)
	rows, cols := g.Rows, g.Cols
	src := make([]int, rows*cols)
	for i := range src {
		src[i] = int(g.Data[i])
	}
	out := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// 3x3 horizontal Sobel with replicated borders.
			var acc int
			for dy := -1; dy <= 1; dy++ {
				yy := clampInt(y+dy, 0, rows-1)
				wy := 1
				if dy == 0 {
					wy = 2
				}
				xl := clampInt(x-1, 0, cols-1)
				xr := clampInt(x+1, 0, cols-1)
				acc += wy * (src[yy*cols+xr] - src[yy*cols+xl])
			}
			if acc < -capVal {
				acc = -capVal
			}
			if acc > capVal {
				acc = capVal
			}
			out.Data[y*cols+x] = uint8(acc + capVal)
		}
	}
	return out
}

// ApplyNormalizedPrefilter applies the normalized-response pre-filter: it
// subtracts the local mean over a preFilterSize×preFilterSize box window and
// clamps the residual, recentred to preFilterCap:
//
//	out = clamp(image - boxmean(image, preFilterSize), -capVal, capVal) + capVal
//
// preFilterSize must be a positive odd integer (default 9); preFilterCap
// defaults to 31 and is clamped to [1, 63]. Borders are replicated. It panics on
// empty or unsupported input, or an even preFilterSize.
func ApplyNormalizedPrefilter(m *cv.Mat, preFilterSize, preFilterCap int) *cv.Mat {
	if preFilterSize <= 0 {
		preFilterSize = 9
	}
	requireOdd(preFilterSize, "ApplyNormalizedPrefilter.preFilterSize")
	capVal := normalizeCap(preFilterCap)
	g := grayMat(m)
	rows, cols := g.Rows, g.Cols
	src := make([]int, rows*cols)
	for i := range src {
		src[i] = int(g.Data[i])
	}
	half := preFilterSize / 2
	area := preFilterSize * preFilterSize
	out := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			sum := 0
			for dy := -half; dy <= half; dy++ {
				yy := clampInt(y+dy, 0, rows-1)
				rowBase := yy * cols
				for dx := -half; dx <= half; dx++ {
					xx := clampInt(x+dx, 0, cols-1)
					sum += src[rowBase+xx]
				}
			}
			residual := src[y*cols+x] - sum/area
			if residual < -capVal {
				residual = -capVal
			}
			if residual > capVal {
				residual = capVal
			}
			out.Data[y*cols+x] = uint8(residual + capVal)
		}
	}
	return out
}

// applyPrefilter dispatches on t, returning a pre-filtered single-channel Mat.
func applyPrefilter(m *cv.Mat, t PrefilterType, preFilterSize, preFilterCap int) *cv.Mat {
	switch t {
	case PrefilterXSobel:
		return ApplyXSobelPrefilter(m, preFilterCap)
	case PrefilterNormalizedResponse:
		return ApplyNormalizedPrefilter(m, preFilterSize, preFilterCap)
	default:
		return grayMat(m)
	}
}

// normalizeCap applies the OpenCV default and range for the pre-filter capVal.
func normalizeCap(preFilterCap int) int {
	if preFilterCap <= 0 {
		preFilterCap = 31
	}
	return clampInt(preFilterCap, 1, 63)
}

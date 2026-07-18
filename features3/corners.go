package features3

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// features3structureTensor computes the windowed sums of the Sobel gradient
// products (Sxx, Syy, Sxy) over a blockSize×blockSize box with a replicated
// border, the common input to the Harris and Shi–Tomasi corner measures.
func features3structureTensor(g *features3gray, blockSize int) (sxx, syy, sxy []float64) {
	if blockSize < 1 {
		blockSize = 3
	}
	gx, gy := features3sobel(g)
	n := g.Rows * g.Cols
	ixx := make([]float64, n)
	iyy := make([]float64, n)
	ixy := make([]float64, n)
	for i := 0; i < n; i++ {
		ixx[i] = gx[i] * gx[i]
		iyy[i] = gy[i] * gy[i]
		ixy[i] = gx[i] * gy[i]
	}
	sxx = features3boxSum(ixx, g.Rows, g.Cols, blockSize)
	syy = features3boxSum(iyy, g.Rows, g.Cols, blockSize)
	sxy = features3boxSum(ixy, g.Rows, g.Cols, blockSize)
	return
}

// features3boxSum sums a per-pixel field over a blockSize×blockSize window
// centred on each pixel with a replicated border.
func features3boxSum(field []float64, rows, cols, blockSize int) []float64 {
	out := make([]float64, rows*cols)
	a := blockSize / 2
	clamp := func(v, hi int) int {
		if v < 0 {
			return 0
		}
		if v >= hi {
			return hi - 1
		}
		return v
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for dy := -a; dy <= a; dy++ {
				ry := clamp(y+dy, rows)
				for dx := -a; dx <= a; dx++ {
					rx := clamp(x+dx, cols)
					s += field[ry*cols+rx]
				}
			}
			out[y*cols+x] = s
		}
	}
	return out
}

// HarrisResponse computes the Harris corner response of an image. For each pixel
// it forms the structure tensor M summed over a blockSize window of Sobel
// gradients and returns R = det(M) - k*trace(M)^2 as a cv.FloatMat, where large
// positive values indicate corners. A typical k is 0.04. Colour input is
// converted to grayscale first.
func HarrisResponse(img *cv.Mat, blockSize int, k float64) *cv.FloatMat {
	g := features3ToGray(img)
	sxx, syy, sxy := features3structureTensor(g, blockSize)
	res := cv.NewFloatMat(g.Rows, g.Cols)
	for i := range res.Data {
		det := sxx[i]*syy[i] - sxy[i]*sxy[i]
		trace := sxx[i] + syy[i]
		res.Data[i] = det - k*trace*trace
	}
	return res
}

// CornerMinEigenVal computes the Shi–Tomasi corner measure: the smaller
// eigenvalue of the windowed structure tensor at each pixel, returned as a
// cv.FloatMat. Large values indicate corners. Colour input is converted to
// grayscale first.
func CornerMinEigenVal(img *cv.Mat, blockSize int) *cv.FloatMat {
	g := features3ToGray(img)
	sxx, syy, sxy := features3structureTensor(g, blockSize)
	res := cv.NewFloatMat(g.Rows, g.Cols)
	for i := range res.Data {
		a := sxx[i]
		b := sxy[i]
		c := syy[i]
		tr := a + c
		disc := math.Sqrt((a-c)*(a-c) + 4*b*b)
		res.Data[i] = (tr - disc) / 2
	}
	return res
}

// CornerEigenVals computes both eigenvalues of the windowed structure tensor at
// every pixel, returning the larger eigenvalue field lambda1 and the smaller
// lambda2 as separate cv.FloatMats. Colour input is converted to grayscale.
func CornerEigenVals(img *cv.Mat, blockSize int) (lambda1, lambda2 *cv.FloatMat) {
	g := features3ToGray(img)
	sxx, syy, sxy := features3structureTensor(g, blockSize)
	lambda1 = cv.NewFloatMat(g.Rows, g.Cols)
	lambda2 = cv.NewFloatMat(g.Rows, g.Cols)
	for i := range lambda1.Data {
		a := sxx[i]
		b := sxy[i]
		c := syy[i]
		tr := a + c
		disc := math.Sqrt((a-c)*(a-c) + 4*b*b)
		lambda1.Data[i] = (tr + disc) / 2
		lambda2.Data[i] = (tr - disc) / 2
	}
	return lambda1, lambda2
}

// PreCornerDetect computes OpenCV's preCornerDetect feature map, the quantity
// Ix^2*Iyy + Iy^2*Ixx - 2*Ix*Iy*Ixy formed from first and second image
// derivatives. Its zero crossings mark ridges and its extrema mark corners.
// Colour input is converted to grayscale first.
func PreCornerDetect(img *cv.Mat) *cv.FloatMat {
	g := features3ToGray(img)
	rows, cols := g.Rows, g.Cols
	res := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			c := g.atClamped(x, y)
			l := g.atClamped(x-1, y)
			r := g.atClamped(x+1, y)
			u := g.atClamped(x, y-1)
			d := g.atClamped(x, y+1)
			ul := g.atClamped(x-1, y-1)
			ur := g.atClamped(x+1, y-1)
			dl := g.atClamped(x-1, y+1)
			dr := g.atClamped(x+1, y+1)
			ix := (r - l) / 2
			iy := (d - u) / 2
			ixx := r - 2*c + l
			iyy := d - 2*c + u
			ixy := (dr - dl - ur + ul) / 4
			res.Data[y*cols+x] = ix*ix*iyy + iy*iy*ixx - 2*ix*iy*ixy
		}
	}
	return res
}

// HarrisCorners detects corners with the Harris measure. It computes
// [HarrisResponse], keeps responses above qualityLevel times the maximum
// response, applies 3×3 non-maximum suppression and returns the survivors as
// keypoints sorted by descending response. When maxCorners > 0 only the
// strongest maxCorners are returned.
func HarrisCorners(img *cv.Mat, blockSize int, k, qualityLevel float64, maxCorners int) []KeyPoint {
	resp := HarrisResponse(img, blockSize, k)
	return features3peaks(resp, qualityLevel, maxCorners)
}

// ShiTomasiCorners detects corners with the Shi–Tomasi minimum-eigenvalue
// measure. It computes [CornerMinEigenVal], keeps responses above qualityLevel
// times the maximum, applies 3×3 non-maximum suppression and returns the
// survivors as keypoints sorted by descending response. When maxCorners > 0 only
// the strongest maxCorners are returned.
func ShiTomasiCorners(img *cv.Mat, blockSize int, qualityLevel float64, maxCorners int) []KeyPoint {
	resp := CornerMinEigenVal(img, blockSize)
	return features3peaks(resp, qualityLevel, maxCorners)
}

// features3peaks extracts local maxima of a response image above qualityLevel
// times its maximum positive value, using 3×3 non-maximum suppression, sorted by
// descending response and truncated to maxCorners when positive.
func features3peaks(resp *cv.FloatMat, qualityLevel float64, maxCorners int) []KeyPoint {
	rows, cols := resp.Rows, resp.Cols
	var maxResp float64
	for _, v := range resp.Data {
		if v > maxResp {
			maxResp = v
		}
	}
	if maxResp <= 0 {
		return nil
	}
	threshold := qualityLevel * maxResp
	var kps []KeyPoint
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			v := resp.Data[y*cols+x]
			if v < threshold {
				continue
			}
			isMax := true
			for dy := -1; dy <= 1 && isMax; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					if resp.Data[(y+dy)*cols+(x+dx)] > v {
						isMax = false
						break
					}
				}
			}
			if isMax {
				kps = append(kps, NewKeyPoint(float64(x), float64(y), v))
			}
		}
	}
	SortKeyPointsByResponse(kps)
	if maxCorners > 0 && maxCorners < len(kps) {
		kps = kps[:maxCorners]
	}
	return kps
}

// GoodFeaturesToTrack finds strong corners with the Shi–Tomasi measure, the
// Go analogue of OpenCV's cv::goodFeaturesToTrack. Corners weaker than
// qualityLevel times the strongest are discarded; the survivors are taken in
// descending strength and each accepted corner suppresses others within
// minDistance pixels. At most maxCorners points are returned (all when
// maxCorners <= 0). Colour input is converted to grayscale first.
func GoodFeaturesToTrack(img *cv.Mat, maxCorners int, qualityLevel, minDistance float64, blockSize int) []KeyPoint {
	resp := CornerMinEigenVal(img, blockSize)
	cands := features3peaks(resp, qualityLevel, 0)
	kept := KeypointNMS(cands, minDistance)
	if maxCorners > 0 && maxCorners < len(kept) {
		kept = kept[:maxCorners]
	}
	return kept
}

// CornerSubPix refines integer corner locations to sub-pixel accuracy using the
// classic iterative method: the refined point is the location where gradient
// vectors in a (2*winSize+1) neighbourhood are most nearly orthogonal to the
// vectors from the point to the sample. Iteration stops after maxIter steps or
// when the update falls below epsilon pixels. Colour input is converted to
// grayscale first.
func CornerSubPix(img *cv.Mat, corners []cv.Point2f, winSize, maxIter int, epsilon float64) []cv.Point2f {
	g := features3ToGray(img)
	gx, gy := features3sobel(g)
	if winSize < 1 {
		winSize = 5
	}
	if maxIter < 1 {
		maxIter = 10
	}
	out := make([]cv.Point2f, len(corners))
	sampleGrad := func(field []float64, x, y int) float64 {
		if x < 0 {
			x = 0
		} else if x >= g.Cols {
			x = g.Cols - 1
		}
		if y < 0 {
			y = 0
		} else if y >= g.Rows {
			y = g.Rows - 1
		}
		return field[y*g.Cols+x]
	}
	for ci, c := range corners {
		cx, cy := c.X, c.Y
		for iter := 0; iter < maxIter; iter++ {
			var a11, a12, a22, b1, b2 float64
			bx := int(math.Round(cx))
			by := int(math.Round(cy))
			for dy := -winSize; dy <= winSize; dy++ {
				for dx := -winSize; dx <= winSize; dx++ {
					px := bx + dx
					py := by + dy
					gxv := sampleGrad(gx, px, py)
					gyv := sampleGrad(gy, px, py)
					a11 += gxv * gxv
					a12 += gxv * gyv
					a22 += gyv * gyv
					b1 += gxv*gxv*float64(px) + gxv*gyv*float64(py)
					b2 += gxv*gyv*float64(px) + gyv*gyv*float64(py)
				}
			}
			det := a11*a22 - a12*a12
			if math.Abs(det) < 1e-12 {
				break
			}
			nx := (a22*b1 - a12*b2) / det
			ny := (a11*b2 - a12*b1) / det
			shift := math.Hypot(nx-cx, ny-cy)
			cx, cy = nx, ny
			if shift < epsilon {
				break
			}
		}
		out[ci] = cv.Point2f{X: cx, Y: cy}
	}
	return out
}

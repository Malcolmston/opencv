package stitch

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// grayValue returns the luminance of pixel (x, y) of img. For single-channel
// images it is the sample itself; for three or more channels it is the Rec. 601
// weighted sum of the first three channels (treated as RGB).
func grayValue(img *cv.Mat, x, y int) float64 {
	i := (y*img.Cols + x) * img.Channels
	if img.Channels == 1 {
		return float64(img.Data[i])
	}
	r := float64(img.Data[i])
	g := float64(img.Data[i+1])
	b := float64(img.Data[i+2])
	return 0.299*r + 0.587*g + 0.114*b
}

// HarrisResponse computes the Harris corner response map of img over a
// (2*window+1)² aggregation neighbourhood, using the sensitivity parameter k
// (typically 0.04–0.06). The returned FloatMat has the same dimensions as img;
// larger values indicate stronger corners. Gradients use the central difference
// and border pixels replicate the edge.
func HarrisResponse(img *cv.Mat, window int, k float64) *cv.FloatMat {
	rows, cols := img.Rows, img.Cols
	ix := make([]float64, rows*cols)
	iy := make([]float64, rows*cols)
	clampX := func(x int) int {
		if x < 0 {
			return 0
		}
		if x >= cols {
			return cols - 1
		}
		return x
	}
	clampY := func(y int) int {
		if y < 0 {
			return 0
		}
		if y >= rows {
			return rows - 1
		}
		return y
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gx := grayValue(img, clampX(x+1), y) - grayValue(img, clampX(x-1), y)
			gy := grayValue(img, x, clampY(y+1)) - grayValue(img, x, clampY(y-1))
			ix[y*cols+x] = gx * 0.5
			iy[y*cols+x] = gy * 0.5
		}
	}
	resp := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var sxx, syy, sxy float64
			for wy := -window; wy <= window; wy++ {
				yy := clampY(y + wy)
				for wx := -window; wx <= window; wx++ {
					xx := clampX(x + wx)
					gx := ix[yy*cols+xx]
					gy := iy[yy*cols+xx]
					sxx += gx * gx
					syy += gy * gy
					sxy += gx * gy
				}
			}
			det := sxx*syy - sxy*sxy
			trace := sxx + syy
			resp.Data[y*cols+x] = det - k*trace*trace
		}
	}
	return resp
}

// HarrisCorners detects up to maxCorners strong Harris corners in img. quality
// sets the response threshold as a fraction of the strongest response (for
// example 0.01), and minDistance is the minimum spacing in pixels between
// returned corners, enforced greedily from strongest to weakest. Corners are
// returned strongest-first.
func HarrisCorners(img *cv.Mat, maxCorners int, quality float64, minDistance int) []PointF {
	resp := HarrisResponse(img, 2, 0.04)
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
	thresh := quality * maxResp

	type cand struct {
		x, y int
		v    float64
	}
	var cands []cand
	// Keep only local maxima above threshold (3×3 non-maximum suppression).
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := resp.Data[y*cols+x]
			if v < thresh {
				continue
			}
			isMax := true
			for dy := -1; dy <= 1 && isMax; dy++ {
				ny := y + dy
				if ny < 0 || ny >= rows {
					continue
				}
				for dx := -1; dx <= 1; dx++ {
					nx := x + dx
					if nx < 0 || nx >= cols || (dx == 0 && dy == 0) {
						continue
					}
					if resp.Data[ny*cols+nx] > v {
						isMax = false
						break
					}
				}
			}
			if isMax {
				cands = append(cands, cand{x, y, v})
			}
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].v != cands[j].v {
			return cands[i].v > cands[j].v
		}
		if cands[i].y != cands[j].y {
			return cands[i].y < cands[j].y
		}
		return cands[i].x < cands[j].x
	})
	var out []PointF
	minD2 := float64(minDistance * minDistance)
	for _, c := range cands {
		if maxCorners > 0 && len(out) >= maxCorners {
			break
		}
		ok := true
		if minDistance > 0 {
			for _, p := range out {
				dx := p.X - float64(c.x)
				dy := p.Y - float64(c.y)
				if dx*dx+dy*dy < minD2 {
					ok = false
					break
				}
			}
		}
		if ok {
			out = append(out, PointF{float64(c.x), float64(c.y)})
		}
	}
	return out
}

// patchStats returns the mean and standard deviation of the grayscale patch of
// half-size half centred at (cx, cy) of img, and whether the patch fits entirely
// inside the image.
func patchStats(img *cv.Mat, cx, cy, half int) (mean, std float64, ok bool) {
	if cx-half < 0 || cy-half < 0 || cx+half >= img.Cols || cy+half >= img.Rows {
		return 0, 0, false
	}
	n := (2*half + 1) * (2*half + 1)
	var sum, sumSq float64
	for dy := -half; dy <= half; dy++ {
		for dx := -half; dx <= half; dx++ {
			v := grayValue(img, cx+dx, cy+dy)
			sum += v
			sumSq += v * v
		}
	}
	mean = sum / float64(n)
	variance := sumSq/float64(n) - mean*mean
	if variance < 0 {
		variance = 0
	}
	return mean, math.Sqrt(variance), true
}

// NormalizedCrossCorrelation returns the zero-mean normalized cross-correlation
// of the grayscale patches of half-size half centred at (ax, ay) in a and
// (bx, by) in b. The result lies in [-1, 1]; 1 is a perfect match. It returns 0
// (and false) if either patch falls outside its image or has zero variance.
func NormalizedCrossCorrelation(a, b *cv.Mat, ax, ay, bx, by, half int) (float64, bool) {
	ma, sa, oka := patchStats(a, ax, ay, half)
	mb, sb, okb := patchStats(b, bx, by, half)
	if !oka || !okb || sa == 0 || sb == 0 {
		return 0, false
	}
	n := (2*half + 1) * (2*half + 1)
	var acc float64
	for dy := -half; dy <= half; dy++ {
		for dx := -half; dx <= half; dx++ {
			va := grayValue(a, ax+dx, ay+dy) - ma
			vb := grayValue(b, bx+dx, by+dy) - mb
			acc += va * vb
		}
	}
	return acc / (float64(n) * sa * sb), true
}

// MatchCornersNCC matches corners between images a and b by normalized
// cross-correlation of local patches. For each corner in cornersA it searches
// every corner in cornersB within searchRadius pixels of the same location and
// keeps the best correlation above minCorrelation, using a patch of half-size
// half. A correspondence is accepted only if it is mutually the best match in
// both directions, which rejects most ambiguous pairings. Results are returned
// in the order of cornersA.
func MatchCornersNCC(a, b *cv.Mat, cornersA, cornersB []PointF, half, searchRadius int, minCorrelation float64) []Match {
	bestForA := make([]int, len(cornersA))
	scoreForA := make([]float64, len(cornersA))
	for i := range bestForA {
		bestForA[i] = -1
		scoreForA[i] = minCorrelation
	}
	bestForB := make([]int, len(cornersB))
	scoreForB := make([]float64, len(cornersB))
	for j := range bestForB {
		bestForB[j] = -1
		scoreForB[j] = minCorrelation
	}
	r2 := float64(searchRadius * searchRadius)
	for i, pa := range cornersA {
		for j, pb := range cornersB {
			dx := pa.X - pb.X
			dy := pa.Y - pb.Y
			if searchRadius > 0 && dx*dx+dy*dy > r2 {
				continue
			}
			s, ok := NormalizedCrossCorrelation(a, b, int(pa.X), int(pa.Y), int(pb.X), int(pb.Y), half)
			if !ok {
				continue
			}
			if s > scoreForA[i] {
				scoreForA[i] = s
				bestForA[i] = j
			}
			if s > scoreForB[j] {
				scoreForB[j] = s
				bestForB[j] = i
			}
		}
	}
	var out []Match
	for i := range cornersA {
		j := bestForA[i]
		if j >= 0 && bestForB[j] == i {
			out = append(out, Match{Src: cornersA[i], Dst: cornersB[j]})
		}
	}
	return out
}

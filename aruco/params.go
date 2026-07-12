package aruco

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// CornerRefinementMethod selects how [DetectMarkersWithParams] refines the
// detected marker corners.
type CornerRefinementMethod int

const (
	// CornerRefineNone leaves corners at their contour-derived integer pixels.
	CornerRefineNone CornerRefinementMethod = iota
	// CornerRefineSubpix refines each corner to subpixel accuracy with
	// [CornerSubPix].
	CornerRefineSubpix
)

// DetectorParameters tunes the marker-detection pipeline of
// [DetectMarkersWithParams]. It exposes the knobs that matter most in practice:
// the adaptive-threshold window sweep, the polygon-approximation accuracy, the
// candidate perimeter and corner-spacing bounds, and optional subpixel corner
// refinement. Obtain a sensible starting point from [DefaultDetectorParameters]
// and adjust individual fields.
type DetectorParameters struct {
	// AdaptiveThreshWinSizeMin, AdaptiveThreshWinSizeMax and
	// AdaptiveThreshWinSizeStep define the range of adaptive-threshold window
	// sizes (in pixels) that are tried; detections from every window are merged.
	// A wider range detects markers across a wider range of sizes and lighting at
	// the cost of speed. Sizes are forced odd and at least 3.
	AdaptiveThreshWinSizeMin  int
	AdaptiveThreshWinSizeMax  int
	AdaptiveThreshWinSizeStep int
	// AdaptiveThreshConstant is subtracted from the local mean when thresholding.
	AdaptiveThreshConstant float64
	// MinMarkerPerimeterRate and MaxMarkerPerimeterRate bound a candidate's
	// contour perimeter as a fraction of the larger image dimension. Candidates
	// outside the band are discarded, which rejects both specks and structures
	// far too large to be a marker.
	MinMarkerPerimeterRate float64
	MaxMarkerPerimeterRate float64
	// PolygonalApproxAccuracyRate is the polygon-approximation tolerance as a
	// fraction of the contour perimeter (the epsilon passed to Douglas-Peucker).
	// Smaller values demand straighter marker edges.
	PolygonalApproxAccuracyRate float64
	// MinCornerDistanceRate is the smallest allowed distance between adjacent
	// candidate corners, as a fraction of the contour perimeter.
	MinCornerDistanceRate float64
	// MinMarkerDistanceRate is the centre distance, relative to marker size, below
	// which two detections of the same id are treated as duplicates.
	MinMarkerDistanceRate float64
	// CornerRefinementMethod selects optional subpixel corner refinement.
	CornerRefinementMethod CornerRefinementMethod
	// CornerRefinementWinSize is the half-size, in pixels, of the subpixel
	// refinement window.
	CornerRefinementWinSize int
	// CornerRefinementMaxIterations bounds the subpixel refinement iterations.
	CornerRefinementMaxIterations int
	// CornerRefinementMinAccuracy stops subpixel refinement once a step moves the
	// corner by less than this many pixels.
	CornerRefinementMinAccuracy float64
}

// DefaultDetectorParameters returns detector parameters equivalent to the
// package's built-in [DetectMarkers] behaviour: a moderate adaptive-threshold
// window sweep, a 5% polygon-approximation tolerance, generous perimeter bounds
// and no corner refinement.
func DefaultDetectorParameters() DetectorParameters {
	return DetectorParameters{
		AdaptiveThreshWinSizeMin:      7,
		AdaptiveThreshWinSizeMax:      27,
		AdaptiveThreshWinSizeStep:     10,
		AdaptiveThreshConstant:        7,
		MinMarkerPerimeterRate:        0.03,
		MaxMarkerPerimeterRate:        4.0,
		PolygonalApproxAccuracyRate:   0.05,
		MinCornerDistanceRate:         0.05,
		MinMarkerDistanceRate:         0.05,
		CornerRefinementMethod:        CornerRefineNone,
		CornerRefinementWinSize:       5,
		CornerRefinementMaxIterations: 30,
		CornerRefinementMinAccuracy:   0.01,
	}
}

// DetectMarkersWithParams is a configurable variant of [DetectMarkers] driven by
// params. It sweeps the configured adaptive-threshold window sizes, applies the
// perimeter, polygon-accuracy and corner-spacing filters, reads and matches each
// candidate against dict, and merges detections across window sizes. When
// params.CornerRefinementMethod is [CornerRefineSubpix] the accepted corners are
// refined to subpixel accuracy. The returned corners and ids are parallel, as in
// [DetectMarkers]; both are nil when nothing is found.
func DetectMarkersWithParams(img *cv.Mat, dict *Dictionary, params DetectorParameters) (corners [][4]cv.Point, ids []int) {
	if img == nil || dict == nil || img.Empty() {
		return nil, nil
	}
	gray := toGray(img)
	maxDim := gray.Rows
	if gray.Cols > maxDim {
		maxDim = gray.Cols
	}
	minPeri := params.MinMarkerPerimeterRate * float64(maxDim)
	maxPeri := params.MaxMarkerPerimeterRate * float64(maxDim)

	step := params.AdaptiveThreshWinSizeStep
	if step < 1 {
		step = 1
	}
	winMax := params.AdaptiveThreshWinSizeMax
	if winMax < params.AdaptiveThreshWinSizeMin {
		winMax = params.AdaptiveThreshWinSizeMin
	}
	minDim := gray.Rows
	if gray.Cols < minDim {
		minDim = gray.Cols
	}

	for win := params.AdaptiveThreshWinSizeMin; win <= winMax; win += step {
		block := win
		if block < 3 {
			block = 3
		}
		if block%2 == 0 {
			block++
		}
		if block >= minDim {
			block = minDim - 1
			if block%2 == 0 {
				block--
			}
			if block < 3 {
				continue
			}
		}
		bin := cv.AdaptiveThreshold(gray, 255, cv.AdaptiveThreshMeanC, cv.ThreshBinaryInv, block, params.AdaptiveThreshConstant)
		contours, _ := cv.FindContours(bin, cv.RetrExternal, cv.ChainApproxNone)
		for _, c := range contours {
			quad, ok := candidateQuadParams(c, minPeri, maxPeri, params.PolygonalApproxAccuracyRate, params.MinCornerDistanceRate)
			if !ok {
				continue
			}
			read, ok := readMarkerGrid(gray, quad, dict.bitsPerSide)
			if !ok {
				continue
			}
			id, k, ok := matchGrid(dict, read)
			if !ok {
				continue
			}
			ordered := rotateCorners(quad, k)
			if isDuplicateRate(corners, ids, ordered, id, params.MinMarkerDistanceRate) {
				continue
			}
			corners = append(corners, ordered)
			ids = append(ids, id)
		}
	}

	if params.CornerRefinementMethod == CornerRefineSubpix && len(corners) > 0 {
		win := params.CornerRefinementWinSize
		if win < 1 {
			win = 1
		}
		for i := range corners {
			pts := make([][2]float64, 4)
			for j := 0; j < 4; j++ {
				pts[j] = [2]float64{float64(corners[i][j].X), float64(corners[i][j].Y)}
			}
			pts = CornerSubPix(gray, pts, win, params.CornerRefinementMaxIterations, params.CornerRefinementMinAccuracy)
			for j := 0; j < 4; j++ {
				corners[i][j] = cv.Point{X: int(math.Round(pts[j][0])), Y: int(math.Round(pts[j][1]))}
			}
		}
	}
	return corners, ids
}

// candidateQuadParams is candidateQuad with the thresholds supplied by
// DetectorParameters: a perimeter band, a polygon-approximation rate and a
// minimum corner-spacing rate.
func candidateQuadParams(c cv.Contour, minPeri, maxPeri, approxRate, minCornerRate float64) ([4]cv.Point, bool) {
	pts := []cv.Point(c)
	if len(pts) < 4 {
		return [4]cv.Point{}, false
	}
	peri := cv.ArcLength(pts, true)
	if peri < minPeri || peri > maxPeri {
		return [4]cv.Point{}, false
	}
	approx := cv.ApproxPolyDP(pts, approxRate*peri, true)
	if len(approx) != 4 || !isConvex(approx) {
		return [4]cv.Point{}, false
	}
	minCorner := minCornerRate * peri
	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		d := math.Hypot(float64(approx[j].X-approx[i].X), float64(approx[j].Y-approx[i].Y))
		if d < minCorner {
			return [4]cv.Point{}, false
		}
	}
	var quad [4]cv.Point
	copy(quad[:], approx)
	orientClockwise(&quad)
	return quad, true
}

// isDuplicateRate reports whether an accepted detection with the same id already
// lies within minRate*markerSize of the candidate's centre.
func isDuplicateRate(corners [][4]cv.Point, ids []int, cand [4]cv.Point, id int, minRate float64) bool {
	if minRate <= 0 {
		minRate = 0.05
	}
	ccx, ccy := quadCenter(cand)
	side := quadSide(cand)
	for i := range corners {
		if ids[i] != id {
			continue
		}
		ex, ey := quadCenter(corners[i])
		if math.Hypot(ccx-ex, ccy-ey) < math.Max(0.5*side, minRate*side) {
			return true
		}
	}
	return false
}

// CornerSubPix refines each of the given image corners to subpixel accuracy by
// the standard iterative method (OpenCV's cv::cornerSubPix): around each corner
// it finds the point q for which the image gradient at every nearby pixel is
// orthogonal to the vector from q to that pixel, which is exactly a corner or an
// edge saddle. winSize is the half-size in pixels of the square search window,
// maxIter bounds the iterations and epsilon stops refinement once a step moves
// the corner by fewer than epsilon pixels. image must be single-channel; corners
// are returned in the same order, unchanged where refinement is degenerate.
func CornerSubPix(image *cv.Mat, corners [][2]float64, winSize, maxIter int, epsilon float64) [][2]float64 {
	out := make([][2]float64, len(corners))
	copy(out, corners)
	if image == nil || image.Channels != 1 || winSize < 1 {
		return out
	}
	if maxIter < 1 {
		maxIter = 1
	}
	sigma2 := float64(winSize) * float64(winSize) / 4
	if sigma2 <= 0 {
		sigma2 = 1
	}
	for idx := range out {
		cx, cy := out[idx][0], out[idx][1]
		for iter := 0; iter < maxIter; iter++ {
			var a11, a12, a22, b1, b2 float64
			for dy := -winSize; dy <= winSize; dy++ {
				for dx := -winSize; dx <= winSize; dx++ {
					px := cx + float64(dx)
					py := cy + float64(dy)
					gx := (bilinear(image, px+1, py) - bilinear(image, px-1, py)) / 2
					gy := (bilinear(image, px, py+1) - bilinear(image, px, py-1)) / 2
					w := math.Exp(-(float64(dx*dx) + float64(dy*dy)) / (2 * sigma2))
					gxx := gx * gx * w
					gxy := gx * gy * w
					gyy := gy * gy * w
					a11 += gxx
					a12 += gxy
					a22 += gyy
					b1 += gxx*px + gxy*py
					b2 += gxy*px + gyy*py
				}
			}
			det := a11*a22 - a12*a12
			if math.Abs(det) < 1e-12 {
				break
			}
			nx := (a22*b1 - a12*b2) / det
			ny := (a11*b2 - a12*b1) / det
			move := math.Hypot(nx-cx, ny-cy)
			cx, cy = nx, ny
			if move < epsilon {
				break
			}
		}
		out[idx] = [2]float64{cx, cy}
	}
	return out
}

// bilinear samples the single-channel image at the floating-point position
// (x, y) with bilinear interpolation, clamping to the image edge.
func bilinear(m *cv.Mat, x, y float64) float64 {
	if x < 0 {
		x = 0
	} else if x > float64(m.Cols-1) {
		x = float64(m.Cols - 1)
	}
	if y < 0 {
		y = 0
	} else if y > float64(m.Rows-1) {
		y = float64(m.Rows - 1)
	}
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 > m.Cols-1 {
		x1 = m.Cols - 1
	}
	if y1 > m.Rows-1 {
		y1 = m.Rows - 1
	}
	fx := x - float64(x0)
	fy := y - float64(y0)
	v00 := float64(m.At(y0, x0, 0))
	v01 := float64(m.At(y0, x1, 0))
	v10 := float64(m.At(y1, x0, 0))
	v11 := float64(m.At(y1, x1, 0))
	return v00*(1-fx)*(1-fy) + v01*fx*(1-fy) + v10*(1-fx)*fy + v11*fx*fy
}

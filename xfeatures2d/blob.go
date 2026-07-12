package xfeatures2d

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// SimpleBlobDetector extracts blobs from an image and describes each as a
// [KeyPoint]. It is the port of OpenCV's cv::SimpleBlobDetector.
//
// The detector binarises the (grayscale) image at a sweep of thresholds from
// MinThreshold to MaxThreshold in steps of ThresholdStep. At each threshold it
// extracts connected components with cv.FindContours and, for every component,
// computes its area, centroid, circularity, convexity and inertia ratio. A
// component is kept only if it passes every enabled filter. Centers that recur
// across thresholds within MinDistBetweenBlobs are merged into a single blob;
// a blob is reported only if it appears in at least MinRepeatability thresholds.
//
// Whether dark or light blobs are sought is controlled by BlobColor: a value
// below 128 detects dark blobs on a lighter background (a pixel is foreground
// when it is darker than the current threshold), while a value of 128 or above
// detects light blobs. Build a detector with sensible defaults using
// [NewSimpleBlobDetector].
type SimpleBlobDetector struct {
	// MinThreshold, MaxThreshold and ThresholdStep define the inclusive
	// binarisation sweep. MinThreshold must be < MaxThreshold and ThresholdStep
	// positive.
	MinThreshold  float64
	MaxThreshold  float64
	ThresholdStep float64

	// MinRepeatability is the minimum number of thresholds at which a blob must
	// appear to be reported.
	MinRepeatability int
	// MinDistBetweenBlobs is the minimum separation, in pixels, between the
	// centers of distinct blobs.
	MinDistBetweenBlobs float64
	// BlobColor selects dark (<128) or light (>=128) blobs.
	BlobColor uint8

	// FilterByArea enables the [MinArea, MaxArea] area filter (in pixels²).
	FilterByArea bool
	MinArea      float64
	MaxArea      float64

	// FilterByCircularity enables the [MinCircularity, MaxCircularity] filter.
	// Circularity is 4·π·area / perimeter²; a perfect circle scores 1.
	FilterByCircularity bool
	MinCircularity      float64
	MaxCircularity      float64

	// FilterByConvexity enables the [MinConvexity, MaxConvexity] filter.
	// Convexity is area / convex-hull-area; a convex shape scores 1.
	FilterByConvexity bool
	MinConvexity      float64
	MaxConvexity      float64

	// FilterByInertia enables the [MinInertia, MaxInertia] filter on the ratio of
	// the smaller to the larger second central moment (elongation): a disc scores
	// 1 and a line segment scores 0.
	FilterByInertia bool
	MinInertia      float64
	MaxInertia      float64
}

// NewSimpleBlobDetector returns a SimpleBlobDetector configured with defaults
// close to OpenCV's: a threshold sweep of 50–220 in steps of 10, a minimum
// repeatability of 2, dark blobs, and area, convexity and inertia filtering
// enabled.
func NewSimpleBlobDetector() *SimpleBlobDetector {
	return &SimpleBlobDetector{
		MinThreshold:        50,
		MaxThreshold:        220,
		ThresholdStep:       10,
		MinRepeatability:    2,
		MinDistBetweenBlobs: 10,
		BlobColor:           0,

		FilterByArea: true,
		MinArea:      25,
		MaxArea:      5000,

		FilterByCircularity: false,
		MinCircularity:      0.8,
		MaxCircularity:      math.Inf(1),

		FilterByConvexity: true,
		MinConvexity:      0.95,
		MaxConvexity:      math.Inf(1),

		FilterByInertia: true,
		MinInertia:      0.1,
		MaxInertia:      math.Inf(1),
	}
}

// blobCenter is one candidate blob center found at a single threshold.
type blobCenter struct {
	x, y       float64
	radius     float64
	confidence float64
}

// Detect finds blobs in img and returns them as keypoints, sorted by descending
// response (repeatability). The keypoint Size is the blob diameter and Angle is
// -1. img may be single- or three-channel; a colour image is converted to gray.
func (d *SimpleBlobDetector) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	step := d.ThresholdStep
	if step <= 0 {
		step = 1
	}

	// Groups of centers that survive across thresholds. Each group is kept sorted
	// by ascending radius so its median radius is the middle element.
	var groups [][]blobCenter

	for t := d.MinThreshold; t < d.MaxThreshold; t += step {
		cur := d.findBlobs(gray, t)
		for _, c := range cur {
			isNew := true
			for gi := range groups {
				g := groups[gi]
				mid := g[len(g)/2]
				dist := math.Hypot(mid.x-c.x, mid.y-c.y)
				isNew = dist >= d.MinDistBetweenBlobs && dist >= mid.radius && dist >= c.radius
				if !isNew {
					// Insert c keeping the group sorted by radius.
					groups[gi] = append(g, c)
					k := len(groups[gi]) - 1
					for k > 0 && c.radius < groups[gi][k-1].radius {
						groups[gi][k] = groups[gi][k-1]
						k--
					}
					groups[gi][k] = c
					break
				}
			}
			if isNew {
				groups = append(groups, []blobCenter{c})
			}
		}
	}

	var kps []KeyPoint
	for _, g := range groups {
		if len(g) < d.MinRepeatability {
			continue
		}
		var sx, sy, norm float64
		for _, c := range g {
			sx += c.x * c.confidence
			sy += c.y * c.confidence
			norm += c.confidence
		}
		if norm <= 0 {
			continue
		}
		radius := g[len(g)/2].radius
		kps = append(kps, KeyPoint{
			Pt:       cv.Point{X: int(math.Round(sx / norm)), Y: int(math.Round(sy / norm))},
			Size:     radius * 2,
			Angle:    -1,
			Response: float64(len(g)),
		})
	}
	sort.SliceStable(kps, func(i, j int) bool { return kps[i].Response > kps[j].Response })
	return kps
}

// findBlobs binarises gray at threshold t, extracts external contours and
// returns the centers of the components that pass every enabled filter.
func (d *SimpleBlobDetector) findBlobs(gray *cv.Mat, t float64) []blobCenter {
	bin := cv.NewMat(gray.Rows, gray.Cols, 1)
	light := d.BlobColor >= 128
	for i, v := range gray.Data {
		var fg bool
		if light {
			fg = float64(v) > t
		} else {
			fg = float64(v) < t
		}
		if fg {
			bin.Data[i] = 255
		}
	}

	contours, _ := cv.FindContours(bin, cv.RetrExternal, cv.ChainApproxNone)
	var out []blobCenter
	for _, c := range contours {
		m := contourMoments(c)
		if m.area <= 0 {
			continue
		}
		if d.FilterByArea && (m.area < d.MinArea || m.area >= d.MaxArea) {
			continue
		}
		if d.FilterByCircularity {
			perim := cv.ArcLength([]cv.Point(c), true)
			if perim <= 0 {
				continue
			}
			circ := 4 * math.Pi * m.area / (perim * perim)
			if circ < d.MinCircularity || circ > d.MaxCircularity {
				continue
			}
		}
		if d.FilterByConvexity {
			hull := cv.ConvexHull([]cv.Point(c))
			hullArea := cv.ContourArea(cv.Contour(hull))
			if hullArea <= 0 {
				continue
			}
			conv := m.area / hullArea
			if conv < d.MinConvexity || conv > d.MaxConvexity {
				continue
			}
		}
		if d.FilterByInertia {
			ratio := m.inertiaRatio()
			if ratio < d.MinInertia || ratio > d.MaxInertia {
				continue
			}
		}

		// Radius: median distance from the centroid to the contour points.
		dists := make([]float64, len(c))
		for i, p := range c {
			dists[i] = math.Hypot(float64(p.X)-m.cx, float64(p.Y)-m.cy)
		}
		sort.Float64s(dists)
		radius := dists[len(dists)/2]
		out = append(out, blobCenter{x: m.cx, y: m.cy, radius: radius, confidence: 1})
	}
	return out
}

// moments holds the spatial and central second-order moments of a contour,
// computed from its polygon boundary via Green's theorem.
type moments struct {
	area             float64
	cx, cy           float64
	mu20, mu11, mu02 float64
}

// contourMoments computes the polygon moments of a contour. area is the (always
// non-negative) enclosed area, (cx, cy) is the centroid, and mu20/mu11/mu02 are
// the second central moments used to estimate elongation.
func contourMoments(c cv.Contour) moments {
	n := len(c)
	if n < 3 {
		var m moments
		if n > 0 {
			m.cx = float64(c[0].X)
			m.cy = float64(c[0].Y)
		}
		return m
	}
	var a00, a10, a01, a20, a11, a02 float64
	xPrev := float64(c[n-1].X)
	yPrev := float64(c[n-1].Y)
	for i := 0; i < n; i++ {
		xi := float64(c[i].X)
		yi := float64(c[i].Y)
		dxy := xPrev*yi - xi*yPrev
		xii := xPrev + xi
		yii := yPrev + yi
		a00 += dxy
		a10 += dxy * xii
		a01 += dxy * yii
		a20 += dxy * (xPrev*xii + xi*xi)
		a11 += dxy * (xPrev*(yii+yPrev) + xi*(yii+yi))
		a02 += dxy * (yPrev*yii + yi*yi)
		xPrev, yPrev = xi, yi
	}

	var m moments
	if a00 == 0 {
		return m
	}
	// Apply the Green's-theorem normalisers.
	m00 := a00 / 2
	m10 := a10 / 6
	m01 := a01 / 6
	m20 := a20 / 12
	m11 := a11 / 24
	m02 := a02 / 12

	// The border traversal may be clockwise, giving a negative signed area.
	// Negating every moment yields a positive area and positive central second
	// moments (so the inertia ratio comes out in [0,1]) while leaving the
	// centroid, which is a ratio, unchanged.
	if m00 < 0 {
		m00, m10, m01, m20, m11, m02 = -m00, -m10, -m01, -m20, -m11, -m02
	}

	invArea := 1 / m00
	cx := m10 * invArea
	cy := m01 * invArea
	m.area = math.Abs(m00)
	m.cx = cx
	m.cy = cy
	m.mu20 = m20 - cx*m10
	m.mu11 = m11 - cx*m01
	m.mu02 = m02 - cy*m01
	return m
}

// inertiaRatio returns the ratio of the smaller to the larger principal second
// moment (in [0,1]): 1 for an isotropic blob, approaching 0 for a line.
func (m moments) inertiaRatio() float64 {
	denom := math.Sqrt((m.mu20-m.mu02)*(m.mu20-m.mu02) + 4*m.mu11*m.mu11)
	if denom < 1e-2 {
		return 1
	}
	cosmin := (m.mu20 - m.mu02) / denom
	sinmin := 2 * m.mu11 / denom
	imin := 0.5*(m.mu20+m.mu02) - 0.5*(m.mu20-m.mu02)*cosmin - m.mu11*sinmin
	imax := 0.5*(m.mu20+m.mu02) + 0.5*(m.mu20-m.mu02)*cosmin + m.mu11*sinmin
	if imax <= 0 {
		return 1
	}
	ratio := imin / imax
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	return ratio
}

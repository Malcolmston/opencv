package features2d

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// FastFeatureDetector wraps the FAST-9 corner detector (cv.FASTCorners) in the
// OpenCV cv::FastFeatureDetector class form, producing [KeyPoint]s whose
// Response is the FAST cornerness score (the summed absolute contrast around the
// Bresenham circle). The zero value is usable and applies the defaults;
// construct a customised instance with [NewFastFeatureDetector].
type FastFeatureDetector struct {
	// Threshold is the intensity threshold on the centre-vs-ring difference.
	// Zero means the default (10).
	Threshold int
	// NonmaxSuppression enables 3×3 non-maximum suppression of the responses.
	NonmaxSuppression bool
}

// NewFastFeatureDetector returns a FAST detector with the given threshold and
// non-maximum suppression enabled.
func NewFastFeatureDetector(threshold int) *FastFeatureDetector {
	return &FastFeatureDetector{Threshold: threshold, NonmaxSuppression: true}
}

func (d *FastFeatureDetector) threshold() int {
	if d.Threshold > 0 {
		return d.Threshold
	}
	return 10
}

// Detect returns the FAST corners of img as keypoints of Size 7 (the diameter
// of the FAST detection circle) and Angle -1 (FAST assigns no orientation).
// Response is the FAST score, so the result can be ranked or thinned with
// [KeyPointsFilter].
func (d *FastFeatureDetector) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	pts := cv.FASTCorners(gray, d.threshold(), d.NonmaxSuppression)
	kps := make([]KeyPoint, len(pts))
	for i, p := range pts {
		kps[i] = KeyPoint{
			Pt:       p,
			Size:     7,
			Angle:    -1,
			Response: fastScore(gray, p.X, p.Y, d.threshold()),
			Octave:   0,
		}
	}
	return kps
}

// fastCircle16 holds the radius-3 Bresenham circle offsets used by AGAST/FAST,
// clockwise from the top.
var fastCircle16 = [16][2]int{
	{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
	{0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3},
}

// fastScore returns the summed absolute contrast between the centre pixel and
// the 16 circle samples, matching the score cv.FASTCorners uses internally for
// non-maximum suppression.
func fastScore(gray *cv.Mat, x, y, _ int) float64 {
	if x < 3 || y < 3 || x >= gray.Cols-3 || y >= gray.Rows-3 {
		return 0
	}
	p := int(gray.Data[y*gray.Cols+x])
	var s float64
	for k := 0; k < 16; k++ {
		v := int(gray.Data[(y+fastCircle16[k][1])*gray.Cols+(x+fastCircle16[k][0])])
		s += math.Abs(float64(v - p))
	}
	return s
}

// AgastFeatureDetector implements the AGAST corner detector using the OAST 9_16
// mask (an arc of at least 9 contiguous circle pixels all brighter than centre+t
// or all darker than centre-t, the same criterion as FAST-9). Its distinguishing
// feature versus FAST is the response: AGAST scores each corner by the largest
// threshold for which the pixel still qualifies as a corner, found by binary
// search, which yields a more stable ranking.
//
// This is a genuine AGAST *detector*; it differs from OpenCV's implementation
// only in that OpenCV uses a precompiled ternary decision tree (the "AGAST"
// machine) to evaluate the mask, whereas this reads the 16 samples directly. The
// set of detected corners is the same; only the internal evaluation order
// differs. The zero value is usable; construct one with [NewAgastFeatureDetector].
type AgastFeatureDetector struct {
	// Threshold is the base corner threshold. Zero means the default (10).
	Threshold int
	// NonmaxSuppression enables 3×3 non-maximum suppression on the AGAST score.
	NonmaxSuppression bool
}

// NewAgastFeatureDetector returns an AGAST detector with the given threshold and
// non-maximum suppression enabled.
func NewAgastFeatureDetector(threshold int) *AgastFeatureDetector {
	return &AgastFeatureDetector{Threshold: threshold, NonmaxSuppression: true}
}

func (d *AgastFeatureDetector) threshold() int {
	if d.Threshold > 0 {
		return d.Threshold
	}
	return 10
}

// isAgastCorner reports whether pixel (x, y) is an OAST 9_16 corner at the given
// threshold.
func isAgastCorner(gray *cv.Mat, x, y, t int) bool {
	p := int(gray.Data[y*gray.Cols+x])
	hi := p + t
	lo := p - t
	var vals [16]int
	brighter, darker := 0, 0
	for k := 0; k < 16; k++ {
		v := int(gray.Data[(y+fastCircle16[k][1])*gray.Cols+(x+fastCircle16[k][0])])
		vals[k] = v
		if v > hi {
			brighter++
		} else if v < lo {
			darker++
		}
	}
	if brighter < 9 && darker < 9 {
		return false
	}
	// Contiguous arc of length >= 9, wrapping the ring.
	for _, bright := range [2]bool{true, false} {
		run := 0
		for k := 0; k < 24; k++ {
			v := vals[k%16]
			ok := v > hi
			if !bright {
				ok = v < lo
			}
			if ok {
				run++
				if run >= 9 {
					return true
				}
			} else {
				run = 0
			}
		}
	}
	return false
}

// agastScore returns the largest threshold (>= base) for which (x, y) is still a
// corner. This is the AGAST cornerness measure.
func agastScore(gray *cv.Mat, x, y, base int) float64 {
	if !isAgastCorner(gray, x, y, base) {
		return 0
	}
	lo, hi := base, base
	// Exponentially grow an upper bound that is not a corner.
	for isAgastCorner(gray, x, y, hi) && hi < 255 {
		lo = hi
		if hi == 0 {
			hi = 1
		} else {
			hi *= 2
		}
		if hi > 255 {
			hi = 255
			break
		}
	}
	// Binary search the boundary in (lo, hi].
	for lo+1 < hi {
		mid := (lo + hi) / 2
		if isAgastCorner(gray, x, y, mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	return float64(lo)
}

// Detect returns the AGAST corners of img. Keypoints have Size 7, Angle -1 and
// Response equal to the AGAST score (the maximal qualifying threshold).
func (d *AgastFeatureDetector) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	t := d.threshold()
	rows, cols := gray.Rows, gray.Cols
	scores := make([]float64, rows*cols)
	var raw []cv.Point
	for y := 3; y < rows-3; y++ {
		for x := 3; x < cols-3; x++ {
			if !isAgastCorner(gray, x, y, t) {
				continue
			}
			scores[y*cols+x] = agastScore(gray, x, y, t)
			raw = append(raw, cv.Point{X: x, Y: y})
		}
	}
	kps := make([]KeyPoint, 0, len(raw))
	for _, p := range raw {
		if d.NonmaxSuppression {
			s := scores[p.Y*cols+p.X]
			suppressed := false
			for dy := -1; dy <= 1 && !suppressed; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					if scores[(p.Y+dy)*cols+(p.X+dx)] > s {
						suppressed = true
						break
					}
				}
			}
			if suppressed {
				continue
			}
		}
		kps = append(kps, KeyPoint{Pt: p, Size: 7, Angle: -1, Response: scores[p.Y*cols+p.X]})
	}
	return kps
}

// GFTTDetector wraps the Shi–Tomasi "good features to track" corner detector
// (cv.GoodFeaturesToTrack) in OpenCV's cv::GFTTDetector class form. The zero
// value is usable and applies the defaults; construct one with [NewGFTTDetector].
type GFTTDetector struct {
	// MaxCorners is the maximum number of corners to return. Zero means the
	// default (1000); a negative value returns all.
	MaxCorners int
	// QualityLevel is the minimum accepted corner strength as a fraction of the
	// strongest corner. Zero means the default (0.01).
	QualityLevel float64
	// MinDistance is the minimum spacing in pixels between returned corners.
	// Zero means the default (1).
	MinDistance float64
	// BlockSize is the structure-tensor window size. Zero means the default (3).
	BlockSize int
}

// NewGFTTDetector returns a GFTT detector retaining up to maxCorners corners
// with the default quality level, spacing and block size.
func NewGFTTDetector(maxCorners int) *GFTTDetector {
	return &GFTTDetector{MaxCorners: maxCorners}
}

// Detect returns the strong Shi–Tomasi corners of img as keypoints of Size
// equal to the block size and Angle -1. Response is left 0 (OpenCV also reports
// 0 unless useHarrisDetector is set); rank instead by detection order, which is
// descending corner strength.
func (d *GFTTDetector) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	maxCorners := d.MaxCorners
	if maxCorners == 0 {
		maxCorners = 1000
	}
	quality := d.QualityLevel
	if quality <= 0 {
		quality = 0.01
	}
	minDist := d.MinDistance
	if minDist <= 0 {
		minDist = 1
	}
	block := d.BlockSize
	if block <= 0 {
		block = 3
	}
	pts := cv.GoodFeaturesToTrack(gray, maxCorners, quality, minDist, block)
	kps := make([]KeyPoint, len(pts))
	for i, p := range pts {
		kps[i] = KeyPoint{Pt: p, Size: float64(block), Angle: -1, Octave: 0}
	}
	return kps
}

// SimpleBlobDetector detects blobs — connected regions that are darker or
// brighter than their surroundings — following the algorithm of OpenCV's
// cv::SimpleBlobDetector. The image is thresholded at a series of levels; in
// each binary image the connected components (via cv.ConnectedComponents) whose
// area lies in [MinArea, MaxArea] contribute a candidate centre; candidates that
// recur at nearby locations across thresholds are merged into one blob whose
// Size reflects the mean component diameter and Response the number of
// supporting thresholds. The zero value is usable; construct one with
// [NewSimpleBlobDetector].
type SimpleBlobDetector struct {
	// MinThreshold, MaxThreshold and ThresholdStep define the threshold sweep.
	// Zero values default to 50, 220 and 10 respectively.
	MinThreshold  float64
	MaxThreshold  float64
	ThresholdStep float64
	// MinArea and MaxArea bound the accepted blob area in pixels. Zero defaults
	// to 25 and 5000.
	MinArea float64
	MaxArea float64
	// MinRepeatability is the minimum number of thresholds a blob must appear in.
	// Zero means the default (2).
	MinRepeatability int
	// MinDistBetweenBlobs is the merge radius for centres across thresholds.
	// Zero means the default (10).
	MinDistBetweenBlobs float64
	// DetectDark selects dark-on-light blobs (true, the default behaviour of the
	// zero value) versus bright-on-dark blobs (false).
	DetectDark bool
}

// NewSimpleBlobDetector returns a blob detector configured for dark blobs on a
// light background with default thresholds and area bounds.
func NewSimpleBlobDetector() *SimpleBlobDetector {
	return &SimpleBlobDetector{DetectDark: true}
}

func (d *SimpleBlobDetector) params() (minT, maxT, step, minA, maxA, minDist float64, minRep int) {
	minT, maxT, step = d.MinThreshold, d.MaxThreshold, d.ThresholdStep
	if minT == 0 {
		minT = 50
	}
	if maxT == 0 {
		maxT = 220
	}
	if step == 0 {
		step = 10
	}
	minA, maxA = d.MinArea, d.MaxArea
	if minA == 0 {
		minA = 25
	}
	if maxA == 0 {
		maxA = 5000
	}
	minDist = d.MinDistBetweenBlobs
	if minDist == 0 {
		minDist = 10
	}
	minRep = d.MinRepeatability
	if minRep == 0 {
		minRep = 2
	}
	return
}

// center is one blob candidate at a single threshold.
type blobCenter struct {
	x, y   float64
	radius float64
}

// Detect returns the detected blobs as keypoints, with Pt at the blob centroid,
// Size the mean blob diameter and Response the number of thresholds that
// supported the blob.
func (d *SimpleBlobDetector) Detect(img *cv.Mat) []KeyPoint {
	gray := toGray(img)
	minT, maxT, step, minA, maxA, minDist, minRep := d.params()

	// Collect candidate centres per threshold, then agglomerate across
	// thresholds into stable blobs (OpenCV's approach).
	var blobs [][]blobCenter
	for thr := minT; thr <= maxT; thr += step {
		var binary *cv.Mat
		if d.DetectDark {
			// Dark blobs: pixels below threshold become foreground.
			b, _ := cv.Threshold(gray, thr, 255, cv.ThreshBinaryInv)
			binary = b
		} else {
			b, _ := cv.Threshold(gray, thr, 255, cv.ThreshBinary)
			binary = b
		}
		centers := blobCentersFromBinary(binary, minA, maxA)
		for _, c := range centers {
			matched := false
			for i := range blobs {
				last := blobs[i][len(blobs[i])-1]
				dx, dy := c.x-last.x, c.y-last.y
				if math.Hypot(dx, dy) < minDist {
					blobs[i] = append(blobs[i], c)
					matched = true
					break
				}
			}
			if !matched {
				blobs = append(blobs, []blobCenter{c})
			}
		}
	}

	var kps []KeyPoint
	for _, group := range blobs {
		if len(group) < minRep {
			continue
		}
		var sx, sy, sr float64
		for _, c := range group {
			sx += c.x
			sy += c.y
			sr += c.radius
		}
		n := float64(len(group))
		kps = append(kps, KeyPoint{
			Pt:       cv.Point{X: int(math.Round(sx / n)), Y: int(math.Round(sy / n))},
			Size:     2 * sr / n,
			Angle:    -1,
			Response: n,
		})
	}
	// Deterministic order: by position.
	sort.SliceStable(kps, func(i, j int) bool {
		if kps[i].Pt.Y != kps[j].Pt.Y {
			return kps[i].Pt.Y < kps[j].Pt.Y
		}
		return kps[i].Pt.X < kps[j].Pt.X
	})
	return kps
}

// blobCentersFromBinary labels connected foreground components and returns the
// centroid and equivalent-circle radius of each component whose area is in
// [minArea, maxArea].
func blobCentersFromBinary(binary *cv.Mat, minArea, maxArea float64) []blobCenter {
	_, count, stats := cv.ConnectedComponentsWithStats(binary, cv.Connectivity8)
	var out []blobCenter
	for l := 1; l < count; l++ {
		st := stats[l]
		a := float64(st.Area)
		if a < minArea || a > maxArea {
			continue
		}
		out = append(out, blobCenter{
			x:      st.CentroidX,
			y:      st.CentroidY,
			radius: math.Sqrt(a / math.Pi),
		})
	}
	return out
}

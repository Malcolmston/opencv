package features3

import (
	"math"
	"math/bits"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// KeyPoint is a salient image point produced by a detector in this package. It
// mirrors OpenCV's cv::KeyPoint but stores its location as a sub-pixel
// cv.Point2f so that refined corners and scale-space blobs share one type.
type KeyPoint struct {
	// Pt is the keypoint location in pixel coordinates (X is the column, Y the
	// row); it may be fractional for refined or interpolated points.
	Pt cv.Point2f
	// Size is the diameter of the meaningful keypoint neighbourhood in pixels.
	Size float64
	// Angle is the keypoint orientation in degrees in the range [0, 360); a
	// value of -1 means no orientation was computed.
	Angle float64
	// Response is the detector response used to rank keypoints; larger is
	// stronger.
	Response float64
	// Octave is the scale-space layer the keypoint was detected in; single-scale
	// detectors leave it 0.
	Octave int
}

// NewKeyPoint returns a KeyPoint at the given column/row with the supplied
// response, a size of 7, an unset orientation (-1) and octave 0.
func NewKeyPoint(x, y, response float64) KeyPoint {
	return KeyPoint{Pt: cv.Point2f{X: x, Y: y}, Size: 7, Angle: -1, Response: response}
}

// Point returns the keypoint location rounded to the nearest integer cv.Point.
func (k KeyPoint) Point() cv.Point {
	return cv.Point{X: int(math.Round(k.Pt.X)), Y: int(math.Round(k.Pt.Y))}
}

// DistanceTo returns the Euclidean distance in pixels between two keypoints.
func (k KeyPoint) DistanceTo(other KeyPoint) float64 {
	dx := k.Pt.X - other.Pt.X
	dy := k.Pt.Y - other.Pt.Y
	return math.Hypot(dx, dy)
}

// Match records a correspondence between a query descriptor and a train
// descriptor, mirroring OpenCV's cv::DMatch. Distance is the Hamming distance
// for the binary descriptors in this package.
type Match struct {
	// QueryIdx is the index of the descriptor in the query set.
	QueryIdx int
	// TrainIdx is the index of the matched descriptor in the train set.
	TrainIdx int
	// Distance is the descriptor distance; smaller means more similar.
	Distance int
}

// Blob is a scale-space interest region: a location and the Gaussian scale
// (sigma) at which it responded most strongly, as returned by [LoGBlobs],
// [DoGBlobs] and [DoHBlobs].
type Blob struct {
	// X and Y are the blob centre in pixel coordinates.
	X, Y float64
	// Sigma is the Gaussian standard deviation of the detecting scale.
	Sigma float64
	// Response is the (absolute) operator response at the detection.
	Response float64
}

// Radius returns the characteristic blob radius sigma*sqrt(2), the radius of the
// bright/dark region a Laplacian-of-Gaussian of this scale responds to.
func (b Blob) Radius() float64 {
	return b.Sigma * math.Sqrt2
}

// Area returns the area of the blob disc, pi*Radius^2.
func (b Blob) Area() float64 {
	r := b.Radius()
	return math.Pi * r * r
}

// ToKeyPoint converts the blob to a KeyPoint whose Size is the blob diameter
// (2*Radius) and whose Response carries the operator response.
func (b Blob) ToKeyPoint() KeyPoint {
	return KeyPoint{
		Pt:       cv.Point2f{X: b.X, Y: b.Y},
		Size:     2 * b.Radius(),
		Angle:    -1,
		Response: b.Response,
	}
}

// HammingDistance returns the number of differing bits between two equal-length
// byte slices (their Hamming distance). It panics if the lengths differ.
func HammingDistance(a, b []byte) int {
	if len(a) != len(b) {
		panic("features3: HammingDistance requires equal-length slices")
	}
	var d int
	for i := range a {
		d += bits.OnesCount8(a[i] ^ b[i])
	}
	return d
}

// HammingDistanceUint64 returns the Hamming distance between two 64-bit words.
func HammingDistanceUint64(a, b uint64) int {
	return bits.OnesCount64(a ^ b)
}

// BitCount returns the number of set bits (population count) of x.
func BitCount(x uint64) int {
	return bits.OnesCount64(x)
}

// SortKeyPointsByResponse sorts keypoints in place in descending order of
// Response, breaking ties by position for a deterministic ordering.
func SortKeyPointsByResponse(kps []KeyPoint) {
	sort.SliceStable(kps, func(i, j int) bool {
		if kps[i].Response != kps[j].Response {
			return kps[i].Response > kps[j].Response
		}
		if kps[i].Pt.Y != kps[j].Pt.Y {
			return kps[i].Pt.Y < kps[j].Pt.Y
		}
		return kps[i].Pt.X < kps[j].Pt.X
	})
}

// RetainBest returns the n strongest keypoints by Response. When n <= 0 or n >=
// len(kps) a sorted copy of every keypoint is returned. The input slice is not
// modified.
func RetainBest(kps []KeyPoint, n int) []KeyPoint {
	out := make([]KeyPoint, len(kps))
	copy(out, kps)
	SortKeyPointsByResponse(out)
	if n > 0 && n < len(out) {
		out = out[:n]
	}
	return out
}

// FilterByResponse returns the keypoints whose Response is at least minResponse,
// preserving their order. The input slice is not modified.
func FilterByResponse(kps []KeyPoint, minResponse float64) []KeyPoint {
	var out []KeyPoint
	for _, k := range kps {
		if k.Response >= minResponse {
			out = append(out, k)
		}
	}
	return out
}

// FilterByBorder returns the keypoints lying at least border pixels away from
// every edge of a rows×cols image, preserving their order.
func FilterByBorder(kps []KeyPoint, rows, cols, border int) []KeyPoint {
	var out []KeyPoint
	for _, k := range kps {
		x, y := k.Pt.X, k.Pt.Y
		if x >= float64(border) && x < float64(cols-border) &&
			y >= float64(border) && y < float64(rows-border) {
			out = append(out, k)
		}
	}
	return out
}

// ConvertKeyPointsToPoints returns the rounded integer locations of the given
// keypoints.
func ConvertKeyPointsToPoints(kps []KeyPoint) []cv.Point {
	out := make([]cv.Point, len(kps))
	for i, k := range kps {
		out[i] = k.Point()
	}
	return out
}

// ConvertPointsToKeyPoints wraps integer points as keypoints with the given
// size, zero response and unset orientation.
func ConvertPointsToKeyPoints(pts []cv.Point, size float64) []KeyPoint {
	out := make([]KeyPoint, len(pts))
	for i, p := range pts {
		out[i] = KeyPoint{Pt: cv.Point2f{X: float64(p.X), Y: float64(p.Y)}, Size: size, Angle: -1}
	}
	return out
}

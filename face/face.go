package face

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FaceRecognizer is the common interface implemented by every recognizer in
// this package: [EigenFaceRecognizer], [FisherFaceRecognizer] and
// [LBPHFaceRecognizer]. It mirrors the shape of OpenCV's
// cv::face::FaceRecognizer.
//
// Train fits the model to a set of labelled face images. Every image is
// reduced to single-channel luma; the holistic recognizers (Eigen, Fisher)
// additionally resample each image to a common geometry taken from the first
// training image. labels are arbitrary non-negative integers identifying the
// subject of each image; the same label may (and should) appear more than once.
//
// Predict classifies a query face and returns the predicted label together with
// a confidence score. Following OpenCV's convention the confidence is a
// distance in the recognizer's feature space, so smaller is better and a
// perfect match scores 0. The unit differs per recognizer (Euclidean distance
// in the eigen/fisher subspace, chi-square histogram distance for LBPH).
//
// Both methods panic on malformed input (no images, a label/image count
// mismatch, a nil or empty image, or Predict before Train) rather than
// returning an error, matching the panic-on-programmer-error style of the root
// package's pixel accessors.
type FaceRecognizer interface {
	// Train fits the recognizer to the labelled images.
	Train(images []*cv.Mat, labels []int)
	// Predict returns the best-matching label and its distance-based
	// confidence (lower is more confident).
	Predict(img *cv.Mat) (label int, confidence float64)
}

// Compile-time assertions that every recognizer satisfies FaceRecognizer.
var (
	_ FaceRecognizer = (*EigenFaceRecognizer)(nil)
	_ FaceRecognizer = (*FisherFaceRecognizer)(nil)
	_ FaceRecognizer = (*LBPHFaceRecognizer)(nil)
)

// toGrayMat returns a single-channel view of img. A one-channel Mat is returned
// unchanged (no copy); any other channel count is reduced to BT.601 luma
// (0.299R + 0.587G + 0.114B) from its first three channels, matching the root
// package's RGB convention. A two-channel Mat uses its first channel. It panics
// on a nil or empty image.
func toGrayMat(img *cv.Mat) *cv.Mat {
	if img == nil || img.Empty() {
		panic("face: nil or empty image")
	}
	if img.Channels == 1 {
		return img
	}
	ch := img.Channels
	g := cv.NewMat(img.Rows, img.Cols, 1)
	n := img.Total()
	for i := 0; i < n; i++ {
		base := i * ch
		if ch >= 3 {
			r := float64(img.Data[base])
			gg := float64(img.Data[base+1])
			b := float64(img.Data[base+2])
			g.Data[i] = uint8(0.299*r + 0.587*gg + 0.114*b + 0.5)
		} else {
			g.Data[i] = img.Data[base]
		}
	}
	return g
}

// imageVector reduces img to luma, resamples it to rows×cols (bilinearly, via
// the root [cv.Resize]) when its size differs, and flattens it row-major into a
// float64 vector of length rows*cols. This is the shared front end of the
// holistic (Eigen/Fisher) recognizers.
func imageVector(img *cv.Mat, rows, cols int) []float64 {
	g := toGrayMat(img)
	if g.Rows != rows || g.Cols != cols {
		g = cv.Resize(g, cols, rows, cv.InterLinear)
	}
	v := make([]float64, rows*cols)
	for i := range v {
		v[i] = float64(g.Data[i])
	}
	return v
}

// validateTraining performs the common Train argument checks and panics on any
// violation.
func validateTraining(images []*cv.Mat, labels []int) {
	if len(images) == 0 {
		panic("face: Train requires at least one image")
	}
	if len(images) != len(labels) {
		panic("face: Train image and label counts differ")
	}
	for i, im := range images {
		if im == nil || im.Empty() {
			panic("face: Train given a nil or empty image at index " + itoa(i))
		}
	}
}

// itoa is a tiny, allocation-light integer formatter used only in panic
// messages, avoiding a strconv import for this one call site.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// euclidean returns the Euclidean distance between two equal-length vectors.
func euclidean(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}

// nearestNeighbor returns the index of the projection in db closest to query
// under the supplied distance function, together with that distance. It panics
// if db is empty.
func nearestNeighbor(db [][]float64, query []float64, dist func(a, b []float64) float64) (int, float64) {
	if len(db) == 0 {
		panic("face: no trained samples")
	}
	best := 0
	bestDist := dist(db[0], query)
	for i := 1; i < len(db); i++ {
		d := dist(db[i], query)
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best, bestDist
}

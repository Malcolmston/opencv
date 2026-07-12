package face

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// This file implements a landmark localiser in the spirit of OpenCV's
// cv::face::FacemarkLBF: given a face bounding box it places a set of facial
// landmarks. Rather than the local-binary-feature random forests of the paper,
// it uses the closely related Supervised Descent Method (Xiong & De la Torre,
// 2013): starting from a learned mean shape it applies a cascade of linear
// regressors, each predicting a shape update from shape-indexed local image
// features. The result is a genuine, trainable cascade — not a stub — that
// converges from the mean shape toward the true landmarks, and it is fully
// deterministic (no randomness in training or fitting).

// FacemarkLBF is a trainable facial-landmark localiser. Construct one with
// [NewFacemarkLBF]; the zero value is not usable. After [FacemarkLBF.Train] it
// places numLandmarks landmarks inside any supplied face rectangle via
// [FacemarkLBF.Fit].
type FacemarkLBF struct {
	numLandmarks int
	stages       int
	radius       int     // patch half-extent (grid is (2r+1)×(2r+1))
	lambda       float64 // ridge regularisation

	meanShape []float64     // normalised, length 2*numLandmarks (x,y interleaved)
	regs      [][][]float64 // per stage: feature×(2L) weight matrix
	featLen   int
	trained   bool
}

// NewFacemarkLBF returns an untrained localiser for numLandmarks landmarks. It
// uses sensible defaults: a five-stage cascade and a 5×5 shape-indexed sampling
// patch per landmark. Tune these with [NewFacemarkLBFParams]. It panics if
// numLandmarks is not positive.
func NewFacemarkLBF(numLandmarks int) *FacemarkLBF {
	return NewFacemarkLBFParams(numLandmarks, 5, 2)
}

// NewFacemarkLBFParams returns an untrained localiser with an explicit number of
// cascade stages and patch radius (the sampling grid is (2*radius+1) square). It
// panics on non-positive parameters.
func NewFacemarkLBFParams(numLandmarks, stages, radius int) *FacemarkLBF {
	if numLandmarks < 1 {
		panic("face: NewFacemarkLBF requires at least one landmark")
	}
	if stages < 1 || radius < 1 {
		panic("face: NewFacemarkLBF requires positive stages and radius")
	}
	return &FacemarkLBF{
		numLandmarks: numLandmarks,
		stages:       stages,
		radius:       radius,
		lambda:       1.0,
	}
}

// NumLandmarks returns the number of landmarks the localiser predicts.
func (f *FacemarkLBF) NumLandmarks() int { return f.numLandmarks }

// Train fits the cascade to labelled faces. images[i] carries a face inside
// rects[i], and shapes[i] gives that face's numLandmarks ground-truth landmark
// points in image coordinates. Landmarks are stored relative to their face
// rectangle so the model generalises across face positions and sizes. It panics
// on mismatched slice lengths, an empty set, or a shape with the wrong landmark
// count.
func (f *FacemarkLBF) Train(images []*cv.Mat, rects []cv.Rect, shapes [][]cv.Point) {
	if len(images) == 0 {
		panic("face: FacemarkLBF.Train requires at least one image")
	}
	if len(images) != len(rects) || len(images) != len(shapes) {
		panic("face: FacemarkLBF.Train image, rect and shape counts differ")
	}
	n := len(images)
	L := f.numLandmarks
	f.featLen = L*(2*f.radius+1)*(2*f.radius+1) + 1 // +1 bias term

	grays := make([]*cv.Mat, n)
	targets := make([][]float64, n) // normalised ground-truth shapes
	for i := range images {
		if len(shapes[i]) != L {
			panic("face: FacemarkLBF.Train shape has the wrong landmark count")
		}
		grays[i] = toGrayMat(images[i])
		targets[i] = normalizeShape(shapes[i], rects[i])
	}

	// Mean shape: average of the normalised ground truths.
	f.meanShape = make([]float64, 2*L)
	for _, t := range targets {
		for j := range t {
			f.meanShape[j] += t[j]
		}
	}
	for j := range f.meanShape {
		f.meanShape[j] /= float64(n)
	}

	// Initialise every sample's current estimate at the mean shape.
	current := make([][]float64, n)
	for i := range current {
		current[i] = append([]float64(nil), f.meanShape...)
	}

	f.regs = make([][][]float64, f.stages)
	for s := 0; s < f.stages; s++ {
		X := make([][]float64, n) // features
		Y := make([][]float64, n) // target deltas (target − current)
		for i := 0; i < n; i++ {
			X[i] = f.features(grays[i], rects[i], current[i])
			d := make([]float64, 2*L)
			for j := 0; j < 2*L; j++ {
				d[j] = targets[i][j] - current[i][j]
			}
			Y[i] = d
		}
		W := ridgeSolve(X, Y, f.lambda)
		f.regs[s] = W
		// Apply the freshly learned stage to advance every current estimate.
		for i := 0; i < n; i++ {
			applyUpdate(current[i], X[i], W)
		}
	}
	f.trained = true
}

// Fit places the landmarks in img inside the face rectangle rect and returns
// them as image-coordinate points. It starts from the mean shape and runs the
// trained cascade. It panics if the localiser is untrained.
func (f *FacemarkLBF) Fit(img *cv.Mat, rect cv.Rect) []cv.Point {
	if !f.trained {
		panic("face: FacemarkLBF.Fit before Train")
	}
	g := toGrayMat(img)
	shape := append([]float64(nil), f.meanShape...)
	for s := 0; s < f.stages; s++ {
		x := f.features(g, rect, shape)
		applyUpdate(shape, x, f.regs[s])
	}
	return denormalizeShape(shape, rect)
}

// MeanShapeAt returns the learned mean shape placed inside rect, as image-
// coordinate points. This is the localiser's starting guess before regression
// and a useful baseline. It panics if the localiser is untrained.
func (f *FacemarkLBF) MeanShapeAt(rect cv.Rect) []cv.Point {
	if !f.trained {
		panic("face: FacemarkLBF.MeanShapeAt before Train")
	}
	return denormalizeShape(f.meanShape, rect)
}

// features extracts the shape-indexed feature vector for one image at the given
// (normalised) shape: around each landmark it samples a (2r+1)² grid of pixels
// spaced proportionally to the face size, contrast-normalises each landmark's
// patch (zero mean, unit standard deviation) for illumination tolerance, and
// concatenates them, appending a constant bias term.
func (f *FacemarkLBF) features(g *cv.Mat, rect cv.Rect, shape []float64) []float64 {
	L := f.numLandmarks
	r := f.radius
	side := 2*r + 1
	feat := make([]float64, f.featLen)
	stepX := rect.Width / 16
	stepY := rect.Height / 16
	if stepX < 1 {
		stepX = 1
	}
	if stepY < 1 {
		stepY = 1
	}
	pos := 0
	patch := make([]float64, side*side)
	for l := 0; l < L; l++ {
		cx := float64(rect.X) + shape[2*l]*float64(rect.Width)
		cy := float64(rect.Y) + shape[2*l+1]*float64(rect.Height)
		k := 0
		var mean float64
		for dy := -r; dy <= r; dy++ {
			for dx := -r; dx <= r; dx++ {
				px := int(cx+0.5) + dx*stepX
				py := int(cy+0.5) + dy*stepY
				v := float64(replicateAt(g, py, px))
				patch[k] = v
				mean += v
				k++
			}
		}
		mean /= float64(len(patch))
		var sd float64
		for _, v := range patch {
			d := v - mean
			sd += d * d
		}
		sd = math.Sqrt(sd/float64(len(patch))) + 1e-6
		for _, v := range patch {
			feat[pos] = (v - mean) / sd
			pos++
		}
	}
	feat[pos] = 1 // bias
	return feat
}

// applyUpdate adds the regressor's predicted delta (features · W) to shape.
func applyUpdate(shape, feat []float64, W [][]float64) {
	out := len(shape)
	for k := 0; k < out; k++ {
		var s float64
		for j := range feat {
			s += feat[j] * W[j][k]
		}
		shape[k] += s
	}
}

// normalizeShape maps image-coordinate landmarks into rect-relative coordinates
// in roughly [0,1], interleaved x,y.
func normalizeShape(pts []cv.Point, rect cv.Rect) []float64 {
	out := make([]float64, 2*len(pts))
	w := float64(rect.Width)
	h := float64(rect.Height)
	if w == 0 {
		w = 1
	}
	if h == 0 {
		h = 1
	}
	for i, p := range pts {
		out[2*i] = (float64(p.X) - float64(rect.X)) / w
		out[2*i+1] = (float64(p.Y) - float64(rect.Y)) / h
	}
	return out
}

// denormalizeShape is the inverse of normalizeShape, rounding to integer pixels.
func denormalizeShape(shape []float64, rect cv.Rect) []cv.Point {
	pts := make([]cv.Point, len(shape)/2)
	for i := range pts {
		x := float64(rect.X) + shape[2*i]*float64(rect.Width)
		y := float64(rect.Y) + shape[2*i+1]*float64(rect.Height)
		pts[i] = cv.Point{X: int(math.Round(x)), Y: int(math.Round(y))}
	}
	return pts
}

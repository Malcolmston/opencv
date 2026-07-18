package shapefit

import cv "github.com/malcolmston/opencv"

// PrimitiveKind identifies which geometric primitive a [Primitive] holds.
type PrimitiveKind int

const (
	// PrimitiveNone indicates no primitive could be fit.
	PrimitiveNone PrimitiveKind = iota
	// PrimitiveLine indicates the primitive is a [Line].
	PrimitiveLine
	// PrimitiveCircle indicates the primitive is a [Circle].
	PrimitiveCircle
	// PrimitiveEllipse indicates the primitive is an [Ellipse].
	PrimitiveEllipse
)

// String returns the primitive kind name.
func (k PrimitiveKind) String() string {
	switch k {
	case PrimitiveLine:
		return "line"
	case PrimitiveCircle:
		return "circle"
	case PrimitiveEllipse:
		return "ellipse"
	default:
		return "none"
	}
}

// Primitive is the result of [FitBestPrimitive]: the geometric shape that best
// explains a point set, along with its RANSAC inlier support. Only the field
// selected by Kind is meaningful.
type Primitive struct {
	// Kind selects which of Line, Circle or Ellipse is populated.
	Kind PrimitiveKind
	// Line holds the model when Kind is PrimitiveLine.
	Line Line
	// Circle holds the model when Kind is PrimitiveCircle.
	Circle Circle
	// Ellipse holds the model when Kind is PrimitiveEllipse.
	Ellipse Ellipse
	// Inliers is the number of points supporting the chosen model.
	Inliers int
}

// FitBestPrimitive fits a line, a circle and an ellipse to the point set with
// RANSAC and returns whichever explains the most points as inliers. Ties are
// broken toward the simpler model (line over circle over ellipse) via a small
// support margin, so a set of near-collinear points is reported as a line
// rather than an over-fit ellipse. It returns a Primitive with Kind
// PrimitiveNone when no model can be fit.
func FitBestPrimitive(pts []cv.Point2f, params RANSACParams) Primitive {
	best := Primitive{Kind: PrimitiveNone}

	if line, in, err := RANSACLine(pts, params); err == nil {
		if len(in) > best.Inliers {
			best = Primitive{Kind: PrimitiveLine, Line: line, Inliers: len(in)}
		}
	}
	// A circle must beat the line's support by more than a small margin to be
	// preferred, since three degrees of freedom can always match at least as
	// many points as two.
	const margin = 2
	if circle, in, err := RANSACCircle(pts, params); err == nil {
		if len(in) > best.Inliers+margin || best.Kind == PrimitiveNone {
			best = Primitive{Kind: PrimitiveCircle, Circle: circle, Inliers: len(in)}
		}
	}
	if ell, in, err := RANSACEllipse(pts, params); err == nil {
		if len(in) > best.Inliers+margin || best.Kind == PrimitiveNone {
			best = Primitive{Kind: PrimitiveEllipse, Ellipse: ell, Inliers: len(in)}
		}
	}
	return best
}

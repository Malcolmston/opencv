package videostab

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MotionModel enumerates the 2-D transformation models that the global-motion
// estimators in this package can fit between two frames. The models are ordered
// by increasing number of degrees of freedom and mirror OpenCV's
// cv::videostab::MotionModel constants.
type MotionModel int

const (
	// MotionModelTranslation is a pure 2-D translation (2 DOF).
	MotionModelTranslation MotionModel = iota
	// MotionModelTranslationAndScale is translation plus a uniform scale (3 DOF).
	MotionModelTranslationAndScale
	// MotionModelRotation is a pure rotation about the image origin (1 DOF).
	MotionModelRotation
	// MotionModelRigid is a Euclidean transform: rotation and translation with
	// unit scale (3 DOF).
	MotionModelRigid
	// MotionModelSimilarity is a similarity transform: uniform scale, rotation
	// and translation (4 DOF).
	MotionModelSimilarity
	// MotionModelAffine is a general affine transform (6 DOF).
	MotionModelAffine
	// MotionModelHomography is a projective transform (8 DOF).
	MotionModelHomography
	// MotionModelUnknown marks an unspecified model.
	MotionModelUnknown
)

// String returns the canonical name of the motion model.
func (m MotionModel) String() string {
	switch m {
	case MotionModelTranslation:
		return "translation"
	case MotionModelTranslationAndScale:
		return "translationAndScale"
	case MotionModelRotation:
		return "rotation"
	case MotionModelRigid:
		return "rigid"
	case MotionModelSimilarity:
		return "similarity"
	case MotionModelAffine:
		return "affine"
	case MotionModelHomography:
		return "homography"
	default:
		return "unknown"
	}
}

// minPoints returns the minimum number of correspondences required to fit the
// model exactly.
func (m MotionModel) minPoints() int {
	switch m {
	case MotionModelTranslation, MotionModelRotation:
		return 1
	case MotionModelTranslationAndScale, MotionModelRigid, MotionModelSimilarity:
		return 2
	case MotionModelAffine:
		return 3
	case MotionModelHomography:
		return 4
	default:
		return 3
	}
}

// Motion is a 2-D transformation stored as a row-major 3×3 homogeneous matrix:
//
//	[ m0 m1 m2 ]
//	[ m3 m4 m5 ]
//	[ m6 m7 m8 ]
//
// A point (x, y) is mapped to ((m0·x+m1·y+m2)/w, (m3·x+m4·y+m5)/w) with
// w = m6·x+m7·y+m8. All the fitted models except MotionModelHomography leave
// the bottom row equal to [0 0 1]. The zero value is not a valid transform; use
// [IdentityMotion] or one of the constructor helpers.
type Motion [9]float64

// IdentityMotion returns the identity transform.
func IdentityMotion() Motion {
	return Motion{1, 0, 0, 0, 1, 0, 0, 0, 1}
}

// TranslationMotion returns a pure translation by (tx, ty).
func TranslationMotion(tx, ty float64) Motion {
	return Motion{1, 0, tx, 0, 1, ty, 0, 0, 1}
}

// SimilarityMotion returns a similarity transform with the given uniform scale,
// rotation angle (radians, counter-clockwise) and translation.
func SimilarityMotion(scale, angle, tx, ty float64) Motion {
	c := scale * math.Cos(angle)
	s := scale * math.Sin(angle)
	return Motion{c, -s, tx, s, c, ty, 0, 0, 1}
}

// Mul returns the matrix product m·n (apply n first, then m).
func (m Motion) Mul(n Motion) Motion {
	var out Motion
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			var sum float64
			for k := 0; k < 3; k++ {
				sum += m[r*3+k] * n[k*3+c]
			}
			out[r*3+c] = sum
		}
	}
	return out
}

// Apply maps the point (x, y) through the transform, performing the homogeneous
// divide.
func (m Motion) Apply(x, y float64) (float64, float64) {
	xp := m[0]*x + m[1]*y + m[2]
	yp := m[3]*x + m[4]*y + m[5]
	w := m[6]*x + m[7]*y + m[8]
	if w == 0 {
		return xp, yp
	}
	return xp / w, yp / w
}

// Determinant returns the determinant of the 3×3 matrix.
func (m Motion) Determinant() float64 {
	return m[0]*(m[4]*m[8]-m[5]*m[7]) -
		m[1]*(m[3]*m[8]-m[5]*m[6]) +
		m[2]*(m[3]*m[7]-m[4]*m[6])
}

// Inverse returns the matrix inverse and reports whether the transform is
// invertible.
func (m Motion) Inverse() (Motion, bool) {
	det := m.Determinant()
	if math.Abs(det) < 1e-15 {
		return Motion{}, false
	}
	inv := 1 / det
	var out Motion
	out[0] = (m[4]*m[8] - m[5]*m[7]) * inv
	out[1] = (m[2]*m[7] - m[1]*m[8]) * inv
	out[2] = (m[1]*m[5] - m[2]*m[4]) * inv
	out[3] = (m[5]*m[6] - m[3]*m[8]) * inv
	out[4] = (m[0]*m[8] - m[2]*m[6]) * inv
	out[5] = (m[2]*m[3] - m[0]*m[5]) * inv
	out[6] = (m[3]*m[7] - m[4]*m[6]) * inv
	out[7] = (m[1]*m[6] - m[0]*m[7]) * inv
	out[8] = (m[0]*m[4] - m[1]*m[3]) * inv
	return out, true
}

// Affine returns the top two rows of the transform as a cv.AffineMatrix,
// dropping the projective row. This is exact for every model except
// MotionModelHomography, for which it is only the affine part.
func (m Motion) Affine() cv.AffineMatrix {
	return cv.AffineMatrix{m[0], m[1], m[2], m[3], m[4], m[5]}
}

// scaled returns m with every element multiplied by k.
func (m Motion) scaled(k float64) Motion {
	var out Motion
	for i := range m {
		out[i] = m[i] * k
	}
	return out
}

// add returns the element-wise sum of m and n.
func (m Motion) add(n Motion) Motion {
	var out Motion
	for i := range m {
		out[i] = m[i] + n[i]
	}
	return out
}

// warp applies the transform to a frame, producing an output of the same size.
// The projective row is ignored (the affine part is used) because cv.WarpAffine
// operates on 2×3 matrices; for the affine and lower models this is exact.
func (m Motion) warp(src *cv.Mat) *cv.Mat {
	return cv.WarpAffine(src, m.Affine(), src.Cols, src.Rows, cv.InterLinear)
}

package calib3d

import "math"

// ConvertPointsToHomogeneous appends a unit homogeneous coordinate to each 2D
// point, converting an inhomogeneous point set (x, y) into its homogeneous
// representation (x, y, 1). It mirrors OpenCV's convertPointsToHomogeneous for
// the 2D→3D case.
func ConvertPointsToHomogeneous(pts [][2]float64) [][3]float64 {
	out := make([][3]float64, len(pts))
	for i, p := range pts {
		out[i] = [3]float64{p[0], p[1], 1}
	}
	return out
}

// ConvertPointsFromHomogeneous performs the perspective division that maps each
// homogeneous 3-vector (x, y, w) back to the inhomogeneous 2D point (x/w, y/w).
// A point whose homogeneous weight w is negligible is passed through as (x, y)
// unchanged, matching OpenCV's behaviour of leaving points at infinity finite.
func ConvertPointsFromHomogeneous(pts [][3]float64) [][2]float64 {
	out := make([][2]float64, len(pts))
	for i, p := range pts {
		w := p[2]
		if math.Abs(w) < 1e-15 {
			out[i] = [2]float64{p[0], p[1]}
			continue
		}
		out[i] = [2]float64{p[0] / w, p[1] / w}
	}
	return out
}

// Epiline selectors for [ComputeCorrespondEpilines]. They match OpenCV's
// whichImage argument: pass points observed in image 1 or image 2 respectively.
const (
	// Image1 means the input points were observed in the first image; the
	// returned epipolar lines live in the second image (l' = F·x).
	Image1 = 1
	// Image2 means the input points were observed in the second image; the
	// returned epipolar lines live in the first image (l = Fᵀ·x').
	Image2 = 2
)

// ComputeCorrespondEpilines computes, for each input image point, the epipolar
// line it induces in the other image of a stereo pair described by the
// fundamental matrix F. Each returned line is the coefficient triple (a, b, c)
// of a·x + b·y + c = 0, normalised so that a² + b² = 1, so that the signed
// distance from a point to the line is simply a·x + b·y + c.
//
// whichImage selects which image the input points come from: [Image1] treats
// pts as observed in the first image and returns lines in the second (l = F·x);
// [Image2] treats pts as observed in the second image and returns lines in the
// first (l = Fᵀ·x). Any other value is treated as [Image1].
func ComputeCorrespondEpilines(pts [][2]float64, whichImage int, F [3][3]float64) [][3]float64 {
	m := F
	if whichImage == Image2 {
		m = transpose3(F)
	}
	out := make([][3]float64, len(pts))
	for i, p := range pts {
		v := [3]float64{p[0], p[1], 1}
		l := matVec3(m, v)
		n := math.Hypot(l[0], l[1])
		if n > 1e-15 {
			l[0] /= n
			l[1] /= n
			l[2] /= n
		}
		out[i] = l
	}
	return out
}

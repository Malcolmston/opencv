package stitching

import (
	"image"
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// CameraParams holds the intrinsic and extrinsic parameters recovered for one
// image of a rotational panorama: the focal length and principal point (the
// intrinsics) and a 3×3 rotation matrix and translation (the extrinsics). For a
// camera rotating about its optical centre the translation is zero and the images
// are related purely by K·R, which is what the estimator and bundle adjuster
// solve for.
type CameraParams struct {
	// Focal is the focal length in pixels.
	Focal float64
	// Aspect is the pixel aspect ratio (fy/fx); 1 for square pixels.
	Aspect float64
	// PPX and PPY are the principal point coordinates in pixels.
	PPX, PPY float64
	// R is the row-major 3×3 camera rotation matrix.
	R [9]float64
	// T is the camera translation (zero for a purely rotating camera).
	T [3]float64
}

// defaultCamera returns a camera with identity rotation, unit aspect and the
// principal point at the centre of a width×height image.
func defaultCamera(focal float64, width, height int) CameraParams {
	return CameraParams{
		Focal:  focal,
		Aspect: 1,
		PPX:    float64(width) / 2,
		PPY:    float64(height) / 2,
		R:      [9]float64(mat3Ident()),
	}
}

// K returns the 3×3 intrinsic matrix K = [f 0 ppx; 0 f·aspect ppy; 0 0 1].
func (c CameraParams) K() cv.PerspectiveMatrix {
	aspect := c.Aspect
	if aspect == 0 {
		aspect = 1
	}
	return cv.PerspectiveMatrix{
		c.Focal, 0, c.PPX,
		0, c.Focal * aspect, c.PPY,
		0, 0, 1,
	}
}

// rot returns the rotation as a mat3.
func (c CameraParams) rot() mat3 { return mat3(c.R) }

// kMat returns the intrinsics as a mat3.
func (c CameraParams) kMat() mat3 { return mat3(c.K()) }

// MatchesInfo carries the inlier point correspondences between two images, used
// by the [Estimator] and [BundleAdjuster] to relate the views. Correspondence i
// pairs SrcPoints[i] in image Src with DstPoints[i] in image Dst.
type MatchesInfo struct {
	// Src and Dst are the indices of the two images.
	Src, Dst int
	// SrcPoints and DstPoints are the matched pixel locations, index-aligned.
	SrcPoints, DstPoints []image.Point
}

// homography fits the homography mapping this match's source points onto its
// destination points, reporting whether the fit succeeded.
func (mi MatchesInfo) homography() (mat3, bool) {
	if len(mi.SrcPoints) < 4 || len(mi.SrcPoints) != len(mi.DstPoints) {
		return mat3{}, false
	}
	src := pointsToF(mi.SrcPoints)
	dst := pointsToF(mi.DstPoints)
	h, ok := computeHomographyDLT(src, dst)
	if !ok {
		return mat3{}, false
	}
	return mat3(h), true
}

// pointsToF converts integer image points to floating-point coordinates.
func pointsToF(pts []image.Point) []pointF {
	out := make([]pointF, len(pts))
	for i, p := range pts {
		out[i] = pointF{float64(p.X), float64(p.Y)}
	}
	return out
}

// Estimator recovers per-image [CameraParams] (focal lengths and rotations) from
// the pairwise correspondences of a panorama. The result is the starting point
// that [BundleAdjuster] then refines globally. The implementation is
// [HomographyBasedEstimator].
type Estimator interface {
	// Estimate returns one camera per image. sizes[i] gives image i's dimensions
	// as (X:width, Y:height); matches connect the images into a spanning tree
	// rooted at image 0. The bool result is false when the cameras cannot be
	// recovered.
	Estimate(sizes []image.Point, matches []MatchesInfo) ([]CameraParams, bool)
}

// HomographyBasedEstimator recovers focal lengths from the inter-image
// homographies (via [EstimateFocalsFromHomography]) and chains the pairwise
// rotations outward from image 0 to give every camera an absolute rotation. It
// implements [Estimator].
type HomographyBasedEstimator struct {
	// FocalOverride, when positive, is used for every camera instead of estimating
	// the focal length from the homographies.
	FocalOverride float64
}

// Estimate implements [Estimator].
func (e HomographyBasedEstimator) Estimate(sizes []image.Point, matches []MatchesInfo) ([]CameraParams, bool) {
	n := len(sizes)
	if n == 0 {
		return nil, false
	}

	// Gather every valid pairwise homography and the focal estimates it yields.
	type edge struct {
		other int
		h     mat3 // maps this node's points onto other's frame
	}
	adj := make([][]edge, n)
	var focals []float64
	for _, mi := range matches {
		if mi.Src < 0 || mi.Src >= n || mi.Dst < 0 || mi.Dst >= n {
			continue
		}
		h, ok := mi.homography()
		if !ok {
			continue
		}
		hInv, okInv := h.inv3()
		if !okInv {
			continue
		}
		// h maps Src → Dst; the reverse maps Dst → Src.
		adj[mi.Src] = append(adj[mi.Src], edge{other: mi.Dst, h: h})
		adj[mi.Dst] = append(adj[mi.Dst], edge{other: mi.Src, h: hInv})
		f0, f1, ok0, ok1 := EstimateFocalsFromHomography(cv.PerspectiveMatrix(h))
		// The individual per-camera focals recovered from a single homography are
		// noisy, but their geometric mean is a robust per-pair estimate (the same
		// quantity OpenCV medians over all pairs).
		switch {
		case ok0 && ok1:
			focals = append(focals, math.Sqrt(f0*f1))
		case ok0:
			focals = append(focals, f0)
		case ok1:
			focals = append(focals, f1)
		}
	}

	focal := e.FocalOverride
	if focal <= 0 {
		focal = medianFocal(focals, sizes)
	}

	cams := make([]CameraParams, n)
	for i := range cams {
		cams[i] = defaultCamera(focal, sizes[i].X, sizes[i].Y)
	}

	// Chain rotations outward from image 0 by breadth-first traversal. For an
	// edge with homography H (this → other): H = K_o R_o R_t⁻¹ K_t⁻¹, so
	// R_o = orthonormalize( K_o⁻¹ H K_t R_t ).
	visited := make([]bool, n)
	visited[0] = true
	queue := []int{0}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		for _, ed := range adj[u] {
			if visited[ed.other] {
				continue
			}
			koInv, ok := cams[ed.other].kMat().inv3()
			if !ok {
				continue
			}
			// R_other = orthonormalize( K_other⁻¹ · H · K_u · R_u ).
			m := koInv.mul(ed.h).mul(cams[u].kMat()).mul(cams[u].rot())
			cams[ed.other].R = [9]float64(m.orthonormalize())
			visited[ed.other] = true
			queue = append(queue, ed.other)
		}
	}
	return cams, true
}

// medianFocal returns the median of the estimated focals, falling back to the
// mean image dimension when no focal could be estimated.
func medianFocal(focals []float64, sizes []image.Point) float64 {
	if len(focals) > 0 {
		sort.Float64s(focals)
		mid := len(focals) / 2
		if len(focals)%2 == 1 {
			return focals[mid]
		}
		return (focals[mid-1] + focals[mid]) / 2
	}
	var sum float64
	for _, s := range sizes {
		sum += float64(s.X+s.Y) / 2
	}
	return sum / float64(len(sizes))
}

// EstimateFocalsFromHomography recovers the focal lengths of the two cameras
// related by the homography h (which must be normalised so h8 = 1), following the
// closed-form solution used by OpenCV. It returns the focal of the source camera
// (f0) and the destination camera (f1) along with a validity flag for each; a
// flag is false when the corresponding focal cannot be recovered from h.
func EstimateFocalsFromHomography(h cv.PerspectiveMatrix) (f0, f1 float64, ok0, ok1 bool) {
	// f1: focal of the destination ("to") camera.
	d1 := h[6] * h[7]
	d2 := (h[7] - h[6]) * (h[7] + h[6])
	v1 := -(h[0]*h[1] + h[3]*h[4]) / d1
	v2 := (h[0]*h[0] + h[3]*h[3] - h[1]*h[1] - h[4]*h[4]) / d2
	if v1 < v2 {
		v1, v2 = v2, v1
	}
	switch {
	case v1 > 0 && v2 > 0:
		if math.Abs(d1) > math.Abs(d2) {
			f1, ok1 = math.Sqrt(v1), true
		} else {
			f1, ok1 = math.Sqrt(v2), true
		}
	case v1 > 0:
		f1, ok1 = math.Sqrt(v1), true
	}

	// f0: focal of the source ("from") camera.
	d1 = h[0]*h[3] + h[1]*h[4]
	d2 = h[0]*h[0] + h[1]*h[1] - h[3]*h[3] - h[4]*h[4]
	v1 = -h[2] * h[5] / d1
	v2 = (h[5]*h[5] - h[2]*h[2]) / d2
	if v1 < v2 {
		v1, v2 = v2, v1
	}
	switch {
	case v1 > 0 && v2 > 0:
		if math.Abs(d1) > math.Abs(d2) {
			f0, ok0 = math.Sqrt(v1), true
		} else {
			f0, ok0 = math.Sqrt(v2), true
		}
	case v1 > 0:
		f0, ok0 = math.Sqrt(v1), true
	}
	return f0, f1, ok0, ok1
}

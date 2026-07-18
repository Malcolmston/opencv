package matching2

import (
	"math"

	"github.com/malcolmston/opencv/core"
)

// FindHomographyDLT estimates the 3×3 projective transform H mapping the source
// points onto the destination points using the normalized Direct Linear
// Transform, so that for each i the homogeneous product H·(srcᵢ,1) is
// proportional to (dstᵢ,1). src and dst must have equal length of at least four.
// With exactly four points the fit is exact; with more it is the least-squares
// solution. H is scaled so that H[2][2] == 1 when possible. It reports false
// when the inputs are too few or degenerate.
func FindHomographyDLT(src, dst []core.Point2d) ([3][3]float64, bool) {
	if len(src) != len(dst) || len(src) < 4 {
		return [3][3]float64{}, false
	}
	nsrc, Ts := NormalizePoints2D(src)
	ndst, Td := NormalizePoints2D(dst)
	a := make([][]float64, 0, 2*len(src))
	for i := range nsrc {
		x, y := nsrc[i].X, nsrc[i].Y
		u, v := ndst[i].X, ndst[i].Y
		a = append(a,
			[]float64{-x, -y, -1, 0, 0, 0, u * x, u * y, u},
			[]float64{0, 0, 0, -x, -y, -1, v * x, v * y, v},
		)
	}
	h := matching2nullVector(a)
	Hn := [3][3]float64{
		{h[0], h[1], h[2]},
		{h[3], h[4], h[5]},
		{h[6], h[7], h[8]},
	}
	// Denormalize: H = Td⁻¹ · Hn · Ts.
	TdInv, ok := Mat3Inverse(Td)
	if !ok {
		return [3][3]float64{}, false
	}
	H := Mat3Mul(TdInv, Mat3Mul(Hn, Ts))
	if math.Abs(H[2][2]) > 1e-300 {
		H = Mat3Scale(H, 1/H[2][2])
	}
	if !matching2finite9(H) {
		return [3][3]float64{}, false
	}
	return H, true
}

// ApplyHomography maps a single point through the homography H, returning the
// dehomogenized image (H·(p,1) divided by its third coordinate). The returned
// point is the origin when the third coordinate is zero (a point at infinity).
func ApplyHomography(H [3][3]float64, p core.Point2d) core.Point2d {
	x := H[0][0]*p.X + H[0][1]*p.Y + H[0][2]
	y := H[1][0]*p.X + H[1][1]*p.Y + H[1][2]
	w := H[2][0]*p.X + H[2][1]*p.Y + H[2][2]
	if math.Abs(w) < 1e-300 {
		return core.Point2d{}
	}
	return core.Point2d{X: x / w, Y: y / w}
}

// PerspectiveTransform maps every point in pts through the homography H,
// returning the transformed points in the same order.
func PerspectiveTransform(H [3][3]float64, pts []core.Point2d) []core.Point2d {
	out := make([]core.Point2d, len(pts))
	for i, p := range pts {
		out[i] = ApplyHomography(H, p)
	}
	return out
}

// HomographyReprojectionError returns the forward reprojection error
// ‖H·src − dst‖ for a single correspondence, in destination pixels.
func HomographyReprojectionError(H [3][3]float64, src, dst core.Point2d) float64 {
	p := ApplyHomography(H, src)
	return math.Hypot(p.X-dst.X, p.Y-dst.Y)
}

// SymmetricTransferError returns the symmetric transfer error for a
// correspondence: the sum of the forward reprojection error ‖H·src − dst‖ and
// the backward error ‖H⁻¹·dst − src‖, both in pixels. It is the residual used by
// [FindHomographyRANSAC]. A non-invertible H yields the forward error alone.
func SymmetricTransferError(H [3][3]float64, src, dst core.Point2d) float64 {
	fwd := HomographyReprojectionError(H, src, dst)
	Hinv, ok := Mat3Inverse(H)
	if !ok {
		return fwd
	}
	back := HomographyReprojectionError(Hinv, dst, src)
	return fwd + back
}

// FindHomographyRANSAC robustly estimates a homography from correspondences that
// may contain outliers. A correspondence is an inlier when its symmetric
// transfer error is at most threshold pixels. iters bounds the number of random
// four-point samples. The returned mask flags the inliers, and the final model
// is refit over all inliers with the normalized DLT. It reports Ok false when no
// four-point sample yields at least four inliers.
func FindHomographyRANSAC(src, dst []core.Point2d, threshold float64, iters int, seed int64) RANSACResult[[3][3]float64] {
	var empty RANSACResult[[3][3]float64]
	if len(src) != len(dst) || len(src) < 4 {
		return empty
	}
	fit := func(sample []int) ([3][3]float64, bool) {
		s := make([]core.Point2d, len(sample))
		d := make([]core.Point2d, len(sample))
		for i, idx := range sample {
			s[i] = src[idx]
			d[i] = dst[idx]
		}
		return FindHomographyDLT(s, d)
	}
	inliers := func(H [3][3]float64) []bool {
		mask := make([]bool, len(src))
		for i := range src {
			mask[i] = SymmetricTransferError(H, src[i], dst[i]) <= threshold
		}
		return mask
	}
	return RANSAC(len(src), 4, iters, 4, seed, fit, inliers, fit)
}

// matching2finite9 reports whether every entry of a 3×3 matrix is finite.
func matching2finite9(m [3][3]float64) bool {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.IsNaN(m[i][j]) || math.IsInf(m[i][j], 0) {
				return false
			}
		}
	}
	return true
}

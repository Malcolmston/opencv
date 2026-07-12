package calib3d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SolvePnPPlanar recovers the pose (rotation vector and translation) of a
// calibrated camera observing a planar object. The object points objPts must be
// coplanar; their Z coordinate is ignored and the plane is taken to be Z = 0.
// imgPts are the corresponding image observations and K is the 3×3 intrinsic
// matrix. objPts and imgPts must have equal length of at least four.
//
// The method estimates the object-to-image homography with [FindHomography],
// factors it through K⁻¹ into the first two rotation columns and the
// translation, reconstructs the third column by orthogonality, and orthonormalises
// the result with an SVD so that the returned rotation is a proper rotation
// matrix. rvec is that rotation as an axis-angle vector (see [RodriguesToVector])
// and tvec is the translation. ok is false when the input is insufficient or the
// homography is degenerate.
func SolvePnPPlanar(objPts [][3]float64, imgPts []cv.Point, K [3][3]float64) (rvec, tvec [3]float64, ok bool) {
	if len(objPts) != len(imgPts) || len(objPts) < 4 {
		return [3]float64{}, [3]float64{}, false
	}
	// Object plane coordinates (X, Y) and their image observations.
	src := make([][2]float64, len(objPts))
	for i, p := range objPts {
		src[i] = [2]float64{p[0], p[1]}
	}
	dst := pointsToF(imgPts)
	h, ok := dltHomography(src, dst)
	if !ok {
		return [3]float64{}, [3]float64{}, false
	}
	kInv, ok := inv3(K)
	if !ok {
		return [3]float64{}, [3]float64{}, false
	}
	// B = K⁻¹·H; its columns encode r1, r2 and t up to a common scale.
	b := mul3(kInv, h)
	b1 := [3]float64{b[0][0], b[1][0], b[2][0]}
	b2 := [3]float64{b[0][1], b[1][1], b[2][1]}
	b3 := [3]float64{b[0][2], b[1][2], b[2][2]}
	n1 := norm3(b1)
	n2 := norm3(b2)
	if n1 < 1e-15 || n2 < 1e-15 {
		return [3]float64{}, [3]float64{}, false
	}
	lambda := 2 / (n1 + n2)
	r1 := scale3(b1, lambda)
	r2 := scale3(b2, lambda)
	t := scale3(b3, lambda)
	// Ensure the object is in front of the camera (positive depth).
	if t[2] < 0 {
		r1 = scale3(r1, -1)
		r2 = scale3(r2, -1)
		t = scale3(t, -1)
	}
	r3 := cross3(r1, r2)
	rApprox := [3][3]float64{
		{r1[0], r2[0], r3[0]},
		{r1[1], r2[1], r3[1]},
		{r1[2], r2[2], r3[2]},
	}
	// Orthonormalise: the closest rotation to rApprox is U·Vᵀ from its SVD.
	u, _, v := svd3(rApprox)
	r := mul3(u, transpose3(v))
	if det3(r) < 0 {
		// Flip the last column of U to keep a proper (det = +1) rotation.
		u[0][2] = -u[0][2]
		u[1][2] = -u[1][2]
		u[2][2] = -u[2][2]
		r = mul3(u, transpose3(v))
	}
	return RodriguesToVector(r), t, true
}

// TriangulatePoints reconstructs 3D points from their projections in two views
// by linear (Direct Linear Transform) triangulation. P1 and P2 are the 3×4
// projection matrices of the two cameras and pts1, pts2 are the corresponding
// image points; the two point slices must have equal length.
//
// For each correspondence the four homogeneous constraints x·(P·X) are stacked
// into a 4×4 system whose null space is the reconstructed point; the result is
// dehomogenised to a Euclidean [3]float64. Points whose homogeneous weight
// vanishes are returned as the origin.
func TriangulatePoints(P1, P2 [3][4]float64, pts1, pts2 []cv.Point) [][3]float64 {
	n := len(pts1)
	if len(pts2) < n {
		n = len(pts2)
	}
	out := make([][3]float64, n)
	for i := 0; i < n; i++ {
		x1, y1 := float64(pts1[i].X), float64(pts1[i].Y)
		x2, y2 := float64(pts2[i].X), float64(pts2[i].Y)
		// Rows: x·P₃ − P₁ and y·P₃ − P₂ for each view.
		rows := [4][4]float64{
			subRow4(scaleRow4(P1[2], x1), P1[0]),
			subRow4(scaleRow4(P1[2], y1), P1[1]),
			subRow4(scaleRow4(P2[2], x2), P2[0]),
			subRow4(scaleRow4(P2[2], y2), P2[1]),
		}
		var ata [4][4]float64
		for _, r := range rows {
			for a := 0; a < 4; a++ {
				for b := 0; b < 4; b++ {
					ata[a][b] += r[a] * r[b]
				}
			}
		}
		dyn := make([][]float64, 4)
		for a := 0; a < 4; a++ {
			dyn[a] = make([]float64, 4)
			copy(dyn[a], ata[a][:])
		}
		x := smallestEigvec(dyn)
		if math.Abs(x[3]) < 1e-15 {
			out[i] = [3]float64{0, 0, 0}
			continue
		}
		out[i] = [3]float64{x[0] / x[3], x[1] / x[3], x[2] / x[3]}
	}
	return out
}

// scale3 multiplies a 3-vector by a scalar.
func scale3(v [3]float64, s float64) [3]float64 {
	return [3]float64{v[0] * s, v[1] * s, v[2] * s}
}

// scaleRow4 multiplies a length-4 row by a scalar.
func scaleRow4(r [4]float64, s float64) [4]float64 {
	return [4]float64{r[0] * s, r[1] * s, r[2] * s, r[3] * s}
}

// subRow4 returns a − b for two length-4 rows.
func subRow4(a, b [4]float64) [4]float64 {
	return [4]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]}
}

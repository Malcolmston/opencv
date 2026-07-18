package matching2

import (
	"github.com/malcolmston/opencv/core"
)

// ProjectionMatrix assembles the 3×4 camera projection matrix P = K·[R | t] from
// the intrinsic matrix K, rotation R and translation t. A world point X projects
// to the image as P·(X,1).
func ProjectionMatrix(K, R [3][3]float64, t [3]float64) [3][4]float64 {
	var Rt [3][4]float64
	for i := 0; i < 3; i++ {
		Rt[i][0] = R[i][0]
		Rt[i][1] = R[i][1]
		Rt[i][2] = R[i][2]
		Rt[i][3] = t[i]
	}
	var P [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			P[i][j] = K[i][0]*Rt[0][j] + K[i][1]*Rt[1][j] + K[i][2]*Rt[2][j]
		}
	}
	return P
}

// TriangulatePoint reconstructs the 3-D world point that projects to p1 in the
// first camera (matrix P1) and to p2 in the second (matrix P2) using the
// homogeneous Direct Linear Transform. P1 and P2 are 3×4 projection matrices and
// p1, p2 are image points in the same units as those matrices (pixels for
// P = K·[R|t], or normalized coordinates for P = [R|t]).
func TriangulatePoint(P1, P2 [3][4]float64, p1, p2 core.Point2d) core.Point3d {
	a := [][]float64{
		{p1.X*P1[2][0] - P1[0][0], p1.X*P1[2][1] - P1[0][1], p1.X*P1[2][2] - P1[0][2], p1.X*P1[2][3] - P1[0][3]},
		{p1.Y*P1[2][0] - P1[1][0], p1.Y*P1[2][1] - P1[1][1], p1.Y*P1[2][2] - P1[1][2], p1.Y*P1[2][3] - P1[1][3]},
		{p2.X*P2[2][0] - P2[0][0], p2.X*P2[2][1] - P2[0][1], p2.X*P2[2][2] - P2[0][2], p2.X*P2[2][3] - P2[0][3]},
		{p2.Y*P2[2][0] - P2[1][0], p2.Y*P2[2][1] - P2[1][1], p2.Y*P2[2][2] - P2[1][2], p2.Y*P2[2][3] - P2[1][3]},
	}
	x := matching2nullVector(a)
	if x[3] == 0 {
		return core.Point3d{X: x[0], Y: x[1], Z: x[2]}
	}
	return core.Point3d{X: x[0] / x[3], Y: x[1] / x[3], Z: x[2] / x[3]}
}

// TriangulatePoints triangulates a batch of correspondences with
// [TriangulatePoint], returning one reconstructed world point per pair. pts1 and
// pts2 must have equal length.
func TriangulatePoints(P1, P2 [3][4]float64, pts1, pts2 []core.Point2d) []core.Point3d {
	if len(pts1) != len(pts2) {
		return nil
	}
	out := make([]core.Point3d, len(pts1))
	for i := range pts1 {
		out[i] = TriangulatePoint(P1, P2, pts1[i], pts2[i])
	}
	return out
}

// project34 applies a 3×4 projection matrix to a world point and returns the
// third (depth) coordinate together with the dehomogenized image point.
func project34(P [3][4]float64, X core.Point3d) (depth float64, img core.Point2d) {
	x := P[0][0]*X.X + P[0][1]*X.Y + P[0][2]*X.Z + P[0][3]
	y := P[1][0]*X.X + P[1][1]*X.Y + P[1][2]*X.Z + P[1][3]
	w := P[2][0]*X.X + P[2][1]*X.Y + P[2][2]*X.Z + P[2][3]
	if w == 0 {
		return 0, core.Point2d{}
	}
	return w, core.Point2d{X: x / w, Y: y / w}
}

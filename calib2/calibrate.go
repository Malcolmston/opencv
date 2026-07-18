package calib2

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// normalizePoints computes an isotropic normalizing similarity for a point set,
// translating the centroid to the origin and scaling so the mean distance to
// the origin is √2. It returns the transform and the normalized points; ok is
// false when the points are all coincident.
func normalizePoints(pts [][2]float64) (t [3][3]float64, out [][2]float64, ok bool) {
	n := len(pts)
	var cx, cy float64
	for _, p := range pts {
		cx += p[0]
		cy += p[1]
	}
	cx /= float64(n)
	cy /= float64(n)
	var meanDist float64
	for _, p := range pts {
		meanDist += math.Hypot(p[0]-cx, p[1]-cy)
	}
	meanDist /= float64(n)
	if meanDist < 1e-12 {
		return Mat3Identity(), nil, false
	}
	s := math.Sqrt2 / meanDist
	t = [3][3]float64{{s, 0, -s * cx}, {0, s, -s * cy}, {0, 0, 1}}
	out = make([][2]float64, n)
	for i, p := range pts {
		out[i] = [2]float64{s * (p[0] - cx), s * (p[1] - cy)}
	}
	return t, out, true
}

// homographyDLT estimates the 3×3 homography mapping src to dst by the
// normalized direct linear transform. It requires at least four
// correspondences and returns ok=false when the configuration is degenerate.
func homographyDLT(src, dst [][2]float64) (h [3][3]float64, ok bool) {
	if len(src) < 4 || len(src) != len(dst) {
		return h, false
	}
	ts, ns, ok1 := normalizePoints(src)
	td, nd, ok2 := normalizePoints(dst)
	if !ok1 || !ok2 {
		return h, false
	}
	a := NewMatrix(2*len(src), 9)
	for i := range ns {
		X, Y := ns[i][0], ns[i][1]
		x, y := nd[i][0], nd[i][1]
		r0 := (2 * i) * 9
		a.Data[r0+0] = -X
		a.Data[r0+1] = -Y
		a.Data[r0+2] = -1
		a.Data[r0+6] = x * X
		a.Data[r0+7] = x * Y
		a.Data[r0+8] = x
		r1 := (2*i + 1) * 9
		a.Data[r1+3] = -X
		a.Data[r1+4] = -Y
		a.Data[r1+5] = -1
		a.Data[r1+6] = y * X
		a.Data[r1+7] = y * Y
		a.Data[r1+8] = y
	}
	hv := smallestEigenvector(a.gram())
	hn := [3][3]float64{
		{hv[0], hv[1], hv[2]},
		{hv[3], hv[4], hv[5]},
		{hv[6], hv[7], hv[8]},
	}
	// Denormalize: H = Td⁻¹ · Hn · Ts.
	tdInv, ok3 := Mat3Inverse(td)
	if !ok3 {
		return h, false
	}
	h = Mat3Mul(Mat3Mul(tdInv, hn), ts)
	if math.Abs(h[2][2]) > 1e-15 {
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				h[i][j] /= h[2][2]
			}
		}
	}
	return h, true
}

// FindHomography estimates the 3×3 projective transform H that maps the source
// points to the destination points, using the normalized direct linear
// transform over all correspondences (an exact fit for four points, a
// least-squares fit for more). At least four correspondences are required. The
// returned homography is scaled so that H[2][2] = 1; ok is false when the
// configuration is degenerate.
func FindHomography(src, dst []cv.Point2f) (h [3][3]float64, ok bool) {
	if len(src) != len(dst) {
		return h, false
	}
	s := make([][2]float64, len(src))
	d := make([][2]float64, len(dst))
	for i := range src {
		s[i] = [2]float64{src[i].X, src[i].Y}
		d[i] = [2]float64{dst[i].X, dst[i].Y}
	}
	return homographyDLT(s, d)
}

// ApplyHomography maps a single point through a 3×3 homography, performing the
// perspective division. It returns the mapped point.
func ApplyHomography(h [3][3]float64, p cv.Point2f) cv.Point2f {
	w := h[2][0]*p.X + h[2][1]*p.Y + h[2][2]
	if w == 0 {
		w = 1e-12
	}
	return cv.Point2f{
		X: (h[0][0]*p.X + h[0][1]*p.Y + h[0][2]) / w,
		Y: (h[1][0]*p.X + h[1][1]*p.Y + h[1][2]) / w,
	}
}

// GenerateChessboardCorners returns the 3D object coordinates of the inner
// corners of a chessboard calibration target lying in the z = 0 plane. The grid
// has patternRows × patternCols corners spaced squareSize apart; corner (r, c)
// is at (c·squareSize, r·squareSize, 0), enumerated row-major.
func GenerateChessboardCorners(patternRows, patternCols int, squareSize float64) [][3]float64 {
	out := make([][3]float64, 0, patternRows*patternCols)
	for r := 0; r < patternRows; r++ {
		for c := 0; c < patternCols; c++ {
			out = append(out, [3]float64{float64(c) * squareSize, float64(r) * squareSize, 0})
		}
	}
	return out
}

// zhangConstraintRow builds the 6-vector v_ij used in Zhang's intrinsic
// constraint from two columns of a homography.
func zhangConstraintRow(hi, hj [3]float64) [6]float64 {
	return [6]float64{
		hi[0] * hj[0],
		hi[0]*hj[1] + hi[1]*hj[0],
		hi[1] * hj[1],
		hi[2]*hj[0] + hi[0]*hj[2],
		hi[2]*hj[1] + hi[1]*hj[2],
		hi[2] * hj[2],
	}
}

// IntrinsicsFromHomographies recovers the camera intrinsics from a set of
// plane-to-image homographies using Zhang's closed-form solution. At least
// three homographies (from three non-parallel views of a planar target) are
// needed for a full solve with skew; two suffice when skew is assumed zero. It
// returns an error when the linear system is degenerate or the recovered
// parameters are non-physical.
func IntrinsicsFromHomographies(homographies [][3][3]float64) (CameraMatrix, error) {
	if len(homographies) < 2 {
		return CameraMatrix{}, errors.New("calib2: need at least two homographies")
	}
	v := NewMatrix(2*len(homographies), 6)
	for i, h := range homographies {
		h1 := [3]float64{h[0][0], h[1][0], h[2][0]}
		h2 := [3]float64{h[0][1], h[1][1], h[2][1]}
		v12 := zhangConstraintRow(h1, h2)
		v11 := zhangConstraintRow(h1, h1)
		v22 := zhangConstraintRow(h2, h2)
		for j := 0; j < 6; j++ {
			v.Data[(2*i)*6+j] = v12[j]
			v.Data[(2*i+1)*6+j] = v11[j] - v22[j]
		}
	}
	b := smallestEigenvector(v.gram())
	// b = [B11, B12, B22, B13, B23, B33] for B = K⁻ᵀ K⁻¹ (up to scale).
	B11, B12, B22, B13, B23, B33 := b[0], b[1], b[2], b[3], b[4], b[5]
	// B must be positive-definite; flip global sign if needed.
	if B11 < 0 || B22 < 0 || B33 < 0 {
		B11, B12, B22, B13, B23, B33 = -B11, -B12, -B22, -B13, -B23, -B33
	}
	denom := B11*B22 - B12*B12
	if math.Abs(denom) < 1e-18 || math.Abs(B11) < 1e-18 {
		return CameraMatrix{}, errors.New("calib2: degenerate intrinsic solve")
	}
	v0 := (B12*B13 - B11*B23) / denom
	lambda := B33 - (B13*B13+v0*(B12*B13-B11*B23))/B11
	if lambda/B11 <= 0 || lambda*B11/denom <= 0 {
		return CameraMatrix{}, errors.New("calib2: non-physical intrinsics")
	}
	alpha := math.Sqrt(lambda / B11)
	beta := math.Sqrt(lambda * B11 / denom)
	gamma := -B12 * alpha * alpha * beta / lambda
	u0 := gamma*v0/beta - B13*alpha*alpha/lambda
	return CameraMatrix{Fx: alpha, Fy: beta, Cx: u0, Cy: v0, Skew: gamma}, nil
}

// nearestRotation returns the proper rotation matrix closest to q (in the
// Frobenius sense) by computing q·(qᵀq)^-1/2 through a symmetric eigenvalue
// decomposition and flipping the sign when the determinant is negative.
func nearestRotation(q [3][3]float64) [3][3]float64 {
	qt := Mat3Transpose(q)
	m := Mat3Mul(qt, q)
	g := [][]float64{
		{m[0][0], m[0][1], m[0][2]},
		{m[1][0], m[1][1], m[1][2]},
		{m[2][0], m[2][1], m[2][2]},
	}
	vals, vecs := jacobiEigen(g)
	// Build M^{-1/2} = V diag(1/sqrt(w)) Vᵀ.
	var inv [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var s float64
			for k := 0; k < 3; k++ {
				w := vals[k]
				if w < 1e-18 {
					w = 1e-18
				}
				s += vecs[i][k] * (1 / math.Sqrt(w)) * vecs[j][k]
			}
			inv[i][j] = s
		}
	}
	r := Mat3Mul(q, inv)
	if Mat3Det(r) < 0 {
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				r[i][j] = -r[i][j]
			}
		}
	}
	return r
}

// ExtrinsicsFromHomography recovers the camera pose (rotation and translation)
// for a planar target from its plane-to-image homography and the known
// intrinsics, following Zhang. The rotation is orthonormalized to the nearest
// proper rotation matrix, and the translation sign is chosen so the target lies
// in front of the camera.
func ExtrinsicsFromHomography(h [3][3]float64, k CameraMatrix) Pose {
	kinv := k.Inverse()
	h1 := [3]float64{h[0][0], h[1][0], h[2][0]}
	h2 := [3]float64{h[0][1], h[1][1], h[2][1]}
	h3 := [3]float64{h[0][2], h[1][2], h[2][2]}
	kh1 := Mat3VecMul(kinv, h1)
	kh2 := Mat3VecMul(kinv, h2)
	kh3 := Mat3VecMul(kinv, h3)
	lambda := 1.0 / Vec3Norm(kh1)
	r1 := Vec3Scale(kh1, lambda)
	r2 := Vec3Scale(kh2, lambda)
	r3 := Vec3Cross(r1, r2)
	t := Vec3Scale(kh3, lambda)
	if t[2] < 0 {
		r1 = Vec3Scale(r1, -1)
		r2 = Vec3Scale(r2, -1)
		r3 = Vec3Cross(r1, r2)
		t = Vec3Scale(t, -1)
	}
	q := [3][3]float64{
		{r1[0], r2[0], r3[0]},
		{r1[1], r2[1], r3[1]},
		{r1[2], r2[2], r3[2]},
	}
	return Pose{R: nearestRotation(q), T: t}
}

// estimateRadialDistortion fits the two leading radial coefficients k1, k2 by
// linear least squares, given the intrinsics and the per-view poses, by
// comparing the distortion-free projections against the observed image points.
func estimateRadialDistortion(objectPoints [][][3]float64, imagePoints [][]cv.Point2f, k CameraMatrix, poses []Pose) DistortionCoeffs {
	// Normal-equations accumulators for the 2×2 system.
	var a00, a01, a11, b0, b1 float64
	for v := range objectPoints {
		for i, obj := range objectPoints[v] {
			cam := poses[v].Apply(obj)
			z := cam[2]
			if z == 0 {
				continue
			}
			x := cam[0] / z
			y := cam[1] / z
			r2 := x*x + y*y
			u := k.Fx*x + k.Skew*y + k.Cx
			vv := k.Fy*y + k.Cy
			du := u - k.Cx
			dv := vv - k.Cy
			// Rows: [du*r2, du*r2^2]·[k1;k2] = obs_u - u  (and same for v).
			c0u := du * r2
			c1u := du * r2 * r2
			ru := imagePoints[v][i].X - u
			c0v := dv * r2
			c1v := dv * r2 * r2
			rv := imagePoints[v][i].Y - vv
			a00 += c0u*c0u + c0v*c0v
			a01 += c0u*c1u + c0v*c1v
			a11 += c1u*c1u + c1v*c1v
			b0 += c0u*ru + c0v*rv
			b1 += c1u*ru + c1v*rv
		}
	}
	det := a00*a11 - a01*a01
	if math.Abs(det) < 1e-18 {
		return DistortionCoeffs{}
	}
	k1 := (b0*a11 - b1*a01) / det
	k2 := (a00*b1 - a01*b0) / det
	return DistortionCoeffs{K1: k1, K2: k2}
}

// CalibrateCamera performs intrinsic camera calibration from several views of a
// planar target using Zhang's method. objectPoints[v] holds the 3D coordinates
// of the target corners for view v (all with z = 0), imagePoints[v] the
// corresponding detected image points, and the imageSize is unused by the
// closed-form solve but accepted for API symmetry. It returns the recovered
// intrinsics, the leading radial distortion coefficients (k1, k2; tangential
// and higher-order terms are left zero), the per-view poses and the RMS
// reprojection error in pixels. At least two views are required and every view
// must use the same number of points.
func CalibrateCamera(objectPoints [][][3]float64, imagePoints [][]cv.Point2f) (k CameraMatrix, d DistortionCoeffs, poses []Pose, rmsError float64, err error) {
	if len(objectPoints) < 2 || len(objectPoints) != len(imagePoints) {
		return k, d, nil, 0, errors.New("calib2: need at least two matched views")
	}
	homs := make([][3][3]float64, len(objectPoints))
	for v := range objectPoints {
		if len(objectPoints[v]) != len(imagePoints[v]) || len(objectPoints[v]) < 4 {
			return k, d, nil, 0, errors.New("calib2: each view needs at least four matched points")
		}
		src := make([][2]float64, len(objectPoints[v]))
		dst := make([][2]float64, len(imagePoints[v]))
		for i := range objectPoints[v] {
			src[i] = [2]float64{objectPoints[v][i][0], objectPoints[v][i][1]}
			dst[i] = [2]float64{imagePoints[v][i].X, imagePoints[v][i].Y}
		}
		h, ok := homographyDLT(src, dst)
		if !ok {
			return k, d, nil, 0, errors.New("calib2: homography estimation failed")
		}
		homs[v] = h
	}
	k, err = IntrinsicsFromHomographies(homs)
	if err != nil {
		return k, d, nil, 0, err
	}
	poses = make([]Pose, len(homs))
	for v := range homs {
		poses[v] = ExtrinsicsFromHomography(homs[v], k)
	}
	d = estimateRadialDistortion(objectPoints, imagePoints, k, poses)
	rmsError = calibrationRMS(objectPoints, imagePoints, k, d, poses)
	return k, d, poses, rmsError, nil
}

// calibrationRMS computes the root-mean-square reprojection error over all
// views using the full projection model.
func calibrationRMS(objectPoints [][][3]float64, imagePoints [][]cv.Point2f, k CameraMatrix, d DistortionCoeffs, poses []Pose) float64 {
	var sum float64
	var n int
	for v := range objectPoints {
		for i, obj := range objectPoints[v] {
			proj := ProjectPoint(obj, poses[v], k, d)
			dx := proj.X - imagePoints[v][i].X
			dy := proj.Y - imagePoints[v][i].Y
			sum += dx*dx + dy*dy
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return math.Sqrt(sum / float64(n))
}

// ReprojectionError returns the mean Euclidean reprojection error, in pixels,
// between the observed image points and the projections of the object points
// under the given pose, intrinsics and distortion. It panics if the slices
// differ in length.
func ReprojectionError(objectPoints [][3]float64, imagePoints []cv.Point2f, pose Pose, k CameraMatrix, d DistortionCoeffs) float64 {
	if len(objectPoints) != len(imagePoints) {
		panic("calib2: ReprojectionError length mismatch")
	}
	if len(objectPoints) == 0 {
		return 0
	}
	var sum float64
	for i, obj := range objectPoints {
		proj := ProjectPoint(obj, pose, k, d)
		sum += math.Hypot(proj.X-imagePoints[i].X, proj.Y-imagePoints[i].Y)
	}
	return sum / float64(len(objectPoints))
}

// ReprojectionErrorRMS returns the root-mean-square reprojection error, in
// pixels, between the observed image points and the projections of the object
// points. It panics if the slices differ in length.
func ReprojectionErrorRMS(objectPoints [][3]float64, imagePoints []cv.Point2f, pose Pose, k CameraMatrix, d DistortionCoeffs) float64 {
	if len(objectPoints) != len(imagePoints) {
		panic("calib2: ReprojectionErrorRMS length mismatch")
	}
	if len(objectPoints) == 0 {
		return 0
	}
	var sum float64
	for i, obj := range objectPoints {
		proj := ProjectPoint(obj, pose, k, d)
		dx := proj.X - imagePoints[i].X
		dy := proj.Y - imagePoints[i].Y
		sum += dx*dx + dy*dy
	}
	return math.Sqrt(sum / float64(len(objectPoints)))
}

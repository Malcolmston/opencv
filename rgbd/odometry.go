package rgbd

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// OdometryOptions configures the frame-to-frame odometry solvers
// ([ICPOdometry], [RgbdOdometry] and [RgbdICPOdometry]).
type OdometryOptions struct {
	// MaxIterations caps the Gauss–Newton iterations.
	MaxIterations int
	// MinDepth and MaxDepth bound the depths (in metres) that take part; pixels
	// outside the band are ignored. MaxDepth ≤ 0 disables the upper bound.
	MinDepth float64
	MaxDepth float64
	// MaxDepthDiff gates the geometric (ICP) data association: a src point is
	// paired with the dst surface only if their depths agree to within this many
	// metres, which rejects occlusions and disocclusions.
	MaxDepthDiff float64
	// ConvergenceEps stops the iteration once the twist update's norm drops below
	// it.
	ConvergenceEps float64
	// ICPWeight and RgbdWeight scale the geometric and photometric normal
	// equations when both are combined in [RgbdICPOdometry].
	ICPWeight  float64
	RgbdWeight float64
	// InitialPose seeds the estimate; the zero value is treated as the identity.
	InitialPose Pose
}

// DefaultOdometryOptions returns sensible defaults: 20 iterations, a 0.1–4 m
// depth band, a 0.07 m association gate, a 1e-8 convergence threshold, equal ICP
// and RGB-D weights and an identity initial pose.
func DefaultOdometryOptions() OdometryOptions {
	return OdometryOptions{
		MaxIterations:  20,
		MinDepth:       0.1,
		MaxDepth:       4.0,
		MaxDepthDiff:   0.07,
		ConvergenceEps: 1e-8,
		ICPWeight:      1.0,
		RgbdWeight:     1.0,
		InitialPose:    IdentityPose(),
	}
}

// OdometryResult reports the outcome of an odometry solve.
type OdometryResult struct {
	// Pose is the recovered rigid motion mapping a point seen in the source
	// camera to the same point seen in the destination camera (p_dst = R·p_src +
	// T).
	Pose Pose
	// Iterations is the number of Gauss–Newton steps actually run.
	Iterations int
	// RMSError is the final root-mean-square residual (metres for [ICPOdometry],
	// intensity units for [RgbdOdometry], a blend for [RgbdICPOdometry]).
	RMSError float64
	// Converged reports whether the iteration stopped on the convergence
	// threshold rather than exhausting MaxIterations.
	Converged bool
}

// normalizedOptions fills in an initial pose and clamps degenerate settings.
func normalizedOptions(o OdometryOptions) OdometryOptions {
	if o.MaxIterations <= 0 {
		o.MaxIterations = 20
	}
	if o.ConvergenceEps <= 0 {
		o.ConvergenceEps = 1e-8
	}
	if o.MaxDepthDiff <= 0 {
		o.MaxDepthDiff = 0.07
	}
	// A zero-value pose (all-zero rotation) is not valid; treat it as identity.
	if o.InitialPose.R == ([3][3]float64{}) {
		o.InitialPose = IdentityPose()
	}
	return o
}

// depthOK reports whether z lies inside the configured depth band.
func depthOK(z float64, o OdometryOptions) bool {
	if z <= 0 || z < o.MinDepth {
		return false
	}
	if o.MaxDepth > 0 && z > o.MaxDepth {
		return false
	}
	return true
}

// ICPOdometry estimates the rigid motion between two depth frames with dense
// projective point-to-plane ICP, the geometric arm of OpenCV's rgbd odometry.
// Working from the current estimate, every valid source pixel is back-projected,
// transformed into the destination camera and projected to a destination pixel;
// the destination depth and surface normal there define a tangent plane, and the
// point-to-plane error is linearised and accumulated into a 6×6 Gauss–Newton
// system solved for the incremental twist. Associations whose depths disagree by
// more than MaxDepthDiff are discarded.
//
// srcDepth and dstDepth must share the intrinsics K and the same size. It
// returns the recovered motion (p_dst = R·p_src + T) and convergence
// diagnostics. It panics on nil/empty or mismatched inputs or a zero focal
// length.
func ICPOdometry(srcDepth, dstDepth *cv.FloatMat, k [3][3]float64, opts OdometryOptions) OdometryResult {
	requireDepthPair(srcDepth, dstDepth)
	validK(k)
	o := normalizedOptions(opts)
	dstNormals := Compute3DNormals(dstDepth, k)
	pose := o.InitialPose
	var rms float64
	converged := false
	iter := 0
	for ; iter < o.MaxIterations; iter++ {
		var a [6][6]float64
		var b [6]float64
		sumSq, used := icpAccumulate(&a, &b, srcDepth, dstDepth, dstNormals, k, pose, o)
		if used > 0 {
			rms = math.Sqrt(sumSq / float64(used))
		}
		xi, ok := solveGaussNewton(a, b)
		if !ok {
			break
		}
		pose = applyTwist(xi, pose)
		if twistNorm(xi) < o.ConvergenceEps {
			converged = true
			iter++
			break
		}
	}
	return OdometryResult{Pose: pose, Iterations: iter, RMSError: rms, Converged: converged}
}

// icpAccumulate folds the point-to-plane constraints of one iteration into the
// normal equations and returns the summed squared residual and the count of
// constraints used.
func icpAccumulate(a *[6][6]float64, b *[6]float64, srcDepth, dstDepth *cv.FloatMat, dstNormals [][3]float64, k [3][3]float64, pose Pose, o OdometryOptions) (sumSq float64, used int) {
	fx, fy := k[0][0], k[1][1]
	cx, cy := k[0][2], k[1][2]
	rows, cols := dstDepth.Rows, dstDepth.Cols
	for v := 0; v < srcDepth.Rows; v++ {
		for u := 0; u < srcDepth.Cols; u++ {
			z := srcDepth.At(v, u)
			if !depthOK(z, o) {
				continue
			}
			p := pose.Apply(backProject(u, v, z, k))
			if p[2] <= 0 {
				continue
			}
			du := fx*p[0]/p[2] + cx
			dv := fy*p[1]/p[2] + cy
			iu := int(math.Round(du))
			iv := int(math.Round(dv))
			if iu < 0 || iu >= cols || iv < 0 || iv >= rows {
				continue
			}
			zq := dstDepth.At(iv, iu)
			if !depthOK(zq, o) {
				continue
			}
			q := backProject(iu, iv, zq, k)
			if math.Abs(p[2]-q[2]) > o.MaxDepthDiff {
				continue
			}
			n := dstNormals[iv*cols+iu]
			if norm3(n) < 1e-9 {
				continue
			}
			r := dot3(n, sub3(p, q))
			pn := cross3(p, n)
			jac := [6]float64{pn[0], pn[1], pn[2], n[0], n[1], n[2]}
			accumulateNormal(a, b, jac, r, o.ICPWeight)
			sumSq += r * r
			used++
		}
	}
	return sumSq, used
}

// RgbdOdometry estimates the rigid motion between two frames from photometric
// (brightness-constancy) constraints, the direct/dense arm of RGB-D odometry.
// Each valid source pixel is back-projected with its depth, transformed by the
// current estimate and re-projected into the destination image; the difference
// between the destination intensity there and the source intensity is the
// residual, and its Jacobian chains the destination image gradient through the
// projection and the motion twist. The constraints form a 6×6 Gauss–Newton
// system solved per iteration.
//
// srcGray/dstGray are single-channel intensity images and srcDepth/dstDepth
// their depths; all four share the intrinsics K and the same size. It returns
// the recovered motion (p_dst = R·p_src + T) with an intensity-unit RMS. It
// panics on nil/empty or mismatched inputs or a zero focal length.
func RgbdOdometry(srcGray, srcDepth, dstGray, dstDepth *cv.FloatMat, k [3][3]float64, opts OdometryOptions) OdometryResult {
	requireDepthPair(srcDepth, dstDepth)
	requireSameSize(srcGray, srcDepth)
	requireSameSize(dstGray, dstDepth)
	validK(k)
	o := normalizedOptions(opts)
	pose := o.InitialPose
	var rms float64
	converged := false
	iter := 0
	for ; iter < o.MaxIterations; iter++ {
		var a [6][6]float64
		var b [6]float64
		sumSq, used := rgbdAccumulate(&a, &b, srcGray, srcDepth, dstGray, k, pose, o)
		if used > 0 {
			rms = math.Sqrt(sumSq / float64(used))
		}
		xi, ok := solveGaussNewton(a, b)
		if !ok {
			break
		}
		pose = applyTwist(xi, pose)
		if twistNorm(xi) < o.ConvergenceEps {
			converged = true
			iter++
			break
		}
	}
	return OdometryResult{Pose: pose, Iterations: iter, RMSError: rms, Converged: converged}
}

// rgbdAccumulate folds the photometric constraints of one iteration into the
// normal equations and returns the summed squared residual and constraint count.
func rgbdAccumulate(a *[6][6]float64, b *[6]float64, srcGray, srcDepth, dstGray *cv.FloatMat, k [3][3]float64, pose Pose, o OdometryOptions) (sumSq float64, used int) {
	fx, fy := k[0][0], k[1][1]
	cx, cy := k[0][2], k[1][2]
	rows, cols := dstGray.Rows, dstGray.Cols
	for v := 0; v < srcDepth.Rows; v++ {
		for u := 0; u < srcDepth.Cols; u++ {
			z := srcDepth.At(v, u)
			if !depthOK(z, o) {
				continue
			}
			p := pose.Apply(backProject(u, v, z, k))
			if p[2] <= 0 {
				continue
			}
			du := fx*p[0]/p[2] + cx
			dv := fy*p[1]/p[2] + cy
			if !inBounds(du, dv, rows, cols) {
				continue
			}
			r := sampleBilinear(dstGray, du, dv) - srcGray.At(v, u)
			gx, gy := gradientAt(dstGray, du, dv)
			jac := photometricJacobian(p, gx, gy, fx, fy)
			accumulateNormal(a, b, jac, r, o.RgbdWeight)
			sumSq += r * r
			used++
		}
	}
	return sumSq, used
}

// photometricJacobian returns the 6-vector Jacobian of a brightness-constancy
// residual w.r.t. the twist (ω, ν), chaining the image gradient (gx, gy) through
// the projection derivative at the transformed point p and the motion of p.
func photometricJacobian(p [3]float64, gx, gy, fx, fy float64) [6]float64 {
	invZ := 1 / p[2]
	invZ2 := invZ * invZ
	// Projection Jacobian ∂π/∂P (2×3).
	j00, j02 := fx*invZ, -fx*p[0]*invZ2
	j11, j12 := fy*invZ, -fy*p[1]*invZ2
	// grad·∂π/∂P gives the 1×3 sensitivity of intensity to the point P.
	gpx := gx * j00
	gpy := gy * j11
	gpz := gx*j02 + gy*j12
	gP := [3]float64{gpx, gpy, gpz}
	// ∂P/∂ν = I, ∂P/∂ω = −[p]ₓ, so the rotation part is gP·(−[p]ₓ) = (p×gP).
	rot := cross3(p, gP)
	return [6]float64{rot[0], rot[1], rot[2], gP[0], gP[1], gP[2]}
}

// RgbdICPOdometry estimates motion by jointly minimising the photometric and the
// geometric point-to-plane residuals, the full RGB-D + ICP odometry. Each
// iteration accumulates both constraint sets — brightness constancy weighted by
// RgbdWeight and point-to-plane weighted by ICPWeight — into a single 6×6
// system, so texture and shape complement each other where either alone is
// under-constrained.
//
// All four images share the intrinsics K and the same size. It returns the
// recovered motion (p_dst = R·p_src + T); RMSError is the combined
// per-constraint RMS. It panics on nil/empty or mismatched inputs or a zero
// focal length.
func RgbdICPOdometry(srcGray, srcDepth, dstGray, dstDepth *cv.FloatMat, k [3][3]float64, opts OdometryOptions) OdometryResult {
	requireDepthPair(srcDepth, dstDepth)
	requireSameSize(srcGray, srcDepth)
	requireSameSize(dstGray, dstDepth)
	validK(k)
	o := normalizedOptions(opts)
	dstNormals := Compute3DNormals(dstDepth, k)
	pose := o.InitialPose
	var rms float64
	converged := false
	iter := 0
	for ; iter < o.MaxIterations; iter++ {
		var a [6][6]float64
		var b [6]float64
		icpSq, icpN := icpAccumulate(&a, &b, srcDepth, dstDepth, dstNormals, k, pose, o)
		rgbdSq, rgbdN := rgbdAccumulate(&a, &b, srcGray, srcDepth, dstGray, k, pose, o)
		if n := icpN + rgbdN; n > 0 {
			rms = math.Sqrt((icpSq + rgbdSq) / float64(n))
		}
		xi, ok := solveGaussNewton(a, b)
		if !ok {
			break
		}
		pose = applyTwist(xi, pose)
		if twistNorm(xi) < o.ConvergenceEps {
			converged = true
			iter++
			break
		}
	}
	return OdometryResult{Pose: pose, Iterations: iter, RMSError: rms, Converged: converged}
}

// applyTwist composes the incremental motion encoded by the twist ξ = (ω, ν)
// onto the current pose (left multiplication, as the residuals were linearised
// in the destination frame).
func applyTwist(xi [6]float64, pose Pose) Pose {
	delta := Pose{
		R: Rodrigues([3]float64{xi[0], xi[1], xi[2]}),
		T: [3]float64{xi[3], xi[4], xi[5]},
	}
	return delta.Compose(pose)
}

// twistNorm returns the Euclidean norm of a 6-vector twist.
func twistNorm(xi [6]float64) float64 {
	var s float64
	for _, v := range xi {
		s += v * v
	}
	return math.Sqrt(s)
}

// requireDepthPair panics unless both depth maps are non-empty and identically
// sized.
func requireDepthPair(src, dst *cv.FloatMat) {
	if src == nil || dst == nil || len(src.Data) == 0 || len(dst.Data) == 0 {
		panic("rgbd: odometry given an empty depth map")
	}
	if src.Rows != dst.Rows || src.Cols != dst.Cols {
		panic("rgbd: odometry source and destination sizes differ")
	}
}

// requireSameSize panics unless a and b are non-nil and identically sized.
func requireSameSize(a, b *cv.FloatMat) {
	if a == nil || b == nil || a.Rows != b.Rows || a.Cols != b.Cols {
		panic("rgbd: odometry intensity and depth sizes differ")
	}
}

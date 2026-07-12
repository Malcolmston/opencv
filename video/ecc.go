package video

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MotionType selects the parametric motion model estimated by
// [FindTransformECC].
type MotionType int

const (
	// MotionTranslation is a pure translation (2 parameters). The warp is a 2x3
	// matrix [[1,0,tx],[0,1,ty]].
	MotionTranslation MotionType = iota
	// MotionEuclidean is a rigid rotation + translation (3 parameters). The warp
	// is a 2x3 matrix [[cosθ,-sinθ,tx],[sinθ,cosθ,ty]].
	MotionEuclidean
	// MotionAffine is a general affine transform (6 parameters). The warp is a
	// full 2x3 matrix.
	MotionAffine
	// MotionHomography is a projective transform (8 parameters). The warp is a
	// 3x3 matrix with the bottom-right entry fixed at 1.
	MotionHomography
)

// numParams returns the number of free parameters for a motion type.
func (mt MotionType) numParams() int {
	switch mt {
	case MotionTranslation:
		return 2
	case MotionEuclidean:
		return 3
	case MotionAffine:
		return 6
	case MotionHomography:
		return 8
	default:
		panic("video: unknown MotionType")
	}
}

// identityParams returns the parameter vector of the identity warp for a motion
// type.
func (mt MotionType) identityParams() []float64 {
	switch mt {
	case MotionTranslation:
		return []float64{0, 0}
	case MotionEuclidean:
		return []float64{0, 0, 0} // theta, tx, ty
	case MotionAffine:
		return []float64{1, 0, 0, 0, 1, 0} // a b c d e f: wx=ax+by+c, wy=dx+ey+f
	case MotionHomography:
		return []float64{1, 0, 0, 0, 1, 0, 0, 0} // h0..h7, h8=1
	default:
		panic("video: unknown MotionType")
	}
}

// warpPoint maps template coordinates (x, y) into the input image using the
// current parameters for a motion type.
func (mt MotionType) warpPoint(p []float64, x, y float64) (wx, wy float64) {
	switch mt {
	case MotionTranslation:
		return x + p[0], y + p[1]
	case MotionEuclidean:
		c, s := math.Cos(p[0]), math.Sin(p[0])
		return c*x - s*y + p[1], s*x + c*y + p[2]
	case MotionAffine:
		return p[0]*x + p[1]*y + p[2], p[3]*x + p[4]*y + p[5]
	case MotionHomography:
		d := p[6]*x + p[7]*y + 1
		return (p[0]*x + p[1]*y + p[2]) / d, (p[3]*x + p[4]*y + p[5]) / d
	default:
		panic("video: unknown MotionType")
	}
}

// jacobian returns the 2xN Jacobian of the warped coordinate with respect to the
// parameters at template point (x, y). jx holds ∂wx/∂p and jy holds ∂wy/∂p.
func (mt MotionType) jacobian(p []float64, x, y float64) (jx, jy []float64) {
	switch mt {
	case MotionTranslation:
		return []float64{1, 0}, []float64{0, 1}
	case MotionEuclidean:
		c, s := math.Cos(p[0]), math.Sin(p[0])
		jx = []float64{-s*x - c*y, 1, 0}
		jy = []float64{c*x - s*y, 0, 1}
		return jx, jy
	case MotionAffine:
		jx = []float64{x, y, 1, 0, 0, 0}
		jy = []float64{0, 0, 0, x, y, 1}
		return jx, jy
	case MotionHomography:
		d := p[6]*x + p[7]*y + 1
		wx := (p[0]*x + p[1]*y + p[2]) / d
		wy := (p[3]*x + p[4]*y + p[5]) / d
		jx = []float64{x / d, y / d, 1 / d, 0, 0, 0, -wx * x / d, -wx * y / d}
		jy = []float64{0, 0, 0, x / d, y / d, 1 / d, -wy * x / d, -wy * y / d}
		return jx, jy
	default:
		panic("video: unknown MotionType")
	}
}

// warpMatrix converts a parameter vector into the OpenCV-style warp matrix: a
// 2x3 matrix for translation/euclidean/affine, or a 3x3 matrix for homography.
func (mt MotionType) warpMatrix(p []float64) [][]float64 {
	switch mt {
	case MotionTranslation:
		return [][]float64{{1, 0, p[0]}, {0, 1, p[1]}}
	case MotionEuclidean:
		c, s := math.Cos(p[0]), math.Sin(p[0])
		return [][]float64{{c, -s, p[1]}, {s, c, p[2]}}
	case MotionAffine:
		return [][]float64{{p[0], p[1], p[2]}, {p[3], p[4], p[5]}}
	case MotionHomography:
		return [][]float64{{p[0], p[1], p[2]}, {p[3], p[4], p[5]}, {p[6], p[7], 1}}
	default:
		panic("video: unknown MotionType")
	}
}

// FindTransformECC estimates the geometric transform that best aligns
// inputImage to templateImage by maximising the Enhanced Correlation Coefficient
// (Evangelidis & Psarakis, 2008), mirroring cv::findTransformECC. It runs the
// forward-additive ECC Gauss-Newton iteration for the chosen motion model,
// returning the achieved correlation coefficient (in [-1, 1], higher is better)
// and the recovered warp matrix that maps template coordinates into the input
// image (2x3 for translation/euclidean/affine, 3x3 for homography).
//
// The warp is initialised to identity unless initWarp is supplied (a matrix of
// the right shape for the motion type). Both images are converted to grayscale.
// Iteration stops when crit is satisfied (its Epsilon compares the increment of
// the correlation coefficient). It panics on empty or mismatched images.
func FindTransformECC(templateImage, inputImage *cv.Mat, initWarp [][]float64, motionType MotionType, crit TermCriteria) (cc float64, warp [][]float64) {
	if templateImage == nil || inputImage == nil || templateImage.Empty() || inputImage.Empty() {
		panic("video: FindTransformECC requires non-empty images")
	}
	if templateImage.Rows != inputImage.Rows || templateImage.Cols != inputImage.Cols {
		panic("video: FindTransformECC requires equal-sized images")
	}
	tGrayMat := toGray(templateImage)
	iGrayMat := toGray(inputImage)
	tmpl := gridFromMat(tGrayMat)
	img := gridFromMat(iGrayMat)
	gx, gy := gradients(iGrayMat)

	n := motionType.numParams()
	var p []float64
	if initWarp != nil {
		p = paramsFromWarp(motionType, initWarp)
	} else {
		p = motionType.identityParams()
	}

	rows, cols := tGrayMat.Rows, tGrayMat.Cols
	// Keep a small margin so warped samples stay well inside the input image.
	const margin = 1
	maxIter := crit.iterCap(50)

	var prevCC float64
	for iter := 0; iter < maxIter; iter++ {
		var (
			tv  []float64   // template intensities
			iv  []float64   // warped input intensities
			sds [][]float64 // steepest-descent vectors (N x n)
		)
		var sumT, sumI float64
		for y := margin; y < rows-margin; y++ {
			for x := margin; x < cols-margin; x++ {
				fx, fy := float64(x), float64(y)
				wx, wy := motionType.warpPoint(p, fx, fy)
				if wx < 0 || wx > float64(cols-1) || wy < 0 || wy > float64(rows-1) {
					continue
				}
				iVal := img.bilinear(wx, wy)
				gxv := gx.bilinear(wx, wy)
				gyv := gy.bilinear(wx, wy)
				jx, jy := motionType.jacobian(p, fx, fy)
				sd := make([]float64, n)
				for k := 0; k < n; k++ {
					sd[k] = gxv*jx[k] + gyv*jy[k]
				}
				tVal := tmpl.bilinear(fx, fy)
				tv = append(tv, tVal)
				iv = append(iv, iVal)
				sds = append(sds, sd)
				sumT += tVal
				sumI += iVal
			}
		}
		N := len(tv)
		if N < n+1 {
			break
		}
		meanT := sumT / float64(N)
		meanI := sumI / float64(N)

		// Zero-mean images and accumulate ECC quantities.
		hess := zeros(n, n)
		imgProj := make([]float64, n)
		tmplProj := make([]float64, n)
		var correlation, imgNorm2, tmplNorm2 float64
		tzm := make([]float64, N)
		izm := make([]float64, N)
		for i := 0; i < N; i++ {
			tzm[i] = tv[i] - meanT
			izm[i] = iv[i] - meanI
			correlation += tzm[i] * izm[i]
			imgNorm2 += izm[i] * izm[i]
			tmplNorm2 += tzm[i] * tzm[i]
			sd := sds[i]
			for a := 0; a < n; a++ {
				imgProj[a] += sd[a] * izm[i]
				tmplProj[a] += sd[a] * tzm[i]
				ha := hess[a]
				for b := a; b < n; b++ {
					ha[b] += sd[a] * sd[b]
				}
			}
		}
		// Symmetrise the Hessian.
		for a := 0; a < n; a++ {
			for b := 0; b < a; b++ {
				hess[a][b] = hess[b][a]
			}
		}
		if imgNorm2 <= 0 || tmplNorm2 <= 0 {
			break
		}
		cc = correlation / math.Sqrt(imgNorm2*tmplNorm2)

		hInv, ok := matInverse(hess)
		if !ok {
			break
		}
		hInvImgProj := matVec(hInv, imgProj)
		lambdaN := imgNorm2 - dot(imgProj, hInvImgProj)
		lambdaD := correlation - dot(tmplProj, hInvImgProj)
		if lambdaD <= 1e-12 {
			// Numerically unable to improve further.
			break
		}
		lambda := lambdaN / lambdaD

		errProj := make([]float64, n)
		for i := 0; i < N; i++ {
			e := lambda*tzm[i] - izm[i]
			sd := sds[i]
			for a := 0; a < n; a++ {
				errProj[a] += sd[a] * e
			}
		}
		deltaP := matVec(hInv, errProj)
		for k := 0; k < n; k++ {
			p[k] += deltaP[k]
		}

		if crit.reached(iter, math.Abs(cc-prevCC)) {
			break
		}
		prevCC = cc
	}

	return cc, motionType.warpMatrix(p)
}

// paramsFromWarp extracts a parameter vector from a supplied warp matrix.
func paramsFromWarp(mt MotionType, w [][]float64) []float64 {
	switch mt {
	case MotionTranslation:
		return []float64{w[0][2], w[1][2]}
	case MotionEuclidean:
		theta := math.Atan2(w[1][0], w[0][0])
		return []float64{theta, w[0][2], w[1][2]}
	case MotionAffine:
		return []float64{w[0][0], w[0][1], w[0][2], w[1][0], w[1][1], w[1][2]}
	case MotionHomography:
		return []float64{w[0][0], w[0][1], w[0][2], w[1][0], w[1][1], w[1][2], w[2][0], w[2][1]}
	default:
		panic("video: unknown MotionType")
	}
}

// dot returns the inner product of two equal-length vectors.
func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

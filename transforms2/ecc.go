package transforms2

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// MotionModel selects the parametric warp family estimated by
// [FindTransformECC].
type MotionModel int

const (
	// MotionTranslation estimates a pure translation (2 parameters).
	MotionTranslation MotionModel = iota
	// MotionEuclidean estimates a rigid rotation plus translation
	// (3 parameters).
	MotionEuclidean
	// MotionAffine estimates a full affine transform (6 parameters).
	MotionAffine
)

// ECCCriteria controls the [FindTransformECC] iteration: it stops after
// MaxCount iterations or once the parameter update magnitude falls below
// Epsilon, whichever comes first.
type ECCCriteria struct {
	// MaxCount is the maximum number of Gauss-Newton iterations.
	MaxCount int
	// Epsilon is the convergence threshold on the parameter update norm.
	Epsilon float64
}

// DefaultECCCriteria returns criteria suitable for typical registration tasks
// (200 iterations, epsilon 1e-6).
func DefaultECCCriteria() ECCCriteria {
	return ECCCriteria{MaxCount: 200, Epsilon: 1e-6}
}

// transforms2gray returns the per-pixel channel average of m as a float slice
// in row-major order.
func transforms2gray(m *cv.Mat) []float64 {
	out := make([]float64, m.Rows*m.Cols)
	for i := 0; i < m.Rows*m.Cols; i++ {
		var s float64
		base := i * m.Channels
		for c := 0; c < m.Channels; c++ {
			s += float64(m.Data[base+c])
		}
		out[i] = s / float64(m.Channels)
	}
	return out
}

// transforms2bilinearF bilinearly samples a float image with edge replication.
func transforms2bilinearF(data []float64, w, h int, x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	dx := x - float64(x0)
	dy := y - float64(y0)
	clamp := func(v, n int) int {
		if v < 0 {
			return 0
		}
		if v >= n {
			return n - 1
		}
		return v
	}
	x1 := clamp(x0+1, w)
	y1 := clamp(y0+1, h)
	x0 = clamp(x0, w)
	y0 = clamp(y0, h)
	v00 := data[y0*w+x0]
	v01 := data[y0*w+x1]
	v10 := data[y1*w+x0]
	v11 := data[y1*w+x1]
	return (v00*(1-dx)+v01*dx)*(1-dy) + (v10*(1-dx)+v11*dx)*dy
}

// EnhancedCorrelationCoefficient returns the zero-mean normalised correlation
// coefficient between a and b (each flattened over all samples), a value in
// [-1, 1] where 1 means perfect linear agreement. a and b must have identical
// dimensions and channel counts. It returns 0 if either has zero variance. It
// panics on a dimension mismatch.
func EnhancedCorrelationCoefficient(a, b *cv.Mat) float64 {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("transforms2: EnhancedCorrelationCoefficient dimension mismatch")
	}
	n := len(a.Data)
	var ma, mb float64
	for i := 0; i < n; i++ {
		ma += float64(a.Data[i])
		mb += float64(b.Data[i])
	}
	ma /= float64(n)
	mb /= float64(n)
	var num, da, db float64
	for i := 0; i < n; i++ {
		av := float64(a.Data[i]) - ma
		bv := float64(b.Data[i]) - mb
		num += av * bv
		da += av * av
		db += bv * bv
	}
	if da == 0 || db == 0 {
		return 0
	}
	return num / math.Sqrt(da*db)
}

// FindTransformECC estimates the geometric transform that best aligns image to
// template by maximising the Enhanced Correlation Coefficient (Evangelidis &
// Psarakis, 2008). The returned affine matrix W maps template coordinates to
// image coordinates, so that resampling image through W reproduces template.
// The second return value is the final correlation coefficient. The images may
// be multi-channel (they are reduced to intensity) but must be non-empty. It
// returns an error if the linear system becomes singular before convergence.
func FindTransformECC(template, image *cv.Mat, model MotionModel, criteria ECCCriteria) (cv.AffineMatrix, float64, error) {
	tR, tC := template.Rows, template.Cols
	iR, iC := image.Rows, image.Cols
	tGray := transforms2gray(template)
	iGray := transforms2gray(image)

	// Central-difference gradients of the image.
	gxI := make([]float64, iR*iC)
	gyI := make([]float64, iR*iC)
	for y := 0; y < iR; y++ {
		for x := 0; x < iC; x++ {
			xm := x - 1
			if xm < 0 {
				xm = 0
			}
			xp := x + 1
			if xp >= iC {
				xp = iC - 1
			}
			ym := y - 1
			if ym < 0 {
				ym = 0
			}
			yp := y + 1
			if yp >= iR {
				yp = iR - 1
			}
			gxI[y*iC+x] = 0.5 * (iGray[y*iC+xp] - iGray[y*iC+xm])
			gyI[y*iC+x] = 0.5 * (iGray[yp*iC+x] - iGray[ym*iC+x])
		}
	}

	var np int
	var p []float64
	switch model {
	case MotionTranslation:
		np, p = 2, []float64{0, 0}
	case MotionEuclidean:
		np, p = 3, []float64{0, 0, 0}
	case MotionAffine:
		np, p = 6, []float64{1, 0, 0, 0, 1, 0}
	default:
		panic("transforms2: FindTransformECC unknown motion model")
	}

	buildW := func(p []float64) cv.AffineMatrix {
		switch model {
		case MotionTranslation:
			return cv.AffineMatrix{1, 0, p[0], 0, 1, p[1]}
		case MotionEuclidean:
			c, s := math.Cos(p[0]), math.Sin(p[0])
			return cv.AffineMatrix{c, -s, p[1], s, c, p[2]}
		default:
			return cv.AffineMatrix{p[0], p[1], p[2], p[3], p[4], p[5]}
		}
	}

	var rho float64
	for iter := 0; iter < criteria.MaxCount; iter++ {
		w := buildW(p)
		var (
			sds        [][]float64
			tv, wv     []float64
			sumT, sumW float64
		)
		theta := 0.0
		if model == MotionEuclidean {
			theta = p[0]
		}
		sinT, cosT := math.Sin(theta), math.Cos(theta)
		for y := 0; y < tR; y++ {
			for x := 0; x < tC; x++ {
				fx := float64(x)
				fy := float64(y)
				ix := w[0]*fx + w[1]*fy + w[2]
				iy := w[3]*fx + w[4]*fy + w[5]
				if ix < 0 || ix > float64(iC-1) || iy < 0 || iy > float64(iR-1) {
					continue
				}
				gx := transforms2bilinearF(gxI, iC, iR, ix, iy)
				gy := transforms2bilinearF(gyI, iC, iR, ix, iy)
				var sd []float64
				switch model {
				case MotionTranslation:
					sd = []float64{gx, gy}
				case MotionEuclidean:
					sd = []float64{gx*(-sinT*fx-cosT*fy) + gy*(cosT*fx-sinT*fy), gx, gy}
				default:
					sd = []float64{gx * fx, gx * fy, gx, gy * fx, gy * fy, gy}
				}
				t := tGray[y*tC+x]
				iw := transforms2bilinearF(iGray, iC, iR, ix, iy)
				sds = append(sds, sd)
				tv = append(tv, t)
				wv = append(wv, iw)
				sumT += t
				sumW += iw
			}
		}
		m := len(sds)
		if m < np+1 {
			return w, rho, errors.New("transforms2: FindTransformECC insufficient overlap")
		}
		meanT := sumT / float64(m)
		meanW := sumW / float64(m)
		// Hessian and projections.
		H := make([][]float64, np)
		for i := range H {
			H[i] = make([]float64, np)
		}
		gt := make([]float64, np)
		gw := make([]float64, np)
		var dotTW, normW2, normT2 float64
		for k := 0; k < m; k++ {
			sd := sds[k]
			tzm := tv[k] - meanT
			wzm := wv[k] - meanW
			dotTW += tzm * wzm
			normW2 += wzm * wzm
			normT2 += tzm * tzm
			for i := 0; i < np; i++ {
				gt[i] += sd[i] * tzm
				gw[i] += sd[i] * wzm
				for j := i; j < np; j++ {
					H[i][j] += sd[i] * sd[j]
				}
			}
		}
		for i := 0; i < np; i++ {
			for j := 0; j < i; j++ {
				H[i][j] = H[j][i]
			}
		}
		rho = 0
		if normT2 > 0 && normW2 > 0 {
			rho = dotTW / math.Sqrt(normT2*normW2)
		}
		// a = H^{-1} gw
		Hcopy := transforms2copyMat(H)
		a, ok := transforms2solve(Hcopy, append([]float64(nil), gw...))
		if !ok {
			return w, rho, errors.New("transforms2: FindTransformECC singular Hessian")
		}
		num := normW2 - transforms2dot(gw, a)
		den := dotTW - transforms2dot(gt, a)
		if math.Abs(den) < 1e-12 {
			break
		}
		lambda := num / den
		rhs := make([]float64, np)
		for i := 0; i < np; i++ {
			rhs[i] = lambda*gt[i] - gw[i]
		}
		dp, ok := transforms2solve(transforms2copyMat(H), rhs)
		if !ok {
			return w, rho, errors.New("transforms2: FindTransformECC singular Hessian")
		}
		for i := 0; i < np; i++ {
			p[i] += dp[i]
		}
		if transforms2norm(dp) < criteria.Epsilon {
			break
		}
	}
	return buildW(p), rho, nil
}

// transforms2copyMat deep-copies a square matrix.
func transforms2copyMat(a [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = append([]float64(nil), a[i]...)
	}
	return out
}

// transforms2dot returns the dot product of two equal-length vectors.
func transforms2dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// transforms2norm returns the Euclidean norm of v.
func transforms2norm(v []float64) float64 {
	return math.Sqrt(transforms2dot(v, v))
}

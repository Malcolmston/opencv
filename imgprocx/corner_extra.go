package imgprocx

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// EigenValsVecs holds the eigen-decomposition of the 2×2 structure tensor at one
// pixel, as produced by [CornerEigenValsAndVecs]. Lambda1 >= Lambda2 are the
// eigenvalues and (X1, Y1), (X2, Y2) the corresponding unit eigenvectors.
type EigenValsVecs struct {
	Lambda1, Lambda2 float64
	X1, Y1           float64
	X2, Y2           float64
}

// structureTensor computes the box-summed products of the Sobel derivatives of
// gray over a blockSize×blockSize window, returning the three tensor components
// sxx = Σ dx², syy = Σ dy², sxy = Σ dx·dy as flat planes. ksize is the Sobel
// aperture and blockSize the (odd or even) averaging window; borders replicate.
func structureTensor(gray []float64, rows, cols, blockSize, ksize int) (sxx, syy, sxy []float64) {
	dx := derivPlane(gray, rows, cols, 1, 0, ksize)
	dy := derivPlane(gray, rows, cols, 0, 1, ksize)
	xx := make([]float64, rows*cols)
	yy := make([]float64, rows*cols)
	xy := make([]float64, rows*cols)
	for i := range dx {
		xx[i] = dx[i] * dx[i]
		yy[i] = dy[i] * dy[i]
		xy[i] = dx[i] * dy[i]
	}
	sxx = boxSum(xx, rows, cols, blockSize)
	syy = boxSum(yy, rows, cols, blockSize)
	sxy = boxSum(xy, rows, cols, blockSize)
	return sxx, syy, sxy
}

// boxSum returns, for every pixel, the unnormalised sum of plane over the
// block×block window centred on it, replicating the border.
func boxSum(plane []float64, rows, cols, block int) []float64 {
	if block <= 0 {
		panic("imgprocx: corner window (blockSize) must be positive")
	}
	half := block / 2
	out := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var s float64
			for dy := -half; dy <= block-1-half; dy++ {
				sy := clampIndex(y+dy, rows)
				for dx := -half; dx <= block-1-half; dx++ {
					sx := clampIndex(x+dx, cols)
					s += plane[sy*cols+sx]
				}
			}
			out[y*cols+x] = s
		}
	}
	return out
}

// eig2x2 returns the eigenvalues (l1 >= l2) and unit eigenvectors of the
// symmetric matrix [[a, b], [b, c]].
func eig2x2(a, b, c float64) (l1, l2, x1, y1, x2, y2 float64) {
	tr := a + c
	d := math.Sqrt((a-c)*(a-c)/4 + b*b)
	l1 = tr/2 + d
	l2 = tr/2 - d
	// Eigenvector for l1: solve (a-l1)·x + b·y = 0.
	x1, y1 = eigVec(a, b, c, l1)
	x2, y2 = eigVec(a, b, c, l2)
	return l1, l2, x1, y1, x2, y2
}

// eigVec returns a unit eigenvector of [[a,b],[b,c]] for eigenvalue l.
func eigVec(a, b, c, l float64) (x, y float64) {
	// (a-l)x + b y = 0  =>  direction (b, l-a); fall back to (l-c, b).
	vx, vy := b, l-a
	if math.Abs(vx) < 1e-12 && math.Abs(vy) < 1e-12 {
		vx, vy = l-c, b
	}
	n := math.Hypot(vx, vy)
	if n < 1e-12 {
		return 1, 0
	}
	return vx / n, vy / n
}

// CornerEigenValsAndVecs computes, for every pixel of a single-channel image,
// the eigenvalues and eigenvectors of the local structure tensor, mirroring
// cv2.cornerEigenValsAndVecs. blockSize is the side of the averaging window and
// ksize the Sobel aperture used for the derivatives; both must be positive
// (ksize odd). It returns a rows×cols grid of [EigenValsVecs]. It panics if src
// is not single-channel.
//
// The structure tensor at a pixel is the window sum of [[dx², dx·dy],
// [dx·dy, dy²]]; its two eigenvalues describe the local intensity variation
// (two large eigenvalues indicate a corner, one indicates an edge), which is the
// information the Harris and Shi–Tomasi detectors reduce to a single score.
func CornerEigenValsAndVecs(src *cv.Mat, blockSize, ksize int) [][]EigenValsVecs {
	requireSingleChannel(src, "CornerEigenValsAndVecs")
	gray, rows, cols := toGrayPlane(src)
	sxx, syy, sxy := structureTensor(gray, rows, cols, blockSize, ksize)
	out := make([][]EigenValsVecs, rows)
	for y := 0; y < rows; y++ {
		out[y] = make([]EigenValsVecs, cols)
		for x := 0; x < cols; x++ {
			i := y*cols + x
			l1, l2, x1, y1, x2, y2 := eig2x2(sxx[i], sxy[i], syy[i])
			out[y][x] = EigenValsVecs{Lambda1: l1, Lambda2: l2, X1: x1, Y1: y1, X2: x2, Y2: y2}
		}
	}
	return out
}

// CornerMinEigenVal computes the smaller eigenvalue of the local structure
// tensor at every pixel of a single-channel image, mirroring
// cv2.cornerMinEigenVal. It is the Shi–Tomasi corner strength: pixels whose
// minimum eigenvalue is large are good features to track. blockSize is the
// averaging window and ksize the Sobel aperture. The result is returned as a
// [cv.FloatMat]. It panics if src is not single-channel.
func CornerMinEigenVal(src *cv.Mat, blockSize, ksize int) *cv.FloatMat {
	requireSingleChannel(src, "CornerMinEigenVal")
	gray, rows, cols := toGrayPlane(src)
	sxx, syy, sxy := structureTensor(gray, rows, cols, blockSize, ksize)
	out := cv.NewFloatMat(rows, cols)
	for i := range sxx {
		_, l2, _, _, _, _ := eig2x2(sxx[i], sxy[i], syy[i])
		out.Data[i] = l2
	}
	return out
}

// PreCornerDetect computes the corner-feature map used to pre-select corner
// candidates, mirroring cv2.preCornerDetect. For each pixel of a single-channel
// image it evaluates
//
//	Dx²·Dyy + Dy²·Dxx - 2·Dx·Dy·Dxy
//
// from the first (Dx, Dy) and second (Dxx, Dyy, Dxy) Sobel derivatives of
// aperture ksize. The result, returned as a [cv.FloatMat], has extrema at corner
// locations. It panics if src is not single-channel.
func PreCornerDetect(src *cv.Mat, ksize int) *cv.FloatMat {
	requireSingleChannel(src, "PreCornerDetect")
	gray, rows, cols := toGrayPlane(src)
	dx := derivPlane(gray, rows, cols, 1, 0, ksize)
	dy := derivPlane(gray, rows, cols, 0, 1, ksize)
	dxx := derivPlane(gray, rows, cols, 2, 0, ksize)
	dyy := derivPlane(gray, rows, cols, 0, 2, ksize)
	dxy := derivPlane(gray, rows, cols, 1, 1, ksize)
	out := cv.NewFloatMat(rows, cols)
	for i := range dx {
		out.Data[i] = dx[i]*dx[i]*dyy[i] + dy[i]*dy[i]*dxx[i] - 2*dx[i]*dy[i]*dxy[i]
	}
	return out
}

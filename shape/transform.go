package shape

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ShapeTransformer is the common interface of the geometric shape transformers
// in this package, mirroring OpenCV's cv::ShapeTransformer. An implementation is
// first fitted to a pair of equally sized, index-corresponding point sets with
// EstimateTransformation (source[i] maps to target[i]); it can then map arbitrary
// points with ApplyTransformation and resample an image through the estimated
// mapping with WarpImage.
type ShapeTransformer interface {
	// EstimateTransformation fits the transform so that each source point maps to
	// the corresponding target point. The two slices must have equal, non-zero
	// length.
	EstimateTransformation(source, target []Point2D)
	// ApplyTransformation maps points through the fitted transform.
	ApplyTransformation(pts []Point2D) []Point2D
	// WarpImage resamples src through the fitted transform, returning a new image
	// the same size as src.
	WarpImage(src *cv.Mat) *cv.Mat
}

// ---------------------------------------------------------------------------
// Thin-plate spline
// ---------------------------------------------------------------------------

// ThinPlateSplineShapeTransformer warps the plane with a thin-plate spline, the
// minimum-bending-energy interpolant through a set of control-point
// correspondences, mirroring OpenCV's cv::ThinPlateSplineShapeTransformer.
//
// After [ThinPlateSplineShapeTransformer.EstimateTransformation] the transform
// interpolates the control points exactly when Regularization is zero; a
// positive Regularization relaxes the interpolation into a smoothing spline that
// trades off fidelity against bending energy. The bending energy of the fitted
// non-affine part is available from
// [ThinPlateSplineShapeTransformer.BendingEnergy].
//
// The zero value is ready to use (interpolating spline). Use
// [NewThinPlateSplineShapeTransformer] to set a regularization parameter.
type ThinPlateSplineShapeTransformer struct {
	// Regularization is the smoothing parameter λ added to the spline kernel's
	// diagonal. Zero gives exact interpolation.
	Regularization float64

	fitted  bool
	control []Point2D
	// Forward coefficients map source → target (weights wx/wy and affine ax/ay).
	wx, wy []float64
	ax, ay [3]float64
	// Inverse coefficients map target → source, used by WarpImage.
	invControl []Point2D
	iwx, iwy   []float64
	iax, iay   [3]float64
	invFitted  bool
	bending    float64
}

// NewThinPlateSplineShapeTransformer returns a thin-plate spline transformer
// with the given regularization parameter (0 for exact interpolation).
func NewThinPlateSplineShapeTransformer(regularization float64) *ThinPlateSplineShapeTransformer {
	return &ThinPlateSplineShapeTransformer{Regularization: regularization}
}

// tpsU is the thin-plate spline radial basis U(r) = r²·log(r²); U(0) = 0.
func tpsU(r2 float64) float64 {
	if r2 <= 1e-12 {
		return 0
	}
	return r2 * math.Log(r2)
}

// solveTPS fits a thin-plate spline mapping control points src to values given by
// dst, returning the per-control weights (for x and y) and the 3-vector affine
// parts. It reports false when the linear system is singular.
func solveTPS(src, dst []Point2D, lambda float64) (wx, wy []float64, ax, ay [3]float64, ok bool) {
	n := len(src)
	dim := n + 3
	a := make([][]float64, dim)
	for i := range a {
		a[i] = make([]float64, dim)
	}
	b := make([][]float64, dim)
	for i := range b {
		b[i] = make([]float64, 2)
	}
	// K block with regularization on the diagonal, plus P and Pᵀ blocks.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			r2 := sq(src[i].X-src[j].X) + sq(src[i].Y-src[j].Y)
			a[i][j] = tpsU(r2)
		}
		a[i][i] += lambda
		a[i][n] = 1
		a[i][n+1] = src[i].X
		a[i][n+2] = src[i].Y
		a[n][i] = 1
		a[n+1][i] = src[i].X
		a[n+2][i] = src[i].Y
		b[i][0] = dst[i].X
		b[i][1] = dst[i].Y
	}
	sol, ok := solveLinearSystem(a, b)
	if !ok {
		return nil, nil, ax, ay, false
	}
	wx = make([]float64, n)
	wy = make([]float64, n)
	for i := 0; i < n; i++ {
		wx[i] = sol[i][0]
		wy[i] = sol[i][1]
	}
	ax = [3]float64{sol[n][0], sol[n+1][0], sol[n+2][0]}
	ay = [3]float64{sol[n][1], sol[n+1][1], sol[n+2][1]}
	return wx, wy, ax, ay, true
}

// EstimateTransformation fits the thin-plate spline so that source[i] maps to
// target[i]. It panics if the slices differ in length, are empty, or the control
// configuration is degenerate (all control points coincident).
func (t *ThinPlateSplineShapeTransformer) EstimateTransformation(source, target []Point2D) {
	if len(source) != len(target) {
		panic("shape: ThinPlateSplineShapeTransformer.EstimateTransformation length mismatch")
	}
	if len(source) == 0 {
		panic("shape: ThinPlateSplineShapeTransformer.EstimateTransformation empty control set")
	}
	wx, wy, ax, ay, ok := solveTPS(source, target, t.Regularization)
	if !ok {
		panic("shape: ThinPlateSplineShapeTransformer.EstimateTransformation degenerate control points")
	}
	t.control = append(t.control[:0:0], source...)
	t.wx, t.wy, t.ax, t.ay = wx, wy, ax, ay
	t.fitted = true

	// Bending energy of the non-affine part: wᵀ K w summed over both coordinates.
	n := len(source)
	var be float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			r2 := sq(source[i].X-source[j].X) + sq(source[i].Y-source[j].Y)
			k := tpsU(r2)
			be += k * (wx[i]*wx[j] + wy[i]*wy[j])
		}
	}
	t.bending = math.Abs(be)

	// Inverse spline (target → source) for image warping; ignore failure and let
	// WarpImage report an unfitted inverse by returning a copy.
	iwx, iwy, iax, iay, iok := solveTPS(target, source, t.Regularization)
	if iok {
		t.invControl = append(t.invControl[:0:0], target...)
		t.iwx, t.iwy, t.iax, t.iay = iwx, iwy, iax, iay
		t.invFitted = true
	} else {
		t.invFitted = false
	}
}

// tpsMap applies a fitted spline (control, weights, affine) to a single point.
func tpsMap(control []Point2D, wx, wy []float64, ax, ay [3]float64, p Point2D) Point2D {
	fx := ax[0] + ax[1]*p.X + ax[2]*p.Y
	fy := ay[0] + ay[1]*p.X + ay[2]*p.Y
	for i, c := range control {
		r2 := sq(p.X-c.X) + sq(p.Y-c.Y)
		u := tpsU(r2)
		fx += wx[i] * u
		fy += wy[i] * u
	}
	return Point2D{X: fx, Y: fy}
}

// ApplyTransformation maps points through the fitted spline. It panics if called
// before EstimateTransformation.
func (t *ThinPlateSplineShapeTransformer) ApplyTransformation(pts []Point2D) []Point2D {
	if !t.fitted {
		panic("shape: ThinPlateSplineShapeTransformer.ApplyTransformation before EstimateTransformation")
	}
	out := make([]Point2D, len(pts))
	for i, p := range pts {
		out[i] = tpsMap(t.control, t.wx, t.wy, t.ax, t.ay, p)
	}
	return out
}

// BendingEnergy returns the bending energy of the fitted spline's non-affine
// part: zero for a purely affine correspondence and larger for more warping. It
// panics if called before EstimateTransformation.
func (t *ThinPlateSplineShapeTransformer) BendingEnergy() float64 {
	if !t.fitted {
		panic("shape: ThinPlateSplineShapeTransformer.BendingEnergy before EstimateTransformation")
	}
	return t.bending
}

// WarpImage resamples src through the fitted spline into a new image of the same
// size. Each destination pixel is mapped back to the source frame through the
// inverse spline and sampled bilinearly, with out-of-range samples read as
// zero. It panics if called before EstimateTransformation.
func (t *ThinPlateSplineShapeTransformer) WarpImage(src *cv.Mat) *cv.Mat {
	if !t.fitted {
		panic("shape: ThinPlateSplineShapeTransformer.WarpImage before EstimateTransformation")
	}
	dst := cv.NewMat(src.Rows, src.Cols, src.Channels)
	if !t.invFitted {
		copy(dst.Data, src.Data)
		return dst
	}
	for y := 0; y < dst.Rows; y++ {
		for x := 0; x < dst.Cols; x++ {
			s := tpsMap(t.invControl, t.iwx, t.iwy, t.iax, t.iay, Point2D{X: float64(x), Y: float64(y)})
			sampleBilinear(src, dst, s.X, s.Y, y, x)
		}
	}
	return dst
}

// ---------------------------------------------------------------------------
// Affine
// ---------------------------------------------------------------------------

// AffineTransformer maps the plane with a single affine (or, optionally,
// similarity) transform fitted by least squares to point correspondences,
// mirroring OpenCV's cv::AffineTransformer. When FullAffine is true the fit is a
// general 2×3 affine (six degrees of freedom); when false it is a similarity
// transform (uniform scale, rotation and translation, four degrees of freedom)
// estimated in closed form.
//
// The zero value is ready to use as a full-affine transformer. Use
// [NewAffineTransformer] to choose the similarity variant.
type AffineTransformer struct {
	// FullAffine selects a general affine fit (true) or a similarity fit (false).
	FullAffine bool

	fitted bool
	m      [2][3]float64 // forward matrix
	inv    [2][3]float64 // inverse matrix for warping
	invOK  bool
}

// NewAffineTransformer returns an affine transformer. When fullAffine is false it
// fits a similarity transform instead of a general affine one.
func NewAffineTransformer(fullAffine bool) *AffineTransformer {
	return &AffineTransformer{FullAffine: fullAffine}
}

// EstimateTransformation fits the affine (or similarity) transform mapping
// source[i] to target[i] in the least-squares sense. It panics on a length
// mismatch or an empty correspondence set.
func (t *AffineTransformer) EstimateTransformation(source, target []Point2D) {
	if len(source) != len(target) {
		panic("shape: AffineTransformer.EstimateTransformation length mismatch")
	}
	if len(source) == 0 {
		panic("shape: AffineTransformer.EstimateTransformation empty correspondence set")
	}
	if t.FullAffine {
		t.m = fitFullAffine(source, target)
	} else {
		t.m = fitSimilarity(source, target)
	}
	t.inv, t.invOK = invertAffine(t.m)
	t.fitted = true
}

// fitFullAffine solves the six-parameter affine least-squares fit.
func fitFullAffine(src, dst []Point2D) [2][3]float64 {
	// Normal equations for [a b c] mapping (x,y,1) → x' and (x,y,1) → y'.
	var ata [3][3]float64
	var atbx, atby [3]float64
	for i := range src {
		row := [3]float64{src[i].X, src[i].Y, 1}
		for a := 0; a < 3; a++ {
			for b := 0; b < 3; b++ {
				ata[a][b] += row[a] * row[b]
			}
			atbx[a] += row[a] * dst[i].X
			atby[a] += row[a] * dst[i].Y
		}
	}
	inv, ok := inv3(ata)
	if !ok {
		// Degenerate configuration: fall back to a pure translation of centroids.
		return translationOnly(src, dst)
	}
	var m [2][3]float64
	for a := 0; a < 3; a++ {
		for k := 0; k < 3; k++ {
			m[0][a] += inv[a][k] * atbx[k]
			m[1][a] += inv[a][k] * atby[k]
		}
	}
	return m
}

// fitSimilarity estimates a uniform-scale rotation plus translation in closed
// form (the 2-D Umeyama solution without reflection).
func fitSimilarity(src, dst []Point2D) [2][3]float64 {
	n := float64(len(src))
	var msx, msy, mdx, mdy float64
	for i := range src {
		msx += src[i].X
		msy += src[i].Y
		mdx += dst[i].X
		mdy += dst[i].Y
	}
	msx, msy, mdx, mdy = msx/n, msy/n, mdx/n, mdy/n
	var sxx, syy, sxy, syx, varS float64
	for i := range src {
		dx := src[i].X - msx
		dy := src[i].Y - msy
		ex := dst[i].X - mdx
		ey := dst[i].Y - mdy
		sxx += ex * dx
		syy += ey * dy
		sxy += ex * dy
		syx += ey * dx
		varS += dx*dx + dy*dy
	}
	// Rotation+scale from the cross-covariance (a b; -b a) closed form.
	a := sxx + syy
	b := syx - sxy
	denom := varS
	if denom < 1e-12 {
		return translationOnly(src, dst)
	}
	ca := a / denom
	sa := b / denom
	var m [2][3]float64
	m[0][0] = ca
	m[0][1] = -sa
	m[1][0] = sa
	m[1][1] = ca
	m[0][2] = mdx - (ca*msx - sa*msy)
	m[1][2] = mdy - (sa*msx + ca*msy)
	return m
}

// translationOnly returns the pure translation matching the centroids.
func translationOnly(src, dst []Point2D) [2][3]float64 {
	n := float64(len(src))
	var msx, msy, mdx, mdy float64
	for i := range src {
		msx += src[i].X
		msy += src[i].Y
		mdx += dst[i].X
		mdy += dst[i].Y
	}
	var m [2][3]float64
	m[0][0], m[1][1] = 1, 1
	m[0][2] = (mdx - msx) / n
	m[1][2] = (mdy - msy) / n
	return m
}

// invertAffine returns the inverse of a 2×3 affine matrix, reporting false when
// the linear part is singular.
func invertAffine(m [2][3]float64) ([2][3]float64, bool) {
	det := m[0][0]*m[1][1] - m[0][1]*m[1][0]
	if math.Abs(det) < 1e-12 {
		return [2][3]float64{}, false
	}
	invDet := 1 / det
	var out [2][3]float64
	out[0][0] = m[1][1] * invDet
	out[0][1] = -m[0][1] * invDet
	out[1][0] = -m[1][0] * invDet
	out[1][1] = m[0][0] * invDet
	out[0][2] = -(out[0][0]*m[0][2] + out[0][1]*m[1][2])
	out[1][2] = -(out[1][0]*m[0][2] + out[1][1]*m[1][2])
	return out, true
}

// Matrix returns the fitted forward affine matrix as a 2×3 array (rows are the x
// and y output equations). It panics if called before EstimateTransformation.
func (t *AffineTransformer) Matrix() [2][3]float64 {
	if !t.fitted {
		panic("shape: AffineTransformer.Matrix before EstimateTransformation")
	}
	return t.m
}

// ApplyTransformation maps points through the fitted affine transform. It panics
// if called before EstimateTransformation.
func (t *AffineTransformer) ApplyTransformation(pts []Point2D) []Point2D {
	if !t.fitted {
		panic("shape: AffineTransformer.ApplyTransformation before EstimateTransformation")
	}
	out := make([]Point2D, len(pts))
	for i, p := range pts {
		out[i] = Point2D{
			X: t.m[0][0]*p.X + t.m[0][1]*p.Y + t.m[0][2],
			Y: t.m[1][0]*p.X + t.m[1][1]*p.Y + t.m[1][2],
		}
	}
	return out
}

// WarpImage resamples src through the fitted affine transform into a new image of
// the same size, using bilinear sampling and a zero border. It panics if called
// before EstimateTransformation.
func (t *AffineTransformer) WarpImage(src *cv.Mat) *cv.Mat {
	if !t.fitted {
		panic("shape: AffineTransformer.WarpImage before EstimateTransformation")
	}
	dst := cv.NewMat(src.Rows, src.Cols, src.Channels)
	if !t.invOK {
		copy(dst.Data, src.Data)
		return dst
	}
	for y := 0; y < dst.Rows; y++ {
		for x := 0; x < dst.Cols; x++ {
			sx := t.inv[0][0]*float64(x) + t.inv[0][1]*float64(y) + t.inv[0][2]
			sy := t.inv[1][0]*float64(x) + t.inv[1][1]*float64(y) + t.inv[1][2]
			sampleBilinear(src, dst, sx, sy, y, x)
		}
	}
	return dst
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

// sq returns v*v.
func sq(v float64) float64 { return v * v }

// sampleBilinear reads the source pixel at fractional (sx, sy) with bilinear
// interpolation (zero border) and writes it to dst pixel (dy, dx).
func sampleBilinear(src, dst *cv.Mat, sx, sy float64, dy, dx int) {
	x0 := int(math.Floor(sx))
	y0 := int(math.Floor(sy))
	fx := sx - float64(x0)
	fy := sy - float64(y0)
	for c := 0; c < src.Channels; c++ {
		v00 := srcSample(src, y0, x0, c)
		v01 := srcSample(src, y0, x0+1, c)
		v10 := srcSample(src, y0+1, x0, c)
		v11 := srcSample(src, y0+1, x0+1, c)
		top := v00*(1-fx) + v01*fx
		bot := v10*(1-fx) + v11*fx
		val := top*(1-fy) + bot*fy
		dst.Data[(dy*dst.Cols+dx)*dst.Channels+c] = clampToByte(val)
	}
}

// srcSample returns sample (y, x, c) of src, or 0 outside the image.
func srcSample(src *cv.Mat, y, x, c int) float64 {
	if y < 0 || y >= src.Rows || x < 0 || x >= src.Cols {
		return 0
	}
	return float64(src.Data[(y*src.Cols+x)*src.Channels+c])
}

// clampToByte rounds and clamps v to the 0–255 uint8 range.
func clampToByte(v float64) uint8 {
	r := math.Round(v)
	if r < 0 {
		return 0
	}
	if r > 255 {
		return 255
	}
	return uint8(r)
}

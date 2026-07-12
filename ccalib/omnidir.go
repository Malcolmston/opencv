package ccalib

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Omnidir is the package's entry point to the omnidirectional (fisheye and
// catadioptric) camera-model routines. It is the Go rendering of OpenCV's
// cv::omnidir namespace: call its methods as Omnidir.ProjectPoints, and so on.
//
// The model is the unified (Mei / Scaramuzza) sphere model. A camera-frame
// point is projected onto the unit sphere, the sphere centre is shifted by the
// mirror parameter xi along the optical axis, the shifted point is projected to
// the normalized image plane, distorted with a Brown–Conrady term, and finally
// mapped to pixels through the intrinsic matrix K. This single model captures
// wide field-of-view lenses (xi ≈ 0 approaches a pinhole; xi ≈ 1 is a typical
// fisheye) that the pinhole model in calib3d cannot represent.
var Omnidir = omnidirNS{}

// omnidirNS is the receiver type behind the [Omnidir] namespace value. It is
// stateless; every method takes the camera parameters explicitly.
type omnidirNS struct{}

// RectifyFlag selects the target projection used when undistorting an
// omnidirectional image to a conventional one.
type RectifyFlag int

const (
	// RectifyPerspective maps the omnidirectional image onto a pinhole
	// (rectilinear) image. Only rays with a forward-facing component survive;
	// the field of view is therefore limited to well under 180°.
	RectifyPerspective RectifyFlag = iota
	// RectifyCylindrical maps the image onto a cylinder wrapped around the
	// vertical axis, preserving a full horizontal field of view.
	RectifyCylindrical
	// RectifyLongLat maps the image onto a longitude–latitude (equirectangular)
	// grid over the sphere.
	RectifyLongLat
)

// OmniModel bundles the intrinsic parameters of an omnidirectional camera: the
// pinhole intrinsics (focal lengths, principal point and skew), the mirror
// parameter Xi and the Brown–Conrady distortion terms K1, K2 (radial) and P1,
// P2 (tangential). The zero value is not a valid camera.
type OmniModel struct {
	Fx   float64
	Fy   float64
	Cx   float64
	Cy   float64
	Skew float64
	Xi   float64
	K1   float64
	K2   float64
	P1   float64
	P2   float64
}

// K returns the 3×3 intrinsic matrix of the model in row-major order.
func (m OmniModel) K() [3][3]float64 {
	return [3][3]float64{
		{m.Fx, m.Skew, m.Cx},
		{0, m.Fy, m.Cy},
		{0, 0, 1},
	}
}

// Dist returns the distortion coefficients [K1, K2, P1, P2].
func (m OmniModel) Dist() []float64 { return []float64{m.K1, m.K2, m.P1, m.P2} }

// NewOmniModel builds an [OmniModel] from a 3×3 intrinsic matrix, the mirror
// parameter xi and a distortion slice ordered [K1, K2, P1, P2] (a nil or short
// slice leaves the trailing coefficients zero).
func NewOmniModel(k [3][3]float64, xi float64, dist []float64) OmniModel {
	k1, k2, p1, p2 := distParams(dist)
	return OmniModel{
		Fx: k[0][0], Fy: k[1][1], Cx: k[0][2], Cy: k[1][2], Skew: k[0][1],
		Xi: xi, K1: k1, K2: k2, P1: p1, P2: p2,
	}
}

// distParams unpacks a distortion slice into named radial/tangential terms
// following the [K1, K2, P1, P2] ordering. Missing entries are zero.
func distParams(dist []float64) (k1, k2, p1, p2 float64) {
	get := func(i int) float64 {
		if i < len(dist) {
			return dist[i]
		}
		return 0
	}
	return get(0), get(1), get(2), get(3)
}

// distortNormalized applies the Brown–Conrady radial+tangential distortion to a
// normalized image coordinate (x, y).
func distortNormalized(x, y, k1, k2, p1, p2 float64) (xd, yd float64) {
	r2 := x*x + y*y
	radial := 1 + k1*r2 + k2*r2*r2
	xd = x*radial + 2*p1*x*y + p2*(r2+2*x*x)
	yd = y*radial + p1*(r2+2*y*y) + 2*p2*x*y
	return xd, yd
}

// undistortNormalized inverts [distortNormalized] by fixed-point iteration,
// recovering the ideal normalized coordinate from a distorted one.
func undistortNormalized(xd, yd, k1, k2, p1, p2 float64) (x, y float64) {
	x, y = xd, yd
	for i := 0; i < 20; i++ {
		r2 := x*x + y*y
		radial := 1 + k1*r2 + k2*r2*r2
		if radial < 1e-9 {
			radial = 1e-9
		}
		dx := 2*p1*x*y + p2*(r2+2*x*x)
		dy := p1*(r2+2*y*y) + 2*p2*x*y
		x = (xd - dx) / radial
		y = (yd - dy) / radial
	}
	return x, y
}

// projectSphere projects a camera-frame direction (need not be unit length)
// through the omnidirectional model and returns the pixel coordinate together
// with a visibility flag (false when the ray falls behind the mirror and cannot
// be imaged). fx, fy, cx, cy, skew are the intrinsics.
func projectSphere(cam [3]float64, xi, fx, fy, cx, cy, skew, k1, k2, p1, p2 float64) (u, v float64, ok bool) {
	n := norm3(cam)
	if n < 1e-15 {
		return 0, 0, false
	}
	xs, ys, zs := cam[0]/n, cam[1]/n, cam[2]/n
	denom := zs + xi
	if denom <= 1e-9 {
		return 0, 0, false
	}
	x := xs / denom
	y := ys / denom
	xd, yd := distortNormalized(x, y, k1, k2, p1, p2)
	u = fx*xd + skew*yd + cx
	v = fy*yd + cy
	return u, v, true
}

// liftToSphere maps a pixel back to the unit-sphere direction it was imaged
// from. It inverts the intrinsics, the distortion and the sphere projection.
// ok is false when the pixel corresponds to no valid ray.
func liftToSphere(u, v, xi, fx, fy, cx, cy, skew, k1, k2, p1, p2 float64) (dir [3]float64, ok bool) {
	yd := (v - cy) / fy
	xd := (u - cx - skew*yd) / fx
	x, y := undistortNormalized(xd, yd, k1, k2, p1, p2)
	r2 := x*x + y*y
	disc := 1 + r2*(1-xi*xi)
	if disc < 0 {
		return [3]float64{}, false
	}
	w := (-r2*xi + math.Sqrt(disc)) / (r2 + 1)
	xs := x * (w + xi)
	ys := y * (w + xi)
	zs := w
	return [3]float64{xs, ys, zs}, true
}

// ProjectPoints projects 3D object points into an omnidirectional image. objPts
// are world points; rvec and tvec are the rotation vector (axis-angle) and
// translation taking world points into the camera frame; K is the 3×3 intrinsic
// matrix; xi is the mirror parameter; and dist holds [K1, K2, P1, P2] (a nil
// slice means no distortion). The returned slice has one [2]float64 pixel per
// input point. Points that cannot be imaged (behind the mirror) are projected
// to NaN so callers can filter them.
func (omnidirNS) ProjectPoints(objPts [][3]float64, rvec, tvec [3]float64, K [3][3]float64, xi float64, dist []float64) [][2]float64 {
	r := rodriguesToMatrix(rvec)
	k1, k2, p1, p2 := distParams(dist)
	fx, fy, cx, cy, skew := K[0][0], K[1][1], K[0][2], K[1][2], K[0][1]
	out := make([][2]float64, len(objPts))
	for i, X := range objPts {
		cam := add3(matVec3(r, X), tvec)
		u, v, ok := projectSphere(cam, xi, fx, fy, cx, cy, skew, k1, k2, p1, p2)
		if !ok {
			out[i] = [2]float64{math.NaN(), math.NaN()}
			continue
		}
		out[i] = [2]float64{u, v}
	}
	return out
}

// Undistort maps distorted omnidirectional pixels to pixels in a rectified
// pinhole image. For each input pixel it recovers the imaging ray, applies the
// rectification rotation R (identity by default), reprojects the ray through the
// new pinhole intrinsics Knew, and returns the rectilinear pixel. Rays without a
// positive depth in the rectified frame yield NaN. This is the point-wise
// counterpart of [omnidirNS.UndistortImage] with [RectifyPerspective].
func (omnidirNS) Undistort(distorted [][2]float64, K [3][3]float64, xi float64, dist []float64, Knew, R [3][3]float64) [][2]float64 {
	k1, k2, p1, p2 := distParams(dist)
	fx, fy, cx, cy, skew := K[0][0], K[1][1], K[0][2], K[1][2], K[0][1]
	out := make([][2]float64, len(distorted))
	for i, p := range distorted {
		dir, ok := liftToSphere(p[0], p[1], xi, fx, fy, cx, cy, skew, k1, k2, p1, p2)
		if !ok {
			out[i] = [2]float64{math.NaN(), math.NaN()}
			continue
		}
		rd := matVec3(R, dir)
		if rd[2] <= 1e-9 {
			out[i] = [2]float64{math.NaN(), math.NaN()}
			continue
		}
		xn := rd[0] / rd[2]
		yn := rd[1] / rd[2]
		u := Knew[0][0]*xn + Knew[0][1]*yn + Knew[0][2]
		v := Knew[1][1]*yn + Knew[1][2]
		out[i] = [2]float64{u, v}
	}
	return out
}

// InitUndistortRectifyMap builds the pixel-remap tables that turn an
// omnidirectional image into a rectified one of the given width and height. For
// every destination pixel the maps hold the fractional source coordinate to
// sample. R is the rectification rotation (identity by default), Knew the
// intrinsics of the destination image, and flags selects the target projection
// (see [RectifyFlag]). The maps are suitable for [github.com/malcolmston/opencv.Remap].
func (omnidirNS) InitUndistortRectifyMap(K [3][3]float64, xi float64, dist []float64, R, Knew [3][3]float64, width, height int, flags RectifyFlag) (mapX, mapY *cv.FloatMat) {
	k1, k2, p1, p2 := distParams(dist)
	fx, fy, cx, cy, skew := K[0][0], K[1][1], K[0][2], K[1][2], K[0][1]
	rt := transpose3(R)
	mapX = cv.NewFloatMat(height, width)
	mapY = cv.NewFloatMat(height, width)
	fxn, fyn, cxn, cyn := Knew[0][0], Knew[1][1], Knew[0][2], Knew[1][2]
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			var dir [3]float64
			switch flags {
			case RectifyCylindrical:
				lon := (float64(col) - cxn) / fxn
				h := (float64(row) - cyn) / fyn
				dir = [3]float64{math.Sin(lon), h, math.Cos(lon)}
			case RectifyLongLat:
				lon := (float64(col) - cxn) / fxn
				lat := (float64(row) - cyn) / fyn
				dir = [3]float64{
					-math.Cos(lat) * math.Cos(lon),
					math.Sin(lat),
					math.Cos(lat) * math.Sin(lon),
				}
			default: // RectifyPerspective
				dir = [3]float64{(float64(col) - cxn) / fxn, (float64(row) - cyn) / fyn, 1}
			}
			cam := matVec3(rt, dir)
			idx := row*width + col
			u, v, ok := projectSphere(cam, xi, fx, fy, cx, cy, skew, k1, k2, p1, p2)
			if !ok {
				mapX.Data[idx] = -1
				mapY.Data[idx] = -1
				continue
			}
			mapX.Data[idx] = u
			mapY.Data[idx] = v
		}
	}
	return mapX, mapY
}

// UndistortImage rectifies an omnidirectional image src to a new image of the
// given size, using the destination intrinsics Knew, the rectification rotation
// R and the projection selected by flags. It is a convenience wrapper that
// builds the maps with [omnidirNS.InitUndistortRectifyMap] and resamples with
// bilinear interpolation.
func (ns omnidirNS) UndistortImage(src *cv.Mat, K [3][3]float64, xi float64, dist []float64, Knew, R [3][3]float64, width, height int, flags RectifyFlag) *cv.Mat {
	mapX, mapY := ns.InitUndistortRectifyMap(K, xi, dist, R, Knew, width, height, flags)
	return cv.Remap(src, mapX, mapY, cv.InterLinear)
}

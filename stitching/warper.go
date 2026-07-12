package stitching

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Warper projects a source image onto a curved surface (a cylinder or sphere) or
// leaves it planar, parameterised by a focal length in pixels. Projecting to a
// cylinder or sphere is what lets many narrow photos taken by rotating a camera
// join into a wide panorama without the extreme stretching a single planar
// homography would introduce near the edges.
//
// Warp resamples src onto the surface and returns the warped image together with
// the top-left corner of the warped region in surface-pixel coordinates (the
// origin is the principal point of src, so the corner is usually negative).
// WarpBackward inverts the projection, reconstructing a planar image of the
// requested size. The implementations are [PlaneWarper], [CylindricalWarper] and
// [SphericalWarper], selected for a [Pipeline] with [Pipeline.SetWarper].
type Warper interface {
	// Name reports the projection surface ("plane", "cylindrical" or
	// "spherical").
	Name() string
	// Warp projects src onto the surface using the given focal length and
	// returns the warped image and its top-left corner.
	Warp(src *cv.Mat, focal float64) (dst *cv.Mat, corner image.Point)
	// WarpBackward reconstructs a planar image of size width×height from a warped
	// image whose top-left corner is at corner (as returned by Warp).
	WarpBackward(warped *cv.Mat, focal float64, corner image.Point, width, height int) *cv.Mat
	// WarpPoint maps the source point (x, y) onto surface coordinates using the
	// given focal length and principal point.
	WarpPoint(x, y, focal, cx, cy float64) (u, v float64)
}

// projector maps between centred image coordinates (relative to the principal
// point) and centred surface coordinates. Each concrete warper embeds one.
type projector interface {
	// forward maps a centred source point onto centred surface coordinates.
	forward(xc, yc, f float64) (u, v float64)
	// backward maps a centred surface point back to centred source coordinates,
	// reporting false when the point does not correspond to any source point.
	backward(u, v, f float64) (xc, yc float64, ok bool)
}

// planeProjector is the identity projection: the "surface" is the image plane
// itself.
type planeProjector struct{}

func (planeProjector) forward(xc, yc, _ float64) (float64, float64) { return xc, yc }
func (planeProjector) backward(u, v, _ float64) (float64, float64, bool) {
	return u, v, true
}

// cylindricalProjector wraps the image around a cylinder of radius f whose axis
// is vertical, so horizontal angle maps linearly to the warped x coordinate.
type cylindricalProjector struct{}

func (cylindricalProjector) forward(xc, yc, f float64) (float64, float64) {
	u := f * math.Atan2(xc, f)
	v := f * yc / math.Hypot(xc, f)
	return u, v
}

func (cylindricalProjector) backward(u, v, f float64) (float64, float64, bool) {
	theta := u / f
	c := math.Cos(theta)
	if math.Abs(c) < 1e-9 {
		return 0, 0, false
	}
	xc := f * math.Tan(theta)
	yc := v / c
	return xc, yc, true
}

// sphericalProjector wraps the image around a unit sphere of radius f, mapping
// longitude to the warped x coordinate and latitude to the warped y coordinate.
type sphericalProjector struct{}

func (sphericalProjector) forward(xc, yc, f float64) (float64, float64) {
	u := f * math.Atan2(xc, f)
	v := f * math.Atan2(yc, math.Hypot(xc, f))
	return u, v
}

func (sphericalProjector) backward(u, v, f float64) (float64, float64, bool) {
	theta := u / f
	phi := v / f
	c := math.Cos(theta)
	cp := math.Cos(phi)
	if math.Abs(c) < 1e-9 || math.Abs(cp) < 1e-9 {
		return 0, 0, false
	}
	xc := f * math.Tan(theta)
	yc := f * math.Tan(phi) / c
	return xc, yc, true
}

// PlaneWarper is the trivial [Warper] whose surface is the image plane; Warp is
// an identity resample. It is the appropriate warper for [ModeScans], where the
// scene is flat and related by planar transforms rather than camera rotation.
type PlaneWarper struct{}

// Name returns "plane".
func (PlaneWarper) Name() string { return "plane" }

// Warp returns a clone of src with a zero corner; the plane projection is the
// identity.
func (PlaneWarper) Warp(src *cv.Mat, _ float64) (*cv.Mat, image.Point) {
	return src.Clone(), image.Point{}
}

// WarpBackward reconstructs a planar image from a plane-warped one. When the sizes
// already match it is a plain clone; otherwise it resamples through the identity
// projection.
func (PlaneWarper) WarpBackward(warped *cv.Mat, focal float64, corner image.Point, width, height int) *cv.Mat {
	if warped.Cols == width && warped.Rows == height && corner.X == 0 && corner.Y == 0 {
		return warped.Clone()
	}
	return warpBackward(planeProjector{}, warped, focal, corner, width, height)
}

// WarpPoint maps a source point through the identity projection.
func (PlaneWarper) WarpPoint(x, y, focal, cx, cy float64) (float64, float64) {
	return warpPoint(planeProjector{}, x, y, focal, cx, cy)
}

// CylindricalWarper projects onto a vertical cylinder. Straight vertical lines
// stay straight and the horizontal field of view can exceed 180°, which is why
// it is a common choice for horizontally-swept panoramas.
type CylindricalWarper struct{}

// Name returns "cylindrical".
func (CylindricalWarper) Name() string { return "cylindrical" }

// Warp projects src onto the cylinder.
func (CylindricalWarper) Warp(src *cv.Mat, focal float64) (*cv.Mat, image.Point) {
	return warpForward(cylindricalProjector{}, src, focal)
}

// WarpBackward reconstructs the planar image from a cylindrical projection.
func (CylindricalWarper) WarpBackward(warped *cv.Mat, focal float64, corner image.Point, width, height int) *cv.Mat {
	return warpBackward(cylindricalProjector{}, warped, focal, corner, width, height)
}

// WarpPoint maps a source point onto the cylinder.
func (CylindricalWarper) WarpPoint(x, y, focal, cx, cy float64) (float64, float64) {
	return warpPoint(cylindricalProjector{}, x, y, focal, cx, cy)
}

// SphericalWarper projects onto a sphere. It handles panoramas that sweep in two
// directions (both horizontally and vertically) at the cost of curving straight
// lines more strongly than the cylinder.
type SphericalWarper struct{}

// Name returns "spherical".
func (SphericalWarper) Name() string { return "spherical" }

// Warp projects src onto the sphere.
func (SphericalWarper) Warp(src *cv.Mat, focal float64) (*cv.Mat, image.Point) {
	return warpForward(sphericalProjector{}, src, focal)
}

// WarpBackward reconstructs the planar image from a spherical projection.
func (SphericalWarper) WarpBackward(warped *cv.Mat, focal float64, corner image.Point, width, height int) *cv.Mat {
	return warpBackward(sphericalProjector{}, warped, focal, corner, width, height)
}

// WarpPoint maps a source point onto the sphere.
func (SphericalWarper) WarpPoint(x, y, focal, cx, cy float64) (float64, float64) {
	return warpPoint(sphericalProjector{}, x, y, focal, cx, cy)
}

// warpPoint maps an absolute source pixel onto absolute surface coordinates
// through the projector, using the principal point (cx, cy) as the centre.
func warpPoint(p projector, x, y, f, cx, cy float64) (float64, float64) {
	u, v := p.forward(x-cx, y-cy, f)
	return u, v
}

// warpForward resamples src onto the projection surface. It forward-maps the
// image border to find the surface bounding box, allocates the destination, then
// inverse-maps each destination pixel back into src and samples it bilinearly.
// The returned corner is the surface coordinate of the destination's (0, 0)
// pixel.
func warpForward(p projector, src *cv.Mat, f float64) (*cv.Mat, image.Point) {
	cx := float64(src.Cols-1) / 2
	cy := float64(src.Rows-1) / 2

	minU, minV := math.Inf(1), math.Inf(1)
	maxU, maxV := math.Inf(-1), math.Inf(-1)
	consider := func(x, y int) {
		u, v := p.forward(float64(x)-cx, float64(y)-cy, f)
		minU, maxU = math.Min(minU, u), math.Max(maxU, u)
		minV, maxV = math.Min(minV, v), math.Max(maxV, v)
	}
	for x := 0; x < src.Cols; x++ {
		consider(x, 0)
		consider(x, src.Rows-1)
	}
	for y := 0; y < src.Rows; y++ {
		consider(0, y)
		consider(src.Cols-1, y)
	}

	uOff := int(math.Floor(minU))
	vOff := int(math.Floor(minV))
	dstW := int(math.Ceil(maxU)) - uOff + 1
	dstH := int(math.Ceil(maxV)) - vOff + 1
	if dstW < 1 {
		dstW = 1
	}
	if dstH < 1 {
		dstH = 1
	}
	dst := cv.NewMat(dstH, dstW, src.Channels)
	buf := make([]float64, src.Channels)
	for row := 0; row < dstH; row++ {
		v := float64(vOff + row)
		for col := 0; col < dstW; col++ {
			u := float64(uOff + col)
			xc, yc, ok := p.backward(u, v, f)
			if !ok {
				continue
			}
			if sampleBilinear(src, xc+cx, yc+cy, buf) {
				di := (row*dstW + col) * src.Channels
				for c := 0; c < src.Channels; c++ {
					dst.Data[di+c] = clampUint8(buf[c] + 0.5)
				}
			}
		}
	}
	return dst, image.Point{X: uOff, Y: vOff}
}

// warpBackward reconstructs a planar image of size width×height by forward-mapping
// each planar pixel onto the surface and sampling the warped image there.
func warpBackward(p projector, warped *cv.Mat, f float64, corner image.Point, width, height int) *cv.Mat {
	cx := float64(width-1) / 2
	cy := float64(height-1) / 2
	dst := cv.NewMat(height, width, warped.Channels)
	buf := make([]float64, warped.Channels)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			u, v := p.forward(float64(x)-cx, float64(y)-cy, f)
			wx := u - float64(corner.X)
			wy := v - float64(corner.Y)
			if sampleBilinear(warped, wx, wy, buf) {
				di := (y*width + x) * warped.Channels
				for c := 0; c < warped.Channels; c++ {
					dst.Data[di+c] = clampUint8(buf[c] + 0.5)
				}
			}
		}
	}
	return dst
}

// sampleBilinear samples m at the continuous coordinate (fx, fy), writing one
// value per channel into out. It reports false (leaving out untouched) when the
// sample centre lies outside the image, so callers can leave uncovered pixels at
// their zero value.
func sampleBilinear(m *cv.Mat, fx, fy float64, out []float64) bool {
	if fx < 0 || fy < 0 || fx > float64(m.Cols-1) || fy > float64(m.Rows-1) {
		return false
	}
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 > m.Cols-1 {
		x1 = m.Cols - 1
	}
	if y1 > m.Rows-1 {
		y1 = m.Rows - 1
	}
	ax := fx - float64(x0)
	ay := fy - float64(y0)
	ch := m.Channels
	i00 := (y0*m.Cols + x0) * ch
	i01 := (y0*m.Cols + x1) * ch
	i10 := (y1*m.Cols + x0) * ch
	i11 := (y1*m.Cols + x1) * ch
	for c := 0; c < ch; c++ {
		top := float64(m.Data[i00+c])*(1-ax) + float64(m.Data[i01+c])*ax
		bot := float64(m.Data[i10+c])*(1-ax) + float64(m.Data[i11+c])*ax
		out[c] = top*(1-ay) + bot*ay
	}
	return true
}

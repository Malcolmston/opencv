package cv

import (
	"fmt"
	"math"
)

// PerspectiveMatrix is a 3×3 projective transform stored row-major (h0..h8). It
// maps a source point (x, y) to (x', y') where
//
//	w  = h6*x + h7*y + h8
//	x' = (h0*x + h1*y + h2) / w
//	y' = (h3*x + h4*y + h5) / w
type PerspectiveMatrix [9]float64

// GetPerspectiveTransform computes the 3×3 homography that maps the four source
// points src to the four destination points dst, solving the resulting 8×8
// linear system (with h8 fixed to 1). The four points in each set should be in
// corresponding order and no three collinear. It panics if the system is
// singular.
func GetPerspectiveTransform(src, dst [4]Point) PerspectiveMatrix {
	// Build the 8×8 system A*h = b for the 8 unknowns h0..h7.
	var a [8][8]float64
	var b [8]float64
	for i := 0; i < 4; i++ {
		x := float64(src[i].X)
		y := float64(src[i].Y)
		u := float64(dst[i].X)
		v := float64(dst[i].Y)
		r0 := 2 * i
		r1 := 2*i + 1
		a[r0] = [8]float64{x, y, 1, 0, 0, 0, -x * u, -y * u}
		b[r0] = u
		a[r1] = [8]float64{0, 0, 0, x, y, 1, -x * v, -y * v}
		b[r1] = v
	}
	h, ok := solveLinear8(a, b)
	if !ok {
		panic("cv: GetPerspectiveTransform points are degenerate")
	}
	return PerspectiveMatrix{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1}
}

// solveLinear8 solves an 8×8 linear system with Gauss–Jordan elimination and
// partial pivoting, reporting whether the matrix was non-singular.
func solveLinear8(a [8][8]float64, b [8]float64) ([8]float64, bool) {
	const n = 8
	for col := 0; col < n; col++ {
		// Partial pivot.
		piv := col
		best := math.Abs(a[col][col])
		for r := col + 1; r < n; r++ {
			if math.Abs(a[r][col]) > best {
				best = math.Abs(a[r][col])
				piv = r
			}
		}
		if best < 1e-12 {
			return [8]float64{}, false
		}
		a[col], a[piv] = a[piv], a[col]
		b[col], b[piv] = b[piv], b[col]
		// Normalise pivot row.
		p := a[col][col]
		for c := col; c < n; c++ {
			a[col][c] /= p
		}
		b[col] /= p
		// Eliminate other rows.
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a[r][col]
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				a[r][c] -= f * a[col][c]
			}
			b[r] -= f * b[col]
		}
	}
	return b, true
}

// invertPerspective returns the inverse of a 3×3 projective transform, reporting
// whether it is invertible.
func invertPerspective(m PerspectiveMatrix) (PerspectiveMatrix, bool) {
	a, b, c := m[0], m[1], m[2]
	d, e, f := m[3], m[4], m[5]
	g, h, i := m[6], m[7], m[8]
	det := a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
	if math.Abs(det) < 1e-15 {
		return PerspectiveMatrix{}, false
	}
	id := 1 / det
	var inv PerspectiveMatrix
	inv[0] = (e*i - f*h) * id
	inv[1] = (c*h - b*i) * id
	inv[2] = (b*f - c*e) * id
	inv[3] = (f*g - d*i) * id
	inv[4] = (a*i - c*g) * id
	inv[5] = (c*d - a*f) * id
	inv[6] = (d*h - e*g) * id
	inv[7] = (b*g - a*h) * id
	inv[8] = (a*e - b*d) * id
	return inv, true
}

// WarpPerspective applies the projective transform m to src, producing an output
// of the given width and height. Each destination pixel is inverse-mapped into
// src and sampled with the chosen interpolation; coordinates outside src are
// left zero. width and height must be positive and m must be invertible.
func WarpPerspective(src *Mat, m PerspectiveMatrix, width, height int, interp InterpolationFlag) *Mat {
	if width <= 0 || height <= 0 {
		panic("cv: WarpPerspective requires positive width and height")
	}
	inv, ok := invertPerspective(m)
	if !ok {
		panic("cv: WarpPerspective transform is not invertible")
	}
	dst := NewMat(height, width, src.Channels)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			w := inv[6]*float64(x) + inv[7]*float64(y) + inv[8]
			if w == 0 {
				continue
			}
			sx := (inv[0]*float64(x) + inv[1]*float64(y) + inv[2]) / w
			sy := (inv[3]*float64(x) + inv[4]*float64(y) + inv[5]) / w
			switch interp {
			case InterNearest:
				rx := int(math.Round(sx))
				ry := int(math.Round(sy))
				if rx < 0 || rx >= src.Cols || ry < 0 || ry >= src.Rows {
					continue
				}
				si := src.index(ry, rx)
				di := dst.index(y, x)
				copy(dst.Data[di:di+src.Channels], src.Data[si:si+src.Channels])
			case InterLinear:
				if sx < -0.5 || sx > float64(src.Cols)-0.5 || sy < -0.5 || sy > float64(src.Rows)-0.5 {
					continue
				}
				for c := 0; c < src.Channels; c++ {
					dst.Data[dst.index(y, x)+c] = clampToUint8(bilinearSample(src, sx, sy, c) + 0.5)
				}
			default:
				panic(fmt.Sprintf("cv: WarpPerspective unknown interpolation %d", interp))
			}
		}
	}
	return dst
}

// Remap resamples src at the per-pixel source coordinates given by mapX and mapY
// (each a [FloatMat] of the output size): the destination pixel (x, y) is taken
// from src at (mapX[y,x], mapY[y,x]) using the chosen interpolation. Out-of-range
// coordinates yield zero. mapX and mapY must have identical dimensions. It panics
// otherwise.
func Remap(src *Mat, mapX, mapY *FloatMat, interp InterpolationFlag) *Mat {
	if mapX.Rows != mapY.Rows || mapX.Cols != mapY.Cols {
		panic("cv: Remap map dimensions differ")
	}
	dst := NewMat(mapX.Rows, mapX.Cols, src.Channels)
	for y := 0; y < mapX.Rows; y++ {
		for x := 0; x < mapX.Cols; x++ {
			sx := mapX.Data[y*mapX.Cols+x]
			sy := mapY.Data[y*mapY.Cols+x]
			switch interp {
			case InterNearest:
				rx := int(math.Round(sx))
				ry := int(math.Round(sy))
				if rx < 0 || rx >= src.Cols || ry < 0 || ry >= src.Rows {
					continue
				}
				si := src.index(ry, rx)
				di := dst.index(y, x)
				copy(dst.Data[di:di+src.Channels], src.Data[si:si+src.Channels])
			case InterLinear:
				if sx < -0.5 || sx > float64(src.Cols)-0.5 || sy < -0.5 || sy > float64(src.Rows)-0.5 {
					continue
				}
				for c := 0; c < src.Channels; c++ {
					dst.Data[dst.index(y, x)+c] = clampToUint8(bilinearSample(src, sx, sy, c) + 0.5)
				}
			default:
				panic(fmt.Sprintf("cv: Remap unknown interpolation %d", interp))
			}
		}
	}
	return dst
}

// pyrKernel is the normalised 5-tap binomial kernel [1 4 6 4 1]/16 used by the
// Gaussian pyramid; its even/odd tap sums are each 1/2, which makes PyrUp's ×4
// scaling exact on a constant image.
var pyrKernel = []float64{1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16}

// PyrDown blurs src with the 5-tap binomial pyramid kernel and drops every other
// row and column, halving each dimension (rounding up). It is one level of a
// Gaussian pyramid.
func PyrDown(src *Mat) *Mat {
	chans := sepFilterFloat(src, pyrKernel, pyrKernel)
	blurred := floatChannelsToMat(src.Rows, src.Cols, chans)
	dw := (src.Cols + 1) / 2
	dh := (src.Rows + 1) / 2
	dst := NewMat(dh, dw, src.Channels)
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			sy := y * 2
			sx := x * 2
			si := blurred.index(sy, sx)
			di := dst.index(y, x)
			copy(dst.Data[di:di+src.Channels], blurred.Data[si:si+src.Channels])
		}
	}
	return dst
}

// PyrUp doubles each dimension of src by inserting zero rows and columns and
// smoothing with a 5-tap Gaussian scaled by 4 (so average brightness is
// preserved). It is the up-sampling step of a Gaussian pyramid.
func PyrUp(src *Mat) *Mat {
	dw := src.Cols * 2
	dh := src.Rows * 2
	up := NewMat(dh, dw, src.Channels)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			si := src.index(y, x)
			di := up.index(y*2, x*2)
			copy(up.Data[di:di+src.Channels], src.Data[si:si+src.Channels])
		}
	}
	// Smooth the injected image; multiply by 4 to compensate for the zeros.
	chans := sepFilterFloat(up, pyrKernel, pyrKernel)
	out := NewMat(dh, dw, src.Channels)
	for c := range chans {
		for i, v := range chans[c] {
			out.Data[i*src.Channels+c] = clampToUint8(v*4 + 0.5)
		}
	}
	return out
}

// DistanceTransform computes, for each non-zero (foreground) pixel of a binary
// single-channel image, the approximate Euclidean distance to the nearest zero
// (background) pixel, returning the distances as a [FloatMat]. It uses a
// two-pass 3×3 chamfer with weights 1 (orthogonal) and √2 (diagonal). Background
// pixels have distance 0. It panics if src is not single-channel.
func DistanceTransform(src *Mat) *FloatMat {
	requireChannels(src, 1, "DistanceTransform")
	rows, cols := src.Rows, src.Cols
	const (
		a = 1.0
		b = math.Sqrt2
	)
	dist := NewFloatMat(rows, cols)
	big := float64(rows+cols) * 2
	for i, v := range src.Data {
		if v == 0 {
			dist.Data[i] = 0
		} else {
			dist.Data[i] = big
		}
	}
	at := func(y, x int) float64 { return dist.Data[y*cols+x] }
	// Forward pass.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if src.Data[y*cols+x] == 0 {
				continue
			}
			d := at(y, x)
			if x > 0 {
				d = math.Min(d, at(y, x-1)+a)
			}
			if y > 0 {
				d = math.Min(d, at(y-1, x)+a)
			}
			if y > 0 && x > 0 {
				d = math.Min(d, at(y-1, x-1)+b)
			}
			if y > 0 && x < cols-1 {
				d = math.Min(d, at(y-1, x+1)+b)
			}
			dist.Data[y*cols+x] = d
		}
	}
	// Backward pass.
	for y := rows - 1; y >= 0; y-- {
		for x := cols - 1; x >= 0; x-- {
			if src.Data[y*cols+x] == 0 {
				continue
			}
			d := at(y, x)
			if x < cols-1 {
				d = math.Min(d, at(y, x+1)+a)
			}
			if y < rows-1 {
				d = math.Min(d, at(y+1, x)+a)
			}
			if y < rows-1 && x < cols-1 {
				d = math.Min(d, at(y+1, x+1)+b)
			}
			if y < rows-1 && x > 0 {
				d = math.Min(d, at(y+1, x-1)+b)
			}
			dist.Data[y*cols+x] = d
		}
	}
	return dist
}

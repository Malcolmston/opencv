package superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Shift2D is a sub-pixel translation in image coordinates, measured in pixels.
// Positive Dx shifts content to the right, positive Dy shifts it downward.
type Shift2D struct {
	// Dx is the horizontal displacement in pixels.
	Dx float64
	// Dy is the vertical displacement in pixels.
	Dy float64
}

// Shift translates src by the sub-pixel offset (dx, dy) using bilinear
// resampling with border replication. The output has the same dimensions as
// src. A positive dx moves image content to the right; a positive dy moves it
// down (so the sample drawn at output (x, y) is taken from source
// (x−dx, y−dy)).
func Shift(src *cv.Mat, dx, dy float64) *cv.Mat {
	ch := src.Channels
	dst := cv.NewMat(src.Rows, src.Cols, ch)
	for y := 0; y < src.Rows; y++ {
		sy := float64(y) - dy
		iy := int(math.Floor(sy))
		fy := sy - float64(iy)
		for x := 0; x < src.Cols; x++ {
			sx := float64(x) - dx
			ix := int(math.Floor(sx))
			fx := sx - float64(ix)
			for c := 0; c < ch; c++ {
				v00 := superresAt(src, iy, ix, c)
				v01 := superresAt(src, iy, ix+1, c)
				v10 := superresAt(src, iy+1, ix, c)
				v11 := superresAt(src, iy+1, ix+1, c)
				top := v00 + (v01-v00)*fx
				bot := v10 + (v11-v10)*fx
				dst.Data[(y*src.Cols+x)*ch+c] = superresClamp8(top + (bot-top)*fy)
			}
		}
	}
	return dst
}

// IntegerShift translates src by a whole number of pixels (dx, dy) with border
// replication, a fast exact special case of [Shift].
func IntegerShift(src *cv.Mat, dx, dy int) *cv.Mat {
	ch := src.Channels
	dst := cv.NewMat(src.Rows, src.Cols, ch)
	for y := 0; y < src.Rows; y++ {
		sy := superresClampInt(y-dy, 0, src.Rows-1)
		for x := 0; x < src.Cols; x++ {
			sx := superresClampInt(x-dx, 0, src.Cols-1)
			for c := 0; c < ch; c++ {
				dst.Data[(y*src.Cols+x)*ch+c] = src.Data[(sy*src.Cols+sx)*ch+c]
			}
		}
	}
	return dst
}

// superresGray returns a single float plane: channel 0 of the image, or the
// unweighted mean of all channels for multi-channel input.
func superresGray(m *cv.Mat) *superresPlane {
	p := newSuperresPlane(m.Rows, m.Cols)
	ch := m.Channels
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			var s float64
			for c := 0; c < ch; c++ {
				s += float64(m.Data[(y*m.Cols+x)*ch+c])
			}
			p.data[y*m.Cols+x] = s / float64(ch)
		}
	}
	return p
}

// EstimateShift estimates the sub-pixel translation that best maps ref onto
// moved, assuming the two images differ by a small global shift (the classic
// single-iteration Lucas-Kanade / Horn-Schunck translational solution). It
// solves the 2×2 normal equations built from spatial and temporal gradients
// over the whole image and returns the displacement such that
// moved ≈ Shift(ref, Dx, Dy). Accuracy is best for shifts below about one
// pixel; use [EstimateShiftRefine] for larger displacements. Both images must
// have identical dimensions. It panics if the shapes differ.
func EstimateShift(ref, moved *cv.Mat) Shift2D {
	superresCheckSame(ref, moved)
	a := superresGray(ref)
	b := superresGray(moved)
	var gxx, gxy, gyy, gxt, gyt float64
	for y := 0; y < a.rows; y++ {
		for x := 0; x < a.cols; x++ {
			ix := 0.5 * (a.at(y, x+1) - a.at(y, x-1))
			iy := 0.5 * (a.at(y+1, x) - a.at(y-1, x))
			it := b.atRaw(y, x) - a.atRaw(y, x)
			gxx += ix * ix
			gxy += ix * iy
			gyy += iy * iy
			gxt += ix * it
			gyt += iy * it
		}
	}
	// Solve [gxx gxy; gxy gyy] [dx dy]^T = [gxt gyt]^T.
	det := gxx*gyy - gxy*gxy
	if math.Abs(det) < 1e-9 {
		return Shift2D{}
	}
	// Solve A·[dx dy]^T = -[gxt gyt]^T with A = [[gxx gxy]; [gxy gyy]].
	dx := (gxy*gyt - gyy*gxt) / det
	dy := (gxy*gxt - gxx*gyt) / det
	return Shift2D{Dx: dx, Dy: dy}
}

// EstimateShiftRefine estimates a larger sub-pixel shift by iterated
// Lucas-Kanade: it repeatedly estimates the residual shift with [EstimateShift]
// and warps ref toward moved, accumulating the total displacement. iterations
// controls the number of refinement passes (4–8 is usually ample). It returns
// the total displacement such that moved ≈ Shift(ref, Dx, Dy). Both images must
// have identical dimensions. It panics if the shapes differ or iterations < 1.
func EstimateShiftRefine(ref, moved *cv.Mat, iterations int) Shift2D {
	superresCheckSame(ref, moved)
	if iterations < 1 {
		panic("superres: EstimateShiftRefine requires iterations >= 1")
	}
	var total Shift2D
	warped := ref
	for i := 0; i < iterations; i++ {
		delta := EstimateShift(warped, moved)
		total.Dx += delta.Dx
		total.Dy += delta.Dy
		if math.Abs(delta.Dx) < 1e-4 && math.Abs(delta.Dy) < 1e-4 {
			break
		}
		warped = Shift(ref, total.Dx, total.Dy)
	}
	return total
}

// PhaseCorrelateShift estimates the integer-plus-sub-pixel translation between
// ref and moved by phase correlation: it computes the cross-power spectrum via
// a discrete Fourier transform, locates the correlation peak, and refines it to
// sub-pixel precision by parabolic interpolation of the neighbouring peaks. It
// returns the displacement such that moved ≈ Shift(ref, Dx, Dy). Phase
// correlation is robust to illumination change and works for shifts up to
// roughly half the image size. Both images must have identical dimensions. It
// panics if the shapes differ.
func PhaseCorrelateShift(ref, moved *cv.Mat) Shift2D {
	superresCheckSame(ref, moved)
	h, w := ref.Rows, ref.Cols
	a := superresGray(ref)
	b := superresGray(moved)
	// Remove the DC component to reduce edge/mean effects.
	superresRemoveMean(a)
	superresRemoveMean(b)

	ar, ai := superresDFT2D(a.data, h, w, false)
	br, bi := superresDFT2D(b.data, h, w, false)

	// Cross-power spectrum R = conj(A)·B / |conj(A)·B|.
	cr := make([]float64, h*w)
	ci := make([]float64, h*w)
	for i := 0; i < h*w; i++ {
		// conj(A) = ar - i*ai.
		rr := ar[i]*br[i] + ai[i]*bi[i]
		ri := ar[i]*bi[i] - ai[i]*br[i]
		mag := math.Hypot(rr, ri)
		if mag < 1e-12 {
			cr[i] = 0
			ci[i] = 0
		} else {
			cr[i] = rr / mag
			ci[i] = ri / mag
		}
	}
	// Inverse DFT gives the correlation surface (real part).
	corrR, _ := superresDFT2D2(cr, ci, h, w, true)

	// Locate peak.
	peak := 0
	best := corrR[0]
	for i := 1; i < h*w; i++ {
		if corrR[i] > best {
			best = corrR[i]
			peak = i
		}
	}
	py := peak / w
	px := peak % w

	// Parabolic sub-pixel refinement in each axis (with wraparound).
	subX := superresParabolic(
		corrR[py*w+((px-1+w)%w)],
		corrR[py*w+px],
		corrR[py*w+((px+1)%w)],
	)
	subY := superresParabolic(
		corrR[((py-1+h)%h)*w+px],
		corrR[py*w+px],
		corrR[((py+1)%h)*w+px],
	)

	fx := float64(px) + subX
	fy := float64(py) + subY
	// Map peak position to signed shift (wrap the far half to negative).
	if fx > float64(w)/2 {
		fx -= float64(w)
	}
	if fy > float64(h)/2 {
		fy -= float64(h)
	}
	return Shift2D{Dx: fx, Dy: fy}
}

// superresParabolic returns the sub-sample offset of the peak of the parabola
// through (−1, ym1), (0, y0), (1, yp1); the result lies in (−0.5, 0.5).
func superresParabolic(ym1, y0, yp1 float64) float64 {
	denom := ym1 - 2*y0 + yp1
	if math.Abs(denom) < 1e-12 {
		return 0
	}
	off := 0.5 * (ym1 - yp1) / denom
	if off > 0.5 {
		off = 0.5
	} else if off < -0.5 {
		off = -0.5
	}
	return off
}

// superresRemoveMean subtracts the plane's mean from every sample in place.
func superresRemoveMean(p *superresPlane) {
	var sum float64
	for _, v := range p.data {
		sum += v
	}
	mean := sum / float64(len(p.data))
	for i := range p.data {
		p.data[i] -= mean
	}
}

// superresDFT2D computes the 2-D discrete Fourier transform of a real input of
// size h×w via separable 1-D transforms. inverse selects the inverse
// transform (with 1/(h·w) normalisation). It returns the real and imaginary
// parts.
func superresDFT2D(real []float64, h, w int, inverse bool) (re, im []float64) {
	imag := make([]float64, len(real))
	return superresDFT2D2(real, imag, h, w, inverse)
}

// superresDFT2D2 is like superresDFT2D but accepts a complex input.
func superresDFT2D2(inRe, inIm []float64, h, w int, inverse bool) (re, im []float64) {
	re = make([]float64, h*w)
	im = make([]float64, h*w)
	copy(re, inRe)
	copy(im, inIm)
	// Transform each row.
	tmpRe := make([]float64, w)
	tmpIm := make([]float64, w)
	for y := 0; y < h; y++ {
		superresDFT1D(re[y*w:y*w+w], im[y*w:y*w+w], tmpRe, tmpIm, inverse)
		copy(re[y*w:y*w+w], tmpRe)
		copy(im[y*w:y*w+w], tmpIm)
	}
	// Transform each column.
	colRe := make([]float64, h)
	colIm := make([]float64, h)
	outRe := make([]float64, h)
	outIm := make([]float64, h)
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			colRe[y] = re[y*w+x]
			colIm[y] = im[y*w+x]
		}
		superresDFT1D(colRe, colIm, outRe, outIm, inverse)
		for y := 0; y < h; y++ {
			re[y*w+x] = outRe[y]
			im[y*w+x] = outIm[y]
		}
	}
	return re, im
}

// superresDFT1D computes the 1-D DFT of the complex sequence (inRe, inIm) into
// (outRe, outIm). inverse selects the inverse transform with 1/n
// normalisation. It is a direct O(n²) transform, adequate for the modest
// images used in registration.
func superresDFT1D(inRe, inIm, outRe, outIm []float64, inverse bool) {
	n := len(inRe)
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for k := 0; k < n; k++ {
		var sr, si float64
		base := sign * 2 * math.Pi * float64(k) / float64(n)
		for t := 0; t < n; t++ {
			ang := base * float64(t)
			cos := math.Cos(ang)
			sin := math.Sin(ang)
			sr += inRe[t]*cos - inIm[t]*sin
			si += inRe[t]*sin + inIm[t]*cos
		}
		if inverse {
			sr /= float64(n)
			si /= float64(n)
		}
		outRe[k] = sr
		outIm[k] = si
	}
}

package dnn_superres

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// reflectIdx maps an index into [0,n) by mirror reflection without repeating the
// edge sample (reflect-101), so out-of-range reads land on genuine interior
// samples of the opposite lattice parity. The ×2 edge-directed methods rely on
// this at borders: a plain clamp would fold a neighbour read back onto the
// still-unfilled target pixel, whereas reflection lands on an already-filled
// neighbour.
func reflectIdx(i, n int) int {
	if n == 1 {
		return 0
	}
	for i < 0 || i >= n {
		if i < 0 {
			i = -i
		}
		if i >= n {
			i = 2*(n-1) - i
		}
	}
	return i
}

// solve4 solves the 4×4 linear system C·x = y by Gaussian elimination with
// partial pivoting. It reports ok=false if the system is (numerically)
// singular, so callers can fall back to a simple average. C is row-major.
func solve4(c [16]float64, y [4]float64) (x [4]float64, ok bool) {
	// Augmented matrix a[row][0..3]=C, a[row][4]=y.
	var a [4][5]float64
	for r := 0; r < 4; r++ {
		for col := 0; col < 4; col++ {
			a[r][col] = c[r*4+col]
		}
		a[r][4] = y[r]
	}
	for col := 0; col < 4; col++ {
		// Pivot.
		piv := col
		best := math.Abs(a[col][col])
		for r := col + 1; r < 4; r++ {
			if v := math.Abs(a[r][col]); v > best {
				best, piv = v, r
			}
		}
		if best < 1e-9 {
			return x, false
		}
		a[col], a[piv] = a[piv], a[col]
		// Eliminate below.
		for r := col + 1; r < 4; r++ {
			f := a[r][col] / a[col][col]
			for k := col; k < 5; k++ {
				a[r][k] -= f * a[col][k]
			}
		}
	}
	// Back-substitute.
	for r := 3; r >= 0; r-- {
		s := a[r][4]
		for k := r + 1; k < 4; k++ {
			s -= a[r][k] * x[k]
		}
		x[r] = s / a[r][r]
	}
	return x, true
}

// nediWeights estimates the four covariance-based interpolation weights for the
// NEDI model at low-resolution location (i,j). Using a (2W+1)² training window,
// it models every window pixel as a weighted sum of its four neighbours at the
// relative offsets offs, accumulates the 4×4 autocorrelation C and the
// cross-correlation vector b, and solves C·α = b. A small Tikhonov term keeps C
// invertible on flat patches, and the weights are renormalised to sum to one so
// a constant region is reproduced exactly. It returns the fallback (uniform
// 0.25) weights when the solve is singular.
func nediWeights(lo *cv.Mat, c, i, j, w int, offs [4][2]int) [4]float64 {
	var cc [16]float64
	var b [4]float64
	var reg float64
	for p := i - w; p <= i+w; p++ {
		for q := j - w; q <= j+w; q++ {
			var a [4]float64
			for k := 0; k < 4; k++ {
				a[k] = sampleReplicate(lo, p+offs[k][0], q+offs[k][1], c)
			}
			t := sampleReplicate(lo, p, q, c)
			for r := 0; r < 4; r++ {
				b[r] += a[r] * t
				for s := 0; s < 4; s++ {
					cc[r*4+s] += a[r] * a[s]
				}
			}
			reg += a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]
		}
	}
	// Tikhonov regularisation proportional to the mean diagonal energy.
	lambda := 1e-3 * reg / 16
	if lambda < 1e-6 {
		lambda = 1e-6
	}
	for d := 0; d < 4; d++ {
		cc[d*4+d] += lambda
	}
	alpha, ok := solve4(cc, b)
	if !ok {
		return [4]float64{0.25, 0.25, 0.25, 0.25}
	}
	sum := alpha[0] + alpha[1] + alpha[2] + alpha[3]
	if math.Abs(sum) > 0.1 {
		for k := range alpha {
			alpha[k] /= sum
		}
	} else {
		return [4]float64{0.25, 0.25, 0.25, 0.25}
	}
	return alpha
}

// nediDouble performs one ×2 New Edge-Directed Interpolation step on lo,
// returning a 2×-larger image. Original samples are placed on the even lattice;
// the diagonal high-resolution samples are then estimated with per-pixel
// covariance weights over the diagonal neighbours, and finally the remaining
// axis samples are estimated the same way over the axis neighbours (now
// including the freshly filled diagonals). This is the full 4×4-covariance NEDI,
// applied independently per channel.
func nediDouble(lo *cv.Mat) *cv.Mat {
	h, w, ch := lo.Rows, lo.Cols, lo.Channels
	hi := cv.NewMat(h*2, w*2, ch)
	// Even lattice: copy originals.
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			for c := 0; c < ch; c++ {
				hi.Data[((2*i)*hi.Cols+2*j)*ch+c] = lo.Data[(i*w+j)*ch+c]
			}
		}
	}
	const win = 2
	diagOffs := [4][2]int{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}} // TL,TR,BL,BR
	// Pass 1: diagonal HR pixels hi[2i+1][2j+1].
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			for c := 0; c < ch; c++ {
				alpha := nediWeights(lo, c, i, j, win, diagOffs)
				n := [4]float64{
					sampleReplicate(lo, i, j, c),
					sampleReplicate(lo, i, j+1, c),
					sampleReplicate(lo, i+1, j, c),
					sampleReplicate(lo, i+1, j+1, c),
				}
				v := alpha[0]*n[0] + alpha[1]*n[1] + alpha[2]*n[2] + alpha[3]*n[3]
				hi.Data[((2*i+1)*hi.Cols+2*j+1)*ch+c] = clampByte(v)
			}
		}
	}
	// hiAt reads the partially filled HR grid, reflecting out-of-range reads onto
	// already-filled interior samples (see reflectIdx).
	hiAt := func(y, x, c int) float64 {
		y = reflectIdx(y, hi.Rows)
		x = reflectIdx(x, hi.Cols)
		return float64(hi.Data[(y*hi.Cols+x)*ch+c])
	}
	axisOffs := [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} // N,S,W,E
	// Pass 2: axis HR pixels (both odd/even and even/odd positions).
	fill := func(yy, xx int) {
		i, j := yy/2, xx/2
		for c := 0; c < ch; c++ {
			alpha := nediWeights(lo, c, i, j, win, axisOffs)
			n := [4]float64{
				hiAt(yy-1, xx, c), hiAt(yy+1, xx, c),
				hiAt(yy, xx-1, c), hiAt(yy, xx+1, c),
			}
			v := alpha[0]*n[0] + alpha[1]*n[1] + alpha[2]*n[2] + alpha[3]*n[3]
			hi.Data[(yy*hi.Cols+xx)*ch+c] = clampByte(v)
		}
	}
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			fill(2*i+1, 2*j) // below an original
			fill(2*i, 2*j+1) // right of an original
		}
	}
	return hi
}

// doubleThenResize applies a ×2 doubling operator repeatedly until doubling
// again would overshoot scale, then lands on the exact target size with a final
// bicubic resample. It is the shared arbitrary-scale driver for the ×2-native
// edge-directed methods (NEDI, DCCI).
func doubleThenResize(src *cv.Mat, scale int, double func(*cv.Mat) *cv.Mat) *cv.Mat {
	cur := src.Clone()
	curScale := 1
	for curScale*2 <= scale {
		cur = double(cur)
		curScale *= 2
	}
	if curScale != scale {
		cur = resampleSeparable(cur, src.Cols*scale, src.Rows*scale, keysCubic, 2)
	}
	return cur
}

// UpsampleNEDI enlarges src by an arbitrary integer scale (>= 2) with New
// Edge-Directed Interpolation. NEDI estimates each interpolated pixel from a
// full 4×4 local covariance model, so reconstruction follows the dominant edge
// orientation and diagonal contours stay crisp instead of staircasing. Powers
// of two are produced by successive ×2 NEDI doublings; other factors finish with
// a bicubic resample to the exact size. It returns an error for an empty image
// or a scale below 2.
func UpsampleNEDI(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	return doubleThenResize(src, scale, nediDouble), nil
}

// catmullRom evaluates the 1-D Catmull-Rom cubic through p0..p3 at parameter t
// in [0,1] (the segment between p1 and p2). It is the directional interpolant
// used by DCCI along the chosen edge orientation.
func catmullRom(p0, p1, p2, p3, t float64) float64 {
	t2 := t * t
	t3 := t2 * t
	return 0.5 * (2*p1 +
		(-p0+p2)*t +
		(2*p0-5*p1+4*p2-p3)*t2 +
		(-p0+3*p1-3*p2+p3)*t3)
}

// dcciDouble performs one ×2 Directional Cubic Convolution Interpolation step.
// Diagonal high-resolution samples are interpolated along whichever of the two
// diagonals is smoother (measured by summed directional gradients over the 4×4
// neighbourhood) using a Catmull-Rom cubic, the two directions blended by
// inverse-gradient weights. The remaining axis samples are then filled the same
// way choosing between the horizontal and vertical directions. It runs per
// channel.
func dcciDouble(lo *cv.Mat) *cv.Mat {
	h, w, ch := lo.Rows, lo.Cols, lo.Channels
	hi := cv.NewMat(h*2, w*2, ch)
	const k = 5.0
	g := func(y, x, c int) float64 { return sampleReplicate(lo, y, x, c) }

	// Even lattice: originals.
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			for c := 0; c < ch; c++ {
				hi.Data[((2*i)*hi.Cols+2*j)*ch+c] = lo.Data[(i*w+j)*ch+c]
			}
		}
	}
	invW := func(d float64) float64 { return 1.0 / (1.0 + math.Pow(d, k)) }

	// Pass 1: diagonal pixels hi[2i+1][2j+1].
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			for c := 0; c < ch; c++ {
				// 4×4 window P[m][n] = lo(i-1+m, j-1+n).
				var d1, d2 float64 // "\" (down-right) and "/" (down-left) roughness
				for m := 0; m < 3; m++ {
					for n := 0; n < 3; n++ {
						p := g(i-1+m, j-1+n, c)
						pdr := g(i-1+m+1, j-1+n+1, c)
						pdl := g(i-1+m+1, j-1+n-1, c)
						d1 += math.Abs(p - pdr)
						d2 += math.Abs(p - pdl)
					}
				}
				// "\" diagonal points: P00,P11,P22,P33.
				v1 := catmullRom(g(i-1, j-1, c), g(i, j, c), g(i+1, j+1, c), g(i+2, j+2, c), 0.5)
				// "/" diagonal points: P03,P12,P21,P30.
				v2 := catmullRom(g(i-1, j+2, c), g(i, j+1, c), g(i+1, j, c), g(i+2, j-1, c), 0.5)
				w1, w2 := invW(d1), invW(d2)
				v := (w1*v1 + w2*v2) / (w1 + w2)
				hi.Data[((2*i+1)*hi.Cols+2*j+1)*ch+c] = clampByte(v)
			}
		}
	}
	hiAt := func(y, x, c int) float64 {
		y = reflectIdx(y, hi.Rows)
		x = reflectIdx(x, hi.Cols)
		return float64(hi.Data[(y*hi.Cols+x)*ch+c])
	}
	// Pass 2: axis pixels, blending horizontal vs vertical Catmull-Rom.
	fill := func(yy, xx int) {
		for c := 0; c < ch; c++ {
			var dv, dh float64
			for s := -1; s <= 1; s++ {
				dv += math.Abs(hiAt(yy-1, xx+2*s, c) - hiAt(yy+1, xx+2*s, c))
				dh += math.Abs(hiAt(yy+2*s, xx-1, c) - hiAt(yy+2*s, xx+1, c))
			}
			vVert := catmullRom(hiAt(yy-3, xx, c), hiAt(yy-1, xx, c), hiAt(yy+1, xx, c), hiAt(yy+3, xx, c), 0.5)
			vHorz := catmullRom(hiAt(yy, xx-3, c), hiAt(yy, xx-1, c), hiAt(yy, xx+1, c), hiAt(yy, xx+3, c), 0.5)
			wv, wh := invW(dv), invW(dh)
			v := (wv*vVert + wh*vHorz) / (wv + wh)
			hi.Data[(yy*hi.Cols+xx)*ch+c] = clampByte(v)
		}
	}
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			fill(2*i+1, 2*j)
			fill(2*i, 2*j+1)
		}
	}
	return hi
}

// UpsampleDCCI enlarges src by an arbitrary integer scale (>= 2) with
// Directional Cubic Convolution Interpolation. Each new pixel is interpolated
// along the locally smoothest direction with a Catmull-Rom cubic, which keeps
// diagonal edges sharp and free of the zig-zag artefacts that non-directional
// cubics produce. Powers of two use successive ×2 DCCI doublings; other factors
// finish with a bicubic resample to the exact size. It returns an error for an
// empty image or a scale below 2.
func UpsampleDCCI(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	return doubleThenResize(src, scale, dcciDouble), nil
}

// EdgeGuidedUpscale enlarges src by an arbitrary integer scale (>= 2) and
// sharpens it with gradient-profile sharpening: a bicubic base is enhanced by
// adding an unsharp detail band whose strength is modulated by the local
// gradient magnitude. Edges — where interpolation blurs the gradient profile
// most — are steepened, while flat regions (zero gradient) are left untouched,
// so the result gains apparent resolution without amplifying noise in smooth
// areas.
//
// strength scales the sharpening (1.0–2.0 is typical; 0 reproduces the bicubic
// base). It returns an error for an empty image or a scale below 2.
func EdgeGuidedUpscale(src *cv.Mat, scale int, strength float64) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	base := resampleSeparable(src, src.Cols*scale, src.Rows*scale, keysCubic, 2)
	blur := cv.GaussianBlur(base, 3, 1.0)
	gray := grayscale(base)
	gx := cv.SobelFloat(gray, 1, 0, 3)[0]
	gy := cv.SobelFloat(gray, 0, 1, 3)[0]
	var maxMag float64
	mag := make([]float64, base.Rows*base.Cols)
	for i := range mag {
		m := math.Hypot(gx[i], gy[i])
		mag[i] = m
		if m > maxMag {
			maxMag = m
		}
	}
	dst := cv.NewMat(base.Rows, base.Cols, base.Channels)
	ch := base.Channels
	for p := 0; p < base.Rows*base.Cols; p++ {
		var g float64
		if maxMag > 1e-9 {
			g = mag[p] / maxMag
		}
		for c := 0; c < ch; c++ {
			idx := p*ch + c
			b := float64(base.Data[idx])
			detail := b - float64(blur.Data[idx])
			dst.Data[idx] = clampByte(b + strength*g*detail)
		}
	}
	return dst, nil
}

// UpsampleGradientProfile is a convenience wrapper over [EdgeGuidedUpscale] with
// a fixed sharpening strength (1.5). It enlarges src by an arbitrary integer
// scale (>= 2), returning an error for an empty image or a scale below 2.
func UpsampleGradientProfile(src *cv.Mat, scale int) (*cv.Mat, error) {
	return EdgeGuidedUpscale(src, scale, 1.5)
}

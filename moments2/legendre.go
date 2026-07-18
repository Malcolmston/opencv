package moments2

import cv "github.com/malcolmston/opencv"

// LegendrePolynomial evaluates the Legendre polynomial P_n(x) using the stable
// Bonnet recurrence. The argument x is normally taken in [-1, 1].
func LegendrePolynomial(n int, x float64) float64 {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return x
	}
	pPrev := 1.0
	pCur := x
	for k := 2; k <= n; k++ {
		fk := float64(k)
		pNext := ((2*fk-1)*x*pCur - (fk-1)*pPrev) / fk
		pPrev = pCur
		pCur = pNext
	}
	return pCur
}

// LegendreMoment computes the Legendre moment L_pq of a single-channel image.
// Pixel centres are mapped onto the square [-1, 1] x [-1, 1] and the moment is
// the intensity-weighted projection onto the product P_p(x)*P_q(y) with the
// standard normalization (2p+1)(2q+1)/4. For a constant image of value c the
// moment L_00 equals c. It panics if src is not single-channel.
func LegendreMoment(src *cv.Mat, p, q int) float64 {
	moments2requireGray(src, "LegendreMoment")
	cols := src.Cols
	rows := src.Rows
	dx := 2.0 / float64(cols)
	dy := 2.0 / float64(rows)
	px := make([]float64, cols)
	for x := 0; x < cols; x++ {
		xn := (2*float64(x) + 1 - float64(cols)) / float64(cols)
		px[x] = LegendrePolynomial(p, xn)
	}
	var sum float64
	for y := 0; y < rows; y++ {
		yn := (2*float64(y) + 1 - float64(rows)) / float64(rows)
		py := LegendrePolynomial(q, yn)
		row := y * cols
		for x := 0; x < cols; x++ {
			v := float64(src.Data[row+x])
			if v == 0 {
				continue
			}
			sum += v * px[x] * py
		}
	}
	norm := float64((2*p+1)*(2*q+1)) / 4
	return norm * sum * dx * dy
}

// LegendreMoments returns all Legendre moments L_pq with p+q <= maxOrder as a
// dense (maxOrder+1) x (maxOrder+1) matrix; entries with p+q > maxOrder are left
// at zero. It panics if src is not single-channel or maxOrder is negative.
func LegendreMoments(src *cv.Mat, maxOrder int) [][]float64 {
	moments2requireGray(src, "LegendreMoments")
	if maxOrder < 0 {
		panic("moments2: LegendreMoments requires maxOrder >= 0")
	}
	out := make([][]float64, maxOrder+1)
	for p := 0; p <= maxOrder; p++ {
		out[p] = make([]float64, maxOrder+1)
		for q := 0; q <= maxOrder-p; q++ {
			out[p][q] = LegendreMoment(src, p, q)
		}
	}
	return out
}

package segment2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ActiveContourParams configures the [ActiveContour] parametric snake.
type ActiveContourParams struct {
	// Alpha is the elasticity (tension) weight penalising contour stretching;
	// larger values shrink the snake.
	Alpha float64
	// Beta is the rigidity (stiffness) weight penalising bending; larger values
	// keep the snake smooth.
	Beta float64
	// Gamma is the time step of the implicit Euler update; larger values take
	// bigger steps.
	Gamma float64
	// EdgeWeight scales the image force that pulls the snake toward edges.
	EdgeWeight float64
	// Sigma is the Gaussian smoothing applied to the image before its gradient
	// is measured; 0 disables smoothing.
	Sigma float64
	// Iterations is the number of evolution steps.
	Iterations int
}

// DefaultActiveContourParams returns reasonable defaults (Alpha 0.1, Beta 0.1,
// Gamma 1.0, EdgeWeight 2.0, Sigma 1.0, Iterations 100).
func DefaultActiveContourParams() ActiveContourParams {
	return ActiveContourParams{
		Alpha: 0.1, Beta: 0.1, Gamma: 1.0, EdgeWeight: 2.0, Sigma: 1.0, Iterations: 100,
	}
}

// segment2invert returns the inverse of the n×n matrix a (row-major) by
// Gauss-Jordan elimination with partial pivoting. It panics if a is singular.
func segment2invert(a []float64, n int) []float64 {
	m := make([]float64, n*2*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			m[i*2*n+j] = a[i*n+j]
		}
		m[i*2*n+n+i] = 1
	}
	for col := 0; col < n; col++ {
		// Partial pivot.
		piv := col
		best := math.Abs(m[col*2*n+col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r*2*n+col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-12 {
			panic("segment2: singular internal-energy matrix")
		}
		if piv != col {
			for j := 0; j < 2*n; j++ {
				m[col*2*n+j], m[piv*2*n+j] = m[piv*2*n+j], m[col*2*n+j]
			}
		}
		pv := m[col*2*n+col]
		for j := 0; j < 2*n; j++ {
			m[col*2*n+j] /= pv
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r*2*n+col]
			if f == 0 {
				continue
			}
			for j := 0; j < 2*n; j++ {
				m[r*2*n+j] -= f * m[col*2*n+j]
			}
		}
	}
	inv := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			inv[i*n+j] = m[i*2*n+n+j]
		}
	}
	return inv
}

// segment2sampleField bilinearly samples the scalar field (rows×cols) at the
// continuous coordinate (x, y), clamping to the border.
func segment2sampleField(field []float64, rows, cols int, x, y float64) float64 {
	if x < 0 {
		x = 0
	}
	if x > float64(cols-1) {
		x = float64(cols - 1)
	}
	if y < 0 {
		y = 0
	}
	if y > float64(rows-1) {
		y = float64(rows - 1)
	}
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 > cols-1 {
		x1 = cols - 1
	}
	if y1 > rows-1 {
		y1 = rows - 1
	}
	fx := x - float64(x0)
	fy := y - float64(y0)
	v00 := field[y0*cols+x0]
	v01 := field[y0*cols+x1]
	v10 := field[y1*cols+x0]
	v11 := field[y1*cols+x1]
	return v00*(1-fx)*(1-fy) + v01*fx*(1-fy) + v10*(1-fx)*fy + v11*fx*fy
}

// ActiveContour evolves a closed parametric snake (Kass, Witkin & Terzopoulos)
// toward image edges. initial is the ordered vertex ring of the starting
// contour; the returned slice is the evolved contour with the same number of
// vertices. Internal elasticity and rigidity forces (Alpha, Beta) keep the
// contour smooth while an external force proportional to the gradient of the
// image edge map (scaled by EdgeWeight) attracts it to strong edges. The update
// is the classic implicit Euler step X <- (A + gamma*I)^-1 (gamma*X + f), with A
// the pentadiagonal internal-energy matrix, solved once per axis.
//
// It panics if img is empty, the contour has fewer than 3 vertices, or the
// parameters make the internal-energy matrix singular.
func ActiveContour(img *cv.Mat, initial []cv.Point, params ActiveContourParams) []cv.Point {
	segment2requireNonEmpty(img, "ActiveContour")
	n := len(initial)
	if n < 3 {
		panic("segment2: ActiveContour requires at least 3 vertices")
	}
	if params.Iterations < 1 {
		params.Iterations = 1
	}
	rows, cols := img.Rows, img.Cols

	// Edge map = gradient magnitude of the (optionally smoothed) image; its own
	// gradient is the external force pulling the snake toward edges.
	sm := segment2gaussianBlur(img, params.Sigma)
	gray := segment2gray(sm)
	mag := segment2sobelMag(gray, rows, cols)
	fx := make([]float64, rows*cols)
	fy := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			xm := x - 1
			if xm < 0 {
				xm = 0
			}
			xp := x + 1
			if xp > cols-1 {
				xp = cols - 1
			}
			ym := y - 1
			if ym < 0 {
				ym = 0
			}
			yp := y + 1
			if yp > rows-1 {
				yp = rows - 1
			}
			fx[y*cols+x] = (mag[y*cols+xp] - mag[y*cols+xm]) * 0.5
			fy[y*cols+x] = (mag[yp*cols+x] - mag[ym*cols+x]) * 0.5
		}
	}

	// Build the cyclic pentadiagonal internal-energy matrix A and P = A+gamma*I.
	a := params.Alpha
	b := params.Beta
	d2 := 2*a + 6*b
	d1 := -a - 4*b
	d0 := b
	P := make([]float64, n*n)
	set := func(i, j, v float64) {
		P[int(i)*n+int(j)] += v
	}
	for i := 0; i < n; i++ {
		set(float64(i), float64(i), d2+params.Gamma)
		set(float64(i), float64((i+1)%n), d1)
		set(float64(i), float64((i-1+n)%n), d1)
		set(float64(i), float64((i+2)%n), d0)
		set(float64(i), float64((i-2+n)%n), d0)
	}
	inv := segment2invert(P, n)

	xs := make([]float64, n)
	ys := make([]float64, n)
	for i, p := range initial {
		xs[i] = float64(p.X)
		ys[i] = float64(p.Y)
	}

	nx := make([]float64, n)
	ny := make([]float64, n)
	for it := 0; it < params.Iterations; it++ {
		for i := 0; i < n; i++ {
			bx := params.Gamma*xs[i] + params.EdgeWeight*segment2sampleField(fx, rows, cols, xs[i], ys[i])
			by := params.Gamma*ys[i] + params.EdgeWeight*segment2sampleField(fy, rows, cols, xs[i], ys[i])
			nx[i] = bx
			ny[i] = by
		}
		for i := 0; i < n; i++ {
			var sx, sy float64
			row := inv[i*n : i*n+n]
			for j := 0; j < n; j++ {
				sx += row[j] * nx[j]
				sy += row[j] * ny[j]
			}
			xs[i] = sx
			ys[i] = sy
		}
		// Clamp to image bounds.
		for i := 0; i < n; i++ {
			if xs[i] < 0 {
				xs[i] = 0
			}
			if xs[i] > float64(cols-1) {
				xs[i] = float64(cols - 1)
			}
			if ys[i] < 0 {
				ys[i] = 0
			}
			if ys[i] > float64(rows-1) {
				ys[i] = float64(rows - 1)
			}
		}
	}

	out := make([]cv.Point, n)
	for i := 0; i < n; i++ {
		out[i] = cv.Point{X: int(xs[i] + 0.5), Y: int(ys[i] + 0.5)}
	}
	return out
}

// ContourPerimeter returns the closed-polygon perimeter (summed edge lengths,
// including the closing edge) of the ordered point ring. It returns 0 for fewer
// than two points.
func ContourPerimeter(pts []cv.Point) float64 {
	if len(pts) < 2 {
		return 0
	}
	var s float64
	for i := 0; i < len(pts); i++ {
		j := (i + 1) % len(pts)
		dx := float64(pts[j].X - pts[i].X)
		dy := float64(pts[j].Y - pts[i].Y)
		s += math.Sqrt(dx*dx + dy*dy)
	}
	return s
}

// ContourArea returns the absolute area enclosed by the ordered point ring using
// the shoelace formula. It returns 0 for fewer than three points.
func ContourArea(pts []cv.Point) float64 {
	if len(pts) < 3 {
		return 0
	}
	var s float64
	for i := 0; i < len(pts); i++ {
		j := (i + 1) % len(pts)
		s += float64(pts[i].X)*float64(pts[j].Y) - float64(pts[j].X)*float64(pts[i].Y)
	}
	return math.Abs(s) / 2
}

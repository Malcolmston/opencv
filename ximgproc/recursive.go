package ximgproc

// causalPass applies the second-order recursive filter
//
//	y[i] = n0·x[i] + n1·x[i-1] + n2·x[i-2] + b1·y[i-1] + b2·y[i-2]
//
// left to right and returns y. Out-of-range inputs use replicate borders
// (x[i<0] = x[0]) and the feedback state is initialised to the steady-state
// response for that constant border, so a constant input yields a constant
// output with no start-up transient. It never mutates x.
func causalPass(x []float64, n0, n1, n2, b1, b2 float64) []float64 {
	n := len(x)
	y := make([]float64, n)
	if n == 0 {
		return y
	}
	den := 1 - b1 - b2
	var yss float64
	if den != 0 {
		yss = (n0 + n1 + n2) * x[0] / den
	}
	ym1, ym2 := yss, yss
	for i := 0; i < n; i++ {
		xm1 := x[0]
		if i >= 1 {
			xm1 = x[i-1]
		}
		xm2 := x[0]
		if i >= 2 {
			xm2 = x[i-2]
		}
		v := n0*x[i] + n1*xm1 + n2*xm2 + b1*ym1 + b2*ym2
		y[i] = v
		ym2 = ym1
		ym1 = v
	}
	return y
}

// anticausalPass applies the mirror-image second-order recursive filter
//
//	y[i] = n0·x[i] + n1·x[i+1] + n2·x[i+2] + b1·y[i+1] + b2·y[i+2]
//
// right to left and returns y, with replicate borders (x[i≥n] = x[n-1]) and
// steady-state feedback initialisation. It never mutates x.
func anticausalPass(x []float64, n0, n1, n2, b1, b2 float64) []float64 {
	n := len(x)
	y := make([]float64, n)
	if n == 0 {
		return y
	}
	den := 1 - b1 - b2
	var yss float64
	if den != 0 {
		yss = (n0 + n1 + n2) * x[n-1] / den
	}
	yp1, yp2 := yss, yss
	for i := n - 1; i >= 0; i-- {
		xp1 := x[n-1]
		if i+1 < n {
			xp1 = x[i+1]
		}
		xp2 := x[n-1]
		if i+2 < n {
			xp2 = x[i+2]
		}
		v := n0*x[i] + n1*xp1 + n2*xp2 + b1*yp1 + b2*yp2
		y[i] = v
		yp2 = yp1
		yp1 = v
	}
	return y
}

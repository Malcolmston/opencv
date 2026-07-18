package filters2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SteerableG1 holds the two basis responses of a first-derivative-of-Gaussian
// steerable filter (Freeman and Adelson): Gx is the response to the x
// derivative and Gy to the y derivative. The response at any orientation is
// recovered analytically by [SteerableG1.Steer] as a linear combination of the
// two, so an arbitrarily oriented first-derivative filter is obtained without
// re-convolving.
type SteerableG1 struct {
	// Gx is the response to the x-derivative-of-Gaussian basis filter.
	Gx *FloatImage
	// Gy is the response to the y-derivative-of-Gaussian basis filter.
	Gy *FloatImage
}

// SteerG1 convolves a single-channel image with the two first-derivative-of-
// Gaussian basis filters of the given standard deviation and returns their
// responses. The kernel size is chosen from sigma unless overridden by a
// positive size. It panics on multi-channel or empty input or a non-positive
// sigma.
func SteerG1(src *cv.Mat, sigma float64, size int) *SteerableG1 {
	requireGray(src, "SteerG1")
	if size <= 0 {
		size = logKernelSize(sigma)
	}
	fi := MatToFloatImage(src)
	gx := Convolve(fi, GaussianDerivativeKernel(size, sigma, 1, 0))
	gy := Convolve(fi, GaussianDerivativeKernel(size, sigma, 0, 1))
	return &SteerableG1{Gx: gx, Gy: gy}
}

// Steer returns the first-derivative-of-Gaussian response oriented at theta
// radians, computed as cos(theta)*Gx + sin(theta)*Gy.
func (s *SteerableG1) Steer(theta float64) *FloatImage {
	ct, st := math.Cos(theta), math.Sin(theta)
	out := NewFloatImage(s.Gx.Rows, s.Gx.Cols)
	for i := range out.Data {
		out.Data[i] = ct*s.Gx.Data[i] + st*s.Gy.Data[i]
	}
	return out
}

// OrientationEnergy returns the orientation-independent gradient energy
// sqrt(Gx^2+Gy^2) of the first-derivative basis, the largest response
// obtainable over all orientations.
func (s *SteerableG1) OrientationEnergy() *FloatImage {
	return Magnitude(s.Gx, s.Gy)
}

// DominantOrientation returns, per pixel, the orientation in radians (in
// (-pi,pi]) that maximises the first-derivative response, namely
// atan2(Gy,Gx).
func (s *SteerableG1) DominantOrientation() *FloatImage {
	out := NewFloatImage(s.Gx.Rows, s.Gx.Cols)
	for i := range out.Data {
		out.Data[i] = math.Atan2(s.Gy.Data[i], s.Gx.Data[i])
	}
	return out
}

// SteerableG2 holds the three basis responses of a second-derivative-of-
// Gaussian steerable filter (Freeman and Adelson): Gxx, Gxy and Gyy are the
// responses to the xx, xy and yy second-derivative basis filters. The response
// at any orientation is recovered analytically by [SteerableG2.Steer].
type SteerableG2 struct {
	// Gxx is the response to the second x-derivative basis filter.
	Gxx *FloatImage
	// Gxy is the response to the mixed xy-derivative basis filter.
	Gxy *FloatImage
	// Gyy is the response to the second y-derivative basis filter.
	Gyy *FloatImage
}

// SteerG2 convolves a single-channel image with the three second-derivative-of-
// Gaussian basis filters of the given standard deviation and returns their
// responses. The kernel size is chosen from sigma unless overridden by a
// positive size. It panics on multi-channel or empty input or a non-positive
// sigma.
func SteerG2(src *cv.Mat, sigma float64, size int) *SteerableG2 {
	requireGray(src, "SteerG2")
	if size <= 0 {
		size = logKernelSize(sigma)
	}
	fi := MatToFloatImage(src)
	return &SteerableG2{
		Gxx: Convolve(fi, GaussianDerivativeKernel(size, sigma, 2, 0)),
		Gxy: Convolve(fi, GaussianDerivativeKernel(size, sigma, 1, 1)),
		Gyy: Convolve(fi, GaussianDerivativeKernel(size, sigma, 0, 2)),
	}
}

// Steer returns the second-derivative-of-Gaussian response oriented at theta
// radians, computed with the Freeman-Adelson interpolation
// cos^2(theta)*Gxx + 2*cos(theta)*sin(theta)*Gxy + sin^2(theta)*Gyy.
func (s *SteerableG2) Steer(theta float64) *FloatImage {
	ct, st := math.Cos(theta), math.Sin(theta)
	ka := ct * ct
	kb := 2 * ct * st
	kc := st * st
	out := NewFloatImage(s.Gxx.Rows, s.Gxx.Cols)
	for i := range out.Data {
		out.Data[i] = ka*s.Gxx.Data[i] + kb*s.Gxy.Data[i] + kc*s.Gyy.Data[i]
	}
	return out
}

// OrientationEnergy returns, per pixel, the maximum squared second-derivative
// response over all orientations. Writing the steered response as a quadratic
// form in (cos theta, sin theta), the extreme value is obtained in closed form
// from the basis responses without sampling orientations.
func (s *SteerableG2) OrientationEnergy() *FloatImage {
	out := NewFloatImage(s.Gxx.Rows, s.Gxx.Cols)
	for i := range out.Data {
		a := s.Gxx.Data[i]
		b := s.Gxy.Data[i]
		c := s.Gyy.Data[i]
		// Steered value r(t)=a*cos^2 t + 2b*cos t*sin t + c*sin^2 t
		//            = (a+c)/2 + ((a-c)/2)cos2t + b*sin2t.
		mean := (a + c) / 2
		amp := math.Hypot((a-c)/2, b)
		// The response of largest magnitude is mean +/- amp.
		hi := mean + amp
		lo := mean - amp
		if math.Abs(lo) > math.Abs(hi) {
			out.Data[i] = lo
		} else {
			out.Data[i] = hi
		}
	}
	return out
}

// DominantOrientation returns, per pixel, the orientation in radians that
// maximises the magnitude of the steered second-derivative response. Writing
// the response as mean + amp*cos(2*theta - phi) with phi = atan2(2*Gxy,
// Gxx-Gyy), the extremum lies at phi/2 or phi/2 + pi/2 depending on the sign of
// mean; the returned angle is folded into (-pi/2, pi/2].
func (s *SteerableG2) DominantOrientation() *FloatImage {
	out := NewFloatImage(s.Gxx.Rows, s.Gxx.Cols)
	for i := range out.Data {
		a := s.Gxx.Data[i]
		b := s.Gxy.Data[i]
		c := s.Gyy.Data[i]
		mean := (a + c) / 2
		amp := math.Hypot((a-c)/2, b)
		phi := math.Atan2(2*b, a-c)
		theta := phi / 2
		if math.Abs(mean-amp) > math.Abs(mean+amp) {
			theta += math.Pi / 2
		}
		// Fold into (-pi/2, pi/2].
		for theta > math.Pi/2 {
			theta -= math.Pi
		}
		for theta <= -math.Pi/2 {
			theta += math.Pi
		}
		out.Data[i] = theta
	}
	return out
}

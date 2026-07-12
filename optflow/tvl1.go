package optflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TVL1Params configures the duality-based TV-L1 optical-flow solver
// ([CalcOpticalFlowDenseTVL1] and [DualTVL1OpticalFlow]). The defaults from
// [DefaultTVL1Params] follow Zach, Pock & Bischof, "A Duality Based Approach for
// Realtime TV-L1 Optical Flow" (2007), and match OpenCV's DualTVL1OpticalFlow.
type TVL1Params struct {
	// Tau is the time step of the dual (TV) ascent update. Stable at 0.25.
	Tau float64
	// Lambda weights the L1 data term against the total-variation regulariser.
	// Smaller Lambda yields smoother flow.
	Lambda float64
	// Theta is the tightness of the quadratic relaxation coupling the flow u to
	// the auxiliary variable v.
	Theta float64
	// Scales is the number of pyramid levels (>= 1). More scales recover larger
	// displacements.
	Scales int
	// Warps is the number of warping (re-linearisation) steps per scale.
	Warps int
	// Iterations is the number of inner primal-dual updates per warp.
	Iterations int
	// Epsilon is the stopping threshold on the mean primal update; a value <= 0
	// disables early stopping and always runs the full Iterations.
	Epsilon float64
}

// DefaultTVL1Params returns the standard TV-L1 configuration
// (Tau=0.25, Lambda=0.15, Theta=0.3, Scales=5, Warps=5, Iterations=30,
// Epsilon=0.01).
func DefaultTVL1Params() TVL1Params {
	return TVL1Params{
		Tau:        0.25,
		Lambda:     0.15,
		Theta:      0.3,
		Scales:     5,
		Warps:      5,
		Iterations: 30,
		Epsilon:    0.01,
	}
}

func (p TVL1Params) validate() {
	if p.Tau <= 0 || p.Lambda <= 0 || p.Theta <= 0 {
		panic("optflow: TVL1 requires Tau, Lambda, Theta > 0")
	}
	if p.Scales < 1 || p.Warps < 1 || p.Iterations < 1 {
		panic("optflow: TVL1 requires Scales, Warps, Iterations >= 1")
	}
}

// DualTVL1OpticalFlow is a reusable TV-L1 dense optical-flow estimator, the
// stdlib port of OpenCV's cv::optflow::DualTVL1OpticalFlow /
// cv::DualTVL1OpticalFlow object. Construct it with [NewDualTVL1OpticalFlow] and
// call [DualTVL1OpticalFlow.Calc] for each frame pair; the configuration is held
// in Params and may be tweaked between calls.
type DualTVL1OpticalFlow struct {
	// Params holds the solver configuration used by Calc.
	Params TVL1Params
}

// NewDualTVL1OpticalFlow returns a DualTVL1OpticalFlow configured with p.
func NewDualTVL1OpticalFlow(p TVL1Params) *DualTVL1OpticalFlow {
	p.validate()
	return &DualTVL1OpticalFlow{Params: p}
}

// Calc computes the dense TV-L1 flow from prev to next. It is equivalent to
// [CalcOpticalFlowDenseTVL1] with the receiver's Params and exists to mirror
// OpenCV's stateful DenseOpticalFlow interface.
func (d *DualTVL1OpticalFlow) Calc(prev, next *cv.Mat) *FlowField {
	return CalcOpticalFlowDenseTVL1(prev, next, d.Params)
}

// CalcOpticalFlowDenseTVL1 computes a dense optical-flow field from prev to next
// with the duality-based TV-L1 method of Zach, Pock & Bischof (2007).
//
// The energy minimised is the total variation of the flow plus an L1
// (robust) brightness-constancy data term:
//
//	E(u) = ∫ |∇u| + λ |I1(x + u(x)) − I0(x)| dx
//
// Direct minimisation is hard because the data term is non-differentiable, so
// the solver splits it via a tight quadratic coupling to an auxiliary field v
// (weight 1/θ) and alternates two closed-form steps: a point-wise soft
// thresholding of the linearised data residual (the L1 "shrinkage" that yields
// v from u) and a Chambolle dual-projection TV step (that yields u from v via
// the divergence of a dual variable p). The whole scheme runs coarse-to-fine on
// a Gaussian pyramid with Warps re-linearisations per level, so displacements
// much larger than one pixel are recovered.
//
// prev and next must be non-empty and identically sized; multi-channel inputs
// are converted to grayscale. p is validated (see [TVL1Params]). The computation
// is fully deterministic. For repeated use prefer a [DualTVL1OpticalFlow] value.
func CalcOpticalFlowDenseTVL1(prev, next *cv.Mat, p TVL1Params) *FlowField {
	requirePair(prev, next, "CalcOpticalFlowDenseTVL1")
	p.validate()

	i0 := grayGrid(prev)
	i1 := grayGrid(next)

	pPyr, nPyr := scalePyramidLevels(i0, i1, p.Scales-1, 4)
	nl := len(pPyr)

	var u1, u2 []float64
	var pr, pc int
	for lvl := nl - 1; lvl >= 0; lvl-- {
		I0 := pPyr[lvl]
		I1 := nPyr[lvl]
		rows, cols := I0.Rows, I0.Cols
		if u1 == nil {
			u1 = make([]float64, rows*cols)
			u2 = make([]float64, rows*cols)
		} else {
			u1, u2 = upscaleFlow(u1, u2, pr, pc, rows, cols)
		}
		tvl1Scale(I0, I1, u1, u2, p)
		pr, pc = rows, cols
	}
	return flowFromPlanes(u1, u2, pr, pc)
}

// tvl1Scale runs the warping loop and primal-dual iterations for a single
// pyramid level, updating u1, u2 in place.
func tvl1Scale(I0, I1 *grid, u1, u2 []float64, p TVL1Params) {
	rows, cols := I0.Rows, I0.Cols
	n := rows * cols
	i1x, i1y := sobelGradients(I1)

	p11 := make([]float64, n)
	p12 := make([]float64, n)
	p21 := make([]float64, n)
	p22 := make([]float64, n)

	v1 := make([]float64, n)
	v2 := make([]float64, n)

	l1t := p.Lambda * p.Theta
	taut := p.Tau / p.Theta

	for warp := 0; warp < p.Warps; warp++ {
		// Motion-compensate the next frame and its gradients by the current flow.
		i1w := warpGrid(I1, u1, u2)
		i1wx := warpGrid(i1x, u1, u2)
		i1wy := warpGrid(i1y, u1, u2)

		grad := make([]float64, n)
		rhoC := make([]float64, n)
		for i := 0; i < n; i++ {
			gx := i1wx.Data[i]
			gy := i1wy.Data[i]
			grad[i] = gx*gx + gy*gy
			rhoC[i] = i1w.Data[i] - gx*u1[i] - gy*u2[i] - I0.Data[i]
		}

		for it := 0; it < p.Iterations; it++ {
			// Thresholding step: solve the point-wise L1 data problem for v.
			for i := 0; i < n; i++ {
				gx := i1wx.Data[i]
				gy := i1wy.Data[i]
				rho := rhoC[i] + gx*u1[i] + gy*u2[i]
				th := l1t * grad[i]
				var d1, d2 float64
				switch {
				case rho < -th:
					d1 = l1t * gx
					d2 = l1t * gy
				case rho > th:
					d1 = -l1t * gx
					d2 = -l1t * gy
				case grad[i] > 1e-10:
					d1 = -rho * gx / grad[i]
					d2 = -rho * gy / grad[i]
				}
				v1[i] = u1[i] + d1
				v2[i] = u2[i] + d2
			}

			// Primal update: u = v + theta * div(p), tracking the mean change so
			// the iteration can stop early once the field settles.
			div1 := divergence(p11, p12, rows, cols)
			div2 := divergence(p21, p22, rows, cols)
			var change float64
			for i := 0; i < n; i++ {
				nu1 := v1[i] + p.Theta*div1[i]
				nu2 := v2[i] + p.Theta*div2[i]
				change += math.Abs(nu1-u1[i]) + math.Abs(nu2-u2[i])
				u1[i] = nu1
				u2[i] = nu2
			}

			// Dual update: gradient ascent on p with re-projection onto the unit
			// ball (Chambolle's semi-implicit scheme).
			u1x, u1y := forwardGradient(u1, rows, cols)
			u2x, u2y := forwardGradient(u2, rows, cols)
			for i := 0; i < n; i++ {
				ng1 := 1.0 + taut*math.Hypot(u1x[i], u1y[i])
				p11[i] = (p11[i] + taut*u1x[i]) / ng1
				p12[i] = (p12[i] + taut*u1y[i]) / ng1
				ng2 := 1.0 + taut*math.Hypot(u2x[i], u2y[i])
				p21[i] = (p21[i] + taut*u2x[i]) / ng2
				p22[i] = (p22[i] + taut*u2y[i]) / ng2
			}

			if p.Epsilon > 0 && change/float64(n) < p.Epsilon {
				break
			}
		}
	}
}

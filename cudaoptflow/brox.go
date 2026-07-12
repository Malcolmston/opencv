package cudaoptflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/optflow"
)

// BroxOpticalFlow is the CPU-backed mirror of cv::cuda::BroxOpticalFlow. Unlike
// the other estimators in this package it does not delegate: it implements a
// genuine, self-contained Brox-style variational optical-flow solver in pure Go.
//
// The energy minimised follows Brox, Bruhn, Papenberg & Weickert, "High Accuracy
// Optical Flow Estimation Based on a Theory for Warping" (2004): a robust
// (Charbonnier) data term combining brightness constancy with gradient
// constancy, plus a robust total-variation smoothness term:
//
//	E(u,v) = ∫ Ψ(|I1(x+w)-I0|² + γ|∇I1(x+w)-∇I0|²) + α Ψ(|∇u|²+|∇v|²) dx
//
// with Ψ(s²)=√(s²+ε²). It is minimised coarse-to-fine on a dyadic image pyramid.
// At each level the current flow warps the second frame toward the first
// (OuterIterations warps); each warp linearises the data term, and the resulting
// nonlinear system is solved by a lagged-diffusivity fixed point
// (InnerIterations) whose inner linear system is relaxed with Gauss-Seidel /
// successive over-relaxation (SolverIterations). The flow is then prolongated to
// the next finer level. This recovers displacements far larger than one pixel
// and yields sub-pixel accuracy.
type BroxOpticalFlow struct {
	// Alpha is the smoothness weight (α). Larger values give smoother flow.
	Alpha float64
	// Gamma is the gradient-constancy weight (γ). Zero disables gradient
	// constancy, leaving pure brightness constancy.
	Gamma float64
	// ScaleFactor is the pyramid downsampling ratio in (0,1). It controls how
	// many coarse levels are built (a value near 0.5 gives a dyadic pyramid,
	// which this implementation uses).
	ScaleFactor float64
	// InnerIterations is the number of lagged-diffusivity fixed-point iterations
	// per warp (>= 1).
	InnerIterations int
	// OuterIterations is the number of warps per pyramid level (>= 1).
	OuterIterations int
	// SolverIterations is the number of Gauss-Seidel/SOR sweeps of the inner
	// linear system (>= 1).
	SolverIterations int
}

// NewBroxOpticalFlow creates a Brox variational estimator with explicit
// parameters, mirroring
// cv::cuda::BroxOpticalFlow::create(alpha, gamma, scaleFactor, innerIterations,
// outerIterations, solverIterations). alpha, gamma must be >= 0, scaleFactor in
// (0,1), and the three iteration counts >= 1.
func NewBroxOpticalFlow(alpha, gamma, scaleFactor float64, inner, outer, solver int) *BroxOpticalFlow {
	if alpha < 0 || gamma < 0 {
		panic("cudaoptflow: NewBroxOpticalFlow requires alpha >= 0 and gamma >= 0")
	}
	if scaleFactor <= 0 || scaleFactor >= 1 {
		panic("cudaoptflow: NewBroxOpticalFlow requires 0 < scaleFactor < 1")
	}
	if inner < 1 || outer < 1 || solver < 1 {
		panic("cudaoptflow: NewBroxOpticalFlow requires inner, outer, solver >= 1")
	}
	return &BroxOpticalFlow{
		Alpha:            alpha,
		Gamma:            gamma,
		ScaleFactor:      scaleFactor,
		InnerIterations:  inner,
		OuterIterations:  outer,
		SolverIterations: solver,
	}
}

// DefaultBroxOpticalFlow returns a Brox estimator with parameters tuned for
// interactive CPU use. It keeps OpenCV's algorithmic defaults for alpha (0.197),
// gamma (50) and scaleFactor (0.5) but uses far fewer outer iterations than the
// GPU default of 150, which would be impractically slow in software: inner 3,
// outer 10, solver 20.
func DefaultBroxOpticalFlow() *BroxOpticalFlow {
	return NewBroxOpticalFlow(0.197, 50, 0.5, 3, 10, 20)
}

// Calc computes a dense flow field from prev to next with the Brox variational
// solver, mirroring cv::cuda::BroxOpticalFlow::calc. The result is a full-
// precision [optflow.FlowField]. stream is accepted for API compatibility and
// ignored. prev and next must be non-empty and equally sized.
func (o *BroxOpticalFlow) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField {
	requireFramePair(prev, next, "BroxOpticalFlow.Calc")
	_ = stream

	i0 := grayGrid(prev.mat)
	i1 := grayGrid(next.mat)

	// Build a dyadic pyramid, coarsest last, stopping before a level becomes
	// smaller than a small minimum.
	const minSize = 8
	pyr0 := []*fgrid{i0}
	pyr1 := []*fgrid{i1}
	for {
		top := pyr0[len(pyr0)-1]
		if top.rows/2 < minSize || top.cols/2 < minSize {
			break
		}
		pyr0 = append(pyr0, top.downsample())
		pyr1 = append(pyr1, pyr1[len(pyr1)-1].downsample())
	}

	var u, v *fgrid
	for lvl := len(pyr0) - 1; lvl >= 0; lvl-- {
		f0 := pyr0[lvl]
		f1 := pyr1[lvl]
		if u == nil {
			u = newFgrid(f0.rows, f0.cols)
			v = newFgrid(f0.rows, f0.cols)
		} else {
			u = u.upsampleTo(f0.rows, f0.cols, float64(f0.cols)/float64(u.cols))
			v = v.upsampleTo(f0.rows, f0.cols, float64(f0.rows)/float64(v.rows))
		}
		o.solveLevel(f0, f1, u, v)
	}

	flow := optflow.NewFlowField(u.rows, u.cols)
	for y := 0; y < u.rows; y++ {
		for x := 0; x < u.cols; x++ {
			flow.Set(y, x, u.at(y, x), v.at(y, x))
		}
	}
	return flow
}

// solveLevel refines the flow (u,v) at a single pyramid level in place.
func (o *BroxOpticalFlow) solveLevel(i0, i1, u, v *fgrid) {
	rows, cols := i0.rows, i0.cols
	i0x, i0y := i0.gradients()

	const eps2 = 1e-6 // Charbonnier epsilon squared

	for warp := 0; warp < o.OuterIterations; warp++ {
		// Warp the second frame and its gradients toward the first using the
		// current flow.
		i1w := i1.warp(u, v)
		i1x, i1y := i1.gradients()
		i1wx := i1x.warp(u, v)
		i1wy := i1y.warp(u, v)
		// Second derivatives of the warped second frame.
		i1wxx, i1wxy := i1wx.gradients()
		i1wyx, i1wyy := i1wy.gradients()

		// Precompute the per-pixel linearisation coefficients.
		iz := make([]float64, rows*cols)  // brightness residual I1w - I0
		ix := make([]float64, rows*cols)  // averaged brightness gradient x
		iy := make([]float64, rows*cols)  // averaged brightness gradient y
		ixz := make([]float64, rows*cols) // gradient-constancy residual x
		iyz := make([]float64, rows*cols) // gradient-constancy residual y
		ixx := make([]float64, rows*cols) // d/dx of warped I1x
		ixy := make([]float64, rows*cols) // mixed second derivative
		iyy := make([]float64, rows*cols) // d/dy of warped I1y
		for i := range iz {
			iz[i] = i1w.data[i] - i0.data[i]
			ix[i] = 0.5 * (i0x.data[i] + i1wx.data[i])
			iy[i] = 0.5 * (i0y.data[i] + i1wy.data[i])
			ixz[i] = i1wx.data[i] - i0x.data[i]
			iyz[i] = i1wy.data[i] - i0y.data[i]
			ixx[i] = i1wxx.data[i]
			ixy[i] = 0.5 * (i1wxy.data[i] + i1wyx.data[i])
			iyy[i] = i1wyy.data[i]
		}

		du := make([]float64, rows*cols)
		dv := make([]float64, rows*cols)

		for inner := 0; inner < o.InnerIterations; inner++ {
			// Lagged diffusivities: recompute the robust weights from the
			// current increment.
			psiData := make([]float64, rows*cols)
			psiSmooth := make([]float64, rows*cols)
			for y := 0; y < rows; y++ {
				for x := 0; x < cols; x++ {
					i := y*cols + x
					rb := iz[i] + ix[i]*du[i] + iy[i]*dv[i]
					rgx := ixz[i] + ixx[i]*du[i] + ixy[i]*dv[i]
					rgy := iyz[i] + ixy[i]*du[i] + iyy[i]*dv[i]
					dataSq := rb*rb + o.Gamma*(rgx*rgx+rgy*rgy)
					psiData[i] = 0.5 / math.Sqrt(dataSq+eps2)

					ux := neighDiff(u, du, x, y, cols, rows, 1, 0)
					uyd := neighDiff(u, du, x, y, cols, rows, 0, 1)
					vx := neighDiff(v, dv, x, y, cols, rows, 1, 0)
					vyd := neighDiff(v, dv, x, y, cols, rows, 0, 1)
					smoothSq := ux*ux + uyd*uyd + vx*vx + vyd*vyd
					psiSmooth[i] = 0.5 / math.Sqrt(smoothSq+eps2)
				}
			}

			for sweep := 0; sweep < o.SolverIterations; sweep++ {
				broxSOR(o.Alpha, o.Gamma, u, v, du, dv,
					ix, iy, iz, ixx, ixy, iyy, ixz, iyz,
					psiData, psiSmooth, cols, rows)
			}
		}

		for i := range du {
			u.data[i] += du[i]
			v.data[i] += dv[i]
		}
	}
}

// broxSOR performs one Gauss-Seidel sweep of the linearised Brox system,
// updating the increments du, dv in place.
func broxSOR(alpha, gamma float64, u, v *fgrid, du, dv []float64,
	ix, iy, iz, ixx, ixy, iyy, ixz, iyz, psiData, psiSmooth []float64,
	cols, rows int) {
	const omega = 1.0 // relaxation factor (1.0 = plain Gauss-Seidel, stable)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			pd := psiData[i]

			a11 := pd * (ix[i]*ix[i] + gamma*(ixx[i]*ixx[i]+ixy[i]*ixy[i]))
			a12 := pd * (ix[i]*iy[i] + gamma*(ixx[i]*ixy[i]+ixy[i]*iyy[i]))
			a22 := pd * (iy[i]*iy[i] + gamma*(ixy[i]*ixy[i]+iyy[i]*iyy[i]))
			b1 := -pd * (ix[i]*iz[i] + gamma*(ixx[i]*ixz[i]+ixy[i]*iyz[i]))
			b2 := -pd * (iy[i]*iz[i] + gamma*(ixy[i]*ixz[i]+iyy[i]*iyz[i]))

			// Diffusion weights (harmonic-ish average of the smoothness
			// diffusivity between the centre pixel and each 4-neighbour).
			var wsum, su, sv float64
			addNeighbour := func(nx, ny int) {
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					return
				}
				j := ny*cols + nx
				w := alpha * 0.5 * (psiSmooth[i] + psiSmooth[j])
				wsum += w
				su += w * (u.data[j] + du[j] - u.data[i])
				sv += w * (v.data[j] + dv[j] - v.data[i])
			}
			addNeighbour(x-1, y)
			addNeighbour(x+1, y)
			addNeighbour(x, y-1)
			addNeighbour(x, y+1)

			denomU := a11 + wsum
			denomV := a22 + wsum
			if denomU <= 0 || denomV <= 0 {
				continue
			}
			newDu := (b1 - a12*dv[i] + su) / denomU
			newDv := (b2 - a12*du[i] + sv) / denomV
			du[i] += omega * (newDu - du[i])
			dv[i] += omega * (newDv - dv[i])
		}
	}
}

// neighDiff returns a forward-difference of the total field (base+increment)
// along (dx,dy), clamped at the boundary (yielding zero gradient there).
func neighDiff(base *fgrid, inc []float64, x, y, cols, rows, dx, dy int) float64 {
	nx, ny := x+dx, y+dy
	if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
		return 0
	}
	i := y*cols + x
	j := ny*cols + nx
	return (base.data[j] + inc[j]) - (base.data[i] + inc[i])
}

// grayGrid converts a Mat to a single-channel float grid of intensities in
// [0,1]. Multi-channel inputs are averaged to grayscale.
func grayGrid(m *cv.Mat) *fgrid {
	g := newFgrid(m.Rows, m.Cols)
	ch := m.Channels
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			var sum float64
			for c := 0; c < ch; c++ {
				sum += float64(m.At(y, x, c))
			}
			g.data[y*m.Cols+x] = sum / float64(ch) / 255.0
		}
	}
	return g
}

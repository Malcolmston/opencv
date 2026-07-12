package stereo

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DisparityWLSFilter is an edge-aware weighted-least-squares post-filter for
// disparity maps, mirroring the role of ximgproc::DisparityWLSFilter. It solves,
// by iterative weighted relaxation, the objective
//
//	minimise  Σ_p c(p)·(u(p) - d(p))²  +  λ Σ_{p,q∈N(p)} w(p,q)·(u(p) - u(q))²
//
// where d is the input disparity, c(p) is a per-pixel data confidence (1 for a
// valid input pixel, 0 for an invalid one so holes are filled purely by
// smoothing), and w(p,q) = exp(-|I(p)-I(q)|/SigmaColor) are guidance weights
// from the reference image I. The result both smooths noise and fills the gaps
// left by the uniqueness and left-right-consistency tests, while preserving
// depth discontinuities that coincide with image edges.
//
// The zero value applies mild smoothing; all fields default when non-positive.
type DisparityWLSFilter struct {
	// Lambda is the smoothness weight λ. Larger values smooth more aggressively.
	// Defaults to 1.0.
	Lambda float64
	// SigmaColor controls edge sensitivity: smaller values respect weaker edges.
	// Defaults to 12.0 (intensity units).
	SigmaColor float64
	// Iterations is the number of Gauss-Seidel relaxation sweeps. Defaults to 48.
	Iterations int
}

// Filter smooths and fills disp using guide as the edge reference. disp is a
// single-channel disparity map (0 = [InvalidDisparity], treated as a hole);
// guide may be single- or three-channel (converted to gray) and must match disp
// in size. The filtered map is returned; disp is not modified. It panics on nil,
// empty, multi-channel disp, or a size mismatch.
func (f DisparityWLSFilter) Filter(disp, guide *cv.Mat) *cv.Mat {
	return f.solve(disp, nil, guide)
}

// FilterWithConfidence is like [DisparityWLSFilter.Filter] but takes an explicit
// per-pixel confidence map (as produced by [ComputeConfidence]); confidence is
// scaled to [0, 1] and used as the data weight c(p), so low-confidence pixels
// are pulled toward their neighbours even when they carry a disparity. conf must
// match disp in size and be single-channel.
func (f DisparityWLSFilter) FilterWithConfidence(disp, conf, guide *cv.Mat) *cv.Mat {
	if conf == nil || conf.Empty() || conf.Channels != 1 {
		panic("stereo: FilterWithConfidence requires a single-channel confidence map")
	}
	if conf.Rows != disp.Rows || conf.Cols != disp.Cols {
		panic("stereo: FilterWithConfidence confidence/disparity size mismatch")
	}
	return f.solve(disp, conf, guide)
}

func (f DisparityWLSFilter) solve(disp, conf, guide *cv.Mat) *cv.Mat {
	if disp == nil || disp.Empty() || disp.Channels != 1 {
		panic("stereo: DisparityWLSFilter requires a single-channel disparity map")
	}
	g := grayMat(guide)
	if g.Rows != disp.Rows || g.Cols != disp.Cols {
		panic("stereo: DisparityWLSFilter guide/disparity size mismatch")
	}
	lambda := f.Lambda
	if lambda <= 0 {
		lambda = 1.0
	}
	sigma := f.SigmaColor
	if sigma <= 0 {
		sigma = 12.0
	}
	iters := f.Iterations
	if iters <= 0 {
		iters = 48
	}
	rows, cols := disp.Rows, disp.Cols
	n := rows * cols

	// Data term d and confidence c.
	d := make([]float64, n)
	c := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = float64(disp.Data[i])
		switch {
		case conf != nil:
			c[i] = float64(conf.Data[i]) / 255.0
		case disp.Data[i] == InvalidDisparity:
			c[i] = 0
		default:
			c[i] = 1
		}
	}

	// Precompute the four directional guidance weights per pixel (right, down are
	// enough since weights are symmetric; store per-edge).
	gi := make([]int, n)
	for i := 0; i < n; i++ {
		gi[i] = int(g.Data[i])
	}
	wRight := make([]float64, n) // weight to (x+1)
	wDown := make([]float64, n)  // weight to (y+1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			p := y*cols + x
			if x+1 < cols {
				wRight[p] = math.Exp(-math.Abs(float64(gi[p]-gi[p+1])) / sigma)
			}
			if y+1 < rows {
				wDown[p] = math.Exp(-math.Abs(float64(gi[p]-gi[p+cols])) / sigma)
			}
		}
	}

	// Initialise the solution with the data where confident, else the mean.
	u := make([]float64, n)
	var sum, cnt float64
	for i := 0; i < n; i++ {
		if c[i] > 0 {
			sum += d[i] * c[i]
			cnt += c[i]
		}
	}
	mean := 0.0
	if cnt > 0 {
		mean = sum / cnt
	}
	for i := 0; i < n; i++ {
		if c[i] > 0 {
			u[i] = d[i]
		} else {
			u[i] = mean
		}
	}

	// Gauss-Seidel sweeps of the weighted normal equations.
	for it := 0; it < iters; it++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				p := y*cols + x
				num := c[p] * d[p]
				den := c[p]
				if x > 0 {
					w := lambda * wRight[p-1]
					num += w * u[p-1]
					den += w
				}
				if x+1 < cols {
					w := lambda * wRight[p]
					num += w * u[p+1]
					den += w
				}
				if y > 0 {
					w := lambda * wDown[p-cols]
					num += w * u[p-cols]
					den += w
				}
				if y+1 < rows {
					w := lambda * wDown[p]
					num += w * u[p+cols]
					den += w
				}
				if den > 0 {
					u[p] = num / den
				}
			}
		}
	}

	out := cv.NewMat(rows, cols, 1)
	for i := 0; i < n; i++ {
		out.Data[i] = uint8(clampInt(int(math.Round(u[i])), 0, 255))
	}
	return out
}

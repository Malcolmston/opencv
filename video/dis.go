package video

import (
	cv "github.com/malcolmston/opencv"
)

// DIS preset identifiers accepted by [NewDISOpticalFlow]. They trade accuracy
// for speed by choosing the finest scale that is computed and how hard the
// per-patch inverse search works.
const (
	// DISPresetUltrafast stops at a coarse scale with few descent iterations.
	DISPresetUltrafast = iota
	// DISPresetFast is a balanced, quick configuration.
	DISPresetFast
	// DISPresetMedium computes down to a finer scale for better accuracy.
	DISPresetMedium
)

// DISOpticalFlow computes dense optical flow with the Dense Inverse Search
// algorithm (Kroeger et al., 2016), mirroring cv::DISOpticalFlow. It builds an
// image pyramid and, from coarse to fine, performs a patch-based inverse search:
// each patch's displacement is refined by Gauss-Newton minimisation of the
// sum-of-squared intensity difference using the patch's fixed structure tensor,
// after which the sparse patch displacements are densified into a per-pixel flow
// field by confidence-weighted averaging. The coarse estimate initialises the
// next finer level.
//
// Construct one with [NewDISOpticalFlow] and call [DISOpticalFlow.Calc]. The
// exported fields may be tuned after construction.
type DISOpticalFlow struct {
	// FinestScale is the finest pyramid level actually optimised (0 = full
	// resolution). Larger values are faster but blur fine motion.
	FinestScale int
	// PatchSize is the side length in pixels of a square search patch.
	PatchSize int
	// PatchStride is the spacing between patch centres; smaller means denser
	// coverage and smoother output.
	PatchStride int
	// GradientDescentIterations is the number of Gauss-Newton steps per patch.
	GradientDescentIterations int
}

// NewDISOpticalFlow returns a DISOpticalFlow configured from one of the
// DISPreset* constants. It panics on an unknown preset.
func NewDISOpticalFlow(preset int) *DISOpticalFlow {
	switch preset {
	case DISPresetUltrafast:
		return &DISOpticalFlow{FinestScale: 2, PatchSize: 8, PatchStride: 4, GradientDescentIterations: 12}
	case DISPresetFast:
		return &DISOpticalFlow{FinestScale: 2, PatchSize: 8, PatchStride: 3, GradientDescentIterations: 16}
	case DISPresetMedium:
		return &DISOpticalFlow{FinestScale: 1, PatchSize: 8, PatchStride: 3, GradientDescentIterations: 25}
	default:
		panic("video: NewDISOpticalFlow unknown preset")
	}
}

// resizeFlow bilinearly resamples a displacement grid to (rows, cols) and scales
// the sampled values by scale (used when moving between pyramid levels).
func resizeFlow(src *grid, rows, cols int, scale float64) *grid {
	dst := newGrid(rows, cols)
	sx := float64(src.Cols) / float64(cols)
	sy := float64(src.Rows) / float64(rows)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			fx := (float64(x)+0.5)*sx - 0.5
			fy := (float64(y)+0.5)*sy - 0.5
			dst.Data[y*cols+x] = src.bilinear(fx, fy) * scale
		}
	}
	return dst
}

// refineLevel runs the patch inverse search and densification for one pyramid
// level, given the previous/next intensity grids, the previous frame's
// gradients, and the initial per-pixel displacement (initU, initV). It returns
// the refined displacement grids.
func (d *DISOpticalFlow) refineLevel(prevG, nextG, gxp, gyp, initU, initV *grid) (*grid, *grid) {
	rows, cols := prevG.Rows, prevG.Cols
	half := d.PatchSize / 2
	stride := d.PatchStride
	if stride < 1 {
		stride = 1
	}
	sumU := newGrid(rows, cols)
	sumV := newGrid(rows, cols)
	sumW := newGrid(rows, cols)

	const reg = 1e-3 // structure-tensor regularisation

	for py := half; py < rows-half; py += stride {
		for px := half; px < cols-half; px += stride {
			u := initU.bilinear(float64(px), float64(py))
			v := initV.bilinear(float64(px), float64(py))

			// Fixed structure tensor of the previous-frame patch.
			var a, b, c float64
			for wy := -half; wy < half; wy++ {
				for wx := -half; wx < half; wx++ {
					ix := gxp.atClamp(px+wx, py+wy)
					iy := gyp.atClamp(px+wx, py+wy)
					a += ix * ix
					b += ix * iy
					c += iy * iy
				}
			}
			a += reg
			c += reg
			det := a*c - b*b
			if det <= 0 {
				continue
			}

			var cost float64
			for it := 0; it < d.GradientDescentIterations; it++ {
				var bx, by, e2 float64
				for wy := -half; wy < half; wy++ {
					for wx := -half; wx < half; wx++ {
						sx := float64(px + wx)
						sy := float64(py + wy)
						diff := prevG.atClamp(px+wx, py+wy) - nextG.bilinear(sx+u, sy+v)
						ix := gxp.atClamp(px+wx, py+wy)
						iy := gyp.atClamp(px+wx, py+wy)
						bx += ix * diff
						by += iy * diff
						e2 += diff * diff
					}
				}
				du := (c*bx - b*by) / det
				dv := (a*by - b*bx) / det
				u += du
				v += dv
				cost = e2
				if du*du+dv*dv < 1e-4 {
					break
				}
			}

			npx := float64((2 * half) * (2 * half))
			weight := 1.0 / (1.0 + cost/npx)
			for wy := -half; wy < half; wy++ {
				for wx := -half; wx < half; wx++ {
					yy := py + wy
					xx := px + wx
					idx := yy*cols + xx
					sumU.Data[idx] += weight * u
					sumV.Data[idx] += weight * v
					sumW.Data[idx] += weight
				}
			}
		}
	}

	outU := newGrid(rows, cols)
	outV := newGrid(rows, cols)
	for i := 0; i < rows*cols; i++ {
		if sumW.Data[i] > 0 {
			outU.Data[i] = sumU.Data[i] / sumW.Data[i]
			outV.Data[i] = sumV.Data[i] / sumW.Data[i]
		} else {
			outU.Data[i] = initU.Data[i]
			outV.Data[i] = initV.Data[i]
		}
	}
	return outU, outV
}

// Calc computes the dense optical flow from prev to next and returns it as a
// [FlowField] with the same dimensions as the inputs. Multi-channel inputs are
// converted to grayscale first. prev and next must be non-empty and identically
// sized.
func (d *DISOpticalFlow) Calc(prev, next *cv.Mat) *FlowField {
	if prev == nil || next == nil || prev.Empty() || next.Empty() {
		panic("video: DISOpticalFlow.Calc requires non-empty images")
	}
	if prev.Rows != next.Rows || prev.Cols != next.Cols {
		panic("video: DISOpticalFlow.Calc requires equal-sized images")
	}
	if d.PatchSize < 2 || d.PatchStride < 1 || d.GradientDescentIterations < 1 {
		panic("video: DISOpticalFlow requires PatchSize>=2, PatchStride>=1, GradientDescentIterations>=1")
	}

	// Build grayscale pyramids until the coarsest level is comfortably larger
	// than a patch.
	prevPyr := []*cv.Mat{toGray(prev)}
	nextPyr := []*cv.Mat{toGray(next)}
	minSide := d.PatchSize * 2
	for {
		last := prevPyr[len(prevPyr)-1]
		if last.Rows/2 < minSide || last.Cols/2 < minSide || last.Rows < 4 || last.Cols < 4 {
			break
		}
		prevPyr = append(prevPyr, cv.PyrDown(last))
		nextPyr = append(nextPyr, cv.PyrDown(nextPyr[len(nextPyr)-1]))
	}
	numLevels := len(prevPyr)
	fin := d.FinestScale
	if fin > numLevels-1 {
		fin = numLevels - 1
	}
	if fin < 0 {
		fin = 0
	}

	// Coarse-to-fine optimisation.
	var u, v *grid
	for l := numLevels - 1; l >= fin; l-- {
		pg := gridFromMat(prevPyr[l])
		ng := gridFromMat(nextPyr[l])
		gxp, gyp := gradients(prevPyr[l])
		rows, cols := pg.Rows, pg.Cols
		if u == nil {
			u = newGrid(rows, cols)
			v = newGrid(rows, cols)
		} else {
			u = resizeFlow(u, rows, cols, 2.0)
			v = resizeFlow(v, rows, cols, 2.0)
		}
		u, v = d.refineLevel(pg, ng, gxp, gyp, u, v)
	}

	// Upscale from the finest optimised level to full resolution.
	for l := fin - 1; l >= 0; l-- {
		u = resizeFlow(u, prevPyr[l].Rows, prevPyr[l].Cols, 2.0)
		v = resizeFlow(v, prevPyr[l].Rows, prevPyr[l].Cols, 2.0)
	}

	rows, cols := prev.Rows, prev.Cols
	flow := NewFlowField(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			flow.set(y, x, u.Data[y*cols+x], v.Data[y*cols+x])
		}
	}
	return flow
}

package optflow

import (
	cv "github.com/malcolmston/opencv"
)

// WarpByFlow warps img by the given flow field and returns a new image of the
// same size. It is the inverse-mapping remap that reconstructs the "next" frame
// from the "prev" frame given the flow computed from prev to next: each output
// pixel (y, x) is sampled from
//
//	img( x − u(y,x), y − v(y,x) )
//
// with bilinear interpolation and border replication. Concretely, if flow was
// obtained as CalcOpticalFlow…(prev, next), then WarpByFlow(prev, flow)
// approximates next.
//
// img and flow must have matching dimensions and img must be non-empty. All
// channels are warped independently. The operation is deterministic.
func WarpByFlow(img *cv.Mat, flow *FlowField) *cv.Mat {
	if img == nil || img.Empty() {
		panic("optflow: WarpByFlow requires a non-empty image")
	}
	if flow == nil || flow.Rows != img.Rows || flow.Cols != img.Cols {
		panic("optflow: WarpByFlow requires flow and image of equal size")
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels
	out := cv.NewMat(rows, cols, ch)

	// Per-channel float planes of the source for bilinear sampling.
	planes := make([]*grid, ch)
	for c := 0; c < ch; c++ {
		g := newGrid(rows, cols)
		for p := 0; p < rows*cols; p++ {
			g.Data[p] = float64(img.Data[p*ch+c])
		}
		planes[c] = g
	}

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := (y*cols + x) * 2
			u := flow.Data[i]
			v := flow.Data[i+1]
			srcX := float64(x) - u
			srcY := float64(y) - v
			di := (y*cols + x) * ch
			for c := 0; c < ch; c++ {
				out.Data[di+c] = clampU8(planes[c].bilinear(srcX, srcY))
			}
		}
	}
	return out
}

package cudaoptflow

// fgrid is a single-channel float64 image used by the Brox variational solver.
// Samples are stored row-major in data with length rows*cols.
type fgrid struct {
	rows int
	cols int
	data []float64
}

// newFgrid allocates a zero-filled fgrid.
func newFgrid(rows, cols int) *fgrid {
	return &fgrid{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

// at returns the sample at (y, x) with no bounds checking on hot paths; callers
// stay in range.
func (g *fgrid) at(y, x int) float64 {
	return g.data[y*g.cols+x]
}

// atClamp returns the sample at (y, x) with coordinates clamped to the border,
// so sampling outside the grid replicates the edge.
func (g *fgrid) atClamp(y, x int) float64 {
	if x < 0 {
		x = 0
	} else if x >= g.cols {
		x = g.cols - 1
	}
	if y < 0 {
		y = 0
	} else if y >= g.rows {
		y = g.rows - 1
	}
	return g.data[y*g.cols+x]
}

// bilinear samples the grid at fractional (fx, fy) with edge-clamped bilinear
// interpolation.
func (g *fgrid) bilinear(fx, fy float64) float64 {
	x0 := int(floor(fx))
	y0 := int(floor(fy))
	ax := fx - float64(x0)
	ay := fy - float64(y0)
	v00 := g.atClamp(y0, x0)
	v10 := g.atClamp(y0, x0+1)
	v01 := g.atClamp(y0+1, x0)
	v11 := g.atClamp(y0+1, x0+1)
	top := v00*(1-ax) + v10*ax
	bot := v01*(1-ax) + v11*ax
	return top*(1-ay) + bot*ay
}

// gradients returns central-difference spatial derivatives (d/dx, d/dy) with
// edge-clamped sampling.
func (g *fgrid) gradients() (gx, gy *fgrid) {
	gx = newFgrid(g.rows, g.cols)
	gy = newFgrid(g.rows, g.cols)
	for y := 0; y < g.rows; y++ {
		for x := 0; x < g.cols; x++ {
			gx.data[y*g.cols+x] = 0.5 * (g.atClamp(y, x+1) - g.atClamp(y, x-1))
			gy.data[y*g.cols+x] = 0.5 * (g.atClamp(y+1, x) - g.atClamp(y-1, x))
		}
	}
	return gx, gy
}

// downsample halves both dimensions with 2x2 box averaging.
func (g *fgrid) downsample() *fgrid {
	nr, nc := g.rows/2, g.cols/2
	if nr < 1 {
		nr = 1
	}
	if nc < 1 {
		nc = 1
	}
	out := newFgrid(nr, nc)
	for y := 0; y < nr; y++ {
		for x := 0; x < nc; x++ {
			sy, sx := y*2, x*2
			v := g.atClamp(sy, sx) + g.atClamp(sy, sx+1) +
				g.atClamp(sy+1, sx) + g.atClamp(sy+1, sx+1)
			out.data[y*nc+x] = v * 0.25
		}
	}
	return out
}

// upsampleTo resamples the grid to (rows, cols) with bilinear interpolation and
// multiplies every sample by scale. It is used to prolongate a coarse flow
// component to a finer level (scale accounts for the change in pixel units).
func (g *fgrid) upsampleTo(rows, cols int, scale float64) *fgrid {
	out := newFgrid(rows, cols)
	sx := float64(g.cols) / float64(cols)
	sy := float64(g.rows) / float64(rows)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			fx := (float64(x)+0.5)*sx - 0.5
			fy := (float64(y)+0.5)*sy - 0.5
			out.data[y*cols+x] = g.bilinear(fx, fy) * scale
		}
	}
	return out
}

// warp returns this grid sampled at each pixel displaced by the flow (u, v):
// out(y,x) = g(x+u, y+v), with edge-clamped bilinear interpolation. It is the
// backward warp used to align the second frame with the first.
func (g *fgrid) warp(u, v *fgrid) *fgrid {
	out := newFgrid(g.rows, g.cols)
	for y := 0; y < g.rows; y++ {
		for x := 0; x < g.cols; x++ {
			i := y*g.cols + x
			out.data[i] = g.bilinear(float64(x)+u.data[i], float64(y)+v.data[i])
		}
	}
	return out
}

// floor is a tiny dependency-free integer floor for the bilinear sampler.
func floor(x float64) float64 {
	i := float64(int(x))
	if x < 0 && i != x {
		i--
	}
	return i
}

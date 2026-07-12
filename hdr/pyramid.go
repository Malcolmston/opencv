package hdr

// Image-pyramid helpers used by Mertens exposure fusion. A Gaussian pyramid
// repeatedly blurs and halves an image; a Laplacian pyramid stores the
// band-pass difference between successive Gaussian levels so it can be blended
// and losslessly collapsed. All operations use mirror borders (via plane).

// pyrRows / pyrCols give the dimensions of pyramid level l (floor halving,
// minimum 1).
func pyrRows(rows, l int) int { return pyrDim(rows, l) }
func pyrCols(cols, l int) int { return pyrDim(cols, l) }

func pyrDim(n, l int) int {
	for i := 0; i < l; i++ {
		n = (n + 1) / 2
		if n < 1 {
			n = 1
		}
	}
	return n
}

// pyramidLevels chooses a level count from the smaller dimension so the coarsest
// level is a handful of pixels across. It is deterministic and bounded.
func pyramidLevels(rows, cols int) int {
	m := rows
	if cols < m {
		m = cols
	}
	levels := 1
	for m > 4 && levels < 8 {
		m = (m + 1) / 2
		levels++
	}
	return levels
}

// downsample blurs then decimates by two, producing a plane of size
// ((rows+1)/2, (cols+1)/2).
func downsample(p *plane) *plane {
	b := p.blur(1.0)
	nr := (p.rows + 1) / 2
	nc := (p.cols + 1) / 2
	out := newPlane(nr, nc)
	for y := 0; y < nr; y++ {
		for x := 0; x < nc; x++ {
			out.set(y, x, b.atReflect(2*y, 2*x))
		}
	}
	return out
}

// upsample expands p to the given target size by nearest-neighbour insertion
// followed by a smoothing blur, approximating the standard pyramid expand.
func upsample(p *plane, rows, cols int) *plane {
	out := newPlane(rows, cols)
	for y := 0; y < rows; y++ {
		sy := y / 2
		if sy >= p.rows {
			sy = p.rows - 1
		}
		for x := 0; x < cols; x++ {
			sx := x / 2
			if sx >= p.cols {
				sx = p.cols - 1
			}
			out.set(y, x, p.at(sy, sx))
		}
	}
	return out.blur(1.0)
}

// gaussianPyramid returns levels planes, each the blurred-and-halved version of
// the previous.
func gaussianPyramid(p *plane, levels int) []*plane {
	pyr := make([]*plane, levels)
	pyr[0] = p.clone()
	for l := 1; l < levels; l++ {
		pyr[l] = downsample(pyr[l-1])
	}
	return pyr
}

// laplacianPyramid returns levels planes: bands 0..levels-2 are the differences
// between a Gaussian level and the upsampled next level; the final band is the
// coarsest Gaussian level itself.
func laplacianPyramid(p *plane, levels int) []*plane {
	g := gaussianPyramid(p, levels)
	lap := make([]*plane, levels)
	for l := 0; l < levels-1; l++ {
		up := upsample(g[l+1], g[l].rows, g[l].cols)
		d := newPlane(g[l].rows, g[l].cols)
		for i := range d.data {
			d.data[i] = g[l].data[i] - up.data[i]
		}
		lap[l] = d
	}
	lap[levels-1] = g[levels-1]
	return lap
}

// collapsePyramid reconstructs an image from a Laplacian pyramid by repeatedly
// upsampling the coarser accumulation and adding the next finer band.
func collapsePyramid(lap []*plane) *plane {
	levels := len(lap)
	acc := lap[levels-1].clone()
	for l := levels - 2; l >= 0; l-- {
		up := upsample(acc, lap[l].rows, lap[l].cols)
		for i := range up.data {
			up.data[i] += lap[l].data[i]
		}
		acc = up
	}
	return acc
}

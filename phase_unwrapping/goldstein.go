package phase_unwrapping

import "math"

// GoldsteinBranchCut unwraps a wrapped phase map with a Goldstein-style
// branch-cut method (Goldstein, Zebker and Werner, 1988). It detects the phase
// residues, places branch cuts that join residues of opposite polarity — and any
// residue left over to the image border — and then integrates the wrapped phase
// by flood fill along paths that never cross a cut. Because no admissible path
// can encircle a lone residue, the integration is single-valued and the result
// is congruent to the input everywhere (Rewrap(result) == wrapped), with the
// unavoidable inconsistency confined to the thin cuts.
//
// Residues are paired with their nearest opposite-charge neighbour no farther
// than maxBoxRadius (in the spirit of Goldstein's growing search box); a
// non-positive maxBoxRadius removes the distance limit. On a residue-free map no
// cuts are placed and the true surface is recovered exactly, up to a global 2*pi
// constant. Input values are wrapped defensively first and the argument is not
// modified. It returns [ErrEmptyInput] for a grid smaller than 2x2.
func GoldsteinBranchCut(wrapped [][]float64, maxBoxRadius int) ([][]float64, error) {
	rows, cols, ok := gridDims(wrapped)
	if !ok || rows < 2 || cols < 2 {
		return nil, ErrEmptyInput
	}
	phase := flatten(wrapped, rows, cols)
	cuts := placeBranchCuts(phase, rows, cols, maxBoxRadius)
	u := floodFillIntegrate(phase, cuts, rows, cols)
	return unflatten(u, rows, cols), nil
}

// placeBranchCuts detects residues and returns a row-major boolean cut mask of
// length rows*cols. Each unbalanced residue is connected either to its nearest
// unbalanced opposite-charge residue (within maxBoxRadius) or, if that is farther
// than the nearest border, to the border.
func placeBranchCuts(phase []float64, rows, cols, maxBoxRadius int) []bool {
	cuts := make([]bool, rows*cols)
	if maxBoxRadius <= 0 {
		maxBoxRadius = rows + cols
	}

	type res struct {
		r, c, charge int
	}
	var list []res
	for i := 0; i < rows-1; i++ {
		for j := 0; j < cols-1; j++ {
			ch := loopCharge(phase, cols, i, j)
			if ch != 0 {
				list = append(list, res{r: i, c: j, charge: ch})
			}
		}
	}

	balanced := make([]bool, len(list))
	for i := range list {
		if balanced[i] {
			continue
		}
		ri := list[i]
		// Nearest unbalanced opposite-charge residue.
		best := -1
		bestD := math.Inf(1)
		for j := range list {
			if j == i || balanced[j] {
				continue
			}
			if sign(list[j].charge) == sign(ri.charge) {
				continue
			}
			dr := float64(list[j].r - ri.r)
			dc := float64(list[j].c - ri.c)
			d := math.Sqrt(dr*dr + dc*dc)
			if d < bestD {
				bestD = d
				best = j
			}
		}
		// Distance to the nearest border pixel.
		br, bc := nearestBorder(ri.r, ri.c, rows, cols)
		dbr := float64(br - ri.r)
		dbc := float64(bc - ri.c)
		borderD := math.Sqrt(dbr*dbr + dbc*dbc)

		if best >= 0 && bestD <= borderD && bestD <= float64(maxBoxRadius) {
			drawLine4(cuts, ri.r, ri.c, list[best].r, list[best].c, cols)
			balanced[i] = true
			balanced[best] = true
		} else {
			drawLine4(cuts, ri.r, ri.c, br, bc, cols)
			balanced[i] = true
		}
	}
	return cuts
}

// loopCharge returns the residue charge of the 2x2 loop with top-left corner
// (i, j): round of the summed wrapped gradients taken clockwise around the loop.
func loopCharge(phase []float64, cols, i, j int) int {
	p00 := phase[i*cols+j]
	p01 := phase[i*cols+j+1]
	p11 := phase[(i+1)*cols+j+1]
	p10 := phase[(i+1)*cols+j]
	s := Wrap(p01-p00) + Wrap(p11-p01) + Wrap(p10-p11) + Wrap(p00-p10)
	return int(math.Round(s / twoPi))
}

// nearestBorder returns the coordinates of the border pixel closest to (r, c).
func nearestBorder(r, c, rows, cols int) (int, int) {
	up := r
	down := rows - 1 - r
	left := c
	right := cols - 1 - c
	best := up
	br, bc := 0, c
	if down < best {
		best = down
		br, bc = rows-1, c
	}
	if left < best {
		best = left
		br, bc = r, 0
	}
	if right < best {
		br, bc = r, cols-1
	}
	return br, bc
}

// drawLine4 marks a 4-connected straight line of pixels between (r0, c0) and
// (r1, c1) on the row-major cut mask. A 4-connected (staircase) line leaves no
// diagonal gap, so a 4-connected flood fill cannot slip across it.
func drawLine4(cuts []bool, r0, c0, r1, c1, cols int) {
	r, c := r0, c0
	cuts[r*cols+c] = true
	adr := absInt(r1 - r0)
	adc := absInt(c1 - c0)
	sr := sign(r1 - r0)
	sc := sign(c1 - c0)
	err := adr - adc
	steps := adr + adc
	for k := 0; k < steps; k++ {
		if err > 0 {
			r += sr
			err -= 2 * adc
		} else {
			c += sc
			err += 2 * adr
		}
		cuts[r*cols+c] = true
	}
}

// floodFillIntegrate integrates the wrapped phase by flood fill, never crossing a
// cut pixel. Every maximal cut-free region is unwrapped from its own seed; cut
// pixels are then adjoined to an already-unwrapped neighbour. The result is
// congruent to phase at every pixel.
func floodFillIntegrate(phase []float64, cuts []bool, rows, cols int) []float64 {
	n := rows * cols
	u := make([]float64, n)
	visited := make([]bool, n)
	queue := make([]int, 0, n)

	neighbors := func(a int) [4]int {
		r := a / cols
		c := a % cols
		res := [4]int{-1, -1, -1, -1}
		if r > 0 {
			res[0] = a - cols
		}
		if r < rows-1 {
			res[1] = a + cols
		}
		if c > 0 {
			res[2] = a - 1
		}
		if c < cols-1 {
			res[3] = a + 1
		}
		return res
	}

	for seed := 0; seed < n; seed++ {
		if visited[seed] || cuts[seed] {
			continue
		}
		u[seed] = phase[seed]
		visited[seed] = true
		queue = queue[:0]
		queue = append(queue, seed)
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			for _, nb := range neighbors(cur) {
				if nb < 0 || visited[nb] || cuts[nb] {
					continue
				}
				u[nb] = u[cur] + Wrap(phase[nb]-phase[cur])
				visited[nb] = true
				queue = append(queue, nb)
			}
		}
	}

	// Adjoin the cut pixels (and anything unreachable) to a neighbour.
	for {
		changed := false
		for a := 0; a < n; a++ {
			if visited[a] {
				continue
			}
			for _, nb := range neighbors(a) {
				if nb < 0 || !visited[nb] {
					continue
				}
				u[a] = u[nb] + Wrap(phase[a]-phase[nb])
				visited[a] = true
				changed = true
				break
			}
		}
		if !changed {
			break
		}
	}
	for a := 0; a < n; a++ {
		if !visited[a] {
			u[a] = phase[a]
		}
	}
	return u
}

// sign returns -1, 0 or 1 according to the sign of x.
func sign(x int) int {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}

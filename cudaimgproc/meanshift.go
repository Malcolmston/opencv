package cudaimgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MeanShiftFiltering performs mean-shift filtering on a colour image, mirroring
// cuda::meanShiftFiltering. Each pixel is iteratively shifted towards the mean
// of neighbours that lie within a spatial window of radius sp and whose colour
// is within sr of the running mean; the pixel is then replaced by the colour it
// converged to. Iteration stops after maxIter passes or when the update falls
// below eps. src must have 3 or 4 channels (the colour is taken from the first
// three; a fourth alpha channel is preserved). The trailing Stream argument is
// accepted and ignored.
func MeanShiftFiltering(src GpuMat, sp int, sr float64, maxIter int, eps float64, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("MeanShiftFiltering")
	dst, _, _ := meanShift(m, sp, sr, maxIter, eps)
	return wrap(dst)
}

// MeanShiftProc performs mean-shift filtering and additionally returns the
// per-pixel converged spatial coordinates, mirroring cuda::meanShiftProc. The
// first result is the filtered colour image (as [MeanShiftFiltering]); the
// second is a two-channel image whose channels hold the converged (x, y)
// position of each pixel, clamped to 8 bits (so it is exact for images up to
// 255×255). src must have 3 or 4 channels. The trailing Stream argument is
// accepted and ignored.
func MeanShiftProc(src GpuMat, sp int, sr float64, maxIter int, eps float64, streams ...Stream) (dstr, dstsp GpuMat) {
	_ = firstStream(streams)
	m := src.requireHost("MeanShiftProc")
	filtered, cx, cy := meanShift(m, sp, sr, maxIter, eps)
	coord := cv.NewMat(m.Rows, m.Cols, 2)
	for i := 0; i < m.Total(); i++ {
		coord.Data[i*2+0] = clampU8(float64(cx[i]))
		coord.Data[i*2+1] = clampU8(float64(cy[i]))
	}
	return wrap(filtered), wrap(coord)
}

// MeanShiftSegmentation segments a colour image with the mean-shift procedure,
// mirroring cuda::meanShiftSegmentation. After mean-shift filtering, spatially
// adjacent pixels whose filtered colours differ by less than sr are merged into
// regions; regions smaller than minsize pixels are absorbed into an adjacent
// region, and every pixel is finally painted with the mean colour of its
// region. src must have 3 or 4 channels. The trailing Stream argument is
// accepted and ignored.
func MeanShiftSegmentation(src GpuMat, sp int, sr float64, minsize, maxIter int, eps float64, streams ...Stream) GpuMat {
	_ = firstStream(streams)
	m := src.requireHost("MeanShiftSegmentation")
	filtered, _, _ := meanShift(m, sp, sr, maxIter, eps)
	rows, cols := filtered.Rows, filtered.Cols
	n := rows * cols

	uf := newUnionFind(n)
	// filtered may be 3- or 4-channel; index colour using its own channel count.
	ch := filtered.Channels
	colorDistCh := func(i, j int) float64 {
		bi, bj := i*ch, j*ch
		dr := float64(filtered.Data[bi+0]) - float64(filtered.Data[bj+0])
		dg := float64(filtered.Data[bi+1]) - float64(filtered.Data[bj+1])
		db := float64(filtered.Data[bi+2]) - float64(filtered.Data[bj+2])
		return math.Sqrt(dr*dr + dg*dg + db*db)
	}
	// Merge 4-connected neighbours below the colour radius.
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if x+1 < cols && colorDistCh(i, i+1) < sr {
				uf.union(i, i+1)
			}
			if y+1 < rows && colorDistCh(i, i+cols) < sr {
				uf.union(i, i+cols)
			}
		}
	}

	// Absorb undersized regions into a neighbouring region.
	sizes := make(map[int]int)
	for i := 0; i < n; i++ {
		sizes[uf.find(i)]++
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			r := uf.find(i)
			if sizes[r] >= minsize {
				continue
			}
			// Attach to the neighbour whose region is largest.
			best := -1
			bestSize := -1
			neighbours := []int{}
			if x+1 < cols {
				neighbours = append(neighbours, i+1)
			}
			if x-1 >= 0 {
				neighbours = append(neighbours, i-1)
			}
			if y+1 < rows {
				neighbours = append(neighbours, i+cols)
			}
			if y-1 >= 0 {
				neighbours = append(neighbours, i-cols)
			}
			for _, nb := range neighbours {
				nr := uf.find(nb)
				if nr == r {
					continue
				}
				if sizes[nr] > bestSize {
					bestSize = sizes[nr]
					best = nb
				}
			}
			if best >= 0 {
				ra, rb := uf.find(i), uf.find(best)
				if ra != rb {
					sz := sizes[ra] + sizes[rb]
					uf.union(i, best)
					root := uf.find(i)
					delete(sizes, ra)
					delete(sizes, rb)
					sizes[root] = sz
				}
			}
		}
	}

	// Paint each region with its mean colour.
	type acc struct {
		sum   [3]float64
		count int
	}
	regions := make(map[int]*acc)
	for i := 0; i < n; i++ {
		r := uf.find(i)
		a := regions[r]
		if a == nil {
			a = &acc{}
			regions[r] = a
		}
		b := i * ch
		a.sum[0] += float64(filtered.Data[b+0])
		a.sum[1] += float64(filtered.Data[b+1])
		a.sum[2] += float64(filtered.Data[b+2])
		a.count++
	}
	dst := cv.NewMat(rows, cols, filtered.Channels)
	for i := 0; i < n; i++ {
		r := uf.find(i)
		a := regions[r]
		b := i * ch
		dst.Data[b+0] = clampU8(a.sum[0]/float64(a.count) + 0.5)
		dst.Data[b+1] = clampU8(a.sum[1]/float64(a.count) + 0.5)
		dst.Data[b+2] = clampU8(a.sum[2]/float64(a.count) + 0.5)
		if ch == 4 {
			dst.Data[b+3] = filtered.Data[b+3]
		}
	}
	return wrap(dst)
}

// meanShift runs the mean-shift procedure and returns the filtered image plus
// the converged x and y coordinate of every pixel (length rows*cols). It panics
// unless src has 3 or 4 channels.
func meanShift(m *cv.Mat, sp int, sr float64, maxIter int, eps float64) (dst *cv.Mat, cx, cy []int) {
	if m.Channels != 3 && m.Channels != 4 {
		panic("cudaimgproc: mean shift requires 3 or 4 channels")
	}
	if sp < 1 {
		sp = 1
	}
	if maxIter < 1 {
		maxIter = 1
	}
	rows, cols, ch := m.Rows, m.Cols, m.Channels
	dst = cv.NewMat(rows, cols, ch)
	cx = make([]int, rows*cols)
	cy = make([]int, rows*cols)
	sr2 := sr * sr
	at := func(y, x, c int) float64 { return float64(m.Data[(y*cols+x)*ch+c]) }
	for y0 := 0; y0 < rows; y0++ {
		for x0 := 0; x0 < cols; x0++ {
			x, yy := x0, y0
			cr := at(y0, x0, 0)
			cg := at(y0, x0, 1)
			cb := at(y0, x0, 2)
			for it := 0; it < maxIter; it++ {
				var sumX, sumY, sumR, sumG, sumB float64
				count := 0
				minX := maxInt(0, x-sp)
				maxX := minInt(cols-1, x+sp)
				minY := maxInt(0, yy-sp)
				maxY := minInt(rows-1, yy+sp)
				for ny := minY; ny <= maxY; ny++ {
					for nx := minX; nx <= maxX; nx++ {
						r := at(ny, nx, 0)
						g := at(ny, nx, 1)
						b := at(ny, nx, 2)
						dr := r - cr
						dg := g - cg
						db := b - cb
						if dr*dr+dg*dg+db*db > sr2 {
							continue
						}
						sumX += float64(nx)
						sumY += float64(ny)
						sumR += r
						sumG += g
						sumB += b
						count++
					}
				}
				if count == 0 {
					break
				}
				newX := int(math.Round(sumX / float64(count)))
				newY := int(math.Round(sumY / float64(count)))
				newR := sumR / float64(count)
				newG := sumG / float64(count)
				newB := sumB / float64(count)
				shift := math.Hypot(float64(newX-x), float64(newY-yy)) +
					math.Abs(newR-cr) + math.Abs(newG-cg) + math.Abs(newB-cb)
				x, yy = newX, newY
				cr, cg, cb = newR, newG, newB
				if shift <= eps {
					break
				}
			}
			i := y0*cols + x0
			b := i * ch
			dst.Data[b+0] = clampU8(cr + 0.5)
			dst.Data[b+1] = clampU8(cg + 0.5)
			dst.Data[b+2] = clampU8(cb + 0.5)
			if ch == 4 {
				dst.Data[b+3] = m.Data[b+3]
			}
			cx[i] = x
			cy[i] = yy
		}
	}
	return dst, cx, cy
}

// unionFind is a disjoint-set structure used by [MeanShiftSegmentation] to
// merge pixels into regions.
type unionFind struct {
	parent []int
	rank   []int
}

func newUnionFind(n int) *unionFind {
	u := &unionFind{parent: make([]int, n), rank: make([]int, n)}
	for i := range u.parent {
		u.parent[i] = i
	}
	return u
}

func (u *unionFind) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

func (u *unionFind) union(a, b int) {
	ra, rb := u.find(a), u.find(b)
	if ra == rb {
		return
	}
	if u.rank[ra] < u.rank[rb] {
		ra, rb = rb, ra
	}
	u.parent[rb] = ra
	if u.rank[ra] == u.rank[rb] {
		u.rank[ra]++
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
